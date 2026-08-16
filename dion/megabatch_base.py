import math
from numbers import Real
import torch
import torch.distributed as dist
from collections import OrderedDict, defaultdict
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Literal, Optional, Tuple, Union

from .newton_schulz_triton import (
    TRITON_AVAILABLE,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from .polar_express import polar_express, polar_express_triton
from .opt_utils import AsyncRuntime, AsyncTask, as_scalar_tensor, to_local
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


class DistributedOrthoBase(Optimizer):
    """
    Shared base class for distributed orthogonalization optimizers (NorMuon, Dion2).
    Handles distributed setup, Newton-Schulz config, step orchestration,
    shard detection, and scalar optimizer tasks (Lion, AdamW).

    Subclasses must implement ``_create_ortho_tasks()``.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]],
        algo_name: str,
        defaults: dict,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
        state_initialization: Literal["eager", "deferred"] = "eager",
    ):
        if state_initialization not in ("eager", "deferred"):
            raise ValueError(
                "state_initialization must be either 'eager' or 'deferred', "
                f"got {state_initialization!r}."
            )
        self._state_initialization = state_initialization
        self._state_lifecycle = "deferred" if state_initialization == "deferred" else "initializing"
        super().__init__(params, defaults)
        self._algo_name = algo_name

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, but got {distributed_mesh.ndim}D. "
                    f"For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. "
                f"Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Orthogonalization function configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_gram_newton_schulz:
            try:
                from gram_newton_schulz import GramNewtonSchulz
            except ImportError:
                raise ImportError(
                    "use_gram_newton_schulz=True requires the optional 'gram-newton-schulz' "
                    'package, which is not installed. Install it with: pip install -e ".[gns]" '
                    "(or pip install gram-newton-schulz)."
                )
            use_polar_express = True
            _gns = GramNewtonSchulz(
                ns_use_kernels=use_triton,
                use_gram_newton_schulz=True,
                gram_newton_schulz_reset_iterations=[2],
                # Some compiler crashes were observed with mode="reduce-overhead" when we also compile the entire optimizer step.
                compile_kwargs=dict(fullgraph=True, mode="default"),
            )
            self._newton_schulz_func = lambda X, epsilon=None: _gns(X)
        elif use_polar_express and use_triton:
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "use_triton=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
            self._newton_schulz_func = polar_express_triton
        elif use_polar_express:
            self._newton_schulz_func = polar_express
        elif use_triton:
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "use_triton=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # Eagerly materialize optimizer state for every parameter, including
        # those that may never receive a gradient (frozen matrices, or MoE
        # experts that get no tokens on a given rank/step). State is otherwise
        # created lazily only for params with a gradient, so the set of keys in
        # state_dict() can differ across ranks (rank-asymmetric gradients) or
        # between save and resume. Distributed checkpointing (DCP) gathers
        # optimizer state collectively and assumes a rank-symmetric key set, so
        # the lazy path can mismatch or hang. Pre-populating keeps state_dict()
        # complete and identical across ranks. The per-step path reuses the same
        # (now non-empty) state, so this changes nothing numerically.
        # Ported from InternLM/xtuner v1/optim/muon.py (the equivalent __init__
        # loop), generalized here to the whole DistributedOrthoBase family via
        # the (possibly overridden) _get_or_initialize_state.
        self._state_prepopulated = False
        # Persistent per-group hyperparameter device tensors, keyed by (group index, name).
        # Held here (not only in the group) so the tensor identity survives a caller
        # reassigning group["lr"]; see _ensure_hyperparam_tensor. Which names are carried
        # this way is per group and latching; see _live_hyperparams.
        self._hyperparam_tensors: dict = {}
        self._live_hyperparams_by_group: dict = {}
        if state_initialization == "eager":
            try:
                for group in self.param_groups:
                    self._prepopulate_group_state(group)
                self._sync_hyperparam_tensors()
                self._state_prepopulated = True
                self._validate_materialized_state()
            except Exception:
                self._state_lifecycle = "failed"
                raise

            self._state_lifecycle = "materialized"

    @property
    def is_state_materialized(self) -> bool:
        return self._state_lifecycle == "materialized"

    def materialize_state(self) -> None:
        if self._state_initialization != "deferred":
            raise RuntimeError("materialize_state() is only valid for deferred optimizers.")
        if self._state_lifecycle != "deferred":
            raise RuntimeError(
                "materialize_state() requires an unmaterialized deferred optimizer; "
                f"current lifecycle is {self._state_lifecycle!r}."
            )

        self._state_lifecycle = "materializing"
        try:
            if self.state:
                raise RuntimeError("Cannot materialize deferred optimizer state over partial or stale state.")
            if self._hyperparam_tensors or self._live_hyperparams_by_group:
                raise RuntimeError("Cannot materialize with partially initialized hyperparameter tensors or caches.")

            parameter_ids: set[int] = set()
            for group_index, group in enumerate(self.param_groups):
                group_parameters = group["params"]
                if not group_parameters:
                    raise RuntimeError(f"Cannot materialize empty parameter group {group_index}.")
                for parameter in group_parameters:
                    parameter_id = id(parameter)
                    if parameter_id in parameter_ids:
                        raise RuntimeError("Cannot materialize duplicate parameter identities.")
                    if parameter.grad is not None:
                        raise RuntimeError("Cannot materialize optimizer state after gradients have been assigned.")
                    parameter_ids.add(parameter_id)

            for group in self.param_groups:
                self._prepopulate_group_state(group)
            self._sync_hyperparam_tensors()
            self._state_prepopulated = True
            self._validate_materialized_state()
        except Exception:
            self._state_lifecycle = "failed"
            raise

        self._state_lifecycle = "materialized"

    def _expected_state_shapes(
        self,
        parameter: Tensor,
        algorithm: str,
    ) -> dict[str, tuple[torch.Size, torch.Size, torch.dtype, bool]]:
        local_parameter = to_local(parameter)
        parameter_shape = parameter.shape
        local_shape = local_parameter.shape
        distributed = isinstance(parameter, DTensor)
        if algorithm == self._algo_name or algorithm == "lion":
            expected = {
                "momentum": (parameter_shape, local_shape, parameter.dtype, distributed),
            }
            if algorithm == "normuon":
                expected["variance_neuron"] = (
                    parameter_shape[:-1] + (1,),
                    local_shape[:-1] + (1,),
                    parameter.dtype,
                    distributed,
                )
            return expected
        if algorithm == "adamw":
            return {
                "momentum": (parameter_shape, local_shape, parameter.dtype, distributed),
                "variance": (parameter_shape, local_shape, parameter.dtype, distributed),
                "step_dev": (torch.Size([]), torch.Size([]), torch.float32, False),
            }
        raise RuntimeError(f"Unsupported optimizer state algorithm {algorithm!r}.")

    def _expected_state_fields(self, algorithm: str) -> set[str]:
        if algorithm == self._algo_name or algorithm == "lion":
            expected = {"momentum"}
            if algorithm == "normuon":
                expected.add("variance_neuron")
            return expected
        if algorithm == "adamw":
            return {"momentum", "variance", "step_dev"}
        raise RuntimeError(f"Unsupported optimizer state algorithm {algorithm!r}.")

    def _expected_state_metadata(
        self,
        parameter: Tensor,
        algorithm: str,
    ) -> dict[str, tuple[torch.Size, torch.dtype, bool]]:
        distributed = isinstance(parameter, DTensor)
        if algorithm == self._algo_name or algorithm == "lion":
            expected = {
                "momentum": (parameter.shape, parameter.dtype, distributed),
            }
            if algorithm == "normuon":
                expected["variance_neuron"] = (
                    parameter.shape[:-1] + (1,),
                    parameter.dtype,
                    distributed,
                )
            return expected
        if algorithm == "adamw":
            return {
                "momentum": (parameter.shape, parameter.dtype, distributed),
                "variance": (parameter.shape, parameter.dtype, distributed),
                "step_dev": (torch.Size([]), torch.float32, False),
            }
        raise RuntimeError(f"Unsupported optimizer state algorithm {algorithm!r}.")

    def _validate_state_entry(
        self,
        parameter: Tensor,
        algorithm: str,
        state: dict,
        *,
        validate_device: bool,
        allow_legacy_adamw_step: bool = False,
    ) -> None:
        expected = self._expected_state_shapes(parameter, algorithm)
        expected_fields = set(expected)
        if allow_legacy_adamw_step and algorithm == "adamw":
            expected_fields.remove("step_dev")
        if set(state) != set(expected) and set(state) != expected_fields:
            raise RuntimeError(
                f"Invalid optimizer state fields for algorithm {algorithm!r}: "
                f"expected {sorted(expected)}, got {sorted(state)}."
            )

        parameter_local = to_local(parameter)
        for name, (global_shape, local_shape, dtype, distributed) in expected.items():
            if name not in state:
                continue
            value = state[name]
            if not isinstance(value, Tensor):
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: expected Tensor, got {type(value).__name__}."
                )
            if isinstance(value, DTensor) != distributed:
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: DTensor identity does not match the parameter."
                )
            if value.shape != global_shape or value.dtype != dtype:
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: expected shape {tuple(global_shape)} and "
                    f"dtype {dtype}, got shape {tuple(value.shape)} and dtype {value.dtype}."
                )

            value_local = to_local(value)
            if value_local.shape != local_shape:
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: expected local shape {tuple(local_shape)}, "
                    f"got {tuple(value_local.shape)}."
                )
            if validate_device and value_local.device != parameter_local.device:
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: expected device {parameter_local.device}, "
                    f"got {value_local.device}."
                )
            if distributed and (value.placements != parameter.placements or value.device_mesh != parameter.device_mesh):
                raise RuntimeError(
                    f"Invalid optimizer state {algorithm}.{name}: DTensor mesh or placements do not match the parameter."
                )

    def _validate_materialized_state(self) -> None:
        parameters, algorithms, expected_cache_keys = self._validate_materialized_structure()
        for parameter_id, parameter in parameters.items():
            self._validate_state_entry(
                parameter,
                algorithms[parameter_id],
                self.state[parameter],
                validate_device=True,
            )

        for index, group in enumerate(self.param_groups):
            if not group["params"]:
                continue
            for name in self._live_hyperparams_by_group[index]:
                cached = self._hyperparam_tensors[(index, name)]
                local_device = to_local(group["params"][0]).device
                if cached.shape != torch.Size([]) or cached.dtype != torch.float32 or cached.device != local_device:
                    raise RuntimeError(f"Invalid optimizer state: parameter group {index} has malformed {name} cache.")
                value = group.get(name)
                if value is not cached and not isinstance(value, (int, float, Tensor)):
                    raise RuntimeError(
                        f"Invalid optimizer state: parameter group {index} has invalid {name} value."
                    )
                if isinstance(value, Tensor) and value.shape != torch.Size([]):
                    raise RuntimeError(
                        f"Invalid optimizer state: parameter group {index} has non-scalar {name}."
                    )
        if set(self._hyperparam_tensors) != expected_cache_keys:
            raise RuntimeError("Invalid optimizer state: hyperparameter cache keys do not match parameter groups.")

    def _validate_materialized_structure(
        self,
    ) -> tuple[dict[int, Tensor], dict[int, str], set[tuple[int, str]]]:
        parameters: dict[int, Tensor] = {}
        algorithms: dict[int, str] = {}
        for group_index, group in enumerate(self.param_groups):
            group_parameters = group["params"]
            step = group.get("step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or not 0 <= step <= torch.iinfo(torch.int64).max
            ):
                raise RuntimeError(f"Invalid optimizer state: parameter group {group_index} has invalid step {step!r}.")
            algorithm = group.get("algorithm")
            for parameter in group_parameters:
                parameter_id = id(parameter)
                if parameter_id in algorithms and algorithms[parameter_id] != algorithm:
                    raise RuntimeError("Invalid optimizer state: one parameter has multiple algorithms.")
                algorithms[parameter_id] = algorithm
                parameters[parameter_id] = parameter
                self._validate_parameter_distribution(parameter, algorithm, group)

        parameter_ids = set(parameters)
        state_ids = {id(parameter) for parameter in self.state}
        missing_count = len(parameter_ids - state_ids)
        orphan_count = len(state_ids - parameter_ids)
        if missing_count or orphan_count or len(self.state) != len(parameters):
            raise RuntimeError(
                "Invalid optimizer state: state keys do not exactly cover "
                "parameter-group identities "
                f"(missing={missing_count}, orphan={orphan_count})."
            )
        for parameter_id, parameter in parameters.items():
            algorithm = algorithms[parameter_id]
            state = self.state[parameter]
            if not isinstance(state, dict):
                raise RuntimeError(
                    f"Invalid optimizer state entry for algorithm {algorithm!r}: expected dict."
                )
            expected_fields = self._expected_state_fields(algorithm)
            if set(state) != expected_fields:
                raise RuntimeError(
                    f"Invalid optimizer state fields for algorithm {algorithm!r}: "
                    f"expected {sorted(expected_fields)}, got {sorted(state)}."
                )
            expected_metadata = self._expected_state_metadata(parameter, algorithm)
            parameter_local_device: torch.device | None = None
            for name, value in state.items():
                if not isinstance(value, Tensor):
                    raise RuntimeError(
                        f"Invalid optimizer state {algorithm}.{name}: "
                        f"expected Tensor, got {type(value).__name__}."
                    )
                expected_shape, expected_dtype, distributed = expected_metadata[name]
                if isinstance(value, DTensor) != distributed:
                    raise RuntimeError(
                        f"Invalid optimizer state {algorithm}.{name}: "
                        "DTensor identity does not match the parameter."
                    )
                if value.shape != expected_shape:
                    raise RuntimeError(
                        f"Invalid optimizer state {algorithm}.{name}: "
                        f"expected shape {tuple(expected_shape)}, got {tuple(value.shape)}."
                    )
                if value.dtype != expected_dtype:
                    raise RuntimeError(
                        f"Invalid optimizer state {algorithm}.{name}: "
                        f"expected dtype {expected_dtype}, got {value.dtype}."
                    )
                if value.device != parameter.device:
                    raise RuntimeError(
                        f"Invalid optimizer state {algorithm}.{name}: "
                        f"expected device {parameter.device}, got {value.device}."
                    )
                if distributed:
                    if parameter_local_device is None:
                        parameter_local_device = parameter.to_local().device
                    value_local_device = value.to_local().device
                    if value_local_device != parameter_local_device:
                        raise RuntimeError(
                            f"Invalid optimizer state {algorithm}.{name}: "
                            f"expected local device {parameter_local_device}, "
                            f"got {value_local_device}."
                        )
                    parameter_group_names = tuple(
                        parameter.device_mesh.get_group(mesh_dim).group_name
                        for mesh_dim in range(parameter.device_mesh.ndim)
                    )
                    value_group_names = tuple(
                        value.device_mesh.get_group(mesh_dim).group_name
                        for mesh_dim in range(value.device_mesh.ndim)
                    )
                    if value_group_names != parameter_group_names:
                        raise RuntimeError(
                            f"Invalid optimizer state {algorithm}.{name}: "
                            "DTensor device mesh does not match the parameter."
                        )
                    if value.placements != parameter.placements:
                        raise RuntimeError(
                            f"Invalid optimizer state {algorithm}.{name}: "
                            "DTensor placements do not match the parameter."
                        )

        expected_cache_keys: set[tuple[int, str]] = set()
        if set(self._live_hyperparams_by_group) != set(range(len(self.param_groups))):
            raise RuntimeError("Invalid optimizer state: live hyperparameter groups are incomplete.")
        for index, group in enumerate(self.param_groups):
            names = self._live_hyperparams_by_group[index]
            if names not in (("lr",), ("lr", "weight_decay")):
                raise RuntimeError(
                    f"Invalid optimizer state: parameter group {index} has invalid live hyperparameters {names!r}."
                )
            if not group["params"]:
                continue
            for name in names:
                key = (index, name)
                expected_cache_keys.add(key)
                cached = self._hyperparam_tensors.get(key)
                if not isinstance(cached, Tensor):
                    raise RuntimeError(f"Invalid optimizer state: parameter group {index} is missing its {name} cache.")
                parameter_device = group["params"][0].device
                if (
                    cached.shape != torch.Size([])
                    or cached.dtype != torch.float32
                    or cached.device != parameter_device
                ):
                    raise RuntimeError(
                        f"Invalid optimizer state: parameter group {index} has malformed {name} cache."
                    )
        if set(self._hyperparam_tensors) != expected_cache_keys:
            raise RuntimeError("Invalid optimizer state: hyperparameter cache keys do not match parameter groups.")

        return parameters, algorithms, expected_cache_keys

    def _validate_parameter_distribution(
        self,
        parameter: Tensor,
        algorithm: str,
        group: dict,
    ) -> None:
        if algorithm != self._algo_name or not isinstance(parameter, DTensor):
            return
        if not isinstance(self._distributed_mesh, DeviceMesh):
            raise RuntimeError(
                "Orthogonal DTensor parameters require an optimizer DeviceMesh."
            )
        if not any(
            parameter.device_mesh.get_group(mesh_dim).group_name
            == self._process_group.group_name
            for mesh_dim in range(parameter.device_mesh.ndim)
        ):
            raise RuntimeError(
                "Orthogonal DTensor parameter device mesh is not bound to the optimizer process group."
            )
        if any(placement.is_partial() for placement in parameter.placements):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support Partial DTensor placements."
            )
        self._get_shard_info(parameter, group)

    def _validate_group_envelope(
        self,
        loaded_groups: object,
    ) -> tuple[list[tuple[Tensor, str, object]], list[object]]:
        if not isinstance(loaded_groups, list):
            raise RuntimeError("Invalid optimizer state_dict parameter-group structure.")
        if len(loaded_groups) != len(self.param_groups):
            raise RuntimeError("Invalid optimizer state_dict parameter-group count.")

        loaded_parameter_keys: list[object] = []
        current_parameter_ids: list[int] = []
        expected_entries: list[tuple[Tensor, str, object]] = []
        for group_index, (current_group, loaded_group) in enumerate(zip(self.param_groups, loaded_groups)):
            if not isinstance(loaded_group, dict):
                raise RuntimeError(f"Invalid optimizer state_dict parameter group {group_index}.")
            current_parameters = current_group["params"]
            loaded_parameters = loaded_group.get("params")
            if not isinstance(loaded_parameters, list) or len(loaded_parameters) != len(current_parameters):
                raise RuntimeError(f"Invalid optimizer state_dict parameter count for group {group_index}.")
            algorithm = loaded_group.get("algorithm")
            if algorithm != current_group.get("algorithm"):
                raise RuntimeError(f"Invalid optimizer state_dict algorithm for group {group_index}.")

            missing_fields = set(current_group) - set(loaded_group)
            if missing_fields:
                raise RuntimeError(
                    f"Invalid optimizer state_dict group {group_index}: missing fields {sorted(missing_fields)}."
                )
            step = loaded_group["step"]
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or not 0 <= step <= torch.iinfo(torch.int64).max
            ):
                raise RuntimeError(f"Invalid optimizer state_dict step for group {group_index}: {step!r}.")

            for name, current_value in current_group.items():
                if name in ("params", "algorithm", "step"):
                    continue
                value = loaded_group[name]
                if isinstance(current_value, bool):
                    if not isinstance(value, bool):
                        raise RuntimeError(
                            f"Invalid optimizer state_dict field {name!r} for group {group_index}."
                        )
                elif isinstance(current_value, Tensor) or (
                    isinstance(current_value, Real) and not isinstance(current_value, bool)
                ):
                    if isinstance(value, Tensor):
                        if (
                            isinstance(value, DTensor)
                            or value.shape != torch.Size([])
                            or value.dtype == torch.bool
                            or value.is_complex()
                            or not torch.isfinite(value).item()
                        ):
                            raise RuntimeError(
                                f"Invalid optimizer state_dict scalar {name!r} for group {group_index}."
                            )
                        scalar = float(value)
                    elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
                        scalar = float(value)
                    else:
                        raise RuntimeError(
                            f"Invalid optimizer state_dict scalar {name!r} for group {group_index}."
                        )

                    if name == "fraction" and not 0.0 < scalar <= 1.0:
                        raise RuntimeError(
                            f"Invalid optimizer state_dict fraction for group {group_index}: {scalar!r}."
                        )
                    if name in ("lr", "mu", "muon_beta2", "beta1", "beta2", "epsilon") and scalar < 0.0:
                        raise RuntimeError(
                            f"Invalid optimizer state_dict scalar {name!r} for group {group_index}: {scalar!r}."
                        )
                elif name == "adjust_lr" and value not in ("spectral_norm", "rms_norm", None):
                    raise RuntimeError(f"Invalid optimizer state_dict adjust_lr for group {group_index}.")
                elif name == "selection_scope" and value not in ("local", "global"):
                    raise RuntimeError(f"Invalid optimizer state_dict selection_scope for group {group_index}.")
                elif current_value is not None and not isinstance(value, type(current_value)):
                    raise RuntimeError(
                        f"Invalid optimizer state_dict field {name!r} for group {group_index}."
                    )

            loaded_parameter_keys.extend(loaded_parameters)
            current_parameter_ids.extend(id(parameter) for parameter in current_parameters)
            expected_entries.extend(zip(current_parameters, [algorithm] * len(current_parameters), loaded_parameters))

        try:
            set(loaded_parameter_keys)
        except TypeError as error:
            raise RuntimeError("Invalid optimizer state_dict parameter keys.") from error

        parameter_id_by_key: dict[object, int] = {}
        key_by_parameter_id: dict[int, object] = {}
        for parameter_id, loaded_key in zip(current_parameter_ids, loaded_parameter_keys):
            if loaded_key in parameter_id_by_key and parameter_id_by_key[loaded_key] != parameter_id:
                raise RuntimeError("Invalid optimizer state_dict parameter-key equivalence.")
            if parameter_id in key_by_parameter_id and key_by_parameter_id[parameter_id] != loaded_key:
                raise RuntimeError("Invalid optimizer state_dict parameter-key equivalence.")
            parameter_id_by_key[loaded_key] = parameter_id
            key_by_parameter_id[parameter_id] = loaded_key

        return expected_entries, loaded_parameter_keys

    def _validate_loaded_state_dict(self, state_dict: dict) -> None:
        loaded_state = state_dict.get("state")
        if not isinstance(loaded_state, dict):
            raise RuntimeError("Invalid optimizer state_dict state structure.")
        expected_entries, loaded_parameter_keys = self._validate_group_envelope(state_dict.get("param_groups"))
        state_keys = set(loaded_state)
        parameter_keys = set(loaded_parameter_keys)
        missing_count = len(parameter_keys - state_keys)
        orphan_count = len(state_keys - parameter_keys)
        if missing_count or orphan_count:
            raise RuntimeError(
                "Invalid optimizer state_dict parameter coverage: "
                f"missing={missing_count}, orphan={orphan_count}."
            )
        for parameter, algorithm, loaded_key in expected_entries:
            self._validate_state_entry(
                parameter,
                algorithm,
                loaded_state[loaded_key],
                validate_device=False,
                allow_legacy_adamw_step=True,
            )

    def _prepopulate_group_state(self, group: dict) -> None:
        algo = group["algorithm"]
        for p in group["params"]:
            self._get_or_initialize_state(p, algo)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        # Keep the pre-population invariant for groups added after construction
        # so state_dict() stays complete and rank-symmetric. The guard skips the
        # add_param_group calls that Optimizer.__init__ makes before this class
        # finishes setup; the __init__ loop above pre-populates those groups.
        if getattr(self, "_state_prepopulated", False):
            index = len(self.param_groups) - 1
            for name in self._live_hyperparams(index):
                self._ensure_hyperparam_tensor(index, name)
            self._prepopulate_group_state(self.param_groups[-1])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if self._state_lifecycle == "failed":
            raise RuntimeError("Optimizer state materialization failed; rebuild the optimizer before step().")
        if not self.is_state_materialized:
            raise RuntimeError(
                "Optimizer state must be materialized successfully before step()."
            )
        try:
            self._validate_materialized_structure()
        except Exception:
            self._state_lifecycle = "failed"
            raise

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # The LR is carried as a device tensor the kernels read directly (see
        # _ensure_hyperparam_tensor); a scheduler updates it in place outside the graph and every
        # replay re-reads the live value. Push in anything a caller assigned to
        # group["lr"] as a plain float since the last step. A no-op (and so safe under
        # graph capture) once the tensor is in place, which the wrapper guarantees by
        # calling this itself before capturing.
        self._sync_hyperparam_tensors()

        ortho_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == self._algo_name:
                ortho_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        ortho_tasks = self._create_ortho_tasks(ortho_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(ortho_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Get optimizer state, or lazy-initialize if it doesn't exist."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        if algo == "adamw" and "step_dev" not in state:
            # Device step counter for the capturable fused AdamW. The step has to live
            # on-device for bias correction to advance under CUDA-graph replay, where
            # step()'s host-side group["step"] increment does not run. Always fp32: the
            # fused kernel requires it, and a param-dtype (bf16) counter would stop
            # incrementing at 256. Initialized here rather than lazily at first use so it
            # is present in state_dict() from construction, keeping the key set complete
            # and rank-symmetric for distributed checkpointing (see __init__).
            state["step_dev"] = torch.zeros(
                (), dtype=torch.float32, device=to_local(param).device
            )
        return state

    def _resolve_num_heads(self, group: dict) -> Optional[int]:
        """Validate the ``num_heads`` option on a param group.

        Returns the group's ``num_heads`` when it is set and > 1 (the only case
        that actually triggers the per-head code path). Returns ``None`` when
        ``num_heads`` is unset or equals 1 (both are no-ops). Raises
        ``ValueError`` for invalid values or incompatible combinations.
        """
        num_heads = group.get("num_heads")
        if num_heads is None:
            return None
        # bool is a subclass of int in Python; reject it explicitly.
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError(
                f"num_heads must be a positive integer if set, got {num_heads!r}."
            )
        if num_heads == 1:
            return None
        if group.get("flatten"):
            raise ValueError(
                "num_heads > 1 is incompatible with flatten=True: flattening "
                "the per-head 3D view collapses heads back into a single 2D "
                "matrix, defeating per-head Newton-Schulz."
            )
        return num_heads

    def _resolve_split_sizes(self, group: dict) -> Optional[Tuple[int, ...]]:
        """Validate the ``split_sizes`` option on a param group.

        ``split_sizes`` partitions dim 0 of a 2D weight into row blocks (e.g. a
        fused QKV projection into its Q, K, and V blocks, which may have
        unequal sizes under GQA). Newton-Schulz and the learning-rate
        adjustment then run independently per block, matching the update that
        separate per-block parameters would receive, while the parameter
        itself stays fused for a single wide GEMM in the model.

        Returns the validated sizes as a tuple, or ``None`` when the option is
        unset. Raises ``ValueError`` for invalid values or incompatible
        combinations. The sizes must sum to dim 0 of every parameter in the
        group; that is checked at task-creation time when shapes are known.
        """
        split_sizes = group.get("split_sizes")
        if split_sizes is None:
            return None
        if not isinstance(split_sizes, (tuple, list)) or len(split_sizes) < 2:
            raise ValueError(
                f"split_sizes must be a tuple or list of at least 2 block sizes, "
                f"got {split_sizes!r}."
            )
        # bool is a subclass of int in Python; reject it explicitly.
        if any(
            isinstance(s, bool) or not isinstance(s, int) or s < 1
            for s in split_sizes
        ):
            raise ValueError(
                f"split_sizes entries must be positive integers, got {split_sizes!r}."
            )
        if self._resolve_num_heads(group) is not None:
            raise ValueError(
                "split_sizes is incompatible with num_heads > 1: use num_heads "
                "for uniform per-head splits or split_sizes for uneven row "
                "blocks, not both."
            )
        if group.get("flatten"):
            raise ValueError(
                "split_sizes is incompatible with flatten=True: split_sizes "
                "applies to 2D parameters only."
            )
        return tuple(split_sizes)

    def _validate_split_shape(
        self, split_sizes: Tuple[int, ...], params: List[Tensor]
    ) -> None:
        """Check that ``split_sizes`` is consistent with a shape group's params."""
        shape = params[0].shape
        if len(shape) != 2:
            raise ValueError(
                f"split_sizes is only supported for 2D parameters, got shape "
                f"{tuple(shape)}."
            )
        if sum(split_sizes) != shape[0]:
            raise ValueError(
                f"split_sizes {split_sizes} must sum to dim 0 of the parameter "
                f"(got shape {tuple(shape)})."
            )

    def _prepare_head_split(
        self,
        num_heads: int,
        params: List[Tensor],
        *extras: List[Tensor],
    ) -> Tuple[List[Tensor], ...]:
        """Reshape 2D params (and same-dim-0 companion tensors) into 3D per-head views.

        A 2D weight of shape ``(num_heads * head_dim, ...)`` is returned as
        a 3D local tensor of shape ``(num_heads_local, head_dim, ...)``. The same
        split is applied to any ``extras`` lists (grads, momentums, and NorMuon's
        per-neuron variance buffer of shape ``(out, 1)``).

        In-place updates on the returned views propagate to the underlying storage.
        Callers must also mark the resulting tensors as batch-sharded (skip NS
        all-to-all) since each rank's shard now holds whole heads.
        """
        first = params[0]
        full_shape = first.shape
        if first.ndim != 2:
            raise ValueError(
                f"num_heads is only supported for 2D parameters, got shape {tuple(full_shape)}."
            )
        if full_shape[0] % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide dim 0 of the parameter "
                f"(got shape {tuple(full_shape)})."
            )
        head_dim = full_shape[0] // num_heads

        if isinstance(first, DTensor):
            shard_placements = [
                (i, p)
                for i, p in enumerate(first.placements)
                if p.is_shard() and first.device_mesh.size(i) > 1
            ]
            if any(p.dim != 0 for _, p in shard_placements):
                raise NotImplementedError(
                    f"num_heads requires sharding on dim 0 (the heads dim) or no sharding; "
                    f"got placements {first.placements}."
                )
            if shard_placements:
                sharded_mesh_dim = shard_placements[0][0]
                world = first.device_mesh.size(sharded_mesh_dim)
                if num_heads % world != 0:
                    raise ValueError(
                        f"num_heads ({num_heads}) must be divisible by the sharding "
                        f"world_size ({world}) so each rank holds whole heads."
                    )

        def _as_3d(t: Tensor) -> Tensor:
            local = t.to_local() if isinstance(t, DTensor) else t
            local_dim0 = local.shape[0]
            if local_dim0 % head_dim != 0:
                raise RuntimeError(
                    f"Local shard dim 0 ({local_dim0}) is not a multiple of head_dim "
                    f"({head_dim}); shard boundaries must align with heads."
                )
            return local.view(local_dim0 // head_dim, head_dim, *local.shape[1:])

        return tuple([_as_3d(t) for t in lst] for lst in (params,) + extras)

    def _get_shard_info(self, param: Tensor, group: dict):
        """Determine sharding info. Returns (is_batch_sharded, is_matrix_sharded, sharded_tensor_dim)."""
        is_batch_sharded = False
        is_matrix_sharded = False
        sharded_tensor_dim = None

        if not isinstance(param, DTensor):
            return is_batch_sharded, is_matrix_sharded, sharded_tensor_dim

        if not isinstance(self._distributed_mesh, DeviceMesh):
            raise RuntimeError(
                "Must create optimizer with DeviceMesh if using DTensor parameters."
            )

        shard_placements = [
            (i, p)
            for i, p in enumerate(param.placements)
            if p.is_shard() and param.device_mesh.size(i) > 1
        ]

        if not group["flatten"]:
            matrix_dims = {param.ndim - 1, param.ndim - 2}
            is_batch_sharded = any(
                p.dim not in matrix_dims for _, p in shard_placements
            )
            shard_placements = [
                (i, p) for i, p in shard_placements if p.dim in matrix_dims
            ]

        if len(shard_placements) == 1:
            is_matrix_sharded = True
            sharded_mesh_dim = shard_placements[0][0]
            sharded_tensor_dim = shard_placements[0][1].dim

            if (
                param.device_mesh.get_group(sharded_mesh_dim).group_name
                != self._process_group.group_name
            ):
                raise RuntimeError(
                    f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} "
                    f"different from the optimizer's device mesh. "
                    f"DTensor has mesh: {param.device_mesh}, placements: {param.placements}, "
                    f"but optimizer was created with mesh: {self._distributed_mesh}."
                )
        elif len(shard_placements) > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support parameters with multiple sharded dimensions."
            )

        return is_batch_sharded, is_matrix_sharded, sharded_tensor_dim

    def _live_hyperparams(self, index: int) -> Tuple[str, ...]:
        """Names of ``param_groups[index]``'s hyperparameters carried as persistent device
        tensors, so a captured graph re-reads them on every replay instead of baking them in.

        ``lr`` is always live -- carrying it as a tensor is what lets a ``torch.optim``
        scheduler drive a captured step. ``weight_decay`` is live only when the caller
        supplied a Tensor for it (at construction, ``weight_decay=torch.tensor(0.01)``, or
        by assigning one to the group any time before the graph is captured). That opt-in
        keeps the default float path bit-identical to before: on the AdamW scalar path a
        live weight decay cannot ride ``torch._fused_adamw_``, whose ``weight_decay``
        argument is a float in every overload, so it has to be applied as a separate pass
        and rounds once more.

        The opt-in *latches*: the check re-runs until a Tensor is seen, and never after.
        So an opt-in made after construction still takes effect, while a later
        ``group["weight_decay"] = 0.05`` refills the persistent tensor (as it does for the
        LR) instead of quietly dropping the group back to a value baked in at capture.
        Opting in after a graph exists is too late to be honored -- the graph already baked
        the float -- and ``CudaGraphOptimizer`` raises rather than let that pass silently.

        Every *other* group scalar (``mu``, ``beta1``, ``epsilon``, ...) reaches the kernels
        as a host value and is baked in at capture. Nothing schedules those today; if
        something does, add it here rather than mutating it behind a captured graph.
        """
        live = self._live_hyperparams_by_group.get(index)
        if live is None or "weight_decay" not in live:
            if isinstance(self.param_groups[index].get("weight_decay"), Tensor):
                live = ("lr", "weight_decay")
            elif live is None:
                live = ("lr",)
            self._live_hyperparams_by_group[index] = live
        return live

    def _ensure_hyperparam_tensor(self, index: int, name: str) -> Optional[Tensor]:
        """Make ``param_groups[index][name]`` this group's persistent 0-d device float32
        tensor and return it (``None`` for a group with no parameters, which has no device
        to put it on and no kernel to read it).

        The value is carried as a device tensor the kernels read directly, so a ``torch.optim``
        LR scheduler -- which updates a tensor ``lr`` in place -- drives a captured step
        natively: the scheduler fills this tensor outside the graph and every replay re-reads
        it, with no refresh plumbing. float32 is what the former ``torch.tensor(group[name])``
        produced and the dtype the fused AdamW ``tensor_lr`` overload requires.

        The tensor is allocated once per group and thereafter only ever filled, never
        replaced. That identity is what makes replay correct: a captured graph reads the
        tensor recorded at capture time, so handing the group a *different* tensor would
        leave every later replay on the stale one. Anything else found under ``name`` -- a
        python float from construction or from the ``group["lr"] = 0.05`` idiom of a
        hand-rolled schedule, or a wrong-device tensor restored from a checkpoint via
        ``map_location="cpu"`` -- is copied into the persistent tensor and the tensor put
        back. The steady state (including right after a scheduler ``fill_``) hits the
        identity check and does no work at all.
        """
        group = self.param_groups[index]
        params = group["params"]
        if not params:
            return None

        value = group[name]
        tensor = self._hyperparam_tensors.get((index, name))
        if value is tensor:
            return tensor

        # Everything below writes to the device, so it must not run under graph capture:
        # a recorded fill_ would re-apply the capture-time value on every replay, silently
        # overwriting whatever the scheduler had set.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"param group {index} has a {name} that is not its persistent "
                f"device tensor ({type(value).__name__}), and CUDA graph capture is already "
                f"underway so it cannot be refreshed. Sync {name} before "
                "capturing (CudaGraphOptimizer does this for you)."
            )

        device = to_local(params[0]).device
        if tensor is None or tensor.device != device:
            tensor = torch.empty((), dtype=torch.float32, device=device)
            self._hyperparam_tensors[(index, name)] = tensor
        if isinstance(value, Tensor):
            tensor.copy_(value)
        else:
            tensor.fill_(value)
        group[name] = tensor
        return tensor

    def _sync_hyperparam_tensors(self) -> None:
        """Push each group's host-side hyperparameters into the persistent device tensors the
        kernels (and any captured CUDA graph) read. Called at the top of ``step()``, and by
        ``CudaGraphOptimizer`` before it captures or replays -- under replay ``step()``
        never runs, so this is the only thing that carries a ``group["lr"] = 0.05``
        assignment through to the graph."""
        for index in range(len(self.param_groups)):
            for name in self._live_hyperparams(index):
                self._ensure_hyperparam_tensor(index, name)

    def _advance_host_step_counters(self):
        """Advance the host-side ``group["step"]`` counters for a step that ran as a CUDA
        graph replay. ``step()``'s python increment only runs when ``step()`` is actually
        traced, so under replay the counter would otherwise freeze at its capture value and
        be written stale into every subsequent checkpoint."""
        for group in self.param_groups:
            group["step"] += 1

    def state_dict(self):
        if self._state_lifecycle == "failed":
            raise RuntimeError("Cannot serialize optimizer state after state materialization failed.")
        if self._state_lifecycle == "deferred" and (
            self.state or self._hyperparam_tensors or self._live_hyperparams_by_group
        ):
            self._state_lifecycle = "failed"
            raise RuntimeError(
                "Deferred optimizer state_dict() requires pristine empty state and hyperparameter caches."
            )
        if self.is_state_materialized:
            try:
                self._validate_materialized_state()
            except Exception:
                self._state_lifecycle = "failed"
                raise

        sd = super().state_dict()
        # lr is carried at runtime as a device tensor; serialize it as a plain python
        # number so checkpoints stay portable (no device baggage, human-readable, matches
        # pre-feature checkpoints), and load_state_dict() refills the device tensor.
        # Serializing the tensor instead would ride a CUDA tensor into every checkpoint and
        # come back on whatever device torch.load() used -- a CPU LR that still trains
        # eagerly but is read once at capture and baked in, silently freezing a scheduled
        # LR under replay. This covers every 0-d tensor hyperparameter in the group, not
        # just "lr": an attached LRScheduler also stashes "initial_lr" there, cloned from
        # (and therefore as device-bound as) the LR tensor itself. super().state_dict()
        # copies each group dict, so rewriting entries here does not touch the live
        # optimizer.
        for group in sd["param_groups"]:
            for key, value in group.items():
                if key != "params" and isinstance(value, Tensor) and value.ndim == 0:
                    group[key] = value.item()
        return sd

    def load_state_dict(self, state_dict):
        if self._state_lifecycle == "failed":
            raise RuntimeError("Cannot load optimizer state after state materialization failed.")
        try:
            transformed_state_dict = state_dict.copy()
            for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
                hook_result = pre_hook(self, transformed_state_dict)
                if hook_result is not None:
                    transformed_state_dict = hook_result

            staging_optimizer = object.__new__(type(self))
            staging_optimizer.__dict__ = self.__dict__.copy()
            staging_optimizer.defaults = self.defaults.copy()
            staging_optimizer._optimizer_load_state_dict_pre_hooks = OrderedDict()
            staging_optimizer._optimizer_load_state_dict_post_hooks = OrderedDict()
            staging_optimizer._hyperparam_tensors = {}
            staging_optimizer._live_hyperparams_by_group = self._live_hyperparams_by_group.copy()

            if self._state_lifecycle == "deferred":
                if self.state or self._hyperparam_tensors or self._live_hyperparams_by_group:
                    raise RuntimeError("Deferred optimizer bootstrap load requires pristine empty state.")
                loaded_state = transformed_state_dict.get("state")
                if not isinstance(loaded_state, dict):
                    raise RuntimeError("Invalid optimizer state_dict state structure.")
                if loaded_state:
                    raise RuntimeError("Deferred optimizer load_state_dict() only accepts empty bootstrap state.")
                self._validate_group_envelope(transformed_state_dict.get("param_groups"))
                Optimizer.load_state_dict(staging_optimizer, transformed_state_dict)
                if staging_optimizer.state:
                    raise RuntimeError("Deferred optimizer bootstrap load unexpectedly created state.")
            else:
                self._validate_loaded_state_dict(transformed_state_dict)
                Optimizer.load_state_dict(staging_optimizer, transformed_state_dict)
                # Refill the device hyperparameter tensors from the loaded values (a float from a normal
                # checkpoint, or a wrong-device tensor from one written by an earlier build).
                staging_optimizer._sync_hyperparam_tensors()
                for group in staging_optimizer.param_groups:
                    if group["algorithm"] != "adamw":
                        continue
                    for param in group["params"]:
                        state = staging_optimizer.state.get(param)
                        if state is not None:
                            staging_optimizer._restore_step_tensor(state, param, group)
                staging_optimizer._validate_materialized_state()

            self.__setstate__(
                {
                    "state": staging_optimizer.state,
                    "param_groups": staging_optimizer.param_groups,
                }
            )
            self._hyperparam_tensors = staging_optimizer._hyperparam_tensors
            self._live_hyperparams_by_group = staging_optimizer._live_hyperparams_by_group
            for post_hook in self._optimizer_load_state_dict_post_hooks.values():
                post_hook(self)
            if self.is_state_materialized:
                self._validate_materialized_state()
            elif self.state or self._hyperparam_tensors or self._live_hyperparams_by_group:
                raise RuntimeError(
                    "Deferred optimizer post-load validation requires pristine empty state and caches."
                )
        except Exception:
            self._state_lifecycle = "failed"
            raise

    def _restore_step_tensor(self, state: dict, param: Tensor, group: dict) -> None:
        """Repair a loaded AdamW device step counter.

        Two ways ``Optimizer.load_state_dict()`` leaves it wrong. It casts state tensors to
        the owning param's dtype (its ``step`` special case keys off the literal name, which
        this is not), and a bf16 counter silently stops incrementing at 256 -- 256 + 1 == 256
        in bf16. And a checkpoint written before the counter moved on-device has no entry at
        all; there the host-side ``group["step"]`` is the value to resume from, since
        restarting at 0 would replay AdamW's bias-correction warmup and spike the effective
        LR on the first steps after every resume.
        """
        step = state.get("step_dev")
        device = to_local(param).device
        if step is None:
            step = torch.empty((), dtype=torch.float32, device=device)
            step.fill_(float(group["step"]))
        elif step.dtype != torch.float32 or step.device != device:
            step = step.to(dtype=torch.float32, device=device)
        state["step_dev"] = step

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Subclasses implement this to create orthogonalization tasks."""
        raise NotImplementedError

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "lion"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            momentums = [s["momentum"] for s in states]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=group["lr"],
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=as_scalar_tensor(group["weight_decay"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "adamw"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]
            step_tensors = [s["step_dev"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=group["lr"],
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=as_scalar_tensor(group["weight_decay"]),
                    state_steps=step_tensors,
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )


def megabatch_orthogonalize_async(
    U: List[Tensor],
    comm_dim: Optional[int],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    global_comm_dim_size: Optional[int],
    split_sizes: Optional[Tuple[int, ...]] = None,
    split_scales: Optional[Tuple[float, ...]] = None,
    return_stacked: bool = False,
    local_comm_size: Optional[int] = None,
) -> Generator[None, None, Union[List[Tensor], Tensor]]:
    """
    Shared megabatch communication + Newton-Schulz orthogonalization.

    This is a generator that yields at async communication points and uses
    ``return value`` to pass the result back to the caller. In Python, ``return``
    inside a generator raises ``StopIteration(value)``, and the caller recovers
    the value via ``result = yield from megabatch_orthogonalize_async(...)``.
    The ``yield from`` transparently forwards intermediate yields to AsyncRuntime.

    Args:
        U: List of tensors to orthogonalize (all same shape).
        comm_dim: Dimension for cat/split in all-to-all (negative index).
            None for non-sharded parameters.
        device_rank: This device's rank.
        world_size: Total number of devices.
        process_group: Distributed process group. None for single-GPU.
        newton_schulz_func: Newton-Schulz orthogonalization function.
        flatten: Whether to flatten 3D+ tensors to 2D.
        epsilon: Small value for numerical stability.
        global_comm_dim_size: Required (non-None) when ``comm_dim is not
            None``; pass ``None`` otherwise. The unsharded (global) size
            along ``comm_dim``, taken from the DTensor's global shape
            (``param.shape[comm_dim]``). Used to compute
            ``padded_local_size = ceil(global / world_size)`` so the
            alltoall sees uniform per-pair sizes across ranks.
        split_sizes: Optional row-block sizes for dim -2. Newton-Schulz runs
            independently per block (on the fully assembled matrices), as if
            each block were a separate parameter.
        split_scales: Optional per-block rescaling applied after Newton-Schulz,
            used to convert the caller's whole-matrix learning-rate adjustment
            into the per-block adjustment.
        return_stacked: If True, return the orthogonalized result as a single
            stacked ``[N, *shape]`` tensor instead of a list of N tensors.
            Callers that immediately re-stack the result (NorMuon, NorDion2 for
            their normalization step) use this to skip an unbind-then-restack
            round-trip. Default False preserves the list contract for callers
            that consume per-parameter tensors directly (Muon, Dion2).
        local_comm_size: Optional explicit per-rank size along ``comm_dim`` to
            pad the local shard to, overriding the ``ceil(global / world_size)``
            derivation. Used by the Dion2/NorDion2 "local" scope, which has
            already selected exactly ``k`` rows per rank before communication:
            passing ``local_comm_size=k`` pads to ``k`` (a no-op when the shard
            is already ``k``, a zero-pad for short shards) so the alltoall stays
            uniform while comm and Newton-Schulz shrink by ``fraction``.
            ``global_comm_dim_size`` still carries the true unsharded size.
    """
    N = len(U)

    def _finalize(flat: Tensor) -> Union[List[Tensor], Tensor]:
        # flat: stacked ``[M, *shape]`` (M >= N) in the caller's parameter
        # order. Return the first N either stacked or as a list of views.
        flat = flat[:N]
        return flat if return_stacked else list(flat.unbind(0))

    # Pad to divisible by world_size (needed by both distributed paths)
    if process_group is not None and (N > 1 or comm_dim is not None):
        pad_n = (world_size - N % world_size) % world_size
        U_work = U + [torch.zeros_like(U[0])] * pad_n if pad_n > 0 else U
        N_total = len(U_work)
        per_rank = N_total // world_size
    else:
        U_work = U

    if comm_dim is not None and process_group is not None:
        # --- Mega-batched sharded FSDP2 path ---

        # Pad each rank's local shard along comm_dim to a rank-consistent
        # ``padded_local_size = ceil(global / world_size)`` so dist.all_to_all
        # sees uniform per-pair sizes. FSDP2 contiguous chunking otherwise
        # leaves some ranks with empty (numel=0) shards when the sharded
        # global dim is smaller than world_size or doesn't divide evenly to
        # fill all ranks (e.g. shape (18, D) over world_size=8: ranks 6 and 7
        # hold (0, D) shards). Without padding the alltoall has mismatched
        # per-pair sizes and hangs. Newton-Schulz preserves zero rows (they
        # contribute nothing to U^T U), so padding doesn't change the
        # orthogonalization of the real rows.
        #
        # NOTE: this assumes FSDP2-style contiguous chunking, where every rank
        # holds at most ceil(global / world_size) elements along comm_dim. If
        # FSDP2 ever switches to a non-contiguous strategy (e.g. block-cyclic),
        # this derivation would be wrong; the size check below catches that.
        if global_comm_dim_size is None:
            raise ValueError(
                "global_comm_dim_size must be passed when comm_dim is not "
                "None; callers should pass the unsharded DTensor's global "
                "size along comm_dim."
            )
        # local_comm_size, when given, is the explicit per-rank padded size (the
        # caller pre-selected exactly this many rows). Otherwise derive it from
        # the global size assuming FSDP2 contiguous chunking.
        if local_comm_size is not None:
            # The local-selection scope pads to k rows per rank, so k*world_size
            # is (deliberately) not global_comm_dim_size. That would spuriously
            # trip the split_sizes guard below, whose divisibility test assumes
            # padded_local_size derives from the global size. The two features
            # have never been combined; reject it explicitly rather than emit a
            # misleading "not divisible by world_size" error.
            if split_sizes is not None:
                raise NotImplementedError(
                    "local_comm_size (row-sharded 'local' selection) is not "
                    "supported together with split_sizes; the per-rank pad-to-k "
                    "would intersperse zero rows within the assembled row blocks."
                )
            padded_local_size = local_comm_size
        else:
            padded_local_size = (global_comm_dim_size + world_size - 1) // world_size
        original_local_size = U_work[0].size(comm_dim)
        if (
            split_sizes is not None
            and comm_dim == -2
            and padded_local_size * world_size != global_comm_dim_size
        ):
            # Shard padding would intersperse zero rows between rank segments
            # of the assembled matrices, scrambling the row-block boundaries.
            raise NotImplementedError(
                f"split_sizes requires dim 0 of the parameter "
                f"({global_comm_dim_size}) to be divisible by the sharding "
                f"world_size ({world_size}) so the assembled matrices have no "
                f"interspersed padding rows."
            )
        if padded_local_size < original_local_size:
            raise RuntimeError(
                f"padded_local_size ({padded_local_size}) < this rank's "
                f"local size ({original_local_size}); FSDP2 contiguous-"
                f"chunking assumption violated (global_comm_dim_size="
                f"{global_comm_dim_size}, world_size={world_size})."
            )

        if padded_local_size != original_local_size:
            # F.pad's pad-spec is built from the LAST dim backwards. comm_dim
            # is negative; pad only the END of comm_dim.
            pad_spec = [0, 0] * (-comm_dim - 1) + [0, padded_local_size - original_local_size]
            U_work = [torch.nn.functional.pad(u, pad_spec) for u in U_work]

        # Pack all per-rank segments with a single stack + view instead of
        # world_size separate torch.stack calls (each dispatched per_rank
        # aten::select). The unbind views are contiguous slices of the stacked
        # buffer, so they are safe to hand to all_to_all.
        input_chunks = list(
            torch.stack(U_work).unflatten(0, (world_size, per_rank)).unbind(0)
        )

        output_chunks = [torch.empty_like(c) for c in input_chunks]
        work = dist.all_to_all(
            output_chunks, input_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        # comm_dim is negative, so it correctly indexes the stacked tensor
        full_matrices = torch.cat(output_chunks, dim=comm_dim)
        full_matrices = muon_update_newton_schulz(
            full_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            split_sizes=split_sizes,
            split_scales=split_scales,
        )

        split_chunks = [
            s.contiguous()
            for s in torch.tensor_split(full_matrices, world_size, dim=comm_dim)
        ]

        recv_chunks = [torch.empty_like(c) for c in split_chunks]
        work = dist.all_to_all(
            recv_chunks, split_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Narrow each per-rank result back to the rank's original local size and
        # flatten the (rank, per_rank) axes in one shot, instead of a
        # world_size x per_rank Python loop of select + narrow + contiguous.
        # comm_dim is negative so it still indexes the matrix dim of the stacked
        # tensor; flatten(0, 1) preserves the rank-major, per_rank-minor order of
        # the original comprehension. On padding-only ranks original_local_size
        # == 0 and the narrowed slices are empty.
        recv_stacked = (
            torch.stack(recv_chunks)
            .narrow(comm_dim, 0, original_local_size)
            .contiguous()
        )
        return _finalize(recv_stacked.flatten(0, 1))

    elif N > 1 and process_group is not None:
        # --- Mega-batched non-sharded path ---
        start = device_rank * per_rank
        my_matrices = torch.stack(U_work[start : start + per_rank])
        my_matrices = muon_update_newton_schulz(
            my_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            split_sizes=split_sizes,
            split_scales=split_scales,
        )

        all_chunks = [torch.empty_like(my_matrices) for _ in range(world_size)]
        work = dist.all_gather(
            all_chunks, my_matrices.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

        # Flatten (rank, per_rank) in one op instead of a nested select loop.
        return _finalize(torch.stack(all_chunks).flatten(0, 1))

    elif N == 1:
        out = muon_update_newton_schulz(
            U[0],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            split_sizes=split_sizes,
            split_scales=split_scales,
        )
        return _finalize(out.unsqueeze(0))

    else:
        # N > 1, no process_group (single GPU or batch-sharded 3D)
        stacked = torch.stack(U)
        stacked = muon_update_newton_schulz(
            stacked,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            split_sizes=split_sizes,
            split_scales=split_scales,
        )
        return _finalize(stacked)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    split_sizes: Optional[Tuple[int, ...]] = None,
    split_scales: Optional[Tuple[float, ...]] = None,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    With ``split_sizes``, orthogonalize row blocks of dim -2 independently.
    """
    if split_sizes is not None:
        assert not flatten, "split_sizes is incompatible with flatten=True"
        return _newton_schulz_row_blocks(
            X, split_sizes, split_scales, newton_schulz_func, epsilon
        )

    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def _newton_schulz_row_blocks(
    X: Tensor,
    split_sizes: Tuple[int, ...],
    split_scales: Optional[Tuple[float, ...]],
    newton_schulz_func: Callable,
    epsilon: Tensor,
) -> Tensor:
    """
    Orthogonalize row blocks of dim -2 independently, as if each block were a
    separate parameter. Blocks of equal height are stacked into one batched
    Newton-Schulz call (e.g. the K and V blocks of a fused QKV weight).

    ``split_scales`` optionally rescales each orthogonalized block, used to
    convert the caller's whole-matrix learning-rate adjustment into the
    per-block adjustment that separate parameters would receive.
    """
    assert X.size(-2) == sum(split_sizes), (
        f"dim -2 of input ({X.size(-2)}) does not match sum of split_sizes "
        f"({split_sizes})"
    )
    blocks = list(X.split(list(split_sizes), dim=-2))

    rows_to_idx = defaultdict(list)
    for i, block in enumerate(blocks):
        rows_to_idx[block.size(-2)].append(i)

    out = [None] * len(blocks)
    for idx in rows_to_idx.values():
        stacked = torch.stack([blocks[i] for i in idx], dim=0)
        # Newton-Schulz functions expect at most 3D input
        batched = stacked.flatten(end_dim=-3) if stacked.ndim > 3 else stacked
        ortho = newton_schulz_func(batched, epsilon=epsilon).reshape(stacked.shape)
        for j, i in enumerate(idx):
            block = ortho[j]
            if split_scales is not None:
                block = block * split_scales[i]
            out[i] = block

    return torch.cat(out, dim=-2)


def compute_split_lr_scales(
    split_sizes: Tuple[int, ...],
    param_shape,
    adjust_lr: Optional[str],
) -> Optional[Tuple[float, ...]]:
    """
    Per-block learning-rate corrections for ``split_sizes`` row blocks.

    The megabatch update functions compute a single adjusted learning rate
    from the whole (fused) matrix shape. Each block of a split parameter
    should instead see the adjustment its own shape would receive as a
    separate parameter, so each block is rescaled by
    ``adjust(block_shape) / adjust(full_shape)`` after Newton-Schulz.
    Returns ``None`` when ``adjust_lr`` is None (no shape-dependent scaling).
    """
    if adjust_lr is None:
        return None
    elif adjust_lr == "spectral_norm":
        adjust_fn = adjust_lr_spectral_norm
    elif adjust_lr == "rms_norm":
        adjust_fn = adjust_lr_rms_norm
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    num_cols = param_shape[-1]
    full_adjust = adjust_fn(1.0, param_shape, flatten=False)
    return tuple(
        adjust_fn(1.0, (rows, num_cols), flatten=False) / full_adjust
        for rows in split_sizes
    )


def adjust_lr_rms_norm(lr, param_shape, flatten):
    """Adjust learning rate for constant element-wise RMS norm."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    return lr * adjusted_ratio


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    """Adjust from spectral norm 1 to RMS operator norm 1."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * math.sqrt(fan_out / fan_in)
