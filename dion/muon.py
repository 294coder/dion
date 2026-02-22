import math
from collections.abc import Callable, Generator
from itertools import chain

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, DeviceMesh
from torch.optim.optimizer import Optimizer, ParamsT

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        reg_mask_mode: Optional inline masking mode. One of {None, "magma", "skipupdate"}.
        reg_mask_survival_prob: Bernoulli survival probability for masking.
        reg_mask_tau: Temperature in Magma alignment score sigmoid(cos/tau).
        reg_mask_ema_beta: EMA coefficient for Magma alignment state.
        reg_mask_seed: Optional RNG seed for reproducible masking.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: DeviceMesh | ProcessGroup | None = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: str | None = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Callable | None = None,
        reg_mask_mode: str | None = None,
        reg_mask_survival_prob: float = 0.5,
        reg_mask_tau: float = 2.0,
        reg_mask_ema_beta: float = 0.9,
        reg_mask_seed: int | None = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None.")
        if reg_mask_mode not in (None, "magma", "skipupdate"):
            raise ValueError(f"Invalid reg_mask_mode value: {reg_mask_mode}. Must be None, 'magma', or 'skipupdate'.")
        if not 0.0 < reg_mask_survival_prob <= 1.0:
            raise ValueError(f"Invalid reg_mask_survival_prob: {reg_mask_survival_prob}. Must be in (0, 1].")
        if reg_mask_tau <= 0.0:
            raise ValueError(f"Invalid reg_mask_tau: {reg_mask_tau}. Must be > 0.")
        if not 0.0 <= reg_mask_ema_beta < 1.0:
            raise ValueError(f"Invalid reg_mask_ema_beta: {reg_mask_ema_beta}. Must be in [0, 1).")

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            weight_update_method="sgd",
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            reg_mask_mode=reg_mask_mode,
            reg_mask_survival_prob=reg_mask_survival_prob,
            reg_mask_tau=reg_mask_tau,
            reg_mask_ema_beta=reg_mask_ema_beta,
            reg_mask_seed=reg_mask_seed,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
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
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}")
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        self._mask_rng = torch.Generator(device="cpu")
        self._mask_rng_seeded = False

    @staticmethod
    def _to_local_tensor(x: Tensor) -> Tensor:
        return x.to_local() if isinstance(x, DTensor) else x

    @staticmethod
    def _to_local_tensor_list(xs: list[Tensor]) -> list[Tensor]:
        return [x.to_local() if isinstance(x, DTensor) else x for x in xs]

    @staticmethod
    def _cosine_similarity(x: Tensor, y: Tensor, eps: float = 1e-12) -> float:
        x32 = x.detach().float().reshape(-1)
        y32 = y.detach().float().reshape(-1)
        x_norm = float(torch.norm(x32))
        y_norm = float(torch.norm(y32))
        if x_norm <= eps or y_norm <= eps:
            return 0.0
        value = float(torch.dot(x32, y32) / (x_norm * y_norm + eps))
        return max(min(value, 1.0), -1.0)

    def _ensure_mask_rng_seeded(self) -> None:
        if self._mask_rng_seeded:
            return
        for group in self.param_groups:
            seed = group.get("reg_mask_seed")
            if seed is not None:
                self._mask_rng.manual_seed(int(seed))
                break
        self._mask_rng_seeded = True

    def _sample_mask(self, survival_prob: float) -> float:
        if survival_prob >= 1.0:
            return 1.0
        return float(torch.rand((), generator=self._mask_rng).item() < survival_prob)

    def _compute_param_mask_scale(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict,
        group: dict,
        algo_name: str,
    ) -> float:
        mode = group.get("reg_mask_mode")
        if mode is None:
            return 1.0

        if mode not in ("magma", "skipupdate"):
            raise ValueError(f"Unknown reg_mask_mode: {mode}")

        survival_prob = float(group.get("reg_mask_survival_prob", 0.5))
        tau = float(group.get("reg_mask_tau", 2.0))
        ema_beta = float(group.get("reg_mask_ema_beta", 0.9))
        if not 0.0 < survival_prob <= 1.0:
            raise ValueError(f"reg_mask_survival_prob must be in (0, 1], got {survival_prob}")
        if tau <= 0.0:
            raise ValueError(f"reg_mask_tau must be > 0, got {tau}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"reg_mask_ema_beta must be in [0, 1), got {ema_beta}")

        mask = self._sample_mask(survival_prob)
        if mode == "skipupdate":
            return mask / survival_prob

        momentum_prev = self._to_local_tensor(state["momentum"])
        grad_local = self._to_local_tensor(grad).to(dtype=momentum_prev.dtype)

        if algo_name == "muon":
            mu_new = momentum_prev * float(group["mu"]) + grad_local
        elif algo_name == "adamw":
            mu_new = momentum_prev.lerp(grad_local, 1.0 - float(group["beta1"]))
        elif algo_name == "lion":
            mu_new = momentum_prev.lerp(grad_local, 1.0 - float(group["beta2"]))
        else:
            raise ValueError(f"Unsupported algo_name for mask scale: {algo_name}")

        cossim = self._cosine_similarity(mu_new, grad_local)
        s_tilde = 1.0 / (1.0 + math.exp(-cossim / tau))
        s_prev = state.get("mask_s_t")
        if s_prev is None:
            s_t = s_tilde
        else:
            s_t = ema_beta * float(s_prev) + (1.0 - ema_beta) * s_tilde
        state["mask_s_t"] = s_t
        return s_t * mask

    def _compute_group_mask_scales(
        self,
        params: list[Tensor],
        gradients: list[Tensor],
        states: list[dict],
        group: dict,
        algo_name: str,
    ) -> Tensor:
        scale_values: list[float] = []
        for p, g, s in zip(params, gradients, states, strict=True):
            scale_value = self._compute_param_mask_scale(p, g, s, group, algo_name)
            scale_values.append(scale_value)
        local_param = self._to_local_tensor(params[0])
        return torch.tensor(scale_values, device=local_param.device, dtype=torch.float32)

    def state_dict(self):
        state = super().state_dict()
        state["_mask_rng_state"] = self._mask_rng.get_state()
        state["_mask_rng_seeded"] = self._mask_rng_seeded
        return state

    def load_state_dict(self, state_dict):
        incoming = dict(state_dict)
        mask_rng_state = incoming.pop("_mask_rng_state", None)
        mask_rng_seeded = bool(incoming.pop("_mask_rng_seeded", False))
        super().load_state_dict(incoming)
        if mask_rng_state is not None:
            if not torch.is_tensor(mask_rng_state):
                raise TypeError("mask rng state must be a torch.Tensor.")
            # `accelerate` may place optimizer state tensors on CUDA; Generator expects CPU ByteTensor.
            mask_rng_state = mask_rng_state.detach().to(device="cpu", dtype=torch.uint8)
            self._mask_rng.set_state(mask_rng_state)
            self._mask_rng_seeded = True
        else:
            self._mask_rng_seeded = mask_rng_seeded

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._ensure_mask_rng_seeded()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = (task for task in chain(muon_tasks, lion_tasks, adamw_tasks))
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_muon_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(p.ndim >= 2 for p in group["params"]), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            weight_update_method = group.get("weight_update_method", "sgd")
            if weight_update_method not in ("sgd", "hyperball"):
                raise ValueError(
                    f"Invalid weight_update_method: {weight_update_method}. Must be one of: sgd, hyperball."
                )

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            momentum = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = bool(group["nesterov"])
            flatten = bool(group["flatten"])
            adjust_lr = group["adjust_lr"]
            cautious_wd = bool(group["cautious_wd"])
            hyperball = weight_update_method == "hyperball"

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(group_params, batch_size=self._world_size):
                gradients = [p.grad for p in params]
                assert all(g is not None for g in gradients)
                gradients = [g for g in gradients if g is not None]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                mask_scales = self._compute_group_mask_scales(
                    params=params,
                    gradients=gradients,
                    states=states,
                    group=group,
                    algo_name=algo_name,
                )

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError("Must create optimizer with DeviceMesh if using DTensor parameters.")

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(p.dim not in matrix_dims for _, p in shard_placements)
                        shard_placements = [(i, p) for i, p in shard_placements if p.dim in matrix_dims]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError("Muon does not support parameters with multiple sharded dimensions.")

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim) != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    per_param_scales = tuple(mask_scales.unbind())
                    for x, g, m, scale in zip(params, gradients, momentums, per_param_scales, strict=True):
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                mask_scales=scale.unsqueeze(0),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                adjust_lr=adjust_lr,
                                device_rank=self._device_rank,
                                world_size=self._world_size,
                                shard_dim=None,  # No sharded matrix dim
                                process_group=self._process_group,
                                newton_schulz_func=self._newton_schulz_func,
                                cautious_wd=cautious_wd,
                                hyperball=hyperball,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    padded_scales = pad_batch_scalars(mask_scales, self._world_size, fill_value=1.0)
                    yield AsyncTask(
                        muon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            mask_scales=padded_scales,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            nesterov=nesterov,
                            flatten=flatten,
                            adjust_lr=adjust_lr,
                            device_rank=self._device_rank,
                            world_size=self._world_size,
                            shard_dim=sharded_tensor_dim,
                            process_group=self._process_group,
                            newton_schulz_func=self._newton_schulz_func,
                            cautious_wd=cautious_wd,
                            hyperball=hyperball,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            assert all(g is not None for g in gradients)
            gradients = [g for g in gradients if g is not None]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            mask_scales = self._compute_group_mask_scales(
                params=params,
                gradients=gradients,
                states=states,
                group=group,
                algo_name=algo_name,
            )

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            cautious_wd = group["cautious_wd"]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=self._to_local_tensor_list(params),
                    G=self._to_local_tensor_list(gradients),
                    M=self._to_local_tensor_list(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    cautious_wd=cautious_wd,
                    mask_scales=mask_scales,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            assert all(g is not None for g in gradients)
            gradients = [g for g in gradients if g is not None]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]
            mask_scales = self._compute_group_mask_scales(
                params=params,
                gradients=gradients,
                states=states,
                group=group,
                algo_name=algo_name,
            )

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            cautious_wd = group["cautious_wd"]
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=self._to_local_tensor_list(params),
                    G=self._to_local_tensor_list(gradients),
                    M=self._to_local_tensor_list(momentums),
                    V=self._to_local_tensor_list(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                    cautious_wd=cautious_wd,
                    mask_scales=mask_scales,
                )
            )


def pad_batch_scalars(batch: Tensor, batch_size: int, fill_value: float) -> Tensor:
    assert batch.ndim == 1
    assert batch.numel() > 0
    assert batch.numel() <= batch_size
    pad_count = batch_size - int(batch.numel())
    if pad_count == 0:
        return batch
    fill = torch.full((pad_count,), fill_value, device=batch.device, dtype=batch.dtype)
    return torch.cat((batch, fill), dim=0)


def muon_update_batch_async(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    mask_scales: Tensor,
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: str | None,  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: int | None = None,  # Shard dimension for DTensor (if applicable)
    process_group: ProcessGroup | None = None,
    newton_schulz_func: Callable | None = None,
    cautious_wd: bool = False,
    hyperball: bool = False,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert mask_scales.ndim == 1
    assert len(X) == int(mask_scales.numel())
    if newton_schulz_func is None:
        raise ValueError("newton_schulz_func must not be None.")

    local_g = [g.to_local() if isinstance(g, DTensor) else g for g in G]
    local_m = [m.to_local() if isinstance(m, DTensor) else m for m in M]

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=local_g,
        M=local_m,
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert process_group is not None, "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert X[0].size(shard_dim) % world_size == 0, (
            f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."
        )

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(single_matrix_shards, U, group=process_group, async_op=True)
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [x.contiguous() for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(U, single_matrix_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U) > 1:
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        U = [torch.empty_like(u) for u in U]

        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(U, single_matrix.contiguous(), group=process_group, async_op=True)
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U) == 1
        U[0] = muon_update_newton_schulz(
            U[0],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Update model parameters with orthogonalized output
    local_x = [x.to_local() if isinstance(x, DTensor) else x for x in X]
    muon_update_post_orthogonalize(
        X=local_x,
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
        epsilon=float(epsilon),
        hyperball=hyperball,
        mask_scales=mask_scales,
    )


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: list[Tensor],
    M: list[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> list[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: list[Tensor],
    U: list[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    epsilon: float,
    mask_scales: Tensor,
    cautious_wd: bool = False,
    hyperball: bool = False,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = base_lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        decay_terms = torch._foreach_mul(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    assert mask_scales.ndim == 1
    assert len(X) == int(mask_scales.numel())
    per_param_scales = tuple(mask_scales.unbind())

    if hyperball:
        # Hyperball update: preserve ||X||_2 (Frobenius for matrices) after the weight update.
        radii = torch._foreach_norm(X)
        u_norms = torch._foreach_norm(U)
        u_denom = torch._foreach_clamp_min(u_norms, epsilon)
        u_unit = torch._foreach_div(U, u_denom)

        step_coeff = torch._foreach_mul(radii, adjusted_lr)
        step_coeff = torch._foreach_mul(step_coeff, per_param_scales)
        step_vec = torch._foreach_mul(u_unit, step_coeff)
        torch._foreach_sub_(X, step_vec)

        x_norms = torch._foreach_norm(X)
        x_denom = torch._foreach_clamp_min(x_norms, epsilon)
        scales = torch._foreach_div(radii, x_denom)
        torch._foreach_mul_(X, scales)
    else:
        scaled_u = torch._foreach_mul(U, adjusted_lr)
        scaled_u = torch._foreach_mul(scaled_u, per_param_scales)
        torch._foreach_sub_(X, scaled_u)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
