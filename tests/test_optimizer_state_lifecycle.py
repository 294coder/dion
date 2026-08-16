import copy
import os
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Placement, Replicate, Shard

from dion import NorDion2


def _make_optimizer(*, state_initialization: str = "eager") -> tuple[NorDion2, dict[str, torch.nn.Parameter]]:
    parameters = {
        "nordion2": torch.nn.Parameter(torch.randn(4, 6)),
        "adamw": torch.nn.Parameter(torch.randn(6)),
        "lion": torch.nn.Parameter(torch.randn(6)),
    }
    optimizer = NorDion2(
        [
            {"params": [parameters["nordion2"]], "algorithm": "nordion2"},
            {"params": [parameters["adamw"]], "algorithm": "adamw"},
            {"params": [parameters["lion"]], "algorithm": "lion"},
        ],
        state_initialization=state_initialization,
    )
    return optimizer, parameters


@pytest.fixture(scope="module")
def cpu_meshes(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[DeviceMesh, DeviceMesh]]:
    initialized_here = not dist.is_initialized()
    previous_interface = os.environ.get("GLOO_SOCKET_IFNAME")
    if initialized_here:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path_factory.mktemp('lifecycle-gloo') / 'store'}",
            rank=0,
            world_size=1,
        )
    secondary_group = dist.new_group([0])
    try:
        yield (
            DeviceMesh.from_group(dist.group.WORLD, "cpu"),
            DeviceMesh.from_group(secondary_group, "cpu"),
        )
    finally:
        dist.destroy_process_group(secondary_group)
        if initialized_here:
            dist.destroy_process_group()
        if previous_interface is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous_interface


def _make_dtensor_parameter(
    mesh: DeviceMesh,
    placements: list[Placement],
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        DTensor.from_local(
            torch.randn(4, 6),
            mesh,
            placements,
            run_check=False,
        )
    )


def _assert_permanently_failed(optimizer: NorDion2) -> None:
    assert optimizer.is_state_materialized is False
    with pytest.raises(RuntimeError, match="failed"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="failed"):
        optimizer.state_dict()


def _capture_live_optimizer(optimizer: NorDion2) -> dict:
    return {
        "groups": [dict(group) for group in optimizer.param_groups],
        "state": {parameter: dict(state) for parameter, state in optimizer.state.items()},
        "state_values": {
            (parameter, name): value.clone()
            for parameter, state in optimizer.state.items()
            for name, value in state.items()
        },
        "cache": dict(optimizer._hyperparam_tensors),
        "cache_values": {key: value.clone() for key, value in optimizer._hyperparam_tensors.items()},
        "live": dict(optimizer._live_hyperparams_by_group),
        "defaults": dict(optimizer.defaults),
        "defaults_object": optimizer.defaults,
    }


def _assert_live_optimizer_unchanged(optimizer: NorDion2, snapshot: dict) -> None:
    assert len(optimizer.param_groups) == len(snapshot["groups"])
    for actual_group, expected_group in zip(optimizer.param_groups, snapshot["groups"]):
        assert set(actual_group) == set(expected_group)
        for name, expected in expected_group.items():
            actual = actual_group[name]
            if name == "params":
                assert len(actual) == len(expected)
                assert all(left is right for left, right in zip(actual, expected))
            elif isinstance(expected, torch.Tensor):
                assert actual is expected
            else:
                assert actual == expected

    assert set(optimizer.state) == set(snapshot["state"])
    for parameter, expected_state in snapshot["state"].items():
        assert set(optimizer.state[parameter]) == set(expected_state)
        for name, expected in expected_state.items():
            assert optimizer.state[parameter][name] is expected
            torch.testing.assert_close(expected, snapshot["state_values"][(parameter, name)])

    assert set(optimizer._hyperparam_tensors) == set(snapshot["cache"])
    for key, expected in snapshot["cache"].items():
        assert optimizer._hyperparam_tensors[key] is expected
        torch.testing.assert_close(expected, snapshot["cache_values"][key])
    assert optimizer._live_hyperparams_by_group == snapshot["live"]
    assert optimizer.defaults is snapshot["defaults_object"]
    assert optimizer.defaults == snapshot["defaults"]


def _invoke_entry(
    entry: str,
    mutation: Callable[[dict, dict], None],
) -> NorDion2:
    optimizer, parameters = _make_optimizer()
    if entry == "load_state_dict":
        checkpoint = copy.deepcopy(optimizer.state_dict())
        mutation(checkpoint["state"], {"nordion2": 0, "adamw": 1, "lion": 2})
        target, _ = _make_optimizer()
        with pytest.raises(RuntimeError, match="optimizer state"):
            target.load_state_dict(checkpoint)
        return target

    mutation(optimizer.state, parameters)
    if entry == "step":
        for parameter in parameters.values():
            parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="optimizer state"):
        if entry == "step":
            optimizer.step()
        else:
            optimizer.state_dict()
    if entry == "step":
        assert all(group["step"] == 0 for group in optimizer.param_groups)
    return optimizer


@pytest.mark.parametrize("entry", ["step", "state_dict", "load_state_dict"])
def test_missing_parameter_state_fails_closed(entry: str) -> None:
    def remove_parameter_state(state: dict, keys: dict) -> None:
        del state[keys["lion"]]

    optimizer = _invoke_entry(entry, remove_parameter_state)
    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize("entry", ["step", "state_dict", "load_state_dict"])
@pytest.mark.parametrize(
    ("algorithm", "field"),
    [
        ("nordion2", "momentum"),
        ("nordion2", "variance_neuron"),
        ("adamw", "momentum"),
        ("adamw", "variance"),
        ("lion", "momentum"),
    ],
)
def test_missing_algorithm_state_field_fails_closed(entry: str, algorithm: str, field: str) -> None:
    def remove_field(state: dict, keys: dict) -> None:
        del state[keys[algorithm]][field]

    optimizer = _invoke_entry(entry, remove_field)
    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize("entry", ["step", "state_dict", "load_state_dict"])
@pytest.mark.parametrize("malformation", ["shape", "dtype"])
def test_runtime_and_checkpoint_boundaries_reject_malformed_state_tensor(
    entry: str,
    malformation: str,
) -> None:
    def malform_momentum(state: dict, keys: dict) -> None:
        momentum = state[keys["nordion2"]]["momentum"]
        if malformation == "shape":
            state[keys["nordion2"]]["momentum"] = momentum[1:]
        else:
            state[keys["nordion2"]]["momentum"] = momentum.to(torch.float64)

    optimizer = _invoke_entry(entry, malform_momentum)
    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize("state_initialization", ["eager", "deferred"])
@pytest.mark.parametrize(
    "malformation",
    [
        "algorithm",
        "step_negative",
        "step_type",
        "step_range",
        "hyperparameter",
        "nonfinite_hyperparameter",
        "negative_weight_decay",
        "missing_field",
        "collapsed_parameter_key",
    ],
)
def test_malformed_group_envelope_load_is_atomic_and_fails_closed(
    state_initialization: str,
    malformation: str,
) -> None:
    source, _ = _make_optimizer(state_initialization=state_initialization)
    checkpoint = copy.deepcopy(source.state_dict())
    if malformation == "algorithm":
        checkpoint["param_groups"][0]["algorithm"] = "lion"
    elif malformation == "step_negative":
        checkpoint["param_groups"][0]["step"] = -1
    elif malformation == "step_type":
        checkpoint["param_groups"][0]["step"] = 1.5
    elif malformation == "step_range":
        checkpoint["param_groups"][0]["step"] = 2**63
    elif malformation == "hyperparameter":
        checkpoint["param_groups"][0]["lr"] = torch.ones(2)
    elif malformation == "nonfinite_hyperparameter":
        checkpoint["param_groups"][0]["weight_decay"] = float("nan")
    elif malformation == "negative_weight_decay":
        checkpoint["param_groups"][0]["weight_decay"] = -0.1
    elif malformation == "missing_field":
        del checkpoint["param_groups"][0]["epsilon"]
    else:
        checkpoint["param_groups"][1]["params"][0] = checkpoint["param_groups"][0]["params"][0]

    target, _ = _make_optimizer(state_initialization=state_initialization)
    snapshot = _capture_live_optimizer(target)
    with pytest.raises(RuntimeError, match="optimizer state_dict"):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_permanently_failed(target)


def test_eager_same_group_duplicate_parameter_state_dict_roundtrip() -> None:
    source_parameter = torch.nn.Parameter(torch.randn(4, 6))
    with pytest.warns(UserWarning, match="duplicate parameters"):
        source = NorDion2(
            [{"params": [source_parameter, source_parameter], "algorithm": "nordion2"}],
            adjust_lr=None,
        )
    source.state[source_parameter]["momentum"].fill_(2.0)
    source.state[source_parameter]["variance_neuron"].fill_(3.0)
    checkpoint = copy.deepcopy(source.state_dict())
    assert checkpoint["param_groups"][0]["params"][0] == checkpoint["param_groups"][0]["params"][1]

    target_parameter = torch.nn.Parameter(torch.randn(4, 6))
    with pytest.warns(UserWarning, match="duplicate parameters"):
        target = NorDion2(
            [{"params": [target_parameter, target_parameter], "algorithm": "nordion2"}],
            adjust_lr=None,
        )
    target.load_state_dict(checkpoint)

    assert target.is_state_materialized is True
    torch.testing.assert_close(target.state[target_parameter]["momentum"], torch.full_like(target_parameter, 2.0))
    torch.testing.assert_close(
        target.state[target_parameter]["variance_neuron"],
        torch.full_like(target_parameter[..., :1], 3.0),
    )
    roundtrip_keys = target.state_dict()["param_groups"][0]["params"]
    assert roundtrip_keys[0] == roundtrip_keys[1]


@pytest.mark.parametrize("state_initialization", ["eager", "deferred"])
def test_load_pre_hook_repairs_payload_and_runs_once(state_initialization: str) -> None:
    source, _ = _make_optimizer(state_initialization=state_initialization)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"][0]["algorithm"] = "lion"

    target, _ = _make_optimizer(state_initialization=state_initialization)
    calls = {"pre": 0, "post": 0}

    def repair_payload(optimizer: NorDion2, payload: dict) -> dict:
        assert optimizer is target
        calls["pre"] += 1
        repaired = copy.deepcopy(payload)
        repaired["param_groups"][0]["algorithm"] = "nordion2"
        return repaired

    def record_post_hook(optimizer: NorDion2) -> None:
        assert optimizer is target
        calls["post"] += 1

    target.register_load_state_dict_pre_hook(repair_payload)
    target.register_load_state_dict_post_hook(record_post_hook)
    target.load_state_dict(checkpoint)

    assert calls == {"pre": 1, "post": 1}
    assert target.is_state_materialized is (state_initialization == "eager")


@pytest.mark.parametrize("state_initialization", ["eager", "deferred"])
def test_load_pre_hook_malformation_is_atomic_and_fails_closed(
    state_initialization: str,
) -> None:
    source, _ = _make_optimizer(state_initialization=state_initialization)
    checkpoint = copy.deepcopy(source.state_dict())
    target, _ = _make_optimizer(state_initialization=state_initialization)
    snapshot = _capture_live_optimizer(target)
    calls = 0

    def malform_payload(optimizer: NorDion2, payload: dict) -> None:
        nonlocal calls
        assert optimizer is target
        calls += 1
        payload["param_groups"][0]["algorithm"] = "lion"

    target.register_load_state_dict_pre_hook(malform_payload)
    with pytest.raises(RuntimeError, match="optimizer state_dict algorithm"):
        target.load_state_dict(checkpoint)

    assert calls == 1
    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_permanently_failed(target)


def test_staging_defaults_are_isolated_when_post_super_validation_fails() -> None:
    source, _ = _make_optimizer()
    checkpoint = copy.deepcopy(source.state_dict())
    target, _ = _make_optimizer()
    assert "differentiable" not in target.defaults
    snapshot = _capture_live_optimizer(target)

    def reject_staged_state() -> None:
        raise RuntimeError("staged validation rejection")

    target._validate_materialized_state = reject_staged_state
    with pytest.raises(RuntimeError, match="staged validation rejection"):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    assert "differentiable" not in target.defaults
    _assert_permanently_failed(target)


def test_load_post_hook_exception_keeps_committed_state_and_fails_closed() -> None:
    source, parameters = _make_optimizer()
    source.state[parameters["nordion2"]]["momentum"].fill_(4.0)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"][0]["step"] = 7

    target, target_parameters = _make_optimizer()
    calls = 0

    def reject_after_commit(optimizer: NorDion2) -> None:
        nonlocal calls
        assert optimizer is target
        calls += 1
        raise RuntimeError("post-hook rejection")

    target.register_load_state_dict_post_hook(reject_after_commit)
    with pytest.raises(RuntimeError, match="post-hook rejection"):
        target.load_state_dict(checkpoint)

    assert calls == 1
    assert target.param_groups[0]["step"] == 7
    torch.testing.assert_close(
        target.state[target_parameters["nordion2"]]["momentum"],
        torch.full_like(target_parameters["nordion2"], 4.0),
    )
    _assert_permanently_failed(target)


def test_load_post_hook_cannot_return_invalid_optimizer() -> None:
    source, _ = _make_optimizer()
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"][0]["step"] = 7

    target, target_parameters = _make_optimizer()
    calls = 0

    def invalidate_committed_state(optimizer: NorDion2) -> None:
        nonlocal calls
        calls += 1
        del optimizer.state[target_parameters["nordion2"]]["momentum"]

    target.register_load_state_dict_post_hook(invalidate_committed_state)
    with pytest.raises(RuntimeError, match="optimizer state fields"):
        target.load_state_dict(checkpoint)

    assert calls == 1
    assert target.param_groups[0]["step"] == 7
    assert "momentum" not in target.state[target_parameters["nordion2"]]
    _assert_permanently_failed(target)


@pytest.mark.parametrize("entry", ["step", "state_dict"])
def test_materialized_adamw_missing_step_dev_fails_closed(entry: str) -> None:
    optimizer, parameters = _make_optimizer()
    del optimizer.state[parameters["adamw"]]["step_dev"]
    if entry == "step":
        parameters["adamw"].grad = torch.zeros_like(parameters["adamw"])

    with pytest.raises(RuntimeError, match="step_dev"):
        if entry == "step":
            optimizer.step()
        else:
            optimizer.state_dict()

    _assert_permanently_failed(optimizer)


def test_legacy_adamw_load_restores_step_dev_from_group_step() -> None:
    source, _ = _make_optimizer()
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"][1]["step"] = 7
    del checkpoint["state"][1]["step_dev"]

    target, parameters = _make_optimizer()
    target.load_state_dict(checkpoint)

    step_dev = target.state[parameters["adamw"]]["step_dev"]
    assert step_dev.shape == torch.Size([])
    assert step_dev.dtype == torch.float32
    assert step_dev.item() == 7
    assert target.is_state_materialized is True
    target.state_dict()


def test_failed_load_never_lazy_rebuilds_state() -> None:
    source, _ = _make_optimizer()
    checkpoint = copy.deepcopy(source.state_dict())
    del checkpoint["state"][0]["variance_neuron"]

    target, parameters = _make_optimizer()
    original_state = dict(target.state[parameters["nordion2"]])
    with pytest.raises(RuntimeError, match="optimizer state"):
        target.load_state_dict(checkpoint)

    parameters["nordion2"].grad = torch.randn_like(parameters["nordion2"])
    _assert_permanently_failed(target)
    assert set(target.state[parameters["nordion2"]]) == set(original_state)
    for name, value in original_state.items():
        assert target.state[parameters["nordion2"]][name] is value


def test_deferred_bootstrap_roundtrip_stays_deferred() -> None:
    source, _ = _make_optimizer(state_initialization="deferred")
    checkpoint = source.state_dict()

    target, _ = _make_optimizer(state_initialization="deferred")
    target.load_state_dict(checkpoint)

    assert target.is_state_materialized is False
    assert target.state == {}


def test_invalid_group_step_fails_closed() -> None:
    optimizer, _ = _make_optimizer()
    optimizer.param_groups[0]["step"] = -1

    with pytest.raises(RuntimeError, match="invalid step"):
        optimizer.state_dict()

    _assert_permanently_failed(optimizer)


def test_step_refreshes_reassigned_hyperparameter_before_optimizer_update() -> None:
    optimizer, _ = _make_optimizer()
    cached_lr = optimizer._hyperparam_tensors[(0, "lr")]
    optimizer.param_groups[0]["lr"] = 0.1

    optimizer.step()

    assert optimizer.is_state_materialized is True
    assert optimizer.param_groups[0]["lr"] is cached_lr
    assert cached_lr.item() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lr", torch.ones(2)),
        ("lr", object()),
        ("lr", float("nan")),
        ("lr", -0.1),
        ("weight_decay", torch.ones(2)),
        ("weight_decay", object()),
        ("weight_decay", float("inf")),
        ("weight_decay", -0.1),
    ],
)
def test_step_rejects_invalid_live_hyperparameter_before_increment(
    name: str,
    value: object,
) -> None:
    if name == "weight_decay":
        parameter = torch.nn.Parameter(torch.randn(8))
        optimizer = NorDion2(
            [
                {
                    "params": [parameter],
                    "algorithm": "lion",
                    "weight_decay": torch.tensor(0.01),
                }
            ]
        )
    else:
        optimizer, _ = _make_optimizer()
    optimizer.param_groups[0][name] = value

    with pytest.raises(RuntimeError, match="live hyperparameter"):
        optimizer.step()

    assert all(group["step"] == 0 for group in optimizer.param_groups)
    _assert_permanently_failed(optimizer)


def test_step_strictly_validates_only_active_parameter_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inactive_parameters = [torch.nn.Parameter(torch.randn(8)) for _ in range(128)]
    active_parameter = torch.nn.Parameter(torch.randn(8))
    with pytest.warns(UserWarning, match="duplicate parameters"):
        optimizer = NorDion2(
            [
                {
                    "params": [*inactive_parameters, active_parameter, active_parameter],
                    "algorithm": "lion",
                }
            ]
        )
    active_parameter.grad = torch.zeros_like(active_parameter)
    validated_parameters: list[torch.Tensor] = []
    runtime_validated_parameters: list[torch.Tensor] = []
    original_validator = optimizer._validate_state_entry

    def record_validation(
        parameter: torch.Tensor,
        algorithm: str,
        state: dict,
        *,
        validate_device: bool,
        allow_legacy_adamw_step: bool = False,
    ) -> None:
        validated_parameters.append(parameter)
        original_validator(
            parameter,
            algorithm,
            state,
            validate_device=validate_device,
            allow_legacy_adamw_step=allow_legacy_adamw_step,
        )

    monkeypatch.setattr(optimizer, "_validate_state_entry", record_validation)
    monkeypatch.setattr(
        optimizer,
        "_validate_state_runtime_metadata",
        lambda parameter, algorithm, state: runtime_validated_parameters.append(parameter),
        raising=False,
    )
    monkeypatch.setattr(optimizer, "_create_ortho_tasks", lambda groups: ())
    monkeypatch.setattr(optimizer, "_create_lion_tasks", lambda groups: ())
    monkeypatch.setattr(optimizer, "_create_adamw_tasks", lambda groups: ())
    optimizer.step()

    assert validated_parameters == [active_parameter]
    assert runtime_validated_parameters == []
    assert optimizer.param_groups[0]["step"] == 1


def test_inactive_state_corruption_is_delayed_until_first_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, parameters = _make_optimizer()
    del optimizer.state[parameters["lion"]]["momentum"]
    parameters["nordion2"].grad = torch.zeros_like(parameters["nordion2"])
    monkeypatch.setattr(optimizer, "_create_ortho_tasks", lambda groups: ())
    monkeypatch.setattr(optimizer, "_create_lion_tasks", lambda groups: ())
    monkeypatch.setattr(optimizer, "_create_adamw_tasks", lambda groups: ())

    optimizer.step()

    assert all(group["step"] == 1 for group in optimizer.param_groups)
    parameters["nordion2"].grad = None
    parameters["lion"].grad = torch.zeros_like(parameters["lion"])
    with pytest.raises(RuntimeError, match="optimizer state fields"):
        optimizer.step()

    assert all(group["step"] == 1 for group in optimizer.param_groups)
    _assert_permanently_failed(optimizer)


def test_inactive_state_corruption_is_rejected_by_state_dict() -> None:
    optimizer, parameters = _make_optimizer()
    del optimizer.state[parameters["lion"]]["momentum"]

    with pytest.raises(RuntimeError, match="optimizer state fields"):
        optimizer.state_dict()

    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize("malformation", ["shape", "dtype", "device"])
def test_step_rejects_malformed_hyperparameter_cache(malformation: str) -> None:
    optimizer, _ = _make_optimizer()
    if malformation == "shape":
        cache = torch.ones(2, dtype=torch.float32)
    elif malformation == "dtype":
        cache = torch.ones((), dtype=torch.float64)
    else:
        cache = torch.ones((), dtype=torch.float32, device="meta")
    optimizer._hyperparam_tensors[(0, "lr")] = cache

    with pytest.raises(RuntimeError, match="malformed lr cache"):
        optimizer.step()

    assert optimizer.param_groups[0]["step"] == 0
    _assert_permanently_failed(optimizer)


def test_step_rejects_plain_state_device_corruption_before_increment() -> None:
    optimizer, parameters = _make_optimizer()
    optimizer.state[parameters["nordion2"]]["momentum"] = torch.zeros(
        parameters["nordion2"].shape,
        dtype=parameters["nordion2"].dtype,
        device="meta",
    )
    parameters["nordion2"].grad = torch.zeros_like(parameters["nordion2"])

    with pytest.raises(RuntimeError, match="device"):
        optimizer.step()

    assert optimizer.param_groups[0]["step"] == 0
    _assert_permanently_failed(optimizer)


def test_state_coverage_error_reports_missing_and_orphan_counts() -> None:
    optimizer, parameters = _make_optimizer()
    del optimizer.state[parameters["lion"]]
    optimizer.state[torch.nn.Parameter(torch.randn(3))] = {}

    with pytest.raises(RuntimeError, match=r"missing=1, orphan=1"):
        optimizer.state_dict()

    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize(
    ("mesh_case", "match"),
    [
        ("missing", "DeviceMesh"),
        ("wrong", "device mesh"),
        ("last_dim", "last dimension"),
    ],
)
def test_deferred_materialization_rejects_unsupported_dtensor_distribution(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
    mesh_case: str,
    match: str,
) -> None:
    parameter_mesh, wrong_mesh = cpu_meshes
    placements: list[Placement] = [Shard(1)] if mesh_case == "last_dim" else [Replicate()]
    parameter = _make_dtensor_parameter(parameter_mesh, placements)
    optimizer_mesh = None if mesh_case == "missing" else wrong_mesh if mesh_case == "wrong" else parameter_mesh
    optimizer = NorDion2(
        [{"params": [parameter], "algorithm": "nordion2"}],
        distributed_mesh=optimizer_mesh,
        state_initialization="deferred",
    )

    with pytest.raises((RuntimeError, NotImplementedError), match=match):
        optimizer.materialize_state()

    _assert_permanently_failed(optimizer)


def test_step_rejects_dtensor_mesh_corruption_before_increment(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
) -> None:
    parameter_mesh, _ = cpu_meshes
    parameter = _make_dtensor_parameter(parameter_mesh, [Replicate()])
    optimizer = NorDion2(
        [{"params": [parameter], "algorithm": "nordion2"}],
        distributed_mesh=parameter_mesh,
    )
    optimizer._distributed_mesh = None
    optimizer._process_group = None
    parameter.grad = DTensor.from_local(
        torch.zeros_like(parameter.to_local()),
        parameter_mesh,
        [Replicate()],
        run_check=False,
    )

    with pytest.raises(RuntimeError, match="DeviceMesh"):
        optimizer.step()

    assert optimizer.param_groups[0]["step"] == 0
    _assert_permanently_failed(optimizer)


@pytest.mark.parametrize(
    ("malformation", "match"),
    [
        ("placement", "placements"),
        ("mesh", "device mesh"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("plain_tensor", "DTensor identity"),
        ("local_device", "device"),
    ],
)
def test_step_rejects_dtensor_state_runtime_metadata_before_increment(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
    malformation: str,
    match: str,
) -> None:
    parameter_mesh, wrong_mesh = cpu_meshes
    parameter = _make_dtensor_parameter(parameter_mesh, [Replicate()])
    optimizer = NorDion2(
        [{"params": [parameter], "algorithm": "nordion2"}],
        distributed_mesh=parameter_mesh,
    )
    local = parameter.to_local()
    if malformation == "placement":
        replacement = DTensor.from_local(
            torch.zeros_like(local),
            parameter_mesh,
            [Shard(0)],
            run_check=False,
        )
    elif malformation == "mesh":
        replacement = DTensor.from_local(
            torch.zeros_like(local),
            wrong_mesh,
            [Replicate()],
            run_check=False,
        )
    elif malformation == "shape":
        replacement = DTensor.from_local(
            torch.zeros(3, 6),
            parameter_mesh,
            [Replicate()],
            run_check=False,
        )
    elif malformation == "dtype":
        replacement = DTensor.from_local(
            torch.zeros_like(local, dtype=torch.float64),
            parameter_mesh,
            [Replicate()],
            run_check=False,
        )
    elif malformation == "local_device":
        replacement = DTensor.from_local(
            torch.zeros(local.shape, dtype=local.dtype, device="meta"),
            parameter_mesh,
            [Replicate()],
            run_check=False,
        )
    else:
        replacement = torch.zeros_like(local)
    optimizer.state[parameter]["momentum"] = replacement
    parameter.grad = DTensor.from_local(
        torch.zeros_like(local),
        parameter_mesh,
        [Replicate()],
        run_check=False,
    )

    with pytest.raises(RuntimeError, match=match):
        optimizer.step()

    assert optimizer.param_groups[0]["step"] == 0
    _assert_permanently_failed(optimizer)


def test_step_accepts_equivalent_dtensor_state_mesh_group(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter_mesh, _ = cpu_meshes
    equivalent_mesh = DeviceMesh.from_group(
        parameter_mesh.get_group(),
        "cpu",
        mesh_dim_names=("equivalent",),
    )
    parameter = _make_dtensor_parameter(parameter_mesh, [Replicate()])
    optimizer = NorDion2(
        [{"params": [parameter], "algorithm": "nordion2"}],
        distributed_mesh=parameter_mesh,
    )
    optimizer.state[parameter]["momentum"] = DTensor.from_local(
        torch.zeros_like(parameter.to_local()),
        equivalent_mesh,
        [Replicate()],
        run_check=False,
    )
    parameter.grad = DTensor.from_local(
        torch.zeros_like(parameter.to_local()),
        parameter_mesh,
        [Replicate()],
        run_check=False,
    )
    monkeypatch.setattr(optimizer, "_create_ortho_tasks", lambda groups: ())

    optimizer.step()

    assert optimizer.param_groups[0]["step"] == 1


def test_load_rejects_dtensor_mesh_corruption_atomically(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
) -> None:
    parameter_mesh, wrong_mesh = cpu_meshes
    source_parameter = _make_dtensor_parameter(parameter_mesh, [Replicate()])
    source = NorDion2(
        [{"params": [source_parameter], "algorithm": "nordion2"}],
        distributed_mesh=parameter_mesh,
    )
    checkpoint = copy.deepcopy(source.state_dict())

    target_parameter = _make_dtensor_parameter(parameter_mesh, [Replicate()])
    target = NorDion2(
        [{"params": [target_parameter], "algorithm": "nordion2"}],
        distributed_mesh=parameter_mesh,
    )
    target._distributed_mesh = wrong_mesh
    target._process_group = wrong_mesh.get_group()
    snapshot = _capture_live_optimizer(target)

    with pytest.raises(RuntimeError, match="device mesh"):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_permanently_failed(target)


def test_dtensor_state_placement_mismatch_fails_closed(
    cpu_meshes: tuple[DeviceMesh, DeviceMesh],
) -> None:
    mesh, _ = cpu_meshes
    parameter = torch.nn.Parameter(
        DTensor.from_local(
            torch.randn(4, 6),
            mesh,
            [Replicate()],
            run_check=False,
        )
    )
    optimizer = NorDion2(
        [{"params": [parameter], "algorithm": "nordion2"}],
        distributed_mesh=mesh,
    )
    optimizer.state[parameter]["momentum"] = DTensor.from_local(
        torch.zeros_like(parameter.to_local()),
        mesh,
        [Shard(0)],
        run_check=False,
    )

    with pytest.raises(RuntimeError, match="placements"):
        optimizer.state_dict()

    _assert_permanently_failed(optimizer)
