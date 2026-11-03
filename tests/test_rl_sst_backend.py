"""Functional applied-update and restart checks for canonical CPU SST training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_rl_torch_backend import (
    TinyCausalLM,
    TinyTokenizer,
    _assert_backend_snapshot,
    _assert_nested_equal,
    _backend_snapshot,
)
from torch import nn

from gepa_mindfulness.training.backends.torch_policy import TorchPolicyBackend
from gepa_mindfulness.training.backends.torch_portable import create_portable_backend
from gepa_mindfulness.training.runtime_config import RLRunConfig
from gepa_mindfulness.training.subspace_tuning import SSTConfig


def _backend(
    *,
    enabled: bool = True,
    rank: float = 0.5,
    refresh: int | None = None,
    optimizer: str = "AdamW",
    seed: int = 7,
) -> TorchPolicyBackend:
    """Construct actual matching native optimizers without external model assets."""
    torch.manual_seed(seed)
    model = TinyCausalLM()
    value_head = nn.Linear(6, 1)
    native = getattr(torch.optim, optimizer)(
        [*model.parameters(), *value_head.parameters()], lr=0.03, weight_decay=0.1
    )
    return TorchPolicyBackend(
        policy_model=model,
        tokenizer=TinyTokenizer(),
        value_head=value_head,
        optimizer=native,
        learning_rate=0.03,
        model_identifier="tiny-sst-local",
        sst_config=SSTConfig(enabled=enabled, rank_fraction=rank, refresh_every=refresh),
    )


def _step(backend: TorchPolicyBackend):
    """Supply reproducible gradients while retaining real moments and weight decay."""
    backend.zero_grad()
    for parameter in [*backend.policy_parameters(), *backend.value_head.parameters()]:
        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
        parameter.grad = (values + 1) / (parameter.numel() + 1)
    return backend.optimizer_step()


def _payload(backend: TorchPolicyBackend, path: Path) -> dict:
    """Read the public versioned backend checkpoint rather than private adapter state."""
    backend.save_checkpoint(path)
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("optimizer", ["SGD", "Adam", "AdamW"])
def test_backend_projects_the_applied_native_update_including_weight_decay(optimizer: str) -> None:
    """Projecting raw gradients or ignoring weight decay must violate the matrix equation."""
    protected = _backend(optimizer=optimizer)
    control = _backend(enabled=False, optimizer=optimizer)
    initial = deepcopy(protected.policy_model.state_dict())
    _step(control)
    result = _step(protected)
    assert result.step == 1 and result.updated is True
    assert result.sst_report["source_reference"] == "REF-SST"
    assert result.sst_report["authority_granted"] is False
    for name, actual in protected.policy_model.state_dict().items():
        anchor = initial[name].double()
        raw_delta = control.policy_model.state_dict()[name].double() - anchor
        left, _, right = torch.linalg.svd(anchor, full_matrices=False)
        rank = min(anchor.shape) // 2
        left, right = left[:, :rank], right[:rank].T
        expected_delta = (
            (torch.eye(anchor.shape[0]) - left @ left.T)
            @ raw_delta
            @ (torch.eye(anchor.shape[1]) - right @ right.T)
        )
        expected = (anchor + expected_delta).to(actual.dtype)
        tolerance = 8 * torch.finfo(actual.dtype).eps * float(anchor.abs().max())
        torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
        assert not torch.equal(actual, control.policy_model.state_dict()[name])
    _assert_nested_equal(protected.value_head.state_dict(), control.value_head.state_dict())


@pytest.mark.parametrize("rank", [0.0, 1.0])
def test_backend_rank_extremes_preserve_native_or_freeze_policy_updates(rank: float) -> None:
    """The two boundary ranks must not become implicit unprotected fallbacks."""
    protected = _backend(rank=rank)
    control = _backend(enabled=False)
    initial = deepcopy(protected.policy_model.state_dict())
    _step(control)
    _step(protected)
    expected = control.policy_model.state_dict() if rank == 0 else initial
    _assert_nested_equal(expected, protected.policy_model.state_dict())
    _assert_nested_equal(protected.value_head.state_dict(), control.value_head.state_dict())


@pytest.mark.parametrize("refresh", [None, 2])
def test_sst_checkpoint_resume_matches_uninterrupted_updates(
    tmp_path: Path, refresh: int | None
) -> None:
    """Restoring weights without the saved basis/counters must change the next applied update."""
    uninterrupted = _backend(refresh=refresh)
    anchors = deepcopy(uninterrupted.policy_model.state_dict())
    _step(uninterrupted)
    checkpoint = tmp_path / "sst.pt"
    saved = _payload(uninterrupted, checkpoint)
    assert saved["format_version"] == 3 and saved["sst_state"]["step"] == 1
    for name, value in anchors.items():
        torch.testing.assert_close(saved["sst_state"]["references"][f"policy.{name}"], value)
    resumed = _backend(refresh=refresh, seed=91)
    before = _backend_snapshot(resumed)
    assert resumed.preflight_checkpoint_bytes(checkpoint.read_bytes()).format_version == 3
    _assert_backend_snapshot(resumed, before)
    assert resumed.load_checkpoint(checkpoint).step == 1
    _step(uninterrupted)
    _step(resumed)
    _assert_nested_equal(uninterrupted.policy_model.state_dict(), resumed.policy_model.state_dict())
    _assert_nested_equal(uninterrupted.value_head.state_dict(), resumed.value_head.state_dict())
    _assert_nested_equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    final = _payload(resumed, tmp_path / "resumed.pt")
    assert final["sst_state"]["basis_step"] == (2 if refresh else 0)


@pytest.mark.parametrize("fault", ["counter", "reference", "config"])
def test_invalid_sst_checkpoint_is_rejected_without_mutating_live_state(
    tmp_path: Path, fault: str
) -> None:
    """Corrupt basis/counter/configuration data must fail before restore side effects."""
    backend = _backend()
    _step(backend)
    original = _payload(backend, tmp_path / "original.pt")
    corrupt = deepcopy(original)
    if fault == "counter":
        corrupt["sst_state"]["step"] += 1
    elif fault == "reference":
        next(iter(corrupt["sst_state"]["references"].values())).fill_(float("nan"))
    else:
        corrupt["sst_state"]["config"]["rank_fraction"] = 0.25
    bad_path = tmp_path / "bad.pt"
    torch.save(corrupt, bad_path)
    before = _backend_snapshot(backend)
    with pytest.raises(ValueError, match="SST|sst"):
        backend.load_checkpoint(bad_path)
    _assert_backend_snapshot(backend, before)
    after = _payload(backend, tmp_path / "after.pt")
    _assert_nested_equal(original["sst_state"], after["sst_state"])


def test_sst_checkpoint_cannot_silently_enable_or_drop_protection(tmp_path: Path) -> None:
    """Enabled and default legacy checkpoint modes cannot be interchanged."""
    enabled, disabled = _backend(), _backend(enabled=False)
    active_path, plain_path = tmp_path / "active.pt", tmp_path / "plain.pt"
    assert _payload(enabled, active_path)["format_version"] == 3
    plain = _payload(disabled, plain_path)
    assert plain["format_version"] == 2 and "sst_state" not in plain
    for backend, path in ((enabled, plain_path), (disabled, active_path)):
        before = _backend_snapshot(backend)
        with pytest.raises(ValueError, match="SST|sst"):
            backend.load_checkpoint(path)
        _assert_backend_snapshot(backend, before)


def test_restore_failure_rolls_back_sst_state_as_well_as_native_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after SST restoration cannot leave an older basis/counter installed."""
    backend = _backend(refresh=2)
    _step(backend)
    checkpoint = tmp_path / "step-one.pt"
    _payload(backend, checkpoint)
    _step(backend)
    original = _payload(backend, tmp_path / "step-two.pt")
    before = _backend_snapshot(backend)
    native_set_rng_state = torch.set_rng_state
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected RNG restore failure")
        return native_set_rng_state(state)

    monkeypatch.setattr(torch, "set_rng_state", fail_once)
    with pytest.raises(ValueError, match="incompatible training state"):
        backend.load_checkpoint(checkpoint)
    _assert_backend_snapshot(backend, before)
    after = _payload(backend, tmp_path / "after-failure.pt")
    _assert_nested_equal(original["sst_state"], after["sst_state"])


@pytest.mark.parametrize("mode", ["lora", "cuda", "autocast"])
def test_direct_backend_rejects_unsupported_sst_before_changing_model(mode: str) -> None:
    """A direct backend call cannot bypass canonical configuration compatibility checks."""
    model = TinyCausalLM()
    before = deepcopy(model.state_dict())
    kwargs = {"training_mode": "lora"} if mode == "lora" else {}
    if mode == "cuda":
        kwargs["device"] = "cuda:0"
    elif mode == "autocast":
        kwargs["autocast_dtype"] = torch.bfloat16
    with pytest.raises(ValueError, match="SST"):
        TorchPolicyBackend(
            policy_model=model,
            tokenizer=TinyTokenizer(),
            sst_config=SSTConfig(enabled=True),
            **kwargs,
        )
    _assert_nested_equal(before, model.state_dict())


def test_portable_factory_passes_enabled_sst_and_rejects_lora_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public portable factory must propagate protection and reject incompatible loading."""
    config = RLRunConfig(sst=SSTConfig(enabled=True, rank_fraction=0.5))
    backend = create_portable_backend(
        config, policy_model=TinyCausalLM(), tokenizer=TinyTokenizer()
    )
    assert _step(backend).sst_report["step"] == 1

    def forbidden_loader(*args):
        pytest.fail("incompatible SST/LoRA request reached model loading")

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_portable._load_transformers_assets",
        forbidden_loader,
    )
    with pytest.raises(ValueError, match="SST"):
        create_portable_backend(config, training_mode="lora")
