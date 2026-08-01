"""Contract tests for the CUDA specialization of the shared PyTorch backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gepa_mindfulness.training.backends.torch_cuda import (
    CudaOutOfMemoryError,
    create_cuda_backend,
    detect_cuda_capabilities,
)
from gepa_mindfulness.training.capability import (
    Capability,
    CapabilityError,
    CapabilityState,
)
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    PolicyConfig,
    RLRunConfig,
    RuntimeConfig,
    load_rl_config,
)


def _cuda_config(*, device: str = "cuda:0", precision: str = "fp32") -> RLRunConfig:
    return RLRunConfig(
        runtime=RuntimeConfig(backend="cuda", device=device, precision=precision),
        policy=PolicyConfig(model_name="LOCAL_MODEL_PATH", max_new_tokens=17),
        algorithm=AlgorithmConfig(batch_size=3, gradient_accumulation_steps=5),
    )


def _available_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_count: int = 1,
    bf16_supported: bool = True,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)


def _never_called() -> tuple[object, object]:
    raise AssertionError("model loading must happen only after CUDA validation")


def test_cuda_factory_rejects_unavailable_runtime_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the availability preflight would invoke model loading on a CPU-only host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(CapabilityError, match="CUDA is unavailable"):
        create_cuda_backend(_cuda_config(), _never_called)


@pytest.mark.parametrize("device", ["cpu", "cuda:-1", "cuda:one", "cuda:0:1"])
def test_cuda_detection_rejects_malformed_device_selector(device: str) -> None:
    """Relaxing CUDA selector parsing would let an ambiguous device reach model loading."""
    with pytest.raises(CapabilityError, match="CUDA device"):
        detect_cuda_capabilities(device, "fp32")


@pytest.mark.parametrize(
    ("backend", "device", "precision", "message"),
    [
        ("cuda", "cpu", "fp32", "CUDA device"),
        ("pytorch", "cpu", "fp16", "mixed precision requires a CUDA device"),
        ("cuda", "cuda:0", "tf32", "runtime.precision"),
    ],
)
def test_runtime_config_rejects_unsupported_cuda_combinations(
    backend: str,
    device: str,
    precision: str,
    message: str,
) -> None:
    """Removing cross-field validation would defer a known failure until model loading."""
    with pytest.raises(ValueError, match=message):
        RuntimeConfig(backend=backend, device=device, precision=precision)


def test_cuda_detection_rejects_unknown_precision() -> None:
    """A public detector must not silently downgrade an unknown precision to FP32."""
    with pytest.raises(CapabilityError, match="precision"):
        detect_cuda_capabilities("cuda:0", "tf32")


def test_cuda_factory_rejects_out_of_range_device_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the index bound would defer a deterministic configuration error to Torch."""
    _available_cuda(monkeypatch, device_count=1)

    with pytest.raises(CapabilityError, match=r"cuda:2.*1 visible"):
        create_cuda_backend(_cuda_config(device="cuda:2"), _never_called)


def test_cuda_factory_rejects_bf16_without_hardware_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claiming BF16 without the runtime probe would expose unsupported mixed precision."""
    _available_cuda(monkeypatch, bf16_supported=False)

    with pytest.raises(CapabilityError, match="BF16"):
        create_cuda_backend(_cuda_config(precision="bf16"), _never_called)


@pytest.mark.parametrize(
    ("precision", "autocast_dtype", "uses_scaler"),
    [
        ("fp32", None, False),
        ("fp16", torch.float16, True),
        ("bf16", torch.bfloat16, False),
    ],
)
def test_cuda_factory_selects_amp_components(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    autocast_dtype: torch.dtype | None,
    uses_scaler: bool,
) -> None:
    """Swapping dtype or loss-scaling branches would silently change numeric behavior."""
    _available_cuda(monkeypatch)
    captured: dict[str, object] = {}
    scaler = object()

    def fake_backend(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fake_backend,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda._create_grad_scaler",
        lambda: scaler,
    )

    backend = create_cuda_backend(
        _cuda_config(precision=precision),
        lambda: (object(), object()),
    )

    assert backend.autocast_dtype is autocast_dtype
    assert captured["gradient_scaler"] is (scaler if uses_scaler else None)
    assert captured["device"] == "cuda:0"


@pytest.mark.parametrize(
    ("precision", "mixed_precision_state"),
    [
        ("fp32", CapabilityState.UNSUPPORTED),
        ("fp16", CapabilityState.SUPPORTED),
        ("bf16", CapabilityState.SUPPORTED),
    ],
)
def test_cuda_detection_reports_evidence_backed_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    mixed_precision_state: CapabilityState,
) -> None:
    """Hard-coded capability claims would hide the selected precision and runtime probes."""
    _available_cuda(monkeypatch)

    capabilities = detect_cuda_capabilities("cuda:0", precision)

    assert capabilities.backend_name == "torch_cuda"
    assert capabilities.state(Capability.SUPPORTS_CUDA) is CapabilityState.SUPPORTED
    assert capabilities.state(Capability.SUPPORTS_MIXED_PRECISION) is mixed_precision_state
    evidence = capabilities.capabilities[Capability.SUPPORTS_CUDA].evidence
    assert "cuda:0" in evidence


def test_cuda_factory_translates_oom_with_actionable_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-raising raw CUDA OOM would omit the settings needed to reduce memory pressure."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("allocation failed")

    def fail_backend(**kwargs: object) -> object:
        del kwargs
        raise original

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fail_backend,
    )

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(precision="fp16"), lambda: (object(), object()))

    error = captured.value
    assert error.operation == "backend_initialization"
    assert error.device == "cuda:0"
    assert error.precision == "fp16"
    assert error.batch_size == 3
    assert error.max_new_tokens == 17
    assert error.gradient_accumulation_steps == 5
    assert error.__cause__ is original
    assert "reduce algorithm.batch_size" in str(error)


def test_cuda_factory_translates_model_loading_oom_with_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An asset-loading OOM must identify its operation without hiding the Torch failure."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("model allocation failed")

    def fail_model_factory() -> tuple[object, object]:
        raise original

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(), fail_model_factory)

    assert captured.value.operation == "model_loading"
    assert captured.value.__cause__ is original


def test_cuda_factory_preserves_non_oom_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broad exception translation would misdiagnose model or tokenizer failures as CUDA OOM."""
    _available_cuda(monkeypatch)
    original = RuntimeError("invalid model assets")

    def fail_model_factory() -> tuple[object, object]:
        raise original

    with pytest.raises(RuntimeError) as captured:
        create_cuda_backend(_cuda_config(), fail_model_factory)

    assert captured.value is original


def test_cuda_single_gpu_template_is_strict_and_uses_bundled_pairs() -> None:
    """A stale key or remote model name would make the checked-in CUDA template misleading."""
    path = Path(__file__).parents[1] / "configs" / "rl" / "cuda_single_gpu.yaml"

    config = load_rl_config(path)

    assert config.runtime == RuntimeConfig(backend="cuda", device="cuda:0", precision="fp32")
    assert config.policy.model_name == "LOCAL_MODEL_PATH"
    assert config.dataset.train_path == "data/synthetic/reward_integrity/rl_pairs_v1.jsonl"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_detect_cuda_capabilities_on_visible_hardware() -> None:
    """A CUDA build that cannot substantiate device zero must fail its hardware smoke test."""
    capabilities = detect_cuda_capabilities("cuda:0", "fp32")

    assert capabilities.state(Capability.SUPPORTS_CUDA) is CapabilityState.SUPPORTED
