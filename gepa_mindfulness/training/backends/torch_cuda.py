"""CUDA validation and precision specialization for the shared PyTorch backend."""

from __future__ import annotations

import re
from collections.abc import Callable

import torch
from torch import nn

from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
)
from gepa_mindfulness.training.runtime_config import Precision, RLRunConfig

from .base import TokenizerLike
from .torch_policy import TorchPolicyBackend

ModelFactory = Callable[[], tuple[nn.Module, TokenizerLike]]
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?$")


class CudaOutOfMemoryError(RuntimeError):
    """A CUDA allocation failure with the run settings needed for remediation."""

    def __init__(self, operation: str, config: RLRunConfig) -> None:
        self.operation = operation
        self.device = config.runtime.device
        self.precision = config.runtime.precision
        self.batch_size = config.algorithm.batch_size
        self.max_new_tokens = config.policy.max_new_tokens
        self.gradient_accumulation_steps = config.algorithm.gradient_accumulation_steps
        super().__init__(
            f"CUDA ran out of memory during {operation} on {self.device} with "
            f"precision={self.precision}, algorithm.batch_size={self.batch_size}, "
            f"policy.max_new_tokens={self.max_new_tokens}, and "
            "algorithm.gradient_accumulation_steps="
            f"{self.gradient_accumulation_steps}; reduce algorithm.batch_size or "
            "policy.max_new_tokens, or increase algorithm.gradient_accumulation_steps."
        )


def detect_cuda_capabilities(
    device: str,
    precision: str,
) -> BackendCapabilities:
    """Validate one CUDA device and return only evidence-backed capability states."""
    if not isinstance(device, str):
        raise TypeError("CUDA device selector must be a string")
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None:
        raise CapabilityError("CUDA device must be 'cuda' or 'cuda:<non-negative index>'")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise CapabilityError("CUDA precision must be 'fp32', 'fp16', or 'bf16'")
    if not torch.cuda.is_available():
        raise CapabilityError(
            "CUDA is unavailable; install a CUDA-enabled PyTorch runtime and expose a device."
        )
    device_count = int(torch.cuda.device_count())
    device_index = int(match.group(1) or 0)
    if device_index >= device_count:
        raise CapabilityError(
            f"CUDA device cuda:{device_index} is out of range with {device_count} visible "
            "device(s)."
        )
    if precision == "bf16":
        try:
            with torch.cuda.device(device_index):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
        except (RuntimeError, TypeError) as error:
            raise CapabilityError(f"BF16 support could not be detected: {error}") from error
        if not bf16_supported:
            raise CapabilityError("BF16 precision is unsupported by the detected CUDA hardware")

    selected_device = f"cuda:{device_index}"
    mixed_precision = precision != "fp32"
    supported = {
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_CUDA,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
        Capability.SUPPORTS_GENERATION,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_REFERENCE_LOG_PROBS,
        Capability.SUPPORTS_TOKEN_LOG_PROBS,
        Capability.SUPPORTS_VALUE_HEAD,
    }
    if mixed_precision:
        supported.add(Capability.SUPPORTS_MIXED_PRECISION)
    capabilities = {
        capability: CapabilityEvidence(
            state=(
                CapabilityState.SUPPORTED
                if capability in supported
                else CapabilityState.UNSUPPORTED
            ),
            evidence=_capability_evidence(
                capability,
                selected_device=selected_device,
                precision=precision,
                supported=capability in supported,
            ),
        )
        for capability in Capability
    }
    return BackendCapabilities(
        backend_name="torch_cuda",
        backend_version=torch.__version__,
        capabilities=capabilities,
    )


def create_cuda_backend(
    config: RLRunConfig,
    model_factory: ModelFactory,
) -> TorchPolicyBackend:
    """Create the shared policy backend only after CUDA validation succeeds."""
    if not isinstance(config, RLRunConfig):
        raise TypeError("config must be an RLRunConfig")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    capabilities = detect_cuda_capabilities(
        config.runtime.device,
        config.runtime.precision,
    )
    required = {Capability.SUPPORTS_CUDA}
    if config.runtime.precision != "fp32":
        required.add(Capability.SUPPORTS_MIXED_PRECISION)
    capabilities.require(required)

    try:
        policy_model, tokenizer = model_factory()
    except torch.OutOfMemoryError as error:
        raise CudaOutOfMemoryError("model_loading", config) from error

    autocast_dtype = _autocast_dtype(config.runtime.precision)
    gradient_scaler = _create_grad_scaler() if config.runtime.precision == "fp16" else None
    try:
        return TorchPolicyBackend(
            policy_model=policy_model,
            tokenizer=tokenizer,
            device=config.runtime.device,
            learning_rate=config.algorithm.learning_rate,
            max_new_tokens=config.policy.max_new_tokens,
            max_grad_norm=config.algorithm.max_grad_norm,
            model_identifier=config.policy.model_name,
            backend_name="torch_cuda",
            autocast_dtype=autocast_dtype,
            gradient_scaler=gradient_scaler,
            oom_error_factory=lambda operation: CudaOutOfMemoryError(operation, config),
        )
    except torch.OutOfMemoryError as error:
        raise CudaOutOfMemoryError("backend_initialization", config) from error


def _autocast_dtype(precision: Precision) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _create_grad_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda")


def _capability_evidence(
    capability: Capability,
    *,
    selected_device: str,
    precision: str,
    supported: bool,
) -> str:
    if capability is Capability.SUPPORTS_CUDA:
        return f"PyTorch detected {selected_device} among the visible CUDA devices."
    if capability is Capability.SUPPORTS_MIXED_PRECISION:
        if supported:
            return f"PyTorch detected support for the selected {precision} CUDA precision."
        return "FP32 was selected, so automatic mixed precision is disabled."
    if supported:
        return f"TorchPolicyBackend implements {capability.value} on {selected_device}."
    return f"The single-GPU CUDA backend does not enable {capability.value}."


__all__ = [
    "CudaOutOfMemoryError",
    "ModelFactory",
    "create_cuda_backend",
    "detect_cuda_capabilities",
]
