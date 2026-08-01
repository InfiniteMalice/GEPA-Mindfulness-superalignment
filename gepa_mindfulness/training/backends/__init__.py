"""Shared trainable policy backend implementations."""

from .base import BackendCheckpointResult, OptimizerStepResult, TorchTensorOps
from .torch_cuda import CudaOutOfMemoryError, create_cuda_backend, detect_cuda_capabilities
from .torch_policy import TorchPolicyBackend
from .torch_portable import create_portable_backend

__all__ = [
    "BackendCheckpointResult",
    "CudaOutOfMemoryError",
    "OptimizerStepResult",
    "TorchPolicyBackend",
    "TorchTensorOps",
    "create_cuda_backend",
    "create_portable_backend",
    "detect_cuda_capabilities",
]
