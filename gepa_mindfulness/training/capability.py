"""Backend capability declarations and fail-closed requirement checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


class Capability(str, Enum):
    """Operations or runtime features that a backend can substantiate."""

    SUPPORTS_GENERATION = "supports_generation"
    SUPPORTS_TOKEN_LOG_PROBS = "supports_token_log_probs"
    SUPPORTS_REFERENCE_LOG_PROBS = "supports_reference_log_probs"
    SUPPORTS_BACKWARD = "supports_backward"
    SUPPORTS_OPTIMIZER_STEP = "supports_optimizer_step"
    SUPPORTS_VALUE_HEAD = "supports_value_head"
    SUPPORTS_FULL_WEIGHT_TRAINING = "supports_full_weight_training"
    SUPPORTS_LORA_TRAINING = "supports_lora_training"
    SUPPORTS_DISTRIBUTED_TRAINING = "supports_distributed_training"
    SUPPORTS_MIXED_PRECISION = "supports_mixed_precision"
    SUPPORTS_VULKAN = "supports_vulkan"
    SUPPORTS_CUDA = "supports_cuda"
    SUPPORTS_GGUF = "supports_gguf"


class CapabilityState(str, Enum):
    """The evidence-backed state of a backend capability."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CapabilityError(ValueError):
    """Raised when a backend cannot prove every required capability."""


@dataclass(frozen=True)
class CapabilityEvidence:
    """A state and the observable evidence that supports that state."""

    state: CapabilityState
    evidence: str


@dataclass(frozen=True)
class BackendCapabilities:
    """A backend's versioned capability report."""

    backend_name: str
    backend_version: str
    capabilities: Mapping[Capability, CapabilityEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy capability evidence into an immutable mapping."""
        object.__setattr__(self, "capabilities", MappingProxyType(dict(self.capabilities)))

    @classmethod
    def unknown(cls, backend_name: str, backend_version: str) -> "BackendCapabilities":
        """Create a report that fails closed until every capability has evidence."""
        evidence = CapabilityEvidence(
            state=CapabilityState.UNKNOWN,
            evidence="No capability evidence is available.",
        )
        return cls(
            backend_name=backend_name,
            backend_version=backend_version,
            capabilities={capability: evidence for capability in Capability},
        )

    def state(self, capability: Capability) -> CapabilityState:
        """Return a capability state, treating a missing entry as unknown."""
        evidence = self.capabilities.get(capability)
        if evidence is None:
            return CapabilityState.UNKNOWN
        return evidence.state

    def require(self, required: Iterable[Capability]) -> None:
        """Raise when any required capability is not explicitly supported."""
        unavailable = sorted(
            (
                f"{capability.value} ({self.state(capability).value})"
                for capability in set(required)
                if self.state(capability) is not CapabilityState.SUPPORTED
            ),
        )
        if unavailable:
            details = ", ".join(unavailable)
            raise CapabilityError(
                f"Backend {self.backend_name!r} version {self.backend_version!r} "
                f"does not satisfy required capabilities: {details}."
            )
