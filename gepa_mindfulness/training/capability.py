"""Backend capability declarations and fail-closed requirement checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Literal, Mapping


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


class PureMojoLearnerGate(str, Enum):
    """The closed set of capabilities required by a pure Mojo learner."""

    TRAINABLE_TENSORS = "trainable tensors"
    AUTODIFF_OR_EXPLICIT_BACKWARD = "autodiff or explicit backward"
    OPTIMIZER_STATE = "optimizer state"
    TRANSFORMER_BACKWARD_KERNELS = "transformer backward kernels"
    LORA_PARAMETER_UPDATES = "LoRA parameter updates"
    TRAINABLE_CHECKPOINT_FORMAT = "trainable checkpoint format"
    LLAMA_CPP_ADAPTER_TRANSFER = "llama.cpp adapter transfer"
    NUMERICAL_PARITY_AGAINST_PYTORCH_REFERENCE = "numerical parity against PyTorch reference"
    HARDWARE_RUNTIME_COVERAGE = "hardware/runtime coverage"


class CapabilityError(ValueError):
    """Raised when a capability report cannot prove every required capability."""


@dataclass(frozen=True)
class CapabilityEvidence:
    """A state and the observable evidence that supports that state."""

    state: CapabilityState
    evidence: str


@dataclass(frozen=True)
class PureMojoLearnerReport:
    """Immutable evidence for every capability required by a pure Mojo learner."""

    gates: Mapping[PureMojoLearnerGate, CapabilityEvidence]
    report_path: str = "docs/rl/mojo_learner_feasibility.md"

    def __post_init__(self) -> None:
        """Require the exact closed gate set and make its evidence immutable."""
        expected = set(PureMojoLearnerGate)
        observed = set(self.gates)
        if observed != expected:
            raise ValueError("report must contain exactly the nine pure-Mojo learner gates")
        object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))

    @property
    def decision(self) -> Literal["go", "no-go"]:
        """Return go only when every gate has explicit supported evidence."""
        if all(evidence.state is CapabilityState.SUPPORTED for evidence in self.gates.values()):
            return "go"
        return "no-go"


PURE_MOJO_LEARNER_REPORT = PureMojoLearnerReport(
    gates={
        PureMojoLearnerGate.TRAINABLE_TENSORS: CapabilityEvidence(
            CapabilityState.SUPPORTED,
            "LayoutTensor has mutable CPU/GPU storage; gradients are a separate requirement.",
        ),
        PureMojoLearnerGate.AUTODIFF_OR_EXPLICIT_BACKWARD: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "No complete autodiff or explicit backward path was documented or tested.",
        ),
        PureMojoLearnerGate.OPTIMIZER_STATE: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "No optimizer state and restart-equivalent update path was documented or tested.",
        ),
        PureMojoLearnerGate.TRANSFORMER_BACKWARD_KERNELS: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "Reviewed attention kernels document forward computation, not a backward suite.",
        ),
        PureMojoLearnerGate.LORA_PARAMETER_UPDATES: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "MAX applies trained PEFT adapters for inference; Mojo updates are unproven.",
        ),
        PureMojoLearnerGate.TRAINABLE_CHECKPOINT_FORMAT: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "Weight loading is documented; complete training-state save/resume is unproven.",
        ),
        PureMojoLearnerGate.LLAMA_CPP_ADAPTER_TRANSFER: CapabilityEvidence(
            CapabilityState.SUPPORTED,
            "llama.cpp converts exact Hugging Face PEFT adapter inputs to inference GGUF.",
        ),
        PureMojoLearnerGate.NUMERICAL_PARITY_AGAINST_PYTORCH_REFERENCE: CapabilityEvidence(
            CapabilityState.UNKNOWN,
            "Forward-logit comparison is documented; gradient and update parity are unproven.",
        ),
        PureMojoLearnerGate.HARDWARE_RUNTIME_COVERAGE: CapabilityEvidence(
            CapabilityState.SUPPORTED,
            "Mojo documents a WSL x86-64-v3 CPU substrate; the local learner is untested.",
        ),
    }
)


def require_pure_mojo_learner(
    report: PureMojoLearnerReport = PURE_MOJO_LEARNER_REPORT,
) -> None:
    """Reject a pure Mojo learner unless every evidence gate is supported."""
    unavailable = [
        f"{gate.value} ({report.gates[gate].state.value})"
        for gate in PureMojoLearnerGate
        if report.gates[gate].state is not CapabilityState.SUPPORTED
    ]
    if unavailable:
        details = ", ".join(unavailable)
        raise CapabilityError(
            f"pure Mojo learner is unsupported; non-supported gates: {details}. "
            f"See {report.report_path}. "
            "Use --backend mojo-vulkan-llamacpp --learner pytorch for the supported "
            "hybrid learner path."
        )


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
