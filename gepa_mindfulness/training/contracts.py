"""Public protocol boundaries for backend-neutral reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .adapter_publication import AdapterCandidate
from .capability import BackendCapabilities, Capability
from .policy_versions import PolicyVersion
from .trajectory import (
    EvidenceReference,
    PolicyEvaluation,
    RolloutRequest,
    Trajectory,
    TrajectoryBatch,
)


@dataclass(frozen=True)
class RewardRequest:
    """A reward-scoring request restricted to trajectory-recorded observable evidence."""

    trajectory: Trajectory
    observable_references: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Reject evidence that was not recorded in the trajectory's typed evidence."""
        if not isinstance(self.observable_references, (list, tuple)) or not all(
            isinstance(reference, EvidenceReference) for reference in self.observable_references
        ):
            raise ValueError(
                "Expected observable references as a sequence of EvidenceReference values."
            )
        observable_references = tuple(self.observable_references)
        if any(not reference.is_observable for reference in observable_references):
            raise ValueError(
                "Each observable reference must use an observable output, action, or external "
                "record source kind."
            )
        if not set(observable_references).issubset(self.trajectory.evidence_references):
            raise ValueError(
                "Each observable reference must be recorded in trajectory.evidence_references."
            )
        object.__setattr__(self, "observable_references", observable_references)


@runtime_checkable
class RolloutBackend(Protocol):
    """A backend that generates backend-neutral trajectories."""

    def generate(self, requests: Sequence[RolloutRequest]) -> Sequence[Trajectory]:
        """Generate trajectories for the supplied rollout requests."""

    def capabilities(self) -> BackendCapabilities:
        """Report the operations that this backend can substantiate."""

    def close(self) -> None:
        """Release backend resources."""


@runtime_checkable
class ActorTransport(Protocol):
    """A bounded transport for a versioned external actor coordinator."""

    def start(self) -> object:
        """Start the actor boundary and return its validated handshake."""

    def generate(
        self,
        requests: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]:
        """Exchange one ordered batch of actor generation requests."""

    def close(self) -> None:
        """Release actor transport resources."""


@runtime_checkable
class TrainablePolicyBackend(RolloutBackend, Protocol):
    """A rollout backend that can evaluate and update a policy."""

    def evaluate(self, batch: TrajectoryBatch) -> PolicyEvaluation:
        """Evaluate the batch under the current policy."""

    def backward(self, loss: object) -> None:
        """Accumulate gradients for an algorithm loss."""

    def optimizer_step(self) -> object:
        """Apply the accumulated gradients and return step evidence."""

    def zero_grad(self) -> None:
        """Clear accumulated gradients."""

    def save_checkpoint(self, destination: Path) -> object:
        """Save trainable state and return its manifest."""

    def load_checkpoint(self, source: Path) -> object:
        """Load trainable state and return its manifest."""


@runtime_checkable
class AdapterExportingPolicyBackend(Protocol):
    """A learner that can substantiate a learner-native adapter-only artifact."""

    def export_adapter(
        self,
        destination: Path,
        *,
        model_id: str,
        policy_version: PolicyVersion,
        parent_policy_version: PolicyVersion,
    ) -> AdapterCandidate:
        """Export one adapter-only artifact with typed version evidence."""


@runtime_checkable
class RewardProvider(Protocol):
    """A provider that scores an observable rollout request."""

    def score(self, request: RewardRequest) -> object:
        """Return a reward result for the supplied request."""


@runtime_checkable
class RLAlgorithm(Protocol):
    """An algorithm that declares needs and computes a backend loss."""

    def required_capabilities(self) -> frozenset[Capability]:
        """Return capabilities needed before algorithm execution begins."""

    def compute_loss(self, batch: TrajectoryBatch, evaluation: PolicyEvaluation) -> object:
        """Compute a differentiable loss from trajectories and policy outputs."""
