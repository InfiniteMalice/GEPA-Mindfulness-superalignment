"""Optimizer-eligible epistemic process component records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .reward_provenance import RewardProvenance


class EpistemicProcessComponent(str, Enum):
    """The independently verifiable process properties eligible for optimizer credit."""

    EVIDENCE_FIDELITY = "evidence_fidelity"
    PUBLIC_RATIONALE_FIDELITY = "public_rationale_fidelity"
    CALIBRATION = "calibration"
    CONTRADICTION_HANDLING = "contradiction_handling"
    CONSEQUENCE_PREDICTION = "consequence_prediction"
    JUSTIFIED_ABSTENTION = "justified_abstention"
    MISSING_EVIDENCE_DETECTION = "missing_evidence_detection"
    BELIEF_UPDATE = "belief_update"
    RECOVERY = "recovery"


def _validated_score(value: object) -> float:
    """Return one finite unit-interval optimizer component score."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite number in [0.0, 1.0].")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a finite number in [0.0, 1.0].")
    return score


@dataclass(frozen=True)
class VerifiedProcessComponent:
    """A bounded epistemic-process score paired with exclusive verification provenance."""

    component: EpistemicProcessComponent
    score: float
    provenance: RewardProvenance

    def __post_init__(self) -> None:
        """Require a typed component, bounded score, and matching provenance name."""
        if not isinstance(self.component, EpistemicProcessComponent):
            raise ValueError("component must be an EpistemicProcessComponent.")
        if not isinstance(self.provenance, RewardProvenance):
            raise ValueError("provenance must be a RewardProvenance.")
        if self.provenance.component_name != self.component.value:
            raise ValueError("provenance component_name must match component.")
        object.__setattr__(self, "score", _validated_score(self.score))


@dataclass(frozen=True)
class EpistemicProcessAssessment:
    """A decomposed collection of independently verified optimizer components."""

    verified_components: tuple[VerifiedProcessComponent, ...] = ()
    reasoning_grounded: bool | None = None

    def __post_init__(self) -> None:
        """Keep assessment inputs typed and prevent duplicate component credit."""
        if not isinstance(self.verified_components, (list, tuple)) or not all(
            isinstance(component, VerifiedProcessComponent)
            for component in self.verified_components
        ):
            raise ValueError("verified_components must contain VerifiedProcessComponent values.")
        verified_components = tuple(self.verified_components)
        component_names = {component.component for component in verified_components}
        if len(component_names) != len(verified_components):
            raise ValueError("verified_components cannot contain duplicate components.")
        if self.reasoning_grounded is not None and not isinstance(self.reasoning_grounded, bool):
            raise ValueError("reasoning_grounded must be a bool or None.")
        object.__setattr__(self, "verified_components", verified_components)

    def optimizer_score(self) -> float:
        """Return zero without verification or the unweighted mean of verified components."""
        if not self.verified_components:
            return 0.0
        return sum(component.score for component in self.verified_components) / len(
            self.verified_components
        )


__all__ = [
    "EpistemicProcessAssessment",
    "EpistemicProcessComponent",
    "VerifiedProcessComponent",
]
