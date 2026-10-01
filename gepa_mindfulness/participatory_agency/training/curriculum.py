"""Curriculum phases for participatory agency training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

_VALID_HEADS = frozenset(("epistemic", "cooperation", "flexibility", "belonging"))


@dataclass(frozen=True)
class CurriculumPhase:
    """Defines an ordered curriculum phase."""

    name: str
    description: str
    active_heads: tuple[str, ...]
    loss_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.active_heads) - _VALID_HEADS
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown heads: {unknown_list}")
        unknown_weights = set(self.loss_weights.keys()) - _VALID_HEADS
        if unknown_weights:
            unknown_list = ", ".join(sorted(unknown_weights))
            raise ValueError(f"Unknown loss_weights keys: {unknown_list}")
        object.__setattr__(self, "active_heads", tuple(self.active_heads))
        object.__setattr__(self, "loss_weights", MappingProxyType(dict(self.loss_weights)))


DEFAULT_CURRICULUM: Sequence[CurriculumPhase] = (
    CurriculumPhase(
        name="phase_1_epistemic",
        description="Focus on epistemic humility signals.",
        active_heads=("epistemic",),
        loss_weights={
            "epistemic": 1.0,
            "cooperation": 0.0,
            "flexibility": 0.0,
            "belonging": 0.0,
        },
    ),
    CurriculumPhase(
        name="phase_2_cooperation",
        description="Add cooperative equilibrium signals.",
        active_heads=("epistemic", "cooperation"),
        loss_weights={
            "epistemic": 0.7,
            "cooperation": 0.3,
            "flexibility": 0.0,
            "belonging": 0.0,
        },
    ),
    CurriculumPhase(
        name="phase_3_flexibility",
        description="Add goal flexibility signals.",
        active_heads=("epistemic", "cooperation", "flexibility"),
        loss_weights={
            "epistemic": 0.5,
            "cooperation": 0.3,
            "flexibility": 0.2,
            "belonging": 0.0,
        },
    ),
    CurriculumPhase(
        name="phase_4_belonging",
        description="Add participatory identity and belonging signals.",
        active_heads=("epistemic", "cooperation", "flexibility", "belonging"),
        loss_weights={
            "epistemic": 0.4,
            "cooperation": 0.25,
            "flexibility": 0.2,
            "belonging": 0.15,
        },
    ),
    CurriculumPhase(
        name="phase_5_integrated",
        description="Jointly fine-tune with all heads balanced.",
        active_heads=("epistemic", "cooperation", "flexibility", "belonging"),
        loss_weights={
            "epistemic": 0.25,
            "cooperation": 0.25,
            "flexibility": 0.25,
            "belonging": 0.25,
        },
    ),
)


def get_default_curriculum() -> Sequence[CurriculumPhase]:
    """Return the default participatory agency curriculum."""
    return DEFAULT_CURRICULUM


class PEOStage(str, Enum):
    """Data-learning stages selected independently of the five head/loss phases."""

    CAUSAL = "causal_epistemic"
    INVARIANCE = "representation_invariance"
    COUNTERFACTUAL = "minimal_relation_changes"
    EVIDENCE = "uncertainty_evidence"
    NORMS = "conditional_norms"
    ADVERSARIAL = "adversarial_pressure"
    TEMPORAL = "temporal_peo"


TEMPORAL_FEATURES = (
    "supersession",
    "stale_evidence",
    "motivated_forgetting",
    "state_drift",
    "correction_after_error",
    "reward_pressure",
    "evidence_change",
    "source_reliability_change",
)


@dataclass(frozen=True)
class PEODataStage:
    """A curriculum focus; the host still reviews whether examples exercise it."""

    stage: PEOStage
    description: str
    temporal_features: tuple[str, ...] = ()


_PEO_CURRICULUM = (
    PEODataStage(PEOStage.CAUSAL, "Clean causal and epistemic primitives."),
    PEODataStage(PEOStage.INVARIANCE, "Invariant judgments across surface representations."),
    PEODataStage(
        PEOStage.COUNTERFACTUAL, "Minimal decisive relation changes and paired judgments."
    ),
    PEODataStage(PEOStage.EVIDENCE, "Hidden information, uncertainty, inquiry and abstention."),
    PEODataStage(PEOStage.NORMS, "Conditional normative conflicts grounded in context."),
    PEODataStage(PEOStage.ADVERSARIAL, "Authority pressure, laundering and evaluator pressure."),
    PEODataStage(
        PEOStage.TEMPORAL, "Longitudinal prediction-execution-observation.", TEMPORAL_FEATURES
    ),
)


def get_peo_curriculum(*, enabled: bool = False) -> tuple[PEODataStage, ...]:
    """Return the opt-in data curriculum without changing default training heads."""
    if enabled is not True:
        raise ValueError("PEO curriculum requires enabled=True")
    return _PEO_CURRICULUM
