"""Curriculum phases for participatory agency training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class CurriculumPhase:
    """Defines an ordered curriculum phase."""

    name: str
    description: str
    active_heads: Sequence[str]
    loss_weights: Dict[str, float]


DEFAULT_CURRICULUM: Sequence[CurriculumPhase] = (
    CurriculumPhase(
        name="phase_1_epistemic",
        description="Focus on epistemic humility signals.",
        active_heads=("epistemic",),
        loss_weights={"epistemic": 1.0},
    ),
    CurriculumPhase(
        name="phase_2_cooperation",
        description="Add cooperative equilibrium signals.",
        active_heads=("epistemic", "cooperation"),
        loss_weights={"epistemic": 0.7, "cooperation": 0.3},
    ),
    CurriculumPhase(
        name="phase_3_flexibility",
        description="Add goal flexibility signals.",
        active_heads=("epistemic", "cooperation", "flexibility"),
        loss_weights={"epistemic": 0.5, "cooperation": 0.3, "flexibility": 0.2},
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
