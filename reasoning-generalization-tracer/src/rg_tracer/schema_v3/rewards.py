"""Reward helpers for Schema V3 additive overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment

    from .case_v3 import RewardComponents


def verified_component_score(
    epistemic_process: EpistemicProcessAssessment | None,
    component_name: str,
) -> float:
    """Return an exact independently verified component score, or zero."""
    if epistemic_process is None:
        return 0.0
    for verified_component in epistemic_process.verified_components:
        if verified_component.component.value == component_name:
            return verified_component.score
    return 0.0


def assert_thought_reward_non_negative(reward_components: RewardComponents) -> None:
    """Raise if a caller attempts to introduce a negative hidden-thought reward."""
    if reward_components.r_thought < 0.0:
        raise ValueError("r_thought must be non-negative: H or 0, never negative")
