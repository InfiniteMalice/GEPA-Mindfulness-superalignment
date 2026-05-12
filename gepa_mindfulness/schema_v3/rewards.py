"""Reward helpers for Schema V3 additive overlays."""

from __future__ import annotations

from .case_v3 import RewardComponents


def assert_thought_reward_non_negative(reward_components: RewardComponents) -> None:
    """Raise if a caller attempts to introduce a negative hidden-thought reward."""
    if reward_components.r_thought < 0.0:
        raise ValueError("r_thought must be positive-only: H or 0, never negative")
