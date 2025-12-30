"""Training utilities for participatory agency."""

from __future__ import annotations

from typing import Any

from .curriculum import DEFAULT_CURRICULUM, CurriculumPhase, get_default_curriculum

__all__ = [
    "combined_value_loss",
    "CurriculumPhase",
    "DEFAULT_CURRICULUM",
    "get_default_curriculum",
    "rl_reward",
    "supervised_head_losses",
]


def __getattr__(name: str) -> Any:
    if name == "combined_value_loss":
        from .objectives import combined_value_loss

        return combined_value_loss
    if name == "rl_reward":
        from .objectives import rl_reward

        return rl_reward
    if name == "supervised_head_losses":
        from .objectives import supervised_head_losses

        return supervised_head_losses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
