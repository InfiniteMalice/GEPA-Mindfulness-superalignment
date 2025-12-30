"""Training utilities for participatory agency."""

from __future__ import annotations

from typing import Any

from .curriculum import CurriculumPhase, DEFAULT_CURRICULUM, get_default_curriculum

__all__ = [
    "CurriculumPhase",
    "DEFAULT_CURRICULUM",
    "get_default_curriculum",
    "combined_value_loss",
    "rl_reward",
    "supervised_head_losses",
]


def __getattr__(name: str) -> Any:
    if name in {"combined_value_loss", "rl_reward", "supervised_head_losses"}:
        from .objectives import combined_value_loss, rl_reward, supervised_head_losses

        mapping = {
            "combined_value_loss": combined_value_loss,
            "rl_reward": rl_reward,
            "supervised_head_losses": supervised_head_losses,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
