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

_cached_attrs: dict[str, object] = {}


def __getattr__(name: str) -> Any:
    if name in _cached_attrs:
        return _cached_attrs[name]
    if name == "combined_value_loss":
        from .objectives import combined_value_loss

        _cached_attrs[name] = combined_value_loss
        return _cached_attrs[name]
    if name == "rl_reward":
        from .objectives import rl_reward

        _cached_attrs[name] = rl_reward
        return _cached_attrs[name]
    if name == "supervised_head_losses":
        from .objectives import supervised_head_losses

        _cached_attrs[name] = supervised_head_losses
        return _cached_attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
