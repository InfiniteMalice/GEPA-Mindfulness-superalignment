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
_LAZY_IMPORTS = {
    "combined_value_loss": "objectives",
    "rl_reward": "objectives",
    "supervised_head_losses": "objectives",
}


def __getattr__(name: str) -> Any:
    if name in _cached_attrs:
        return _cached_attrs[name]
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        module = __import__(f".{module_name}", fromlist=[name], level=1)
        attr = getattr(module, name)
        _cached_attrs[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
