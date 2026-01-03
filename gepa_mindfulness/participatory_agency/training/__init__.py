"""Training utilities for participatory agency."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CurriculumPhase",
    "DEFAULT_CURRICULUM",
    "combined_value_loss",
    "get_default_curriculum",
    "rl_reward",
    "supervised_head_losses",
]

_cached_attrs: dict[str, object] = {}
_LAZY_IMPORTS = {
    "CurriculumPhase": "curriculum",
    "DEFAULT_CURRICULUM": "curriculum",
    "get_default_curriculum": "curriculum",
    "combined_value_loss": "objectives",
    "rl_reward": "objectives",
    "supervised_head_losses": "objectives",
}


def __getattr__(name: str) -> Any:
    if name in _cached_attrs:
        return _cached_attrs[name]
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        module = import_module(f".{module_name}", package=__name__)
        attr = getattr(module, name)
        _cached_attrs[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
