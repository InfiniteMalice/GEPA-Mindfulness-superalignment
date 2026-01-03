"""Participatory agency value decomposition package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ParticipatoryAgencyConfig",
    "ParticipatoryValueHead",
    "ValueComponents",
]

_cached: dict[str, object] = {}


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _cached:
        return _cached[name]
    if name == "ParticipatoryAgencyConfig":
        from .config import ParticipatoryAgencyConfig

        _cached[name] = ParticipatoryAgencyConfig
        return ParticipatoryAgencyConfig
    if name in {"ParticipatoryValueHead", "ValueComponents"}:
        from . import values

        attr = getattr(values, name)
        _cached[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
