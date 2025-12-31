"""Participatory agency value decomposition package."""

from __future__ import annotations

from typing import Any

from .config import ParticipatoryAgencyConfig

__all__ = ["ParticipatoryAgencyConfig", "ParticipatoryValueHead", "ValueComponents"]

_cached: dict[str, object] = {}


def __getattr__(name: str) -> Any:
    if name in _cached:
        return _cached[name]
    if name in {"ParticipatoryValueHead", "ValueComponents"}:
        from . import values

        attr = getattr(values, name)
        _cached[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
