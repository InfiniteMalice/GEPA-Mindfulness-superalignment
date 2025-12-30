"""Participatory agency value decomposition package."""

from __future__ import annotations

from typing import Any

from .config import ParticipatoryAgencyConfig

__all__ = ["ParticipatoryAgencyConfig", "ParticipatoryValueHead", "ValueComponents"]


def __getattr__(name: str) -> Any:
    if name in {"ParticipatoryValueHead", "ValueComponents"}:
        from .values import ParticipatoryValueHead, ValueComponents

        mapping = {
            "ParticipatoryValueHead": ParticipatoryValueHead,
            "ValueComponents": ValueComponents,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
