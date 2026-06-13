"""Helpers for importing optional dependencies safely."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def optional_import(name: str) -> ModuleType | None:
    """Attempt to import *name*, returning ``None`` if unavailable."""

    try:
        module = import_module(name)
    except (ImportError, OSError):
        return None
    return module


__all__ = ["optional_import"]
