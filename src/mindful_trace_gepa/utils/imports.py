"""Helpers for importing optional dependencies safely."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional


def optional_import(name: str) -> Optional[ModuleType]:
    """Attempt to import *name*, returning ``None`` if unavailable."""

    try:
        module = import_module(name)
    except ModuleNotFoundError:
        return None
    except Exception:
        # Some optional dependencies raise custom errors during import when
        # system libraries are missing (e.g. Cairo for WeasyPrint). Treat
        # those the same as an unavailable module.
        return None
    return module


__all__ = ["optional_import"]
