"""Shared helpers for optional imports across packages."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any

_sys_path_lock = Lock()


def optional_repo_module(
    name: str,
    modules_path: Path,
    *,
    allow_missing_submodules: bool = False,
    allow_import_error_name_match: bool = False,
) -> Any | None:
    """Import an optional module from a repository-local ``modules`` directory."""

    inserted_path = str(modules_path)
    with _sys_path_lock:
        added_path = False
        if modules_path.is_dir() and inserted_path not in sys.path:
            sys.path.insert(0, inserted_path)
            added_path = True
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name == name:
                return None
            if allow_missing_submodules and missing_name.startswith(f"{name}."):
                return None
            raise
        except ImportError as exc:
            if allow_import_error_name_match and name in str(exc):
                return None
            raise
        finally:
            if added_path and inserted_path in sys.path:
                sys.path.remove(inserted_path)


__all__ = ["optional_repo_module"]
