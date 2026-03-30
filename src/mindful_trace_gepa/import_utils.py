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

    root = name.split(".", 1)[0]
    inserted_path = str(modules_path)
    with _sys_path_lock:
        added_path = False
        root_dir = modules_path / root
        root_py = modules_path / f"{root}.py"
        root_exists = root_dir.exists() or root_py.exists()
        if root_exists and modules_path.is_dir() and inserted_path not in sys.path:
            sys.path.insert(0, inserted_path)
            added_path = True
        try:
            if not root_exists:
                return None
            return import_module(name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name in {name, root}:
                return None
            if allow_missing_submodules and missing_name.startswith(f"{name}."):
                return None
            raise
        except ImportError as exc:
            if allow_import_error_name_match and getattr(exc, "name", None) == name:
                return None
            raise
        finally:
            if added_path and inserted_path in sys.path:
                sys.path.remove(inserted_path)


__all__ = ["optional_repo_module"]
