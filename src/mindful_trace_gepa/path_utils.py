"""Path validation helpers for CLI workflows."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def require_file(path: Path, label: str) -> Path:
    """Ensure a path exists and is a file.

    Args:
        path: Path to validate.
        label: Human-friendly label for error messages.

    Returns:
        The validated path.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} is not a file: {path}")
    return path


def ensure_dir(path: Path, label: str) -> Path:
    """Ensure a path exists as a directory, creating it if needed.

    Args:
        path: Directory path to validate or create.
        label: Human-friendly label for error messages.

    Returns:
        The ensured directory path.
    """
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text by replacing the target only after a complete temp-file write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON atomically with a trailing newline for stable text artifacts."""

    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
