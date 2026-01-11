"""Path validation helpers for CLI workflows."""

from __future__ import annotations

from pathlib import Path


def require_file(path: Path, label: str) -> Path:
    """Ensure a path exists and is a file.

    Args:
        path: Path to validate.
        label: Human-friendly label for error messages.

    Returns:
        The validated path.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
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
