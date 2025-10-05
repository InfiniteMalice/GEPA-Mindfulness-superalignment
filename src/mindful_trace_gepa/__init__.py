"""Mindful Trace GEPA extensions."""

from __future__ import annotations

from typing import Callable, Optional

try:  # pragma: no cover - defensive import for optional deps
    from .cli import main as _cli_main
except Exception:  # pragma: no cover
    cli_main: Optional[Callable[[list[str] | None], None]] = None
else:
    cli_main = _cli_main

__all__ = ["cli_main", "main"]


def main() -> None:
    """Entry point for ``python -m mindful_trace_gepa``."""
    if cli_main is None:  # pragma: no cover - defensive
        raise RuntimeError("CLI entrypoint unavailable; optional dependencies not installed")
    cli_main(None)
