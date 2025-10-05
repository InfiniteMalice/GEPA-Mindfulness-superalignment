"""Mindful Trace GEPA extensions."""

from __future__ import annotations

try:  # pragma: no cover - defensive import for optional deps
    from .cli import main as cli_main
except Exception:  # pragma: no cover
    cli_main = None

__all__ = ["cli_main", "main"]


def main() -> None:
    """Entry point for ``python -m mindful_trace_gepa``."""
    if cli_main is None:  # pragma: no cover - defensive
        raise RuntimeError("CLI entrypoint unavailable; optional dependencies not installed")
    cli_main()
