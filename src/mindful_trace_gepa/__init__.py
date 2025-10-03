"""Mindful Trace GEPA extensions."""
from __future__ import annotations

from .cli import main as cli_main

__all__ = ["cli_main"]


def main() -> None:
    """Entry point for ``python -m mindful_trace_gepa``."""
    cli_main()
