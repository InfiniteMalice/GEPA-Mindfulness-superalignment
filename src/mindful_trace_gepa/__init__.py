"""Mindful Trace GEPA extensions."""

from __future__ import annotations

__all__ = ["cli_main", "main"]


def cli_main(argv: list[str] | None = None) -> int:
    """Load the command implementation only when the CLI is invoked."""
    from .cli import main as cli

    return cli(argv)


def main() -> int:
    """Entry point for ``python -m mindful_trace_gepa``."""
    return cli_main(None)
