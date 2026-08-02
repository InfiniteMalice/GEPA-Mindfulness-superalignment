"""Mindful Trace GEPA extensions."""

from __future__ import annotations

__all__ = ["cli_main", "main"]


def cli_main(argv: list[str] | None = None) -> int:
    """Load the command implementation only when the CLI is invoked."""
    from .cli import main as cli

    return cli(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the package CLI with explicit arguments or process-global arguments when omitted."""
    return cli_main(argv)
