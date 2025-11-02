"""Lightweight matplotlib stub for test environments without the real library."""

from __future__ import annotations


def use(_backend: str, *, force: bool = False) -> None:
    """Accept backend selection without performing any action."""

    return None


__all__ = ["use"]
