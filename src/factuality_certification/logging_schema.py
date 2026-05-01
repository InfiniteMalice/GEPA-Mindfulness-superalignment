"""Structured logging helpers for attribution and interpretability."""

from __future__ import annotations

import hashlib


def make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
