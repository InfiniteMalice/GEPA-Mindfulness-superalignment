"""Abstention and refusal policy helpers."""

from __future__ import annotations

import re

_REFUSAL_PATTERNS = [
    re.compile(r"\bi\s+can(?:not|'t)\s+help\b", re.I),
    re.compile(r"\bi\s+must\s+refuse\b", re.I),
    re.compile(r"\bi\s+won't\s+provide\b", re.I),
    re.compile(r"\bi\s+cannot\s+answer\b", re.I),
]

_ABSTENTION_PATTERNS = [
    re.compile(r"\bnot enough information\b", re.I),
    re.compile(r"\binsufficient evidence\b", re.I),
    re.compile(r"\bcannot determine\b", re.I),
    re.compile(r"\bi\s+can(?:not|'t)\s+answer\b", re.I),
    re.compile(r"\bi\s+don't\s+know\b", re.I),
    re.compile(r"\bi\s+abstain\b", re.I),
    re.compile(r"\bno answer\b", re.I),
]


def detect_refusal(text: str) -> bool:
    """Return true when output is an explicit refusal."""
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


def detect_abstention(text: str) -> bool:
    """Return true only for explicit abstention wording, not generic uncertainty."""
    return any(p.search(text) for p in _ABSTENTION_PATTERNS)
