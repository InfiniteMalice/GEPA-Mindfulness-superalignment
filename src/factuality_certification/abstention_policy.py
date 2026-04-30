"""Abstention and refusal policy helpers."""

from __future__ import annotations


def detect_refusal(text: str) -> bool:
    lower = text.lower()
    return any(
        k in lower
        for k in [
            "i can't help",
            "i cannot help",
            "i cannot answer",
            "i must refuse",
            "i won't provide",
        ]
    )


def detect_abstention(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ["not enough information", "insufficient evidence", "uncertain"])
