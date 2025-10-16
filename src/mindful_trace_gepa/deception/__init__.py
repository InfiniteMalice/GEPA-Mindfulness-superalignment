"""Deception utilities."""

from .fingerprints import DeceptionFingerprint, FingerprintCollector
from .score import score_deception, summarize_deception_sources

__all__ = [
    "score_deception",
    "summarize_deception_sources",
    "DeceptionFingerprint",
    "FingerprintCollector",
]
