"""Heuristic atomic claim extraction."""

from __future__ import annotations

import re

from .config import FactualityCertificationConfig
from .types import AtomicClaim

_CURRENT_PAT = re.compile(r"\b(current|latest|today|recent|now|as of)\b", re.I)
_NUMERIC_PAT = re.compile(r"\d")


def _claim_type(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["law", "legal", "regulation"]):
        return "legal"
    if any(k in low for k in ["clinical", "disease", "treatment", "medical"]):
        return "medical"
    if any(k in low for k in ["study", "paper", "experiment", "scientific"]):
        return "scientific"
    if _NUMERIC_PAT.search(text):
        return "numeric"
    if _CURRENT_PAT.search(text):
        return "temporal"
    return "factual"


def extract_atomic_claims(answer: str, config: FactualityCertificationConfig) -> list[AtomicClaim]:
    if not config.claim_extraction.enabled:
        return []
    chunks = [c.strip() for c in re.split(r"[.!?]\s+", answer) if c.strip()]
    claims: list[AtomicClaim] = []
    for idx, chunk in enumerate(chunks[: config.claim_extraction.max_claims]):
        if any(
            x in chunk.lower() for x in ["i think", "in my opinion"]
        ) and not _NUMERIC_PAT.search(chunk):
            continue
        claims.append(
            AtomicClaim(
                id=f"c{idx+1}",
                text=chunk,
                claim_type=_claim_type(chunk),
                requires_current_source=bool(_CURRENT_PAT.search(chunk)),
                answer_span=chunk,
            )
        )
    return claims
