"""Baseline evidence matching with lexical overlap heuristics."""

from __future__ import annotations

import re

from .config import FactualityCertificationConfig
from .types import AtomicClaim, ClaimSupport, EvidenceItem

_WORD_PAT = re.compile(r"[a-zA-Z0-9]+")
_NEG = {"not", "never", "no"}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_PAT.findall(text)}


def match_claims_to_evidence(
    claims: list[AtomicClaim],
    evidence: list[EvidenceItem],
    config: FactualityCertificationConfig,
) -> list[ClaimSupport]:
    """Match claims against evidence items unless matching is disabled in config."""
    if not config.evidence_matching.enabled:
        return []

    supports: list[ClaimSupport] = []
    for claim in claims:
        ctoks = _tokens(claim.text)
        best_score = 0.0
        best_contra = 0.0
        best_ids: list[str] = []
        for item in evidence:
            etoks = _tokens(item.text)
            overlap = len(ctoks & etoks) / max(1, len(ctoks))
            if overlap <= 0.0:
                weighted = 0.0
            else:
                weighted = (
                    config.evidence_matching.entailment_weight * overlap
                    + config.evidence_matching.source_quality_weight * item.quality_score
                    + config.evidence_matching.retrieval_weight * item.retrieval_score
                )
            has_neg_mismatch = bool((ctoks & _NEG) ^ (etoks & _NEG))
            contra = min(1.0, overlap if has_neg_mismatch else 0.0)
            if weighted > best_score:
                best_score = min(1.0, weighted)
                best_ids = [item.id]
            best_contra = max(best_contra, contra)
        if best_contra >= config.evidence_matching.contradiction_threshold:
            label = "contradicted"
        elif best_score >= config.evidence_matching.min_support_score:
            label = "supported"
        elif best_score > 0.2:
            label = "partially_supported"
        else:
            label = "unsupported"
        supports.append(
            ClaimSupport(
                claim_id=claim.id,
                support_label=label,
                support_score=best_score,
                contradiction_score=best_contra,
                evidence_ids=best_ids,
                rationale=f"lexical score={best_score:.2f}",
                needs_abstention=label in {"unsupported", "contradicted"},
                needs_qualification=label in {"partially_supported", "unsupported"},
            )
        )
    return supports
