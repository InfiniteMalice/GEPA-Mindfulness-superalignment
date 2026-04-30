"""Core types for factuality certification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CLAIM_TYPES = {
    "factual",
    "numeric",
    "temporal",
    "causal",
    "legal",
    "medical",
    "scientific",
    "citation_required",
    "subjective",
    "instruction",
    "safety_relevant",
}

SUPPORT_LABELS = {
    "supported",
    "partially_supported",
    "contradicted",
    "unsupported",
    "unverifiable",
    "not_applicable",
}


@dataclass
class AtomicClaim:
    id: str
    text: str
    claim_type: str = "factual"
    importance: float = 1.0
    requires_current_source: bool = False
    source_span: str | None = None
    answer_span: str | None = None


@dataclass
class EvidenceItem:
    id: str
    text: str
    source: str | None = None
    citation: str | None = None
    timestamp: str | None = None
    quality_score: float = 0.5
    retrieval_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimSupport:
    claim_id: str
    support_label: str
    support_score: float
    contradiction_score: float
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    needs_abstention: bool = False
    needs_qualification: bool = False


@dataclass
class CertificationResult:
    mode: str
    overall_label: str
    hallucination_risk: float
    overrefusal_risk: float
    useful_answer_retention_score: float
    claim_support: list[ClaimSupport]
    recommended_action: str
    revised_answer: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopedAlternative:
    scoped_answer_possible: bool
    action: str
    rationale: str
    suggested_answer: str | None = None
    clarifying_question: str | None = None
    refusal_required: bool = False
