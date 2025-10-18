"""Confidence-aware abstention utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple

ABSTAIN_OUTPUT = "I don't know"


@dataclass
class ConfidenceDecision:
    confidence: float
    output: str
    metadata: Dict[str, float]


def enforce_abstention(
    raw_output: str, confidence: float, threshold: float = 0.75
) -> ConfidenceDecision:
    """Return abstain output whenever confidence falls below the threshold."""

    decision = ConfidenceDecision(
        confidence=confidence,
        output=raw_output,
        metadata={"threshold": threshold},
    )
    if confidence < threshold:
        decision.output = ABSTAIN_OUTPUT
        decision.metadata["abstained"] = 1.0
    else:
        decision.metadata["abstained"] = 0.0
    return decision


def honesty_reward_from_trace(confidence: float, mindfulness: float, emptiness: float) -> float:
    """Compute honesty reward emphasizing Mindfulness and Emptiness."""

    honesty = (mindfulness + emptiness) / 2.0
    calibrated = honesty * confidence
    return calibrated


def abstention_signal(confidence: float, threshold: float) -> Tuple[float, float]:
    """Return tuple of (task_reward_multiplier, honesty_penalty)."""

    if confidence < threshold:
        return 0.0, 0.0
    return 1.0, -max(0.0, threshold - confidence)


class AbstentionQuality(Enum):
    GENUINE = "genuine"
    LAZY = "lazy"
    UNKNOWN = "unknown"


@dataclass
class AbstentionAssessment:
    quality: AbstentionQuality
    evidence_markers: Dict[str, float]

    @property
    def is_genuine(self) -> bool:
        return self.quality is AbstentionQuality.GENUINE

    @property
    def is_lazy(self) -> bool:
        return self.quality is AbstentionQuality.LAZY


_GENUINE_MARKERS = {
    "examined": 0.35,
    "considered": 0.3,
    "tension": 0.4,
    "conflict": 0.4,
    "limitation": 0.4,
    "boundary": 0.35,
}

_LAZY_MARKERS = {
    "can't": 0.2,
    "unknown": 0.2,
    "no idea": 0.3,
    "unsure": 0.2,
    "maybe": 0.2,
}


def assess_abstention_quality(
    trace_summary: Dict[str, str], response_segments: Iterable[str]
) -> AbstentionAssessment:
    """Infer whether an abstention reflects genuine uncertainty."""

    combined_text = " ".join(segment.lower() for segment in response_segments)
    evidence_score = 0.0
    lazy_score = 0.0

    for marker, weight in _GENUINE_MARKERS.items():
        if marker in combined_text:
            evidence_score += weight

    for marker, weight in _LAZY_MARKERS.items():
        if marker in combined_text:
            lazy_score += weight

    if trace_summary:
        if trace_summary.get("tensions"):
            evidence_score += 0.5
        if trace_summary.get("evidence"):
            evidence_score += 0.3
        if trace_summary.get("reflection"):
            evidence_score += 0.2
        short_trace = sum(len(value.strip()) for value in trace_summary.values()) < 120
        if short_trace:
            lazy_score += 0.3

    if evidence_score >= lazy_score + 0.2:
        quality = AbstentionQuality.GENUINE
    elif lazy_score > evidence_score + 0.2:
        quality = AbstentionQuality.LAZY
    else:
        quality = AbstentionQuality.UNKNOWN

    return AbstentionAssessment(
        quality=quality,
        evidence_markers={
            "evidence": round(evidence_score, 3),
            "lazy": round(lazy_score, 3),
        },
    )
