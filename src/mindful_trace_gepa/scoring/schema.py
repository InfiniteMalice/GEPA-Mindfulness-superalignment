"""Data models for the tiered wisdom scoring pipeline without external deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, cast

DIMENSIONS = ["mindfulness", "compassion", "integrity", "prudence"]
ALLOWED_TIERS = {"heuristic", "judge", "classifier"}


@dataclass
class SpanReference:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Span indices must be non-negative")
        if self.end < self.start:
            raise ValueError("Span end must be >= start")

    def to_dict(self) -> Dict[str, int]:
        return {"start": int(self.start), "end": int(self.end)}


@dataclass
class JudgeDimensionScore:
    score: int
    rationale: str
    uncertainty: float
    spans: List[SpanReference] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= int(self.score) <= 4:
            raise ValueError("Judge scores must be integers between 0 and 4")
        self.score = int(self.score)
        self.uncertainty = float(self.uncertainty)
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("Uncertainty must be within [0,1]")
        self.rationale = str(self.rationale)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "uncertainty": self.uncertainty,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass
class JudgeOutput:
    mindfulness: JudgeDimensionScore
    compassion: JudgeDimensionScore
    integrity: JudgeDimensionScore
    prudence: JudgeDimensionScore
    abstain: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JudgeOutput":
        def _dim(name: str) -> JudgeDimensionScore:
            raw = data.get(name, {})
            spans = [SpanReference(**span) for span in raw.get("spans", [])]
            return JudgeDimensionScore(
                score=raw.get("score", 0),
                rationale=raw.get("rationale", ""),
                uncertainty=raw.get("uncertainty", 1.0),
                spans=spans,
            )

        return cls(
            mindfulness=_dim("mindfulness"),
            compassion=_dim("compassion"),
            integrity=_dim("integrity"),
            prudence=_dim("prudence"),
            abstain=bool(data.get("abstain", False)),
        )

    def to_tier_scores(self, confidence_floor: float = 0.1) -> "TierScores":
        scores = {dim: getattr(self, dim).score for dim in DIMENSIONS}
        confidence = {
            dim: max(confidence_floor, 1.0 - getattr(self, dim).uncertainty) for dim in DIMENSIONS
        }
        meta = {
            "rationales": {dim: getattr(self, dim).rationale for dim in DIMENSIONS},
            "spans": {
                dim: [span.to_dict() for span in getattr(self, dim).spans] for dim in DIMENSIONS
            },
            "abstain": self.abstain,
        }
        tier = cast(Literal["heuristic", "judge", "classifier"], "judge")
        return TierScores(tier=tier, scores=scores, confidence=confidence, meta=meta)


@dataclass
class TierScores:
    tier: Literal["heuristic", "judge", "classifier"]
    scores: Dict[str, int]
    confidence: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TierScores":
        tier_value = data.get("tier", "heuristic")
        scores_value = data.get("scores") or {}
        confidence_value = data.get("confidence") or {}
        meta_value = data.get("meta") or {}

        if not isinstance(scores_value, Mapping):
            try:
                scores_value = dict(scores_value)
            except (TypeError, ValueError):
                scores_value = {}
        else:
            scores_value = dict(scores_value)

        if not isinstance(confidence_value, Mapping):
            try:
                confidence_value = dict(confidence_value)
            except (TypeError, ValueError):
                confidence_value = {}
        else:
            confidence_value = dict(confidence_value)

        if not isinstance(meta_value, Mapping):
            try:
                meta_value = dict(meta_value)
            except (TypeError, ValueError):
                meta_value = {}
        else:
            meta_value = dict(meta_value)

        tier_clean = str(tier_value)
        if tier_clean not in ALLOWED_TIERS:
            tier_clean = "heuristic"

        return cls(
            tier=cast(Literal["heuristic", "judge", "classifier"], tier_clean),
            scores=scores_value,
            confidence=confidence_value,
            meta=meta_value,
        )

    def __post_init__(self) -> None:
        if self.tier not in ALLOWED_TIERS:
            raise ValueError(f"Unknown tier '{self.tier}'")
        if isinstance(self.meta, Mapping):
            self.meta = dict(self.meta)
        else:
            try:
                self.meta = dict(self.meta or {})
            except (TypeError, ValueError):
                self.meta = {}
        for dim in DIMENSIONS:
            value: Any = self.scores.get(dim)
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                numeric = 0
            numeric = max(0, min(4, numeric))
            self.scores[dim] = numeric
        for dim in DIMENSIONS:
            raw_conf: Any = self.confidence.get(dim, 0.0)
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.0:
                conf = 0.0
            elif conf > 1.0:
                conf = 1.0
            self.confidence[dim] = conf

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "scores": dict(self.scores),
            "confidence": dict(self.confidence),
            "meta": dict(self.meta),
        }


@dataclass
class AggregateScores:
    final: Dict[str, int]
    confidence: Dict[str, float]
    per_tier: List[TierScores]
    escalate: bool = False
    reasons: List[str] = field(default_factory=list)

    def as_json(self) -> Dict[str, Any]:
        return {
            "final": dict(self.final),
            "confidence": dict(self.confidence),
            "per_tier": [tier.to_dict() for tier in self.per_tier],
            "escalate": bool(self.escalate),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AggregateScores":
        per_tier = [TierScores(**tier) for tier in data.get("per_tier", [])]
        final: Dict[str, int] = {}
        for dim, value in (data.get("final") or {}).items():
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            final[dim] = max(0, min(4, numeric))

        confidence: Dict[str, float] = {}
        for dim, value in (data.get("confidence") or {}).items():
            try:
                conf = float(value)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.0:
                conf = 0.0
            elif conf > 1.0:
                conf = 1.0
            confidence[dim] = conf

        reasons_value = data.get("reasons")
        if isinstance(reasons_value, list):
            reasons = list(reasons_value)
        elif reasons_value is None:
            reasons = []
        else:
            reasons = [str(reasons_value)]

        return cls(
            final=final,
            confidence=confidence,
            per_tier=per_tier,
            escalate=bool(data.get("escalate", False)),
            reasons=reasons,
        )


__all__ = [
    "AggregateScores",
    "DIMENSIONS",
    "JudgeDimensionScore",
    "JudgeOutput",
    "SpanReference",
    "TierScores",
]
