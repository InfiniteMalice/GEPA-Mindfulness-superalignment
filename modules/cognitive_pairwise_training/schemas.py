"""Typed records for CPT-inspired reasoning-trace comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from semantic_intent_robustness.taxonomy import StrEnum


class PairwiseLabel(StrEnum):
    A_MORE_TRUSTWORTHY = "a_more_trustworthy"
    B_MORE_TRUSTWORTHY = "b_more_trustworthy"
    BOTH_TRUSTWORTHY = "both_trustworthy"
    NEITHER_TRUSTWORTHY = "neither_trustworthy"
    UNCERTAIN_REVIEW = "uncertain_review"


class PairType(StrEnum):
    INTRA_MODEL = "intra_model"
    INTER_MODEL = "inter_model"
    SMALL_CORRECT_LARGE_INCORRECT = "small_correct_large_incorrect"
    ANSWER_PROCESS_MISMATCH = "answer_process_mismatch"
    SAME_ANSWER_DIFFERENT_REASONING_QUALITY = "same_answer_different_reasoning_quality"
    ABSTENTION_VERSUS_GUESS = "abstention_versus_guess"
    SEMANTIC_LAUNDERING_STRESS_PAIR = "semantic_laundering_stress_pair"
    PRINCIPLE_PRESSURE_STRESS_PAIR = "principle_pressure_stress_pair"


@dataclass(frozen=True)
class ReasoningTraceCandidate:
    candidate_id: str
    problem_id: str
    prompt: str
    public_reasoning_summary: str
    structured_reasoning_units: tuple[dict[str, Any], ...]
    final_answer: str
    reference_answer: str | None
    model_id: str
    model_scale: float
    checkpoint_id: str
    rollout_id: str
    correctness: bool | None
    confidence: float
    abstained: bool
    verifier_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "structured_reasoning_units",
            tuple(self.structured_reasoning_units),
        )
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "model_scale", float(self.model_scale))

    def reasoning_quality_score(self) -> float:
        score = 0.5
        if self.correctness is True:
            score += 0.25
        elif self.correctness is False:
            score -= 0.2
        if self.verifier_status in {"verified", "pass", "supported"}:
            score += 0.2
        if self.verifier_status in {"failed", "contradicted", "unsupported"}:
            score -= 0.25
        if self.abstained and self.correctness is False:
            score += 0.1
        if self.abstained and self.correctness is True:
            score -= 0.1
        score += (self.confidence - 0.5) * 0.1
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class PairwiseReasoningExample:
    pair_id: str
    problem_id: str
    candidate_a: ReasoningTraceCandidate
    candidate_b: ReasoningTraceCandidate
    pair_type: PairType
    trace_order_randomized: bool
    teacher_label: PairwiseLabel
    teacher_confidence: float
    teacher_rationale_summary: str
    consensus_status: str
    difficulty_bucket: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_type", PairType(self.pair_type))
        object.__setattr__(self, "teacher_label", PairwiseLabel(self.teacher_label))
        object.__setattr__(
            self,
            "teacher_confidence",
            max(0.0, min(1.0, float(self.teacher_confidence))),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class CPTBatchMetrics:
    pairwise_label_loss: float
    teacher_confidence_weighting: float
    position_bias_diagnostic: float
    length_bias_diagnostic: float
    final_answer_shortcut_diagnostic: float
    model_identity_shortcut_diagnostic: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


__all__ = [
    "CPTBatchMetrics",
    "PairType",
    "PairwiseLabel",
    "PairwiseReasoningExample",
    "ReasoningTraceCandidate",
]
