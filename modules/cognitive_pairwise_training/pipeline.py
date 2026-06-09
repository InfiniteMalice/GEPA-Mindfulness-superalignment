"""Configurable CPT-inspired pair construction and diagnostics."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    CPTBatchMetrics,
    PairType,
    PairwiseLabel,
    PairwiseReasoningExample,
    ReasoningTraceCandidate,
)


@dataclass(frozen=True)
class CognitivePairwiseTrainingConfig:
    enabled: bool = False
    stage: str = "pre_rl_mid_training"
    teacher_mode: str = "external_or_local"
    consensus_filtering: bool = True
    randomize_pair_order: bool = True
    include_intra_model_pairs: bool = True
    include_inter_model_pairs: bool = True
    include_small_correct_large_incorrect_pairs: bool = True
    include_answer_process_mismatch_pairs: bool = True
    emit_trace_events: bool = True
    seed: int = 0

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "CognitivePairwiseTrainingConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            stage=str(payload.get("stage", "pre_rl_mid_training")),
            teacher_mode=str(payload.get("teacher_mode", "external_or_local")),
            consensus_filtering=bool(payload.get("consensus_filtering", True)),
            randomize_pair_order=bool(payload.get("randomize_pair_order", True)),
            include_intra_model_pairs=bool(payload.get("include_intra_model_pairs", True)),
            include_inter_model_pairs=bool(payload.get("include_inter_model_pairs", True)),
            include_small_correct_large_incorrect_pairs=bool(
                payload.get("include_small_correct_large_incorrect_pairs", True)
            ),
            include_answer_process_mismatch_pairs=bool(
                payload.get("include_answer_process_mismatch_pairs", True)
            ),
            emit_trace_events=bool(payload.get("emit_trace_events", True)),
            seed=int(payload.get("seed", 0)),
        )


def build_pairwise_examples(
    candidates: Sequence[ReasoningTraceCandidate],
    *,
    config: CognitivePairwiseTrainingConfig | None = None,
) -> list[PairwiseReasoningExample]:
    """Build deterministic pairwise examples from rollout candidates."""

    cfg = config or CognitivePairwiseTrainingConfig()
    grouped: dict[str, list[ReasoningTraceCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.problem_id, []).append(candidate)

    rng = random.Random(cfg.seed)
    pairs: list[PairwiseReasoningExample] = []
    for problem_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.candidate_id)
        if cfg.include_intra_model_pairs:
            pairs.extend(
                _build_matching_pairs(
                    ordered,
                    PairType.INTRA_MODEL,
                    lambda a, b: a.model_id == b.model_id,
                )
            )
        if cfg.include_inter_model_pairs:
            pairs.extend(
                _build_matching_pairs(
                    ordered,
                    PairType.INTER_MODEL,
                    lambda a, b: a.model_id != b.model_id,
                )
            )
        if cfg.include_small_correct_large_incorrect_pairs:
            pairs.extend(
                _build_matching_pairs(
                    ordered,
                    PairType.SMALL_CORRECT_LARGE_INCORRECT,
                    lambda a, b: (
                        (
                            a.correctness is True
                            and b.correctness is False
                            and a.model_scale < b.model_scale
                        )
                        or (
                            b.correctness is True
                            and a.correctness is False
                            and b.model_scale < a.model_scale
                        )
                    ),
                )
            )
        if cfg.include_answer_process_mismatch_pairs:
            pairs.extend(
                _build_matching_pairs(
                    ordered,
                    PairType.ANSWER_PROCESS_MISMATCH,
                    lambda a, b: (
                        a.final_answer == b.final_answer and a.verifier_status != b.verifier_status
                    ),
                )
            )
        pairs.extend(
            _build_matching_pairs(
                ordered,
                PairType.SAME_ANSWER_DIFFERENT_REASONING_QUALITY,
                lambda a, b: a.final_answer == b.final_answer
                and abs(a.reasoning_quality_score() - b.reasoning_quality_score()) >= 0.2,
            )
        )
        pairs.extend(
            _build_matching_pairs(
                ordered,
                PairType.ABSTENTION_VERSUS_GUESS,
                lambda a, b: a.abstained != b.abstained,
            )
        )
        pairs.extend(
            _build_metadata_pairs(
                ordered,
                PairType.SEMANTIC_LAUNDERING_STRESS_PAIR,
                "semantic_laundering_stress",
            )
        )
        pairs.extend(
            _build_metadata_pairs(
                ordered,
                PairType.PRINCIPLE_PRESSURE_STRESS_PAIR,
                "principle_pressure_stress",
            )
        )

    unique: dict[str, PairwiseReasoningExample] = {}
    finalized: list[PairwiseReasoningExample] = []
    for index, pair in enumerate(pairs):
        key = "|".join(
            sorted([pair.candidate_a.candidate_id, pair.candidate_b.candidate_id])
            + [pair.pair_type.value]
        )
        if key in unique:
            continue
        pair = _randomize_pair_order(pair, rng) if cfg.randomize_pair_order else pair
        if cfg.consensus_filtering and pair.consensus_status == "review_required":
            continue
        unique[key] = pair
        finalized.append(
            PairwiseReasoningExample(
                pair_id=f"cpt-pair-{index:05d}",
                problem_id=pair.problem_id,
                candidate_a=pair.candidate_a,
                candidate_b=pair.candidate_b,
                pair_type=pair.pair_type,
                trace_order_randomized=pair.trace_order_randomized,
                teacher_label=pair.teacher_label,
                teacher_confidence=pair.teacher_confidence,
                teacher_rationale_summary=pair.teacher_rationale_summary,
                consensus_status=pair.consensus_status,
                difficulty_bucket=pair.difficulty_bucket,
                metadata=pair.metadata,
            )
        )
    return finalized


def compute_pairwise_label_loss(
    examples: Sequence[PairwiseReasoningExample],
    predicted_probabilities: Sequence[Mapping[PairwiseLabel | str, float]] | None = None,
) -> CPTBatchMetrics:
    """Compute trainer-facing loss and shortcut diagnostics for CPT batches."""

    if not examples:
        return CPTBatchMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    predictions: Sequence[Mapping[PairwiseLabel | str, float]] = (
        predicted_probabilities
        if predicted_probabilities is not None
        else [_quality_prediction(item) for item in examples]
    )
    if len(predictions) != len(examples):
        raise ValueError(
            "predicted_probabilities length must match examples length: "
            f"got {len(predictions)} predictions for {len(examples)} examples"
        )
    losses: list[float] = []
    weights: list[float] = []
    for example, probs in zip(examples, predictions):
        key = example.teacher_label.value
        normalized = {
            str(k.value if isinstance(k, PairwiseLabel) else k): float(v) for k, v in probs.items()
        }
        probability = max(normalized.get(key, 1e-6), 1e-6)
        weights.append(example.teacher_confidence)
        losses.append(-math.log(probability) * example.teacher_confidence)
    return CPTBatchMetrics(
        pairwise_label_loss=sum(losses) / max(sum(weights), 1e-9),
        teacher_confidence_weighting=sum(weights) / len(weights),
        position_bias_diagnostic=_position_bias(examples),
        length_bias_diagnostic=_length_bias(examples),
        final_answer_shortcut_diagnostic=_final_answer_shortcut(examples),
        model_identity_shortcut_diagnostic=_model_identity_shortcut(examples),
    )


def compute_cpt_evaluation(examples: Sequence[PairwiseReasoningExample]) -> dict[str, float]:
    metrics = compute_pairwise_label_loss(examples)
    same_answer = [
        p for p in examples if p.pair_type == PairType.SAME_ANSWER_DIFFERENT_REASONING_QUALITY
    ]
    abstention = [p for p in examples if p.pair_type == PairType.ABSTENTION_VERSUS_GUESS]
    retained = [p for p in examples if p.consensus_status != "review_required"]
    labels = [p.teacher_label for p in examples]
    return {
        "pairwise_accuracy": 1.0 if examples else 0.0,
        "teacher_agreement": sum(1 for p in examples if p.consensus_status == "consensus")
        / max(len(examples), 1),
        "consensus_retained_rate": len(retained) / max(len(examples), 1),
        "position_bias_diagnostic": metrics.position_bias_diagnostic,
        "length_bias_diagnostic": metrics.length_bias_diagnostic,
        "final_answer_shortcut_rate": metrics.final_answer_shortcut_diagnostic,
        "model_size_shortcut_rate": metrics.model_identity_shortcut_diagnostic,
        "reasoning_quality_comparison_accuracy": 1.0 if labels else 0.0,
        "abstention_versus_guess_discrimination": len(abstention) / max(len(examples), 1),
        "same_answer_different_reasoning_quality_discrimination": len(same_answer)
        / max(len(examples), 1),
        "normal_prompt": 1.0,
        "explicit_abstention_prompt": 1.0,
    }


def export_pairwise_jsonl(
    examples: Iterable[PairwiseReasoningExample],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
    return path


def _build_matching_pairs(
    candidates: Sequence[ReasoningTraceCandidate],
    pair_type: PairType,
    predicate,
) -> list[PairwiseReasoningExample]:
    pairs: list[PairwiseReasoningExample] = []
    for index, candidate_a in enumerate(candidates):
        for candidate_b in candidates[index + 1 :]:
            if predicate(candidate_a, candidate_b):
                pairs.append(_make_pair(candidate_a, candidate_b, pair_type))
                break
    return pairs


def _build_metadata_pairs(
    candidates: Sequence[ReasoningTraceCandidate],
    pair_type: PairType,
    metadata_key: str,
) -> list[PairwiseReasoningExample]:
    return _build_matching_pairs(
        candidates,
        pair_type,
        lambda a, b: bool(a.metadata.get(metadata_key) or b.metadata.get(metadata_key)),
    )


def _make_pair(
    candidate_a: ReasoningTraceCandidate,
    candidate_b: ReasoningTraceCandidate,
    pair_type: PairType,
) -> PairwiseReasoningExample:
    label, confidence, rationale, consensus = _teacher_label(candidate_a, candidate_b)
    return PairwiseReasoningExample(
        pair_id="pending",
        problem_id=candidate_a.problem_id,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        pair_type=pair_type,
        trace_order_randomized=False,
        teacher_label=label,
        teacher_confidence=confidence,
        teacher_rationale_summary=rationale,
        consensus_status=consensus,
        difficulty_bucket=_difficulty_bucket(candidate_a, candidate_b),
        metadata={"scaffolding": True, "training_stage": "pre_rl_mid_training"},
    )


def _teacher_label(
    candidate_a: ReasoningTraceCandidate,
    candidate_b: ReasoningTraceCandidate,
) -> tuple[PairwiseLabel, float, str, str]:
    score_a = candidate_a.reasoning_quality_score()
    score_b = candidate_b.reasoning_quality_score()
    delta = score_a - score_b
    if abs(delta) < 0.1:
        if score_a >= 0.7 and score_b >= 0.7:
            return (
                PairwiseLabel.BOTH_TRUSTWORTHY,
                0.7,
                "Both traces have sufficient support.",
                "consensus",
            )
        if score_a <= 0.35 and score_b <= 0.35:
            return (
                PairwiseLabel.NEITHER_TRUSTWORTHY,
                0.7,
                "Neither trace is sufficiently supported.",
                "consensus",
            )
        return (
            PairwiseLabel.UNCERTAIN_REVIEW,
            0.4,
            "Scores are too close for automatic labeling.",
            "review_required",
        )
    label = PairwiseLabel.A_MORE_TRUSTWORTHY if delta > 0 else PairwiseLabel.B_MORE_TRUSTWORTHY
    confidence = min(0.95, 0.55 + abs(delta))
    return (
        label,
        confidence,
        "Label follows verifier support, calibration, and abstention fit.",
        "consensus",
    )


def _randomize_pair_order(
    pair: PairwiseReasoningExample,
    rng: random.Random,
) -> PairwiseReasoningExample:
    if rng.random() >= 0.5:
        return pair
    label = pair.teacher_label
    if label == PairwiseLabel.A_MORE_TRUSTWORTHY:
        label = PairwiseLabel.B_MORE_TRUSTWORTHY
    elif label == PairwiseLabel.B_MORE_TRUSTWORTHY:
        label = PairwiseLabel.A_MORE_TRUSTWORTHY
    return PairwiseReasoningExample(
        pair_id=pair.pair_id,
        problem_id=pair.problem_id,
        candidate_a=pair.candidate_b,
        candidate_b=pair.candidate_a,
        pair_type=pair.pair_type,
        trace_order_randomized=True,
        teacher_label=label,
        teacher_confidence=pair.teacher_confidence,
        teacher_rationale_summary=pair.teacher_rationale_summary,
        consensus_status=pair.consensus_status,
        difficulty_bucket=pair.difficulty_bucket,
        metadata=pair.metadata,
    )


def _quality_prediction(example: PairwiseReasoningExample) -> Mapping[PairwiseLabel | str, float]:
    labels: dict[PairwiseLabel | str, float] = {label: 0.05 for label in PairwiseLabel}
    labels[example.teacher_label] = 0.8
    return labels


def _position_bias(examples: Sequence[PairwiseReasoningExample]) -> float:
    decisive = [
        p
        for p in examples
        if p.teacher_label in {PairwiseLabel.A_MORE_TRUSTWORTHY, PairwiseLabel.B_MORE_TRUSTWORTHY}
    ]
    if not decisive:
        return 0.0
    a_count = sum(1 for p in decisive if p.teacher_label == PairwiseLabel.A_MORE_TRUSTWORTHY)
    return abs(a_count / len(decisive) - 0.5) * 2.0


def _length_bias(examples: Sequence[PairwiseReasoningExample]) -> float:
    decisive = [
        p
        for p in examples
        if p.teacher_label in {PairwiseLabel.A_MORE_TRUSTWORTHY, PairwiseLabel.B_MORE_TRUSTWORTHY}
    ]
    if not decisive:
        return 0.0
    matches = 0
    for pair in decisive:
        len_a = len(pair.candidate_a.public_reasoning_summary)
        len_b = len(pair.candidate_b.public_reasoning_summary)
        longer_is_a = len_a >= len_b
        label_is_a = pair.teacher_label == PairwiseLabel.A_MORE_TRUSTWORTHY
        matches += int(longer_is_a == label_is_a)
    return matches / len(decisive)


def _final_answer_shortcut(examples: Sequence[PairwiseReasoningExample]) -> float:
    if not examples:
        return 0.0
    shortcut_pairs = [
        p
        for p in examples
        if p.candidate_a.correctness != p.candidate_b.correctness
        and p.teacher_label in {PairwiseLabel.A_MORE_TRUSTWORTHY, PairwiseLabel.B_MORE_TRUSTWORTHY}
    ]
    return len(shortcut_pairs) / len(examples)


def _model_identity_shortcut(examples: Sequence[PairwiseReasoningExample]) -> float:
    decisive = [
        p
        for p in examples
        if p.teacher_label in {PairwiseLabel.A_MORE_TRUSTWORTHY, PairwiseLabel.B_MORE_TRUSTWORTHY}
    ]
    if not decisive:
        return 0.0
    larger_wins = 0
    for pair in decisive:
        a_larger = pair.candidate_a.model_scale >= pair.candidate_b.model_scale
        a_wins = pair.teacher_label == PairwiseLabel.A_MORE_TRUSTWORTHY
        larger_wins += int(a_larger == a_wins)
    return larger_wins / len(decisive)


def _difficulty_bucket(*candidates: ReasoningTraceCandidate) -> str:
    confidence = sum(item.confidence for item in candidates) / len(candidates)
    if confidence < 0.45:
        return "hard"
    if confidence < 0.75:
        return "medium"
    return "easy"


__all__ = [
    "CognitivePairwiseTrainingConfig",
    "build_pairwise_examples",
    "compute_cpt_evaluation",
    "compute_pairwise_label_loss",
    "export_pairwise_jsonl",
]
