"""Scoring model for 13-case v2 decomposed metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .schemas import EvaluationScoresV2


@dataclass(slots=True)
class ScoringInputs:
    """Inputs used to compute decomposed scores."""

    answer_correct: bool
    support_coverage: float
    attribution_precision: float
    provenance_binding: float
    calibration_quality: float
    abstention_quality: float
    routing_quality: float
    trace_utility: float
    failure_localization: float
    taxonomy_coverage: float
    guessing_diagnostic_quality: float


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def compute_scores(inputs: ScoringInputs) -> EvaluationScoresV2:
    """Compute a decomposed score vector without collapsing to one metric."""

    return EvaluationScoresV2(
        answer_correctness_score=_clamp(1.0 if inputs.answer_correct else 0.0),
        atomic_fact_support_score=_clamp(inputs.support_coverage),
        attribution_precision_score=_clamp(inputs.attribution_precision),
        provenance_binding_score=_clamp(inputs.provenance_binding),
        calibration_score=_clamp(inputs.calibration_quality),
        abstention_appropriateness_score=_clamp(inputs.abstention_quality),
        routing_decision_score=_clamp(inputs.routing_quality),
        trace_capture_utility_score=_clamp(inputs.trace_utility),
        failure_mode_localization_score=_clamp(inputs.failure_localization),
        hallucination_taxonomy_coverage_score=_clamp(inputs.taxonomy_coverage),
        guessing_vs_abstention_diagnostic_score=_clamp(inputs.guessing_diagnostic_quality),
    )
