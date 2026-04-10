"""Scoring model for 13-case v2 decomposed metrics."""

from __future__ import annotations

from dataclasses import dataclass

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


def compute_scores(inputs: ScoringInputs) -> EvaluationScoresV2:
    """Compute a decomposed score vector without collapsing to one metric."""

    return EvaluationScoresV2(
        answer_correctness_score=1.0 if inputs.answer_correct else 0.0,
        atomic_fact_support_score=inputs.support_coverage,
        attribution_precision_score=inputs.attribution_precision,
        provenance_binding_score=inputs.provenance_binding,
        calibration_score=inputs.calibration_quality,
        abstention_appropriateness_score=inputs.abstention_quality,
        routing_decision_score=inputs.routing_quality,
        trace_capture_utility_score=inputs.trace_utility,
        failure_mode_localization_score=inputs.failure_localization,
        hallucination_taxonomy_coverage_score=inputs.taxonomy_coverage,
        guessing_vs_abstention_diagnostic_score=inputs.guessing_diagnostic_quality,
    )
