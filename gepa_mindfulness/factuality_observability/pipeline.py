"""Pipeline hooks for 13-case v2 evaluation and instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import ConfidenceSignals, fuse_confidence
from .config import FactualityObservabilityConfig
from .decomposition import AtomicDecompositionResult, decompose_verify_and_repair
from .logging import SampleLogBundle, build_sample_log_bundle
from .routing import RoutingContext, choose_routing_action
from .schemas import (
    CaseOverlayV2,
    EvaluationScoresV2,
    ProvenanceStatus,
    RecommendedAction,
    VerificationStatus,
)
from .scoring import ScoringInputs, compute_scores


@dataclass(slots=True)
class PipelineInputs:
    """Input payload for high-level v2 pipeline execution."""

    sample_id: str
    prompt_id: str
    prompt: str
    answer: str
    model_id: str
    model_version: str
    domain: str
    task_type: str
    base_case_label: int
    declared_confidence: float
    latent_uncertainty_signal: float | None
    evidence_lookup: dict[str, list[str]]
    contradiction_lookup: set[str]
    abstention_viable: bool = True
    guessing_pressure_profile: str = "moderate_pressure"
    knowledge_boundary_risk: float = 0.2
    source_reference_divergence_risk: float = 0.2
    staleness_risk: float = 0.1


@dataclass(slots=True)
class PipelineOutputs:
    """Output payload containing overlay, decomposition, and logs."""

    case_overlay: CaseOverlayV2
    decomposition: AtomicDecompositionResult
    scores: EvaluationScoresV2
    log_bundle: SampleLogBundle


def run_v2_pipeline(
    inputs: PipelineInputs,
    config: FactualityObservabilityConfig,
) -> PipelineOutputs:
    """Run decomposition, calibration, routing, scoring, and logging hooks."""

    decomposition = decompose_verify_and_repair(
        answer=inputs.answer,
        evidence_lookup=inputs.evidence_lookup,
        contradiction_lookup=inputs.contradiction_lookup,
    )

    calibration = fuse_confidence(
        ConfidenceSignals(
            declared_confidence=inputs.declared_confidence,
            latent_uncertainty_signal=inputs.latent_uncertainty_signal,
            external_verification_confidence=decomposition.support_coverage_score,
        )
    )

    routing = choose_routing_action(
        RoutingContext(
            base_case_label=inputs.base_case_label,
            operational_confidence=calibration.final_operational_confidence,
            claim_complexity=min(
                1.0,
                len(decomposition.atomic_fact_list)
                / (config.claim_complexity_divisor if config.claim_complexity_divisor > 0 else 1.0),
            ),
            domain_risk=(
                config.domain_risk_high
                if inputs.domain in {"medical", "legal", "finance"}
                else config.domain_risk_default
            ),
            verification_budget=config.budgets.verification_budget,
            has_provenance=decomposition.attribution_precision_score > config.provenance_threshold,
            trace_worthy=decomposition.fact_risk_score > config.trace_worthy_threshold,
            abstention_viable=inputs.abstention_viable,
            guessing_pressure=config.default_guessing_pressure,
        )
    )

    if decomposition.contradiction_fact_indices:
        verification_status = VerificationStatus.CONTRADICTED
    elif decomposition.support_coverage_score >= 0.99:
        verification_status = VerificationStatus.EXTERNALLY_VERIFIED
    elif decomposition.support_coverage_score > 0.0:
        verification_status = VerificationStatus.WEAKLY_CHECKED
    else:
        verification_status = VerificationStatus.UNVERIFIED

    if decomposition.attribution_precision_score >= 0.99:
        provenance_status = ProvenanceStatus.EXTERNALLY_VALIDATED
    elif decomposition.attribution_precision_score > 0.0:
        provenance_status = ProvenanceStatus.STRUCTURED
    else:
        provenance_status = ProvenanceStatus.NONE

    case_overlay = CaseOverlayV2(
        base_case_label=inputs.base_case_label,
        observability_tier=calibration.observability_tier,
        declared_confidence=inputs.declared_confidence,
        latent_uncertainty_signal=inputs.latent_uncertainty_signal,
        verification_status=verification_status,
        provenance_status=provenance_status,
        recommended_action=routing.recommended_action,
    )

    scores = compute_scores(
        ScoringInputs(
            answer_correct=decomposition.support_coverage_score >= config.answer_correct_threshold,
            support_coverage=decomposition.support_coverage_score,
            attribution_precision=decomposition.attribution_precision_score,
            provenance_binding=decomposition.attribution_precision_score,
            calibration_quality=calibration.final_operational_confidence,
            abstention_quality=(
                config.abstention_quality_if_abstain
                if routing.recommended_action is RecommendedAction.ABSTAIN
                else config.abstention_quality_default
            ),
            routing_quality=(
                config.routing_quality_if_non_accept
                if routing.recommended_action is not RecommendedAction.ACCEPT
                else config.routing_quality_if_accept
            ),
            trace_utility=(
                config.trace_utility_high
                if decomposition.fact_risk_score > config.trace_utility_threshold
                else config.trace_utility_low
            ),
            failure_localization=(
                config.failure_localization_high
                if decomposition.unsupported_fact_indices
                else config.failure_localization_low
            ),
            taxonomy_coverage=config.taxonomy_coverage_default,
            guessing_diagnostic_quality=config.guessing_diagnostic_quality_default,
        )
    )

    log_bundle = build_sample_log_bundle(
        sample_id=inputs.sample_id,
        prompt_id=inputs.prompt_id,
        model_id=inputs.model_id,
        model_version=inputs.model_version,
        domain=inputs.domain,
        task_type=inputs.task_type,
        raw_input=inputs.prompt,
        raw_answer=inputs.answer,
        repaired_answer=decomposition.repaired_answer,
        case_overlay=case_overlay,
        routing_path=routing.routing_path,
        routing_target=routing.routing_target,
        atomic_fact_list=decomposition.atomic_fact_list,
        fact_verdict_per_fact=[item.value for item in decomposition.fact_verdict_per_fact],
        evidence_per_fact=decomposition.evidence_per_fact,
        unsupported_fact_indices=decomposition.unsupported_fact_indices,
        contradiction_fact_indices=decomposition.contradiction_fact_indices,
        answer_correctness_score=scores.answer_correctness_score,
        calibration_score=scores.calibration_score,
        atomic_fact_support_score=scores.atomic_fact_support_score,
        guessing_pressure_profile=inputs.guessing_pressure_profile,
        knowledge_boundary_risk=inputs.knowledge_boundary_risk,
        source_reference_divergence_risk=inputs.source_reference_divergence_risk,
        staleness_risk=inputs.staleness_risk,
    )

    return PipelineOutputs(
        case_overlay=case_overlay,
        decomposition=decomposition,
        scores=scores,
        log_bundle=log_bundle,
    )
