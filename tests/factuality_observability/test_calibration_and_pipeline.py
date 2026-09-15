import json

import pytest

from gepa_mindfulness.factuality_observability.calibration import (
    ConfidenceSignals,
    ConfidenceSource,
    fuse_confidence,
)
from gepa_mindfulness.factuality_observability.config import FactualityObservabilityConfig
from gepa_mindfulness.factuality_observability.logging import to_jsonl_line
from gepa_mindfulness.factuality_observability.pipeline import PipelineInputs, run_v2_pipeline
from gepa_mindfulness.factuality_observability.routing import RoutingContext, choose_routing_action
from gepa_mindfulness.factuality_observability.schemas import ObservabilityTier, RecommendedAction


def test_calibration_fuses_external_priority() -> None:
    output = fuse_confidence(
        ConfidenceSignals(
            declared_confidence=0.2,
            latent_uncertainty_signal=0.8,
            structured_provenance_confidence=0.6,
            external_verification_confidence=0.95,
        )
    )
    assert output.final_operational_confidence > 0.7


def test_calibration_tier_for_latent_only_is_o2() -> None:
    output = fuse_confidence(
        ConfidenceSignals(declared_confidence=0.5, latent_uncertainty_signal=0.2)
    )
    assert output.observability_tier is ObservabilityTier.O2


def test_pipeline_returns_schema_complete_log_bundle() -> None:
    outputs = run_v2_pipeline(
        inputs=PipelineInputs(
            sample_id="s2",
            prompt_id="p2",
            prompt="Where is Paris?",
            answer="Paris is in France.",
            model_id="m1",
            model_version="v1",
            domain="qa",
            task_type="qa",
            base_case_label=1,
            declared_confidence=0.9,
            latent_uncertainty_signal=0.2,
            evidence_lookup={"Paris is in France": ["atlas"]},
            contradiction_lookup=set(),
        ),
        config=FactualityObservabilityConfig(),
    )
    assert outputs.case_overlay.final_case_overlay.startswith("Case1-")
    assert outputs.log_bundle.atomic_fact_list
    assert outputs.log_bundle.fact_verdict_per_fact
    assert len(outputs.log_bundle.fact_verdict_per_fact) == len(outputs.log_bundle.atomic_fact_list)
    assert isinstance(outputs.log_bundle.unsupported_fact_indices, list)
    assert isinstance(outputs.log_bundle.contradiction_fact_indices, list)
    assert outputs.log_bundle.recommended_action


@pytest.mark.parametrize("declared", [0.2, 1.0])
def test_internal_confidence_cannot_raise_estimate_or_authorize_accept(declared: float) -> None:
    output = fuse_confidence(
        ConfidenceSignals(
            declared_confidence=declared,
            latent_uncertainty_signal=0.0,
            mechanistic_risk_indicator=0.0,
        )
    )
    assert output.final_operational_confidence <= declared
    assert output.verification_required
    assert output.confidence_source is ConfidenceSource.FUSED
    assert output.confidence_sources == [
        ConfidenceSource.MODEL_SELF_REPORT,
        ConfidenceSource.INTERNAL_REPRESENTATION,
    ]
    decision = choose_routing_action(
        RoutingContext(
            base_case_label=1,
            operational_confidence=output.final_operational_confidence,
            claim_complexity=0.0,
            domain_risk=0.0,
            verification_budget=1,
            has_provenance=True,
            trace_worthy=False,
            abstention_viable=True,
            guessing_pressure=0.0,
            verification_required=output.verification_required,
        )
    )
    assert decision.recommended_action is RecommendedAction.ROUTE_EXTERNAL


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_invalid_confidence_cannot_be_silently_promoted(invalid: float) -> None:
    with pytest.raises(ValueError, match="declared_confidence"):
        fuse_confidence(ConfidenceSignals(declared_confidence=invalid))


def test_self_report_retains_explicit_source_without_telemetry() -> None:
    output = fuse_confidence(ConfidenceSignals(declared_confidence=0.95))
    assert output.confidence_source is ConfidenceSource.MODEL_SELF_REPORT
    assert output.confidence_sources == [ConfidenceSource.MODEL_SELF_REPORT]
    assert output.verification_required


def test_representation_instability_lowers_confidence_and_routes_independent_check() -> None:
    def run(stability: float):
        return run_v2_pipeline(
            inputs=PipelineInputs(
                sample_id="representation-pair",
                prompt_id="p1",
                prompt="Where is Paris?",
                answer="Paris is in France.",
                model_id="m1",
                model_version="v1",
                domain="qa",
                task_type="qa",
                base_case_label=1,
                declared_confidence=1.0,
                latent_uncertainty_signal=0.0,
                evidence_lookup={"Paris is in France": ["atlas"]},
                contradiction_lookup=set(),
                representation_stability=stability,
            ),
            config=FactualityObservabilityConfig(),
        )

    stable = run(1.0)
    unstable = run(0.4)
    assert stable.case_overlay.base_case_label == unstable.case_overlay.base_case_label == 1
    assert unstable.log_bundle.operational_confidence < stable.log_bundle.operational_confidence
    assert unstable.case_overlay.recommended_action is RecommendedAction.ROUTE_EXTERNAL
    log = json.loads(to_jsonl_line(unstable.log_bundle))
    assert log["representation_sensitive"]
    assert log["verification_required"]
    assert log["confidence_source"] == "FUSED"
    assert "EXTERNAL_VERIFIER" in log["confidence_sources"]
