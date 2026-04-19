from gepa_mindfulness.factuality_observability.calibration import ConfidenceSignals, fuse_confidence
from gepa_mindfulness.factuality_observability.config import FactualityObservabilityConfig
from gepa_mindfulness.factuality_observability.pipeline import PipelineInputs, run_v2_pipeline
from gepa_mindfulness.factuality_observability.schemas import ObservabilityTier


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
