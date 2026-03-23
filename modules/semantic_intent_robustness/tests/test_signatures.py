"""Tests for structured signatures and pipeline outputs."""

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.modules import SemanticIntentPipeline
from semantic_intent_robustness.signatures import ALL_SIGNATURES, DecomposeIntent


def test_signature_registry_contains_required_steps() -> None:
    signature_names = [signature.name for signature in ALL_SIGNATURES]
    assert signature_names[0] == DecomposeIntent.name
    assert "GenerateSafeResponse" in signature_names
    assert "AggregateMultiTurnRisk" in signature_names


def test_pipeline_emits_structured_outputs() -> None:
    clusters, _ = build_example_dataset()
    pipeline = SemanticIntentPipeline()
    result = pipeline.run(clusters[0].records[0], cluster=clusters[0])
    assert result.policy_decision["policy_action"] == "refuse"
    assert "response_text" in result.safe_response
    assert result.consistency_report is not None
