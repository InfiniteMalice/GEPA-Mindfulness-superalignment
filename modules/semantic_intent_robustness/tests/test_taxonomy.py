"""Tests for semantic intent robustness taxonomy and schemas."""

# Local
from semantic_intent_robustness.schemas import SemanticSafetyRecord
from semantic_intent_robustness.taxonomy import IntentPrimary, PolicyAction, VariantType


def test_taxonomy_values_are_stable_strings() -> None:
    assert VariantType.PARAPHRASE.value == "paraphrase"
    assert IntentPrimary.HARMFUL_ENABLEMENT.value == "harmful_enablement"
    assert PolicyAction.ABSTAIN.value == "abstain"


def test_schema_round_trip_serialization() -> None:
    record = SemanticSafetyRecord(
        prompt_id="p1",
        prompt_text="abstract prompt",
        semantic_cluster_id="c1",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
    )
    restored = SemanticSafetyRecord.from_dict(record.to_dict())
    assert restored == record
