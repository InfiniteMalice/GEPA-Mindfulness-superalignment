"""Tests for semantic intent robustness taxonomy and schemas."""

# Third-party
import pytest

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


def test_from_dict_preserves_dataclass_defaults_for_partial_records() -> None:
    record = SemanticSafetyRecord.from_dict(
        {
            "prompt_id": "p-defaults",
            "prompt_text": "abstract prompt",
            "semantic_cluster_id": "c1",
            "parent_example_id": None,
            "variant_type": "original",
            "language": "en",
        }
    )
    assert record.intent_primary == IntentPrimary.UNCERTAIN
    assert record.allowed_high_level_help == ()
    assert record.disallowed_operational_help == ()


def test_negative_turn_index_raises_value_error() -> None:
    with pytest.raises(ValueError, match="turn_index must be non-negative"):
        SemanticSafetyRecord(
            prompt_id="p2",
            prompt_text="abstract prompt",
            semantic_cluster_id="c1",
            parent_example_id=None,
            variant_type=VariantType.ORIGINAL,
            language="en",
            turn_index=-1,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("benign_plausibility", -0.1), ("dual_use_probability", 1.1), ("evasion_likelihood", 1.5)),
)
def test_probability_fields_out_of_range_raise_value_error(field_name: str, value: float) -> None:
    payload = {
        "prompt_id": "p3",
        "prompt_text": "abstract prompt",
        "semantic_cluster_id": "c1",
        "parent_example_id": None,
        "variant_type": VariantType.ORIGINAL,
        "language": "en",
        field_name: value,
    }
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        SemanticSafetyRecord(**payload)
