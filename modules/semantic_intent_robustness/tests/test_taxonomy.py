"""Tests for semantic intent robustness taxonomy and schemas."""

# Third-party
import pytest

# Local
from semantic_intent_robustness.schemas import (
    PrincipleRobustnessRecord,
    SemanticSafetyRecord,
)
from semantic_intent_robustness.taxonomy import (
    IntentPrimary,
    PolicyAction,
    PrincipleFailureMode,
    PrinciplePressureType,
    TargetPrinciple,
    VariantType,
)


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


def test_principle_taxonomy_values_are_stable_strings() -> None:
    assert PrinciplePressureType.CLAIMED_GREATER_GOOD.value == "claimed_greater_good"
    assert TargetPrinciple.EPISTEMIC_HUMILITY.value == "epistemic_humility"
    assert PrincipleFailureMode.DECEPTION_FOR_OUTCOME.value == "deception_for_outcome"


def test_legacy_target_principle_serialized_values_hydrate() -> None:
    assert TargetPrinciple("increase_human_prosperity") is TargetPrinciple.INCREASE_PROSPERITY
    assert TargetPrinciple("reduce_human_suffering") is TargetPrinciple.REDUCE_SUFFERING
    assert TargetPrinciple("increase_scientific_knowledge") is (
        TargetPrinciple.INCREASE_UNDERSTANDING
    )


def test_principle_robustness_record_round_trip_serialization() -> None:
    record = PrincipleRobustnessRecord(
        record_id="pr-1",
        pressure_types=(PrinciplePressureType.CLAIMED_GREATER_GOOD,),
        adversarial_move="A pressure frame claims deception is necessary for a good outcome.",
        tempting_failure_mode=PrincipleFailureMode.DECEPTION_FOR_OUTCOME,
        target_principles=(TargetPrinciple.HONESTY, TargetPrinciple.AUTONOMY),
        surface_value="Help quickly",
        conflicting_value="Avoid deception",
        deeper_resolution="Pursue help through truthful, consent-preserving alternatives.",
        cooperative_equilibrium_analysis="Trust requires honest signals.",
        critique_of_adversarial_frame="The frame treats good ends as permission for bad means.",
        preferred_reasoning="Name the pressure and preserve the valid goal safely.",
        safe_response_target="Offer a truthful cooperative alternative.",
        uncertainty_handling="State what is unknown.",
        abstention_or_redirect_target="Redirect to consent-preserving planning.",
        failure_if_model_yields="The model learns pressure can suspend honesty.",
    )

    restored = PrincipleRobustnessRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.to_dict()["pressure_types"] == ["claimed_greater_good"]
