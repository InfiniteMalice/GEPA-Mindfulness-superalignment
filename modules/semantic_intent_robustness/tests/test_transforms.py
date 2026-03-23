"""Tests for semantic cluster transforms."""

# Third-party
import pytest

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.taxonomy import PolicyAction, VariantType
from semantic_intent_robustness.transforms import build_semantic_cluster


def test_semantic_cluster_contains_expected_variant_labels() -> None:
    clusters, _ = build_example_dataset()
    cluster = clusters[0]
    variant_types = {record.variant_type for record in cluster.records}
    assert VariantType.PARAPHRASE in variant_types
    assert VariantType.MULTILINGUAL_TRANSLATION in variant_types
    assert VariantType.ROLEPLAY_WRAPPER in variant_types


def test_negative_control_separates_policy() -> None:
    clusters, _ = build_example_dataset()
    cluster = clusters[0]
    assert cluster.negative_controls[0].policy_action == PolicyAction.ALLOW
    assert cluster.records[0].policy_action == PolicyAction.REFUSE


def test_negative_spec_policy_action_dict_requires_value() -> None:
    clusters, _ = build_example_dataset()
    seed = clusters[0].records[0]
    with pytest.raises(ValueError, match="Unsupported policy_action: None"):
        build_semantic_cluster(
            seed,
            [],
            negative_specs=[{"prompt_id": "bad", "prompt_text": "bad", "policy_action": {}}],
        )


def test_invalid_override_json_raises_clear_error() -> None:
    clusters, _ = build_example_dataset()
    seed = clusters[0].records[0]
    with pytest.raises(ValueError, match="Failed to parse overrides JSON"):
        build_semantic_cluster(
            seed,
            [
                {
                    "prompt_id": "bad-v",
                    "prompt_text": "bad",
                    "variant_type": "paraphrase",
                    "overrides": "{",
                }
            ],
        )


def test_variant_builder_preserves_nullable_fields_and_turn_index() -> None:
    clusters, _ = build_example_dataset()
    seed = clusters[0].records[0]
    cluster = build_semantic_cluster(
        seed,
        [
            {
                "prompt_id": "nullable-v",
                "prompt_text": "variant",
                "variant_type": "paraphrase",
                "parent_example_id": None,
                "user_goal_summary": None,
                "turn_index": 4,
            }
        ],
        negative_specs=[
            {
                "prompt_id": "nullable-n",
                "prompt_text": "negative",
                "parent_example_id": None,
                "user_goal_summary": None,
                "turn_index": 5,
            }
        ],
    )
    assert cluster.records[1].parent_example_id is None
    assert cluster.records[1].user_goal_summary is None
    assert cluster.records[1].turn_index == 4
    assert cluster.negative_controls[0].parent_example_id is None
    assert cluster.negative_controls[0].user_goal_summary is None
    assert cluster.negative_controls[0].turn_index == 5
