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
