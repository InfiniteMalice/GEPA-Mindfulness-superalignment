"""Tests for semantic cluster transforms."""

# Third-party
import pytest

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.taxonomy import PolicyAction, VariantType
from semantic_intent_robustness.transforms import build_semantic_cluster


def _cluster_by_id(clusters, cluster_id: str):
    cluster = next((item for item in clusters if item.cluster_id == cluster_id), None)
    if cluster is None:
        raise AssertionError(f"Missing cluster: {cluster_id}")
    return cluster


def _record_by_variant(cluster, variant_type: VariantType):
    record = next((item for item in cluster.records if item.variant_type == variant_type), None)
    if record is None:
        raise AssertionError(f"Missing variant: {variant_type.value}")
    return record


def test_semantic_cluster_contains_expected_variant_labels() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    variant_types = {record.variant_type for record in cluster.records}
    assert VariantType.PARAPHRASE in variant_types
    assert VariantType.MULTILINGUAL_TRANSLATION in variant_types
    assert VariantType.ROLEPLAY_WRAPPER in variant_types


def test_negative_control_separates_policy() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    assert cluster.negative_controls[0].policy_action == PolicyAction.ALLOW
    assert _record_by_variant(cluster, VariantType.ORIGINAL).policy_action == PolicyAction.REFUSE


def test_negative_spec_policy_action_dict_requires_value() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    seed = _record_by_variant(cluster, VariantType.ORIGINAL)
    with pytest.raises(ValueError, match="Unsupported policy_action: None"):
        build_semantic_cluster(
            seed,
            [],
            negative_specs=[{"prompt_id": "bad", "prompt_text": "bad", "policy_action": {}}],
        )


def test_invalid_override_json_raises_clear_error() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    seed = _record_by_variant(cluster, VariantType.ORIGINAL)
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
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    seed = _record_by_variant(cluster, VariantType.ORIGINAL)
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
    assert cluster.records[1].user_goal_summary == ""
    assert cluster.records[1].turn_index == 4
    assert cluster.negative_controls[0].parent_example_id is None
    assert cluster.negative_controls[0].user_goal_summary == ""
    assert cluster.negative_controls[0].turn_index == 5


def test_negative_controls_default_to_benign_intent() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    seed = _record_by_variant(cluster, VariantType.ORIGINAL)
    cluster = build_semantic_cluster(
        seed,
        [],
        negative_specs=[
            {
                "prompt_id": "default-benign",
                "prompt_text": "benign overlap",
            }
        ],
    )
    assert cluster.negative_controls[0].intent_primary.value == "benign_information"


def test_negative_controls_reset_risk_defaults() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    seed = _record_by_variant(cluster, VariantType.ORIGINAL)
    cluster = build_semantic_cluster(
        seed,
        [],
        negative_specs=[
            {
                "prompt_id": "default-risk",
                "prompt_text": "benign overlap",
            }
        ],
    )
    control = cluster.negative_controls[0]
    assert control.dual_use_probability == 0.0
    assert control.capability_transfer_risk.value == "low"
    assert control.harm_domain.value == "none"
    assert control.abstain_recommended is False
