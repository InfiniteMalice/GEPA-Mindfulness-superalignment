"""Tests for semantic cluster transforms."""

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.taxonomy import PolicyAction, VariantType


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
