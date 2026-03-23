"""Tests for semantic consistency metrics."""

# Local
from semantic_intent_robustness.consistency import (
    aggregate_multi_turn_risk,
    policy_consistency_score,
    topic_vs_intent_discrimination,
)
from semantic_intent_robustness.dataset_builder import build_example_dataset


def test_policy_stability_across_same_intent_variants() -> None:
    clusters, _ = build_example_dataset()
    score = policy_consistency_score(clusters[0].records)
    assert score == 1.0


def test_negative_controls_are_separated() -> None:
    clusters, _ = build_example_dataset()
    score = topic_vs_intent_discrimination(clusters[0])
    assert score == 1.0


def test_multi_turn_aggregation_detects_escalation() -> None:
    _, conversations = build_example_dataset()
    summary = aggregate_multi_turn_risk(conversations[0])
    assert summary["contains_concealment"] is True
    assert summary["abstain_recommended"] is True
