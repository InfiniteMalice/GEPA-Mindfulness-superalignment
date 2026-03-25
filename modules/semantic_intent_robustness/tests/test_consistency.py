"""Tests for semantic consistency metrics."""

# Local
from semantic_intent_robustness.consistency import (
    aggregate_multi_turn_risk,
    policy_consistency_score,
    topic_vs_intent_discrimination,
)
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.schemas import MultiTurnConversation, SemanticCluster


def _cluster_by_id(clusters, cluster_id: str):
    return next(cluster for cluster in clusters if cluster.cluster_id == cluster_id)


def _conversation_by_id(conversations, conversation_id: str):
    return next(
        conversation
        for conversation in conversations
        if conversation.conversation_id == conversation_id
    )


def test_policy_stability_across_same_intent_variants() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    score = policy_consistency_score(list(cluster.records))
    assert score == 1.0


def test_negative_controls_are_separated() -> None:
    clusters, _ = build_example_dataset()
    cluster = _cluster_by_id(clusters, "cluster-harmful-access")
    score = topic_vs_intent_discrimination(cluster)
    assert score == 1.0


def test_multi_turn_aggregation_detects_escalation() -> None:
    _, conversations = build_example_dataset()
    conversation = _conversation_by_id(conversations, "conv-003")
    summary = aggregate_multi_turn_risk(conversation)
    assert summary["contains_concealment"] is True
    assert summary["abstain_recommended"] is True


def test_empty_multi_turn_aggregation_is_safe() -> None:
    summary = aggregate_multi_turn_risk(MultiTurnConversation(conversation_id="empty", turns=()))
    assert summary["turn_count"] == 0
    assert summary["max_capability_transfer_risk"] is None
    assert summary["contains_concealment"] is False


def test_multi_turn_aggregation_uses_risk_severity_order() -> None:
    _, conversations = build_example_dataset()
    conversation = _conversation_by_id(conversations, "conv-003")
    summary = aggregate_multi_turn_risk(conversation)
    assert summary["max_capability_transfer_risk"] == "high"


def test_topic_vs_intent_discrimination_handles_empty_records() -> None:
    score = topic_vs_intent_discrimination(
        SemanticCluster(cluster_id="empty", records=(), negative_controls=())
    )
    assert score == 0.0
