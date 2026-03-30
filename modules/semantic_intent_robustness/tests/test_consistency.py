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
    assert summary == {
        "conversation_id": "conv-003",
        "turn_count": 2,
        "max_capability_transfer_risk": "high",
        "contains_concealment": True,
        "contains_deception": True,
        "abstain_recommended": True,
        "conversation_blocked": True,
    }


def test_empty_multi_turn_aggregation_is_safe() -> None:
    summary = aggregate_multi_turn_risk(MultiTurnConversation(conversation_id="empty", turns=()))
    assert summary == {
        "conversation_id": "empty",
        "turn_count": 0,
        "max_capability_transfer_risk": None,
        "contains_concealment": False,
        "contains_deception": False,
        "abstain_recommended": False,
        "conversation_blocked": False,
    }


def test_multi_turn_aggregation_uses_risk_severity_order() -> None:
    _, conversations = build_example_dataset()
    source = _conversation_by_id(conversations, "conv-003")
    low_turn = type(source.turns[0]).from_dict(
        {
            **source.turns[0].to_dict(),
            "prompt_id": "risk-low",
            "capability_transfer_risk": "low",
            "policy_action": "allow",
            "abstain_recommended": False,
            "concealment_component": False,
            "deception_component": False,
        }
    )
    high_turn = type(source.turns[1]).from_dict(
        {
            **source.turns[1].to_dict(),
            "prompt_id": "risk-high",
            "capability_transfer_risk": "high",
            "policy_action": "allow",
            "abstain_recommended": False,
            "concealment_component": False,
            "deception_component": False,
        }
    )
    summary = aggregate_multi_turn_risk(
        MultiTurnConversation("conv-risk-order", (low_turn, high_turn))
    )
    assert summary["max_capability_transfer_risk"] == "high"


def test_topic_vs_intent_discrimination_handles_empty_records() -> None:
    score = topic_vs_intent_discrimination(
        SemanticCluster(cluster_id="empty", records=(), negative_controls=())
    )
    assert score == 0.0
