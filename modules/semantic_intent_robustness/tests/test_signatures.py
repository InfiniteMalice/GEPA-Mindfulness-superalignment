"""Tests for structured signatures and pipeline outputs."""

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.evaluators import SemanticRobustnessEvaluator
from semantic_intent_robustness.modules import SemanticIntentPipeline
from semantic_intent_robustness.schemas import MultiTurnConversation, SemanticCluster
from semantic_intent_robustness.signatures import ALL_SIGNATURES, DecomposeIntent


def test_signature_registry_contains_required_steps() -> None:
    signature_names = [signature.name for signature in ALL_SIGNATURES]
    assert signature_names[0] == DecomposeIntent.name
    assert "GenerateSafeResponse" in signature_names
    assert "AggregateMultiTurnRisk" in signature_names


def test_pipeline_emits_structured_outputs() -> None:
    clusters, _ = build_example_dataset()
    pipeline = SemanticIntentPipeline()
    result = pipeline.run(clusters[0].records[0], cluster=clusters[0])
    assert result.policy_decision["policy_action"] == "refuse"
    assert "response_text" in result.safe_response
    assert result.consistency_report is not None


def test_evaluator_compares_variant_policy_against_seed() -> None:
    clusters, _ = build_example_dataset()
    cluster = clusters[0]
    mutated = list(cluster.records)
    mutated[1] = type(mutated[1]).from_dict(
        {
            **mutated[1].to_dict(),
            "policy_action": "allow",
        }
    )
    evaluator = SemanticRobustnessEvaluator()
    score = evaluator.evaluate_cluster(
        SemanticCluster(
            cluster_id=cluster.cluster_id,
            records=tuple(mutated),
            negative_controls=cluster.negative_controls,
            cluster_summary=cluster.cluster_summary,
        )
    )["paraphrase_invariance"]
    assert score == 0.0


def test_run_conversation_honors_refused_turns() -> None:
    clusters, _ = build_example_dataset()
    pipeline = SemanticIntentPipeline()
    conversation = MultiTurnConversation(
        conversation_id="conv-refuse", turns=(clusters[0].records[0],)
    )
    summary = pipeline.run_conversation(conversation)
    assert summary["policy_action"] == "refuse"


def test_harmful_fnr_excludes_original_seed() -> None:
    clusters, _ = build_example_dataset()
    cluster = clusters[0]
    mutated = [
        cluster.records[0],
        type(cluster.records[1]).from_dict(
            {**cluster.records[1].to_dict(), "policy_action": "allow"}
        ),
        type(cluster.records[2]).from_dict(
            {**cluster.records[2].to_dict(), "policy_action": "allow"}
        ),
        type(cluster.records[3]).from_dict(
            {**cluster.records[3].to_dict(), "policy_action": "allow"}
        ),
    ]
    evaluator = SemanticRobustnessEvaluator()
    score = evaluator.summarize(
        [
            SemanticCluster(
                cluster.cluster_id,
                tuple(mutated),
                cluster.negative_controls,
                cluster.cluster_summary,
            )
        ],
        [],
    ).harmful_false_negative_rate_under_rewording
    assert score == 1.0


def test_abstention_score_penalizes_over_abstention() -> None:
    clusters, _ = build_example_dataset()
    cluster = clusters[0]
    mutated = [
        type(record).from_dict({**record.to_dict(), "abstain_recommended": True})
        for record in cluster.records
    ]
    evaluator = SemanticRobustnessEvaluator()
    summary = evaluator.summarize(
        [
            SemanticCluster(
                cluster.cluster_id,
                tuple(mutated),
                cluster.negative_controls,
                cluster.cluster_summary,
            )
        ],
        [],
    )
    assert summary.abstention_calibration_score < 1.0
