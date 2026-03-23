"""Tests for structured signatures and pipeline outputs."""

# Local
from semantic_intent_robustness.dataset_builder import build_example_dataset
from semantic_intent_robustness.evaluators import SemanticRobustnessEvaluator
from semantic_intent_robustness.modules import SemanticIntentPipeline
from semantic_intent_robustness.schemas import MultiTurnConversation, SemanticCluster
from semantic_intent_robustness.signatures import ALL_SIGNATURES, DecomposeIntent
from semantic_intent_robustness.taxonomy import PolicyAction, VariantType


def _record_by_variant(cluster: SemanticCluster, variant_type: VariantType):
    for record in cluster.records:
        if record.variant_type == variant_type:
            return record
    raise AssertionError(f"Missing variant: {variant_type.value}")


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
    paraphrase = _record_by_variant(cluster, VariantType.PARAPHRASE)
    mutated = [
        (
            type(record).from_dict({**record.to_dict(), "policy_action": "allow"})
            if record.prompt_id == paraphrase.prompt_id
            else record
        )
        for record in cluster.records
    ]
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
    harmful_variants = {
        VariantType.PARAPHRASE,
        VariantType.MULTILINGUAL_TRANSLATION,
        VariantType.ROLEPLAY_WRAPPER,
    }
    mutated = []
    for record in cluster.records:
        if record.variant_type in harmful_variants:
            mutated.append(type(record).from_dict({**record.to_dict(), "policy_action": "allow"}))
        else:
            mutated.append(record)
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
    negative_controls = [
        type(record).from_dict({**record.to_dict(), "abstain_recommended": True})
        for record in cluster.negative_controls
    ]
    evaluator = SemanticRobustnessEvaluator()
    summary = evaluator.summarize(
        [
            SemanticCluster(
                cluster.cluster_id,
                tuple(mutated),
                tuple(negative_controls),
                cluster.cluster_summary,
            )
        ],
        [],
    )
    assert summary.abstention_calibration_score < 1.0


def test_run_conversation_preserves_allow_for_benign_turns() -> None:
    clusters, _ = build_example_dataset()
    allow_turn = clusters[0].negative_controls[0]

    conversation = MultiTurnConversation(conversation_id="conv-allow", turns=(allow_turn,))
    pipeline = SemanticIntentPipeline()
    summary = pipeline.run_conversation(conversation)
    assert summary["policy_action"] == "allow"


def test_run_conversation_honors_abstain_turns() -> None:
    _, conversations = build_example_dataset()
    pipeline = SemanticIntentPipeline()
    summary = pipeline.run_conversation(
        MultiTurnConversation("conv-abstain", (conversations[0].turns[0],))
    )
    assert summary["policy_action"] == "abstain"


def test_run_conversation_preserves_bounded_help_before_abstain_summary() -> None:
    clusters, _ = build_example_dataset()
    pipeline = SemanticIntentPipeline()
    conversation = MultiTurnConversation(
        conversation_id="conv-bounded",
        turns=(clusters[1].records[0],),
    )
    summary = pipeline.run_conversation(conversation)
    assert summary["policy_action"] == PolicyAction.ALLOW_WITH_BOUNDARIES.value


def test_evaluator_multi_turn_accuracy_counts_explicit_abstain_turns() -> None:
    _, conversations = build_example_dataset()
    evaluator = SemanticRobustnessEvaluator()
    conversation = MultiTurnConversation(
        conversation_id="conv-explicit-abstain",
        turns=(
            type(conversations[0].turns[0]).from_dict(
                {
                    **conversations[0].turns[0].to_dict(),
                    "abstain_recommended": False,
                    "policy_action": PolicyAction.ABSTAIN.value,
                }
            ),
        ),
    )
    assert evaluator._multi_turn_accuracy(conversation) == 1.0


def test_evaluate_cluster_exposes_all_variant_family_metrics() -> None:
    clusters, _ = build_example_dataset()
    evaluator = SemanticRobustnessEvaluator()
    metrics = evaluator.evaluate_cluster(clusters[1])
    assert "academic_wrapper_robustness" in metrics
    assert "indirect_phrasing_invariance" in metrics
    assert "typo_noise_resilience" in metrics
    assert "benign_wrapper_harmful_core_robustness" in metrics
