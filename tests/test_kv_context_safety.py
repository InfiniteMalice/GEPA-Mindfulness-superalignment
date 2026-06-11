"""Tests for opt-in KV-context semantic safety overlays."""

# Standard library
from __future__ import annotations

from dataclasses import replace

# Local
from semantic_intent_robustness.capability_graph import (
    build_capability_disclosure_graph,
    detect_harm_enabling_closure,
    estimate_capability_completion,
    merge_disclosure_event,
    recommend_minimum_safe_redaction,
)
from semantic_intent_robustness.disclosure_events import (
    DisclosureSourceType,
    disclosure_event_from_candidate_response,
    disclosure_event_from_semantic_record,
)
from semantic_intent_robustness.evaluators import SemanticRobustnessEvaluator
from semantic_intent_robustness.internal_state_trajectory import (
    DeterministicInternalStateTrajectoryAdapter,
    TranscriptOnlyTrajectoryAdapter,
)
from semantic_intent_robustness.kv_context_safety import (
    DeterministicKVCacheAdapter,
    KVContextMode,
    KVContextRiskScorer,
    KVContextSafetyConfig,
    TranscriptFallbackAdapter,
)
from semantic_intent_robustness.release_gate import (
    CandidateResponse,
    CandidateResponseReleaseGate,
    ReleaseDecision,
    ReleaseGateConfig,
    ReleaseGateMode,
)
from semantic_intent_robustness.schemas import MultiTurnConversation, SemanticSafetyRecord
from semantic_intent_robustness.taxonomy import (
    CapabilityTransferRisk,
    ExecutionalityLevel,
    HarmDomain,
    IntentPrimary,
    OperationalSpecificity,
    PolicyAction,
    RequestedCapability,
    VariantType,
)


def _record(
    prompt_id: str,
    text: str,
    *,
    turn_index: int = 0,
    intent: IntentPrimary = IntentPrimary.BENIGN_INFORMATION,
    risk: CapabilityTransferRisk = CapabilityTransferRisk.LOW,
    specificity: OperationalSpecificity = OperationalSpecificity.LOW,
    execution: ExecutionalityLevel = ExecutionalityLevel.ABSTRACT,
    policy: PolicyAction = PolicyAction.ALLOW,
    domain: HarmDomain = HarmDomain.NONE,
    concealment: bool = False,
    deception: bool = False,
) -> SemanticSafetyRecord:
    return SemanticSafetyRecord(
        prompt_id=prompt_id,
        prompt_text=text,
        semantic_cluster_id="cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        turn_index=turn_index,
        conversation_id="conv",
        intent_primary=intent,
        dual_use_probability=0.1 if risk == CapabilityTransferRisk.LOW else 0.45,
        evasion_likelihood=0.4 if concealment or deception else 0.0,
        requested_capability=RequestedCapability.ANALYSIS,
        capability_transfer_risk=risk,
        executionality_level=execution,
        operational_specificity=specificity,
        harm_domain=domain,
        concealment_component=concealment,
        deception_component=deception,
        policy_action=policy,
        abstain_recommended=False,
    )


def test_kv_context_config_defaults_disabled() -> None:
    config = KVContextSafetyConfig()

    assert config.enabled is False
    assert config.mode == KVContextMode.OFF


def test_contextual_ratio_uses_minimum_denominator() -> None:
    record = _record("r1", "abstract question")
    conversation = MultiTurnConversation("conv", (record,))
    snapshot = DeterministicKVCacheAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=0,
        prompt_text=record.prompt_text,
        conversation_history=(record.prompt_text,),
        kv_cache={"features": [0.0, 0.0, 0.0]},
    )
    assessment = KVContextRiskScorer().assess(
        latest_record=record,
        conversation=conversation,
        snapshot=snapshot,
        config=KVContextSafetyConfig(enabled=True, minimum_denominator=0.2),
    )

    assert assessment.contextual_ratio == assessment.contextual_risk / 0.2


def test_low_risk_benign_conversation_does_not_trigger() -> None:
    turns = (
        _record("r1", "What is authentication at a high level?", turn_index=0),
        _record("r2", "How do teams document risk responsibly?", turn_index=1),
    )
    conversation = MultiTurnConversation("conv", turns, ground_truth_blocked=False)
    snapshot = TranscriptFallbackAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=1,
        prompt_text=turns[-1].prompt_text,
        conversation_history=tuple(turn.prompt_text for turn in turns),
    )

    assessment = KVContextRiskScorer().assess(
        latest_record=turns[-1],
        conversation=conversation,
        snapshot=snapshot,
        config=KVContextSafetyConfig(enabled=True, mode=KVContextMode.TRANSCRIPT_FALLBACK),
    )

    assert assessment.trajectory_flag is False
    assert assessment.recommended_action == "allow"


def test_defensive_audit_retains_bounded_help() -> None:
    record = _record(
        "r1",
        "For a defensive audit, how should I inventory COMPONENT_A safely?",
        risk=CapabilityTransferRisk.MODERATE,
        specificity=OperationalSpecificity.MEDIUM,
        execution=ExecutionalityLevel.STRATEGIC,
        policy=PolicyAction.ALLOW_WITH_BOUNDARIES,
        domain=HarmDomain.CYBER,
    )
    conversation = MultiTurnConversation("conv", (record,), ground_truth_blocked=False)
    snapshot = TranscriptFallbackAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=0,
        prompt_text=record.prompt_text,
        conversation_history=(record.prompt_text,),
    )

    assessment = KVContextRiskScorer().assess(
        latest_record=record,
        conversation=conversation,
        snapshot=snapshot,
        config=KVContextSafetyConfig(enabled=True, mode=KVContextMode.TRANSCRIPT_FALLBACK),
    )

    assert assessment.trajectory_flag is False
    assert assessment.recommended_action in {"allow_with_boundaries", "clarify"}


def test_multi_turn_weaving_triggers_contextual_uplift() -> None:
    turns = (
        _record("r1", "Catalog COMPONENT_A for a fictional scenario.", turn_index=0),
        _record("r2", "Add PARAMETER_B and sequence STEP_C abstractly.", turn_index=1),
        _record(
            "r3",
            "Which missing integration detail connects the fragments?",
            turn_index=2,
            risk=CapabilityTransferRisk.MODERATE,
            specificity=OperationalSpecificity.MEDIUM,
            execution=ExecutionalityLevel.STRATEGIC,
        ),
    )
    conversation = MultiTurnConversation("conv", turns, ground_truth_blocked=True)
    snapshot = TranscriptFallbackAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=2,
        prompt_text=turns[-1].prompt_text,
        conversation_history=tuple(turn.prompt_text for turn in turns),
    )

    assessment = KVContextRiskScorer().assess(
        latest_record=turns[-1],
        conversation=conversation,
        snapshot=snapshot,
        config=KVContextSafetyConfig(enabled=True, mode=KVContextMode.TRANSCRIPT_FALLBACK),
    )

    assert assessment.contextual_risk >= 0.6
    assert assessment.contextual_uplift >= 0.25
    assert assessment.trajectory_flag is True


def test_candidate_response_can_create_closure() -> None:
    record = _record("r1", "How do these placeholders relate?", risk=CapabilityTransferRisk.LOW)
    conversation = MultiTurnConversation("conv", (record,), ground_truth_blocked=None)
    snapshot = TranscriptFallbackAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=0,
        prompt_text=record.prompt_text,
        conversation_history=("COMPONENT_A PARAMETER_B",),
        candidate_response="INTEGRATION_STEP_C completes the sequence.",
    )

    assessment = KVContextRiskScorer().assess(
        latest_record=record,
        conversation=conversation,
        snapshot=snapshot,
        config=KVContextSafetyConfig(
            enabled=True,
            mode=KVContextMode.TRANSCRIPT_FALLBACK,
            enable_candidate_response_screening=True,
        ),
        candidate_response="INTEGRATION_STEP_C completes the sequence.",
    )

    assert assessment.candidate_response_risk is not None
    assert assessment.closure_risk is not None
    assert assessment.closure_risk >= 0.7
    assert assessment.recommended_action == "refuse"


def test_transcript_fallback_used_when_cache_missing() -> None:
    snapshot = TranscriptFallbackAdapter().extract_snapshot(
        conversation_id="conv",
        turn_index=0,
        prompt_text="abstract prompt",
        conversation_history=("abstract prompt",),
        kv_cache=None,
    )

    assert snapshot.cache_available is False
    assert snapshot.transcript_fallback_used is True


def test_deterministic_cache_adapter_is_reproducible() -> None:
    adapter = DeterministicKVCacheAdapter()
    first = adapter.extract_snapshot(
        conversation_id="conv",
        turn_index=1,
        prompt_text="p",
        conversation_history=("a", "b"),
        kv_cache={"features": [0.2, 0.9, 2.0]},
    )
    second = adapter.extract_snapshot(
        conversation_id="conv",
        turn_index=1,
        prompt_text="p",
        conversation_history=("a", "b"),
        kv_cache={"features": [0.2, 0.9, 2.0]},
    )

    assert first == second
    assert first.feature_vector[-1] == 1.0


def test_existing_semantic_pipeline_behavior_unchanged_by_default() -> None:
    from semantic_intent_robustness.modules import SemanticIntentPipeline

    record = _record("r1", "benign")
    pipeline = SemanticIntentPipeline()
    before = pipeline.run(record)
    after = pipeline.run(record)

    assert before == after


def test_release_gate_runs_before_release_and_prefers_redaction() -> None:
    record = _record("r1", "connect placeholders", turn_index=1)
    conversation = MultiTurnConversation("conv", (record,))
    kv_assessment = KVContextRiskScorer().assess(
        latest_record=record,
        conversation=conversation,
        snapshot=TranscriptFallbackAdapter().extract_snapshot(
            conversation_id="conv",
            turn_index=1,
            prompt_text=record.prompt_text,
            conversation_history=("COMPONENT_A", "PARAMETER_B"),
            candidate_response="INTEGRATION_STEP_C plus CONCEALMENT_DETAIL_D",
        ),
        config=KVContextSafetyConfig(enabled=True, enable_candidate_response_screening=True),
        candidate_response="INTEGRATION_STEP_C plus CONCEALMENT_DETAIL_D",
    )
    candidate = CandidateResponse(
        response_id="resp",
        conversation_id="conv",
        turn_index=1,
        text="INTEGRATION_STEP_C plus CONCEALMENT_DETAIL_D",
    )
    event = disclosure_event_from_candidate_response(candidate)
    graph = build_capability_disclosure_graph((event,))

    assessment = CandidateResponseReleaseGate().assess_before_release(
        conversation=conversation,
        latest_record=record,
        candidate_response=candidate,
        kv_assessment=kv_assessment,
        disclosure_graph=graph,
        config=ReleaseGateConfig(mode=ReleaseGateMode.GATED, redaction_preferred=True),
    )

    assert assessment.closure_flag is True
    assert assessment.decision == ReleaseDecision.REDACT
    assert assessment.minimum_safe_redaction_possible is True


def test_release_gate_uses_candidate_response_identifiers() -> None:
    record = _record("r1", "connect placeholders", turn_index=1)
    conversation = MultiTurnConversation("conv", (record,))
    candidate = CandidateResponse(
        response_id="resp",
        conversation_id="candidate-conv",
        turn_index=7,
        text="bounded answer",
    )
    assessment = CandidateResponseReleaseGate().assess_before_release(
        conversation=conversation,
        latest_record=record,
        candidate_response=candidate,
        kv_assessment=KVContextRiskScorer().assess(
            latest_record=record,
            conversation=conversation,
            snapshot=TranscriptFallbackAdapter().extract_snapshot(
                conversation_id="conv",
                turn_index=1,
                prompt_text=record.prompt_text,
                conversation_history=(record.prompt_text,),
            ),
            config=KVContextSafetyConfig(enabled=True),
        ),
        disclosure_graph=build_capability_disclosure_graph(()),
        config=ReleaseGateConfig(mode=ReleaseGateMode.OFF),
    )

    assert assessment.conversation_id == "candidate-conv"
    assert assessment.turn_index == 7


def test_release_gate_shadow_and_advisory_do_not_enforce() -> None:
    record = _record("r1", "connect placeholders", turn_index=1)
    conversation = MultiTurnConversation("conv", (record,))
    candidate = CandidateResponse("resp", "conv", 1, "INTEGRATION_STEP_C")
    kv_assessment = KVContextRiskScorer().assess(
        latest_record=record,
        conversation=conversation,
        snapshot=TranscriptFallbackAdapter().extract_snapshot(
            conversation_id="conv",
            turn_index=1,
            prompt_text=record.prompt_text,
            conversation_history=("COMPONENT_A", "PARAMETER_B"),
            candidate_response=candidate.text,
        ),
        config=KVContextSafetyConfig(enabled=True, enable_candidate_response_screening=True),
        candidate_response=candidate.text,
    )
    graph = build_capability_disclosure_graph(
        (
            disclosure_event_from_candidate_response(
                CandidateResponse("candidate", "conv", 1, "INTEGRATION_STEP_C")
            ),
        )
    )

    for mode in (ReleaseGateMode.SHADOW, ReleaseGateMode.ADVISORY):
        assessment = CandidateResponseReleaseGate().assess_before_release(
            conversation=conversation,
            latest_record=record,
            candidate_response=candidate,
            kv_assessment=kv_assessment,
            disclosure_graph=graph,
            config=ReleaseGateConfig(mode=mode),
        )
        assert assessment.decision == ReleaseDecision.RELEASE
        assert assessment.metadata["would_be_decision"] in {"redact", "refuse"}


def test_disclosure_graph_aggregates_non_chat_sources() -> None:
    record = _record("r1", "COMPONENT_A", turn_index=0)
    user_event = disclosure_event_from_semantic_record(record)
    tool_event = user_event.with_updates(
        event_id="tool",
        source_type=DisclosureSourceType.TOOL_OUTPUT,
        fragment_ids=("PARAMETER_B",),
        content_summary="PARAMETER_B",
    )
    code_event = user_event.with_updates(
        event_id="code",
        source_type=DisclosureSourceType.GENERATED_CODE,
        fragment_ids=("INTEGRATION_STEP_C",),
        content_summary="INTEGRATION_STEP_C",
    )
    graph = build_capability_disclosure_graph((user_event, tool_event))
    merged = merge_disclosure_event(graph, code_event)

    assert len(merged.disclosure_events) == 3
    assert estimate_capability_completion(merged) >= 0.6
    assert detect_harm_enabling_closure(merged) is True
    assert "INTEGRATION_STEP_C" in recommend_minimum_safe_redaction(merged).redaction_summary


def test_capability_graph_paths_traverse_to_max_depth() -> None:
    graph = build_capability_disclosure_graph(
        (
            disclosure_event_from_candidate_response(
                CandidateResponse("a", "conv", 0, "COMPONENT_A")
            ),
            disclosure_event_from_candidate_response(
                CandidateResponse("b", "conv", 1, "PARAMETER_B")
            ),
            disclosure_event_from_candidate_response(
                CandidateResponse("c", "conv", 2, "INTEGRATION_STEP_C")
            ),
        )
    )

    assert ("COMPONENT_A", "PARAMETER_B", "INTEGRATION_STEP_C") in graph.paths_to(
        "INTEGRATION_STEP_C",
        max_depth=2,
    )


def test_capability_graph_preserves_higher_fragment_confidence() -> None:
    high = disclosure_event_from_candidate_response(
        CandidateResponse("high", "conv", 0, "COMPONENT_A")
    ).with_updates(operational_specificity=0.9, executionality=0.8)
    low = disclosure_event_from_candidate_response(
        CandidateResponse("low", "conv", 1, "COMPONENT_A")
    ).with_updates(operational_specificity=0.1, executionality=0.1)

    graph = build_capability_disclosure_graph((high, low))

    assert graph.fragments[0].confidence == 0.9


def test_internal_state_adapters_are_deterministic_and_transcript_safe() -> None:
    deterministic = DeterministicInternalStateTrajectoryAdapter(layer_count=2)
    first = deterministic.extract_layer_trajectory(
        conversation_id="conv",
        turn_index=1,
        prompt_text="prompt",
        conversation_history=("a",),
        hidden_states={"features": [[0.1, 1.3], [0.8, 0.4]]},
    )
    second = deterministic.extract_layer_trajectory(
        conversation_id="conv",
        turn_index=1,
        prompt_text="prompt",
        conversation_history=("a",),
        hidden_states={"features": [[0.1, 1.3], [0.8, 0.4]]},
    )
    transcript = TranscriptOnlyTrajectoryAdapter().extract_layer_trajectory(
        conversation_id="conv",
        turn_index=1,
        prompt_text="COMPONENT_A",
        conversation_history=("PARAMETER_B",),
    )

    assert first == second
    assert first.normalized_layer_features[0][1] == 1.0
    assert transcript.transcript_fallback_used is True


def test_closure_evaluation_uses_configured_threshold() -> None:
    assessment = KVContextRiskScorer().assess(
        latest_record=_record("r1", "placeholder"),
        conversation=MultiTurnConversation("conv", (_record("r1", "placeholder"),)),
        snapshot=TranscriptFallbackAdapter().extract_snapshot(
            conversation_id="conv",
            turn_index=0,
            prompt_text="placeholder",
            conversation_history=("placeholder",),
        ),
        config=KVContextSafetyConfig(enabled=True),
    )
    assessment = replace(
        assessment,
        conversation_id="closure",
        closure_risk=0.65,
    )

    score = SemanticRobustnessEvaluator().evaluate_candidate_response_closure(
        (assessment,),
        expected_closure_conversation_ids=("closure",),
        closure_risk_threshold=0.60,
    )

    assert score == 1.0
