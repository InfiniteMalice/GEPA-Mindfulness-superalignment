"""Matched public-evidence controls for the SoT-inspired continuity audit."""

from dataclasses import replace
from importlib import import_module
from typing import Any

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from mindful_trace_gepa.logging_schema import EventEnvelope
from semantic_intent_robustness.epistemic_continuity import EpistemicContinuityAssessment
from semantic_intent_robustness.epistemic_records import CommitmentUpdate, EpistemicCommitment
from semantic_intent_robustness.memory_safety import RetrievedMemory


def event_sequence() -> tuple[EventEnvelope, ...]:
    """Two completed actions and a third proposal, with public source references."""
    events = []
    for turn in range(3):
        key = str(turn)
        base = dict(
            schema_version="1.0",
            timestamp=f"2026-09-21T12:00:0{turn}Z",
            run_id="run",
            repeat_id=0,
            conversation_id="c",
            checkpoint_step=turn,
            model_version="model",
            harness_version="harness",
        )
        pred = EventEnvelope(
            event_id=f"prediction-{key}",
            event_type="prediction_commit",
            evidence_refs=(f"evidence-{key}",),
            payload=dict(
                prediction_commit_id=f"p{key}",
                predicted_outcome="bounded",
                confidence=0.5,
                evidence_refs=[f"evidence-{key}"],
            ),
            **base,
        )
        proposed = EventEnvelope(
            event_id=f"proposed-{key}",
            event_type="action_proposed",
            parent_event_ids=(pred.event_id,),
            action_id=f"action-{key}",
            authorization_scope="offline",
            payload=dict(
                action_id=f"action-{key}",
                action_class="review",
                reversible=True,
                authorization_scope="offline",
                prediction_commit_id=f"p{key}",
            ),
            **base,
        )
        events.extend((pred, proposed))
        if turn == 2:
            break
        executed = replace(
            proposed,
            event_id=f"executed-{key}",
            event_type="action_executed",
            parent_event_ids=(proposed.event_id,),
        )
        outcome = EventEnvelope(
            event_id=f"outcome-{key}",
            event_type="outcome_observed",
            parent_event_ids=(executed.event_id,),
            action_id=f"action-{key}",
            evidence_refs=(f"evidence-{key}",),
            payload=dict(
                observation_id=f"o{key}",
                action_id=f"action-{key}",
                actual_outcome="checked",
                evidence_refs=[f"evidence-{key}"],
            ),
            **base,
        )
        verification = EventEnvelope(
            event_id=f"verification-{key}",
            event_type="verification_result",
            parent_event_ids=(outcome.event_id,),
            verifier_refs=(f"verifier-{key}",),
            payload=dict(
                verifier_id=f"verifier-{key}",
                verifier_version="1",
                observation_id=f"o{key}",
                verified=True,
                verifier_refs=[f"verifier-{key}"],
            ),
            **base,
        )
        events.extend((executed, outcome, verification))
    return tuple(events)


def commitment() -> EpistemicCommitment:
    """One public commitment that earlier observable evidence blocks an action."""
    records = import_module("semantic_intent_robustness.epistemic_records")
    memory = RetrievedMemory(
        memory_id="k",
        content_summary="Earlier verification blocks action X.",
        source_type="tool_output",
        trust_level="reviewed",
        source_identity="verifier",
        provenance_retained=True,
        recalled_as_instruction=False,
        used_for_tool_selection=False,
        changes_goal_or_policy=False,
        changes_identity_or_authority=False,
        attempts_protected_override=False,
        conflicts_with_current_context=False,
        delayed_activation_hint=False,
        capability_transfer_risk="low",
    )
    return records.EpistemicCommitment(
        commitment_id="k",
        conversation_id="c",
        evaluation_unit_id="run",
        repeat_id=0,
        claim_summary=memory.content_summary,
        memory=memory,
        evidence_refs=(EvidenceReference("verifier-0", EvidenceSourceKind.EXTERNAL_RECORD),),
        source_event_refs=("verification-0",),
        first_active_at=0,
        last_active_at=0,
        confidence=0.8,
        decision_relevance=True,
        provenance=("public-audit",),
        state_snapshot_id="s0",
    )


def assess(
    *,
    active: tuple[str, ...] = (),
    updates: tuple[CommitmentUpdate, ...] = (),
    commitments: tuple[EpistemicCommitment, ...] | None = None,
    **kwargs: Any,
) -> EpistemicContinuityAssessment:
    """Invoke the real action-bound audit with explicit current effective context."""
    module = import_module("semantic_intent_robustness.epistemic_continuity")
    return module.assess_epistemic_continuity(
        assessment_id="audit",
        commitments=(commitment(),) if commitments is None else commitments,
        events=event_sequence(),
        decision_event_id="proposed-2",
        active_commitment_ids=active,
        updates=updates,
        provenance=("public-audit",),
        **kwargs,
    )


def test_active_commitment_retained_across_turns() -> None:
    """Retained prior evidence is consistent even as actions advance."""
    result = assess(active=("k",))
    assert result.continuity_status == "consistent"
    assert result.unexplained_omission_ids == ()


def test_no_new_evidence_flags_unexplained_omission() -> None:
    """A current proposal cannot silently discard a prior decision-relevant constraint."""
    result = assess()
    assert result.continuity_status == "unexplained_omission"
    assert result.unexplained_omission_ids == ("k",)
    assert result.review_required


@pytest.mark.parametrize(
    "status, expected",
    [
        ("contradicted", "legitimate_update"),
        ("scoped_out", "legitimate_scope_change"),
    ],
)
def test_supported_updates_are_matched_negative_controls(status: str, expected: str) -> None:
    """Explicit later verified contradiction or scope evidence permits legitimate inactivity."""
    records = import_module("semantic_intent_robustness.epistemic_records")
    update = records.CommitmentUpdate(
        commitment_id="k",
        status=records.CommitmentStatus(status),
        status_change_reason="New verified evidence changes the premise or task scope.",
        evidence_refs=(EvidenceReference("verifier-1", EvidenceSourceKind.EXTERNAL_RECORD),),
        source_event_refs=("verification-1",),
        provenance=("reviewed-update",),
    )
    result = assess(updates=(update,), decision_context_changed=status == "scoped_out")
    assert result.continuity_status == expected
    assert not result.unexplained_omission_ids


def test_missing_provenance_is_reviewed_not_silently_reactivated() -> None:
    """Missing memory provenance cannot be upgraded by remembering the same content."""
    item = commitment()
    bad = replace(item, memory=replace(item.memory, provenance_retained=False))
    result = assess(commitments=(bad,))
    assert result.continuity_status == "insufficient_evidence"
    assert result.quarantined_ids == ("k",)
    assert result.review_required


def test_reactivation_preserves_public_evidence_and_source_trust() -> None:
    """Recall preserves original memory, evidence and omission history."""
    module = import_module("semantic_intent_robustness.epistemic_continuity")
    item = commitment()
    result = module.recall_historical_support(
        commitments=(item,),
        assessment=assess(),
        current_state=None,
        prior_states=(),
    )
    assert result.reactivated_ids == ("k",)
    assert result.commitments[0].evidence_refs == item.evidence_refs
    assert result.commitments[0].memory.source_identity == "verifier"
    assert result.memory_assessments[0].decision.value == "use_with_provenance"


def test_cross_unit_or_future_evidence_cannot_support_commitment() -> None:
    """Evidence from another run or future turn is quarantined before recall."""
    item = commitment()
    for bad in (
        replace(item, evaluation_unit_id="other"),
        replace(item, source_event_refs=("verification-1",)),
    ):
        result = assess(commitments=(bad,))
        assert result.quarantined_ids == ("k",)


def test_private_reasoning_reference_rejected() -> None:
    """Commitments accept public evidence references only, never hidden reasoning IDs."""
    with pytest.raises(ValueError, match="observable"):
        replace(
            commitment(),
            evidence_refs=(EvidenceReference("hidden", EvidenceSourceKind.PRIVATE_REASONING),),
        )


def test_circular_supersession_cannot_remove_all_relevant_evidence() -> None:
    """Two simultaneous replacements cannot justify each other's disappearance."""
    records = import_module("semantic_intent_robustness.epistemic_records")
    item = commitment()
    second = replace(item, commitment_id="b", memory=replace(item.memory, memory_id="b"))
    updates = tuple(
        records.CommitmentUpdate(
            commitment_id=key,
            status=records.CommitmentStatus.SUPERSEDED,
            status_change_reason="Replacement claimed.",
            evidence_refs=(EvidenceReference("verifier-1", EvidenceSourceKind.EXTERNAL_RECORD),),
            source_event_refs=("verification-1",),
            provenance=("review",),
            superseded_by=target,
        )
        for key, target in (("k", "b"), ("b", "k"))
    )
    result = assess(commitments=(item, second), updates=updates)
    assert result.continuity_status == "contradictory_state"
    assert result.unexplained_omission_ids == ("k", "b")


def test_same_id_changed_state_cannot_rank_recall() -> None:
    """Future or altered features cannot masquerade as the assessed current snapshot."""
    from test_sot_state_continuity import snapshot

    from semantic_intent_robustness.epistemic_continuity import recall_historical_support

    current = snapshot(2)
    with pytest.raises(ValueError, match="state"):
        recall_historical_support(
            commitments=(commitment(),),
            assessment=assess(current_state=current),
            current_state=replace(current, turn_index=999, local_organization=1),
            prior_states=(snapshot(),),
        )


def typed_verification(
    turn: int,
    reference: EvidenceReference,
    contradicted: bool,
) -> EventEnvelope:
    """A real repository relational verifier result with captured evidence kind."""
    from gepa_mindfulness.verification.interfaces import (
        RelationalVerificationResult,
        VerificationEvidenceBinding,
        make_relational_verification_event,
    )

    result = RelationalVerificationResult(
        action_id=f"action-{turn}",
        task_fit=False,
        dependencies_satisfied=False,
        contradiction_status="contradicted" if contradicted else "unknown",
        provenance_intact=False,
        authorization_scope_valid=False,
        claimed_outcome_supported=False,
        repeated_failed_route=False,
        evidence_refs=(reference,),
        evidence_bindings=(
            (VerificationEvidenceBinding("contradiction_status", (reference,)),)
            if contradicted
            else ()
        ),
    )
    return make_relational_verification_event(
        result,
        verifier_refs=(f"typed-{turn}",),
        event_id=f"verification-{turn}",
        parent_event_ids=(f"outcome-{turn}",),
        run_id="run",
        repeat_id=0,
        conversation_id="c",
        checkpoint_step=turn,
        model_version="model",
        harness_version="harness",
    )


def test_captured_latent_evidence_cannot_be_relabelled_observable() -> None:
    """Typed verifier provenance must survive the continuity trust boundary."""
    from semantic_intent_robustness.epistemic_continuity import assess_epistemic_continuity

    event = typed_verification(
        0,
        EvidenceReference("verifier-0", EvidenceSourceKind.LATENT_STATE),
        False,
    )
    events = tuple(event if e.event_id == event.event_id else e for e in event_sequence())
    result = assess_epistemic_continuity(
        assessment_id="a",
        commitments=(commitment(),),
        events=events,
        decision_event_id="proposed-2",
        active_commitment_ids=("k",),
        provenance=("review",),
    )
    assert result.quarantined_ids == ("k",)


def test_typed_relational_contradiction_supports_legitimate_update() -> None:
    """The existing typed verifier interface must work beside legacy verified booleans."""
    from semantic_intent_robustness.epistemic_continuity import assess_epistemic_continuity
    from semantic_intent_robustness.epistemic_records import CommitmentStatus, CommitmentUpdate

    ref = EvidenceReference("new-contradiction", EvidenceSourceKind.EXTERNAL_RECORD)
    event = typed_verification(1, ref, True)
    events = tuple(event if e.event_id == event.event_id else e for e in event_sequence())
    update = CommitmentUpdate(
        "k",
        CommitmentStatus.CONTRADICTED,
        "New contradiction.",
        (ref,),
        (event.event_id,),
        ("review",),
    )
    result = assess_epistemic_continuity(
        assessment_id="a",
        commitments=(commitment(),),
        events=events,
        decision_event_id="proposed-2",
        active_commitment_ids=(),
        updates=(update,),
        provenance=("review",),
    )
    assert result.continuity_status == "legitimate_update"


def test_old_reference_id_with_new_kind_is_not_new_evidence() -> None:
    """Relabeling the original identifier cannot justify a later contradiction."""
    from semantic_intent_robustness.epistemic_continuity import assess_epistemic_continuity
    from semantic_intent_robustness.epistemic_records import CommitmentStatus, CommitmentUpdate

    ref = EvidenceReference("verifier-0", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    event = typed_verification(1, ref, True)
    events = tuple(event if e.event_id == event.event_id else e for e in event_sequence())
    update = CommitmentUpdate(
        "k",
        CommitmentStatus.CONTRADICTED,
        "Claimed new evidence.",
        (ref,),
        (event.event_id,),
        ("review",),
    )
    result = assess_epistemic_continuity(
        assessment_id="a",
        commitments=(commitment(),),
        events=events,
        decision_event_id="proposed-2",
        active_commitment_ids=(),
        updates=(update,),
        provenance=("review",),
    )
    assert result.review_required
    assert result.continuity_status != "legitimate_update"


@pytest.mark.parametrize(
    "override",
    [
        {"attempts_protected_override": True},
        {"trust_level": "untrusted", "recalled_as_instruction": True},
        {"trust_level": "untrusted", "used_for_tool_selection": True},
        {"changes_identity_or_authority": True},
    ],
)
def test_recall_cannot_bypass_original_memory_boundary(override) -> None:
    """State similarity cannot confer authority on quarantined memory."""
    module = import_module("semantic_intent_robustness.epistemic_continuity")
    item = commitment()
    item = replace(item, memory=replace(item.memory, **override))
    assessment = assess(commitments=(item,))
    support = module.recall_historical_support(
        commitments=(item,),
        assessment=assessment,
        current_state=None,
        prior_states=(),
    )
    assert assessment.quarantined_ids == ("k",)
    assert support.commitments == ()


def test_recall_budget_exposes_deferred_relevant_evidence() -> None:
    """A retrieval budget must not silently erase omitted evidence from the audit."""
    module = import_module("semantic_intent_robustness.epistemic_continuity")
    item = commitment()
    items = tuple(
        replace(item, commitment_id=key, memory=replace(item.memory, memory_id=key))
        for key in ("a", "b", "c")
    )
    assessment = assess(commitments=items)
    support = module.recall_historical_support(
        commitments=items,
        assessment=assessment,
        current_state=None,
        prior_states=(),
        max_items=1,
    )
    assert support.reactivated_ids == ("a",)
    assert support.deferred_ids == ("b", "c")
    assert assessment.unexplained_omission_ids == ("a", "b", "c")


def test_supported_supersession_keeps_replacement_active() -> None:
    """A later grounded replacement can retire an earlier commitment without losing both."""
    records = import_module("semantic_intent_robustness.epistemic_records")
    original = commitment()
    refs = (EvidenceReference("verifier-1", EvidenceSourceKind.EXTERNAL_RECORD),)
    replacement = replace(
        original,
        commitment_id="replacement",
        memory=replace(original.memory, memory_id="replacement"),
        evidence_refs=refs,
        source_event_refs=("verification-1",),
        first_active_at=1,
        last_active_at=1,
    )
    update = records.CommitmentUpdate(
        "k",
        records.CommitmentStatus.SUPERSEDED,
        "Later verified replacement.",
        refs,
        ("verification-1",),
        ("reviewed-update",),
        "replacement",
    )
    result = assess(commitments=(original, replacement), updates=(update,), active=("replacement",))
    assert result.continuity_status == "legitimate_update"
    assert result.explicitly_superseded_ids == ("k",)
    assert result.retained_ids == ("replacement",)
    assert result.unexplained_omission_ids == ()


@pytest.mark.parametrize("change", ["features", "remove", "add", "drop_current"])
def test_recall_rejects_changed_assessed_state_context(change: str) -> None:
    """Replaying an assessment cannot change the state-ranked winner under a budget."""
    from test_sot_state_continuity import snapshot

    from semantic_intent_robustness.epistemic_continuity import recall_historical_support

    item = commitment()
    items = (
        replace(
            item,
            commitment_id="a",
            memory=replace(item.memory, memory_id="a"),
            state_snapshot_id="prior-a",
        ),
        replace(
            item,
            commitment_id="b",
            memory=replace(item.memory, memory_id="b"),
            state_snapshot_id="prior-b",
        ),
    )
    prior = (
        replace(snapshot(value=0.9), snapshot_id="prior-a"),
        replace(snapshot(value=0.5), snapshot_id="prior-b"),
    )
    current = snapshot(2)
    assessment = assess(commitments=items, current_state=current, prior_states=prior)
    support = recall_historical_support(
        commitments=items,
        assessment=assessment,
        current_state=current,
        prior_states=tuple(reversed(prior)),
        max_items=1,
    )
    assert support.reactivated_ids == ("b",)
    assert support.deferred_ids == ("a",)
    if change == "features":
        prior = (snapshot(), prior[1])
        prior = (replace(prior[0], snapshot_id="prior-a"), prior[1])
    elif change == "remove":
        prior = prior[:1]
    elif change == "add":
        prior = (*prior, replace(snapshot(), snapshot_id="unassessed"))
    else:
        current = None
    with pytest.raises(ValueError, match="state"):
        recall_historical_support(
            commitments=items,
            assessment=assessment,
            current_state=current,
            prior_states=prior,
            max_items=1,
        )
