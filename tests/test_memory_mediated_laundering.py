from __future__ import annotations

from semantic_intent_robustness.memory_safety import (
    MemoryDurability,
    MemoryRetrievalDecision,
    MemorySourceType,
    MemoryTrustLevel,
    MemoryWriteDecision,
    MemoryWriteRequest,
    RetrievedMemory,
    aggregate_memory_mediated_laundering,
    assess_memory_write,
    assess_retrieved_memory,
)
from semantic_intent_robustness.modules import SemanticIntentPipeline
from semantic_intent_robustness.taxonomy import CapabilityTransferRisk


def _write(**overrides) -> MemoryWriteRequest:
    payload = {
        "memory_id": "mem-write",
        "content_summary": "User preference summary.",
        "source_type": MemorySourceType.USER_INPUT,
        "source_identity": "user",
        "trust_level": MemoryTrustLevel.UNVERIFIED,
        "requested_durability": MemoryDurability.SESSION,
        "capability_transfer_risk": CapabilityTransferRisk.LOW,
        "provenance_retained": True,
        "contains_instruction": False,
        "changes_goal_or_policy": False,
        "changes_identity_or_authority": False,
        "attempts_protected_override": False,
        "conflicts_with_existing_memory": False,
        "delayed_activation_hint": False,
        "cross_session_relevance": False,
        "tags": (),
    }
    payload.update(overrides)
    return MemoryWriteRequest(**payload)


def _retrieval(**overrides) -> RetrievedMemory:
    payload = {
        "memory_id": "mem-read",
        "content_summary": "A bounded factual note.",
        "source_type": MemorySourceType.USER_INPUT,
        "trust_level": MemoryTrustLevel.UNVERIFIED,
        "provenance_retained": True,
        "recalled_as_instruction": False,
        "used_for_tool_selection": False,
        "changes_goal_or_policy": False,
        "changes_identity_or_authority": False,
        "attempts_protected_override": False,
        "conflicts_with_current_context": False,
        "delayed_activation_hint": False,
        "capability_transfer_risk": CapabilityTransferRisk.LOW,
    }
    payload.update(overrides)
    return RetrievedMemory(**payload)


def test_untrusted_instruction_cannot_be_silently_promoted_to_durable_memory() -> None:
    assessment = assess_memory_write(
        _write(
            requested_durability=MemoryDurability.DURABLE,
            contains_instruction=True,
            trust_level=MemoryTrustLevel.UNTRUSTED,
        )
    )
    assert assessment.decision == MemoryWriteDecision.QUARANTINE


def test_reviewed_benign_preference_can_be_durably_stored() -> None:
    assessment = assess_memory_write(
        _write(
            memory_id="benign-pref",
            trust_level=MemoryTrustLevel.REVIEWED,
            requested_durability=MemoryDurability.DURABLE,
            cross_session_relevance=True,
        )
    )
    assert assessment.decision == MemoryWriteDecision.ALLOW_DURABLE


def test_protected_constitutional_override_is_rejected() -> None:
    assessment = assess_memory_write(_write(attempts_protected_override=True))
    assert assessment.decision == MemoryWriteDecision.REJECT


def test_untrusted_recalled_instruction_is_quarantined() -> None:
    assessment = assess_retrieved_memory(
        _retrieval(trust_level=MemoryTrustLevel.UNTRUSTED, recalled_as_instruction=True)
    )
    assert assessment.decision == MemoryRetrievalDecision.QUARANTINE


def test_untrusted_memory_cannot_bias_later_tool_selection() -> None:
    assessment = assess_retrieved_memory(
        _retrieval(trust_level=MemoryTrustLevel.UNTRUSTED, used_for_tool_selection=True)
    )
    assert "untrusted_memory_used_for_tool_selection" in assessment.reasons


def test_delayed_activation_memory_is_quarantined() -> None:
    assert assess_retrieved_memory(_retrieval(delayed_activation_hint=True)).decision == (
        MemoryRetrievalDecision.QUARANTINE
    )


def test_memory_without_provenance_is_quarantined() -> None:
    assert assess_memory_write(_write(provenance_retained=False)).decision == (
        MemoryWriteDecision.QUARANTINE
    )


def test_unverified_factual_memory_is_bounded_context_not_authority() -> None:
    assessment = assess_retrieved_memory(_retrieval())
    assert assessment.decision == MemoryRetrievalDecision.TREAT_AS_UNTRUSTED_CONTEXT


def test_pipeline_aggregate_matches_direct_helper() -> None:
    writes = [_write(memory_id="w1", provenance_retained=False)]
    retrievals = [_retrieval(memory_id="r1", recalled_as_instruction=True)]
    assert SemanticIntentPipeline().run_memory_boundary(writes, retrievals) == (
        aggregate_memory_mediated_laundering(writes, retrievals)
    )


def test_aggregate_reports_quarantined_and_rejected_ids() -> None:
    report = aggregate_memory_mediated_laundering(
        [
            _write(memory_id="qw", provenance_retained=False),
            _write(memory_id="rw", attempts_protected_override=True),
        ],
        [
            _retrieval(memory_id="qr", delayed_activation_hint=True),
            _retrieval(memory_id="rr", attempts_protected_override=True),
        ],
    )
    assert report.quarantined_write_ids == ("qw",)
    assert report.rejected_write_ids == ("rw",)
    assert report.quarantined_retrieval_ids == ("qr",)
    assert report.rejected_retrieval_ids == ("rr",)
    assert report.memory_mediated_laundering_detected is True
