# Standard library
from __future__ import annotations

# Third-party
import pytest

# Local
from semantic_intent_robustness.memory_safety import (
    MemoryDurability,
    MemoryRetrievalDecision,
    MemorySourceType,
    MemoryTrustLevel,
    MemoryWriteDecision,
    MemoryWriteRequest,
    RepresentationMemoryProvenance,
    RetrievedMemory,
    aggregate_memory_mediated_laundering,
    assess_memory_write,
    assess_retrieved_memory,
)
from semantic_intent_robustness.modules import SemanticIntentPipeline
from semantic_intent_robustness.representation import (
    RepresentationCandidate,
    RepresentationChannel,
    SourceSpan,
    candidate_id_for,
    source_digest_for,
)
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


def _representation_provenance() -> RepresentationMemoryProvenance:
    candidate = RepresentationCandidate(
        source_span=SourceSpan("source-user-1", 0, 3, "teh"),
        candidate_text="the",
        transform_channel=RepresentationChannel.ORTHOGRAPHIC,
        orthographic_score=0.9,
        phonetic_score=0.0,
        contextual_score=0.5,
        semantic_similarity=0.5,
        confidence=0.9,
        provenance=("orthographic-lexicon:the",),
        generation_reason="Retain a bounded edit-distance hypothesis for evaluation.",
    )
    return RepresentationMemoryProvenance(
        candidate=candidate,
        candidate_id=candidate_id_for(candidate),
        source_identity="source-user-1",
        source_document="teh",
        source_digest=source_digest_for("source-user-1", "teh"),
        transform_provenance=("orthographic-lexicon:the",),
        derived_content="the",
        assessed_content="the",
        assessed_as_derived=True,
    )


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


def test_declared_representation_memory_requires_complete_provenance() -> None:
    assessment = assess_memory_write(
        _write(
            content_summary="the",
            source_identity="source-user-1",
            representation_derived=True,
        )
    )

    assert assessment.decision is MemoryWriteDecision.QUARANTINE
    assert "representation_provenance_missing" in assessment.reasons


def test_valid_representation_write_retains_structured_visible_provenance() -> None:
    provenance = _representation_provenance()
    request = _write(
        content_summary="the",
        source_identity="source-user-1",
        trust_level=MemoryTrustLevel.REVIEWED,
        requested_durability=MemoryDurability.DURABLE,
        representation_derived=True,
        representation_provenance=provenance,
    )

    assert assess_memory_write(request).decision is MemoryWriteDecision.ALLOW_DURABLE
    serialized = request.to_dict()
    assert serialized["representation_derived"] is True
    assert serialized["representation_provenance"]["candidate_id"] == provenance.candidate_id
    assert serialized["representation_provenance"]["transform_provenance"] == [
        "orthographic-lexicon:the"
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_identity": "different-source"},
        {"content_summary": "unbound summary"},
        {"provenance_retained": False},
        {"representation_derived": False},
    ],
)
def test_mismatched_representation_write_is_quarantined(overrides: dict[str, object]) -> None:
    payload = {
        "content_summary": "the",
        "source_identity": "source-user-1",
        "representation_derived": True,
        "representation_provenance": _representation_provenance(),
    }
    payload.update(overrides)
    request = _write(**payload)

    assessment = assess_memory_write(request)
    assert assessment.decision is MemoryWriteDecision.QUARANTINE
    assert any(reason.startswith("representation_") for reason in assessment.reasons)


def test_representation_declaration_requires_exact_boolean() -> None:
    with pytest.raises(TypeError, match="representation_derived must be an exact bool"):
        _write(representation_derived=1)


def test_memory_boundary_revalidates_a_corrupted_representation_boolean() -> None:
    request = _write(
        content_summary="the",
        source_identity="source-user-1",
        representation_derived=True,
        representation_provenance=_representation_provenance(),
    )
    object.__setattr__(request, "representation_derived", 1)

    assessment = assess_memory_write(request)
    assert assessment.decision is MemoryWriteDecision.QUARANTINE
    assert "representation_declaration_invalid" in assessment.reasons


def test_retrieved_representation_memory_stays_visibly_labeled() -> None:
    provenance = _representation_provenance()
    memory = _retrieval(
        content_summary="the",
        source_identity="source-user-1",
        trust_level=MemoryTrustLevel.REVIEWED,
        representation_derived=True,
        representation_provenance=provenance,
    )

    assessment = assess_retrieved_memory(memory)
    assert assessment.decision is MemoryRetrievalDecision.USE_WITH_PROVENANCE
    assert assessment.preserve_provenance_label is True
    assert memory.to_dict()["representation_provenance"]["assessed_as_derived"] is True


def test_representation_provenance_survives_write_to_retrieval_round_trip() -> None:
    provenance = _representation_provenance()
    write = _write(
        content_summary="the",
        source_identity="source-user-1",
        representation_derived=True,
        representation_provenance=provenance,
    )
    retrieval = _retrieval(
        content_summary="the",
        source_identity="source-user-1",
        representation_derived=True,
        representation_provenance=provenance,
    )

    assert (
        write.to_dict()["representation_provenance"]
        == retrieval.to_dict()["representation_provenance"]
    )


def test_retrieved_representation_memory_with_mismatched_identity_is_quarantined() -> None:
    assessment = assess_retrieved_memory(
        _retrieval(
            content_summary="the",
            source_identity="different-source",
            representation_derived=True,
            representation_provenance=_representation_provenance(),
        )
    )

    assert assessment.decision is MemoryRetrievalDecision.QUARANTINE
    assert "representation_source_identity_mismatch" in assessment.reasons


def test_retrieved_declared_representation_without_provenance_is_quarantined() -> None:
    assessment = assess_retrieved_memory(
        _retrieval(
            content_summary="the",
            source_identity="source-user-1",
            representation_derived=True,
        )
    )

    assert assessment.decision is MemoryRetrievalDecision.QUARANTINE
    assert "representation_provenance_missing" in assessment.reasons


def test_inactive_representation_defaults_do_not_change_legacy_serialization() -> None:
    assert "representation_derived" not in _write().to_dict()
    assert "representation_provenance" not in _write().to_dict()
    assert "representation_derived" not in _retrieval().to_dict()
    assert "representation_provenance" not in _retrieval().to_dict()
    assert "source_identity" not in _retrieval().to_dict()


def test_representation_provenance_revalidates_correlated_candidate_mutation() -> None:
    provenance = _representation_provenance()
    request = _write(
        content_summary="the",
        source_identity="source-user-1",
        representation_derived=True,
        representation_provenance=provenance,
    )
    object.__setattr__(provenance.candidate, "provenance", ["spoofed"])

    assessment = assess_memory_write(request)
    assert assessment.decision is MemoryWriteDecision.QUARANTINE
    assert "representation_provenance_invalid" in assessment.reasons
