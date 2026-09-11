"""Tests for semantic-hinge detection and representation disagreement routing."""

# Standard library
from dataclasses import FrozenInstanceError

# Third-party
import pytest

# Local
from semantic_intent_robustness.modules import (
    DecomposeIntentModule,
    RepresentationRobustnessModule,
    SemanticIntentPipeline,
)
from semantic_intent_robustness.representation import (
    RepresentationCandidate,
    RepresentationChannel,
    SourceSpan,
)
from semantic_intent_robustness.representation_routing import (
    RepresentationDecision,
    candidate_id_for,
    locate_semantic_hinges,
    route_representation_disagreement,
    source_digest_for,
)
from semantic_intent_robustness.schemas import SemanticSafetyRecord
from semantic_intent_robustness.taxonomy import PolicyAction, VariantType


def _record(
    candidate_id: str,
    policy_action: PolicyAction,
    *,
    source_id: str = "source-1",
    source_start: int = 0,
    source_end: int = 12,
    raw_text: str = "send it now!",
    source_document: str = "send it now!",
    provenance: tuple[str, ...] = ("literal-source",),
    semantic_cluster_id: str = "representation-cluster",
) -> SemanticSafetyRecord:
    return SemanticSafetyRecord(
        prompt_id=f"assessment-{candidate_id}",
        prompt_text=raw_text,
        semantic_cluster_id=semantic_cluster_id,
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        policy_action=policy_action,
        representation_candidate_id=candidate_id,
        representation_source_id=source_id,
        representation_source_start=source_start,
        representation_source_end=source_end,
        representation_raw_text=raw_text,
        representation_source_document=source_document,
        representation_source_digest=source_digest_for(source_id, source_document),
        representation_provenance=provenance,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("Delete the draft.", "Delete"),
        ("Send the report to the server.", "server"),
        ("Do not transfer it.", "Do not"),
        ("Transfer 250.50 credits.", "250.50"),
        ("Email Alice before noon.", "Alice"),
        ("Only proceed with explicit authorization.", "authorization"),
        ("Can this account execute the command?", "execute"),
        ("Use at most 3 retries.", "at most"),
    ),
)
def test_locate_semantic_hinges_retains_exact_source_slices(text: str, expected: str) -> None:
    spans = locate_semantic_hinges(text)

    assert expected in {span.raw_text for span in spans}
    assert all(text[span.start : span.end] == span.raw_text for span in spans)
    assert all(span.source_id == "semantic-hinges" for span in spans)


def test_semantic_hinges_are_descriptive_spans_without_harm_labels() -> None:
    spans = locate_semantic_hinges("Alice may delete 2 drafts with authorization.")

    assert spans
    assert all(not hasattr(span, "harmful") for span in spans)


def test_semantic_hinges_are_deterministic_nonoverlapping_spans() -> None:
    text = "Alice must not delete 2 production servers."

    first = locate_semantic_hinges(text)
    second = locate_semantic_hinges(text)

    assert first == second
    assert tuple((span.start, span.end) for span in first) == tuple(
        sorted((span.start, span.end) for span in first)
    )
    assert all(left.end <= right.start for left, right in zip(first, first[1:]))


def test_candidate_id_is_stable_and_sensitive_to_provenance() -> None:
    span = SourceSpan("source-1", 0, 4, "send")
    first = RepresentationCandidate(
        source_span=span,
        candidate_text="send",
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=("literal-source",),
        generation_reason="Preserve the literal source.",
    )
    changed = RepresentationCandidate(
        source_span=span,
        candidate_text="send",
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=("literal-source", "independent-observation"),
        generation_reason="Preserve the literal source.",
    )

    assert candidate_id_for(first) == candidate_id_for(first)
    assert candidate_id_for(first).startswith("representation-v1:")
    assert candidate_id_for(first) != candidate_id_for(changed)


def test_source_digest_is_stable_and_sensitive_to_document_identity() -> None:
    assert source_digest_for("source-1", "send it now!") == source_digest_for(
        "source-1", "send it now!"
    )
    assert source_digest_for("source-1", "send it now!").startswith("representation-source-v1:")
    assert source_digest_for("source-1", "send it now!") != source_digest_for(
        "source-1", "send it later!"
    )


def test_policy_agreement_preserves_the_common_policy() -> None:
    decision = route_representation_disagreement(
        (
            _record("representation-v1:literal", PolicyAction.ALLOW),
            _record(
                "representation-v1:phonetic",
                PolicyAction.ALLOW,
                provenance=("phonetic-lexicon:send",),
            ),
        ),
        high_stakes=False,
    )

    assert decision == RepresentationDecision(
        selected_candidate_ids=(
            "representation-v1:literal",
            "representation-v1:phonetic",
        ),
        disagreement=False,
        policy_action=PolicyAction.ALLOW,
        explanation="Candidate assessments agree; preserve their common policy action.",
    )


def test_routing_accepts_consistent_full_source_and_candidate_subspan() -> None:
    decision = route_representation_disagreement(
        (
            _record("representation-v1:literal", PolicyAction.ALLOW),
            _record(
                "representation-v1:word",
                PolicyAction.ALLOW,
                source_start=0,
                source_end=4,
                raw_text="send",
                provenance=("orthographic-lexicon:send",),
            ),
        ),
        high_stakes=False,
    )

    assert decision.disagreement is False
    assert decision.policy_action is PolicyAction.ALLOW


def test_low_stakes_disagreement_routes_to_bounded_caution() -> None:
    decision = route_representation_disagreement(
        (
            _record("representation-v1:literal", PolicyAction.ALLOW),
            _record(
                "representation-v1:phonetic",
                PolicyAction.REFUSE,
                provenance=("phonetic-lexicon:send",),
            ),
        ),
        high_stakes=False,
    )

    assert decision.disagreement is True
    assert decision.policy_action is PolicyAction.ALLOW_WITH_BOUNDARIES
    assert "hypotheses" in decision.explanation
    assert "verified" not in decision.explanation.casefold()


def test_high_stakes_disagreement_routes_to_abstention() -> None:
    decision = route_representation_disagreement(
        (
            _record("representation-v1:literal", PolicyAction.ALLOW),
            _record(
                "representation-v1:phonetic",
                PolicyAction.REFUSE,
                provenance=("phonetic-lexicon:send",),
            ),
        ),
        high_stakes=True,
    )

    assert decision.disagreement is True
    assert decision.policy_action is PolicyAction.ABSTAIN
    assert "clarification" in decision.explanation
    assert "hypotheses" in decision.explanation


def test_representation_decision_is_frozen() -> None:
    decision = route_representation_disagreement(
        (_record("representation-v1:literal", PolicyAction.ALLOW),),
        high_stakes=False,
    )

    with pytest.raises(FrozenInstanceError):
        decision.disagreement = True  # type: ignore[misc]


def test_representation_decision_rejects_noncanonical_candidate_ids() -> None:
    with pytest.raises(ValueError, match="representation-v1 namespace"):
        RepresentationDecision(
            selected_candidate_ids=("outside-namespace",),
            disagreement=False,
            policy_action=PolicyAction.ALLOW,
            explanation="Preserve the common action.",
        )


@pytest.mark.parametrize("assessments", ((), [], "not-a-sequence"))
def test_routing_rejects_empty_or_malformed_assessment_sets(assessments: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        route_representation_disagreement(assessments, high_stakes=False)  # type: ignore[arg-type]


def test_routing_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate representation candidate ID"):
        route_representation_disagreement(
            (
                _record("representation-v1:same", PolicyAction.ALLOW),
                _record("representation-v1:same", PolicyAction.REFUSE),
            ),
            high_stakes=False,
        )


@pytest.mark.parametrize(
    "changed",
    (
        {"source_id": "source-2"},
        {"source_start": 1, "source_end": 13},
        {"raw_text": "SEND IT NOW!"},
    ),
)
def test_routing_rejects_inconsistent_raw_source_spans(changed: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="immutable source"):
        route_representation_disagreement(
            (
                _record("representation-v1:literal", PolicyAction.ALLOW),
                _record(
                    "representation-v1:other",
                    PolicyAction.REFUSE,
                    **changed,
                ),
            ),
            high_stakes=False,
        )


def test_routing_rejects_disjoint_spans_from_colliding_source_ids() -> None:
    with pytest.raises(ValueError, match="same immutable source document"):
        route_representation_disagreement(
            (
                _record(
                    "representation-v1:left",
                    PolicyAction.ALLOW,
                    source_start=0,
                    source_end=4,
                    raw_text="send",
                    source_document="send .......",
                ),
                _record(
                    "representation-v1:right",
                    PolicyAction.REFUSE,
                    source_start=7,
                    source_end=12,
                    raw_text="later",
                    source_document="xxxx...later",
                ),
            ),
            high_stakes=False,
        )


def test_routing_rejects_mixed_provenance_presence() -> None:
    unbound = SemanticSafetyRecord(
        prompt_id="unbound",
        prompt_text="send it now!",
        semantic_cluster_id="representation-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        policy_action=PolicyAction.ALLOW,
    )

    with pytest.raises(ValueError, match="complete representation provenance"):
        route_representation_disagreement(
            (unbound, _record("representation-v1:literal", PolicyAction.ALLOW)),
            high_stakes=False,
        )


def test_routing_rejects_assessments_from_mixed_semantic_contexts() -> None:
    with pytest.raises(ValueError, match="same semantic assessment context"):
        route_representation_disagreement(
            (
                _record("representation-v1:first", PolicyAction.ALLOW),
                _record(
                    "representation-v1:second",
                    PolicyAction.REFUSE,
                    semantic_cluster_id="different-cluster",
                ),
            ),
            high_stakes=False,
        )


def test_routing_rejects_nonexact_stakes_flag() -> None:
    with pytest.raises(TypeError, match="high_stakes must be an exact bool"):
        route_representation_disagreement(
            (_record("representation-v1:literal", PolicyAction.ALLOW),),
            high_stakes=1,  # type: ignore[arg-type]
        )


def test_representation_fields_snapshot_and_round_trip_exactly() -> None:
    provenance = ["literal-source", "unicode-normalization:NFC"]
    record = SemanticSafetyRecord(
        prompt_id="round-trip",
        prompt_text="send it now!",
        semantic_cluster_id="representation-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        representation_candidate_id="representation-v1:literal",
        representation_source_id="source-1",
        representation_source_start=0,
        representation_source_end=12,
        representation_raw_text="send it now!",
        representation_source_document="send it now!",
        representation_source_digest=source_digest_for("source-1", "send it now!"),
        representation_provenance=provenance,  # type: ignore[arg-type]
        representation_disagreement=True,
    )
    provenance.append("caller-mutation")

    payload = record.to_dict()
    restored = SemanticSafetyRecord.from_dict(payload)

    assert record.representation_provenance == (
        "literal-source",
        "unicode-normalization:NFC",
    )
    assert payload["representation_provenance"] == (
        "literal-source",
        "unicode-normalization:NFC",
    )
    assert payload["representation_disagreement"] is True
    assert restored == record


def test_disagreement_flag_without_representation_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete representation provenance"):
        SemanticSafetyRecord(
            prompt_id="unbound-disagreement",
            prompt_text="send it now!",
            semantic_cluster_id="representation-cluster",
            parent_example_id=None,
            variant_type=VariantType.ORIGINAL,
            language="en",
            representation_disagreement=True,
        )


def test_legacy_serialization_omits_inactive_representation_defaults() -> None:
    record = SemanticSafetyRecord(
        prompt_id="legacy-serialization",
        prompt_text="Summarize this public report.",
        semantic_cluster_id="legacy-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
    )

    payload = record.to_dict()

    assert not any(key.startswith("representation_") for key in payload)
    assert SemanticSafetyRecord.from_dict(payload) == record


def test_legacy_record_defaults_do_not_change_pipeline_behavior() -> None:
    record = SemanticSafetyRecord(
        prompt_id="legacy",
        prompt_text="Summarize this public report.",
        semantic_cluster_id="legacy-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        policy_action=PolicyAction.ALLOW,
    )

    result = SemanticIntentPipeline().run(record)

    assert result.decomposition is record
    assert result.policy_decision["policy_action"] == PolicyAction.ALLOW.value
    assert record.representation_candidate_id is None
    assert record.representation_provenance == ()
    assert record.representation_disagreement is False


def test_pipeline_runs_representation_module_before_semantic_decomposition() -> None:
    calls: list[str] = []
    record = _record("representation-v1:literal", PolicyAction.ALLOW)
    pipeline = SemanticIntentPipeline()

    class TrackingRepresentation(RepresentationRobustnessModule):
        def __call__(self, value: SemanticSafetyRecord) -> SemanticSafetyRecord:
            calls.append("representation")
            return super().__call__(value)

    class TrackingDecomposition(DecomposeIntentModule):
        def __call__(
            self, value: SemanticSafetyRecord, _conversation_context: str = ""
        ) -> SemanticSafetyRecord:
            calls.append("decomposition")
            return super().__call__(value, _conversation_context)

    pipeline.representation = TrackingRepresentation()
    pipeline.decompose = TrackingDecomposition()

    result = pipeline.run(record)

    assert result.decomposition is record
    assert calls[:2] == ["representation", "decomposition"]


def test_schema_rejects_partial_representation_provenance() -> None:
    with pytest.raises(ValueError, match="complete representation provenance"):
        SemanticSafetyRecord(
            prompt_id="partial",
            prompt_text="send it now!",
            semantic_cluster_id="representation-cluster",
            parent_example_id=None,
            variant_type=VariantType.ORIGINAL,
            language="en",
            representation_candidate_id="representation-v1:literal",
        )
