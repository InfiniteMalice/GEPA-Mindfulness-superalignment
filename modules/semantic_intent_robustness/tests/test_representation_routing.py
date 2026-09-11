"""Tests for semantic-hinge detection and representation disagreement routing."""

# Standard library
import json
from dataclasses import FrozenInstanceError
from hashlib import sha256

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
from semantic_intent_robustness.transforms import build_variant


def _record(
    candidate_label: str,
    policy_action: PolicyAction,
    *,
    source_id: str = "source-1",
    source_start: int = 0,
    source_end: int = 12,
    raw_text: str = "send it now!",
    source_document: str = "send it now!",
    provenance: tuple[str, ...] = ("literal-source",),
    semantic_cluster_id: str = "representation-cluster",
    candidate_id_override: str | None = None,
) -> SemanticSafetyRecord:
    channel = (
        RepresentationChannel.PHONOLOGICAL
        if "phonetic" in candidate_label
        else RepresentationChannel.LITERAL
    )
    candidate = RepresentationCandidate(
        source_span=SourceSpan(source_id, source_start, source_end, raw_text),
        candidate_text=raw_text,
        transform_channel=channel,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=provenance,
        generation_reason=f"Assess candidate {candidate_label}.",
    )
    return SemanticSafetyRecord(
        prompt_id=f"assessment-{candidate_label}",
        prompt_text=(
            source_document[:source_start] + candidate.candidate_text + source_document[source_end:]
        ),
        semantic_cluster_id=semantic_cluster_id,
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        policy_action=policy_action,
        representation_candidate=candidate,
        representation_candidate_id=candidate_id_override or candidate_id_for(candidate),
        representation_source_id=source_id,
        representation_source_start=source_start,
        representation_source_end=source_end,
        representation_raw_text=raw_text,
        representation_source_document=source_document,
        representation_source_digest=source_digest_for(source_id, source_document),
        representation_provenance=provenance,
    )


def _unchecked_candidate_id(candidate: RepresentationCandidate) -> str:
    span = candidate.source_span
    payload = {
        "candidate_text": candidate.candidate_text,
        "confidence": candidate.confidence,
        "contextual_score": candidate.contextual_score,
        "generation_reason": candidate.generation_reason,
        "orthographic_score": candidate.orthographic_score,
        "outcome": candidate.outcome.value,
        "phonetic_score": candidate.phonetic_score,
        "provenance": list(candidate.provenance),
        "semantic_similarity": candidate.semantic_similarity,
        "source_end": span.end,
        "source_id": span.source_id,
        "source_raw_text": span.raw_text,
        "source_start": span.start,
        "transform_channel": candidate.transform_channel.value,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"representation-v1:{sha256(encoded.encode('utf-8')).hexdigest()}"


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
    literal = _record("literal", PolicyAction.ALLOW)
    phonetic = _record(
        "phonetic",
        PolicyAction.ALLOW,
        provenance=("phonetic-lexicon:send",),
    )
    decision = route_representation_disagreement(
        (literal, phonetic),
        high_stakes=False,
    )

    assert decision == RepresentationDecision(
        selected_candidate_ids=tuple(
            sorted(
                (
                    literal.representation_candidate_id,
                    phonetic.representation_candidate_id,
                )
            )
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
    candidate = RepresentationCandidate(
        source_span=SourceSpan("source-1", 0, 12, "send it now!"),
        candidate_text="send it now!",
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=tuple(provenance),
        generation_reason="Preserve the literal source.",
    )
    record = SemanticSafetyRecord(
        prompt_id="round-trip",
        prompt_text="send it now!",
        semantic_cluster_id="representation-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
        representation_candidate=candidate,
        representation_candidate_id=candidate_id_for(candidate),
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
    candidate_payload = payload["representation_candidate"]
    assert type(candidate_payload["transform_channel"]) is str
    assert type(candidate_payload["outcome"]) is str
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


def test_genuine_candidate_id_cannot_authorize_an_unrelated_assessed_prompt() -> None:
    span = SourceSpan("source-1", 0, 12, "send it now!")
    candidate = RepresentationCandidate(
        source_span=span,
        candidate_text="send it now!",
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=("literal-source",),
        generation_reason="Preserve the literal source.",
    )

    with pytest.raises(ValueError, match="actual assessed representation"):
        SemanticSafetyRecord(
            prompt_id="unrelated-prompt",
            prompt_text="an unrelated prompt",
            semantic_cluster_id="representation-cluster",
            parent_example_id=None,
            variant_type=VariantType.ORIGINAL,
            language="en",
            representation_candidate=candidate,
            representation_candidate_id=candidate_id_for(candidate),
            representation_source_id="source-1",
            representation_source_start=0,
            representation_source_end=12,
            representation_raw_text="send it now!",
            representation_source_document="send it now!",
            representation_source_digest=source_digest_for("source-1", "send it now!"),
            representation_provenance=("literal-source",),
        )


@pytest.mark.parametrize("candidate_id", ("representation-v1:fake-a", "representation-v1:fake-b"))
def test_schema_rejects_fabricated_candidate_ids(candidate_id: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        _record("fabricated", PolicyAction.ALLOW, candidate_id_override=candidate_id)


@pytest.mark.parametrize(
    "actions",
    (
        (PolicyAction.REFUSE, PolicyAction.ABSTAIN),
        (PolicyAction.REFUSE, PolicyAction.REDIRECT),
    ),
)
def test_low_stakes_all_nonpermissive_disagreement_stays_nonpermissive(
    actions: tuple[PolicyAction, PolicyAction],
) -> None:
    decision = route_representation_disagreement(
        (
            _record("representation-v1:first", actions[0]),
            _record("representation-v1:second", actions[1]),
        ),
        high_stakes=False,
    )

    assert decision.policy_action is PolicyAction.REFUSE


def test_semantic_safety_record_has_no_live_instance_dictionary() -> None:
    record = SemanticSafetyRecord(
        prompt_id="immutable-record",
        prompt_text="Summarize this.",
        semantic_cluster_id="legacy-cluster",
        parent_example_id=None,
        variant_type=VariantType.ORIGINAL,
        language="en",
    )

    with pytest.raises(AttributeError):
        record.__dict__["prompt_text"] = "mutated"


def test_routing_revalidates_the_assessed_prompt_at_the_trust_boundary() -> None:
    record = _record("literal", PolicyAction.ALLOW)
    object.__setattr__(record, "prompt_text", "mutated after construction")

    with pytest.raises(ValueError, match="actual assessed representation"):
        route_representation_disagreement((record,), high_stakes=False)


def test_unicode_hinges_preserve_names_grouped_numbers_and_emoji_offsets() -> None:
    text = "🔒 Email Élodie 1,000 reports to O’Connor at 1e3."

    spans = locate_semantic_hinges(text)
    observed = {span.raw_text for span in spans}

    assert {"Email", "Élodie", "1,000", "O’Connor", "1e3"} <= observed
    assert all(text[span.start : span.end] == span.raw_text for span in spans)
    assert next(span for span in spans if span.raw_text == "Email").start == 2


@pytest.mark.parametrize(
    "digest",
    (
        "representation-source-v1:" + ("A" * 64),
        "representation-source-v1:" + ("g" * 64),
    ),
)
def test_schema_rejects_non_lowercase_hex_source_digest(digest: str) -> None:
    candidate = RepresentationCandidate(
        source_span=SourceSpan("source-1", 0, 12, "send it now!"),
        candidate_text="send it now!",
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=("literal-source",),
        generation_reason="Preserve the literal source.",
    )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        SemanticSafetyRecord(
            prompt_id="bad-digest",
            prompt_text="send it now!",
            semantic_cluster_id="representation-cluster",
            parent_example_id=None,
            variant_type=VariantType.ORIGINAL,
            language="en",
            representation_candidate=candidate,
            representation_candidate_id=candidate_id_for(candidate),
            representation_source_id="source-1",
            representation_source_start=0,
            representation_source_end=12,
            representation_raw_text="send it now!",
            representation_source_document="send it now!",
            representation_source_digest=digest,
            representation_provenance=("literal-source",),
        )


def test_transform_does_not_copy_stale_representation_binding_to_new_text() -> None:
    seed = _record("literal", PolicyAction.ALLOW)

    variant = build_variant(
        seed,
        prompt_id="new-variant",
        prompt_text="A genuinely different prompt.",
        variant_type=VariantType.PARAPHRASE,
    )

    assert variant.representation_candidate is None
    assert variant.representation_candidate_id is None
    assert variant.representation_provenance == ()


@pytest.mark.parametrize("corrupted_score", (1, float("nan")))
def test_routing_rejects_correlated_hash_for_corrupted_candidate_score(
    corrupted_score: object,
) -> None:
    record = _record("literal", PolicyAction.ALLOW)
    candidate = record.representation_candidate
    assert candidate is not None
    object.__setattr__(candidate, "orthographic_score", corrupted_score)
    object.__setattr__(record, "representation_candidate_id", _unchecked_candidate_id(candidate))

    with pytest.raises((TypeError, ValueError)):
        route_representation_disagreement((record,), high_stakes=False)


def test_candidate_id_rejects_a_corrupted_source_span() -> None:
    record = _record("literal", PolicyAction.ALLOW)
    candidate = record.representation_candidate
    assert candidate is not None
    object.__setattr__(candidate.source_span, "end", candidate.source_span.end + 1)

    with pytest.raises(ValueError, match="raw_text length"):
        candidate_id_for(candidate)
