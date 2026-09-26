"""Contracts for opt-in structural communication diagnostics."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from evaluation.serialization_roundtrip import (
    EquivalenceResult,
    EquivalenceStatus,
    Expression,
    FaultStatus,
    JsonTreeCodec,
    LaunderingClassification,
    PropositionalTreeVerifier,
    RoundTripResult,
    StageEvidence,
    StructuredSource,
    audit_roundtrip,
    classify_laundering,
)


def source() -> StructuredSource:
    return StructuredSource(
        "source:1",
        Expression("and", (Expression("atom", atom="rain"), Expression("atom", atom="cold"))),
        ("trace:source",),
    )


class FixedSerializer:
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def serialize(self, original: StructuredSource) -> str:
        return JsonTreeCodec().serialize(replace(original, expression=self.expression))


class FixedExtractor:
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def extract(self, text: str) -> Expression:
        return self.expression


def stage_evidence(text: str, represented: Expression) -> StageEvidence:
    return StageEvidence(
        sha256(text.encode("utf-8")).hexdigest(), represented, "reference:reader", ("trace:stage",)
    )


def test_exact_roundtrip_retains_structure_complexity_and_provenance() -> None:
    original = source()
    result = audit_roundtrip(
        original, JsonTreeCodec(), JsonTreeCodec(), PropositionalTreeVerifier(), enabled=True
    )
    assert result is not None
    assert result.equivalence.status is EquivalenceStatus.EXACT_EQUIVALENCE
    assert result.reconstructed == original.expression
    assert result.node_count == 3
    assert result.depth == 2
    assert result.evidence_refs == ("trace:source",)
    assert result.roundtrip_failure is False
    # Successful composition alone does not prove that two stage errors did not cancel.
    assert result.serialization_fault is FaultStatus.UNKNOWN


def test_disabled_audit_does_not_invoke_any_adapter() -> None:
    class ForbiddenCodec:
        def serialize(self, original: StructuredSource) -> str:
            raise AssertionError("disabled adapter was invoked")

        def extract(self, text: str) -> Expression:
            raise AssertionError("disabled adapter was invoked")

    assert audit_roundtrip(source(), ForbiddenCodec(), ForbiddenCodec(), None) is None
    with pytest.raises(ValueError, match="enabled"):
        audit_roundtrip(source(), ForbiddenCodec(), ForbiddenCodec(), None, enabled=1)


def test_serializer_lost_node_is_identified_only_with_independent_stage_evidence() -> None:
    original = source()
    lost = original.expression.children[0]
    serializer = FixedSerializer(lost)
    result = audit_roundtrip(
        original,
        serializer,
        JsonTreeCodec(),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(serializer.serialize(original), lost),
    )
    assert result is not None
    assert result.serialization_fault is FaultStatus.FAULT
    assert result.extraction_fault is FaultStatus.NO_FAULT
    assert result.roundtrip_failure is True
    assert "trace:stage" in result.evidence_refs
    assert classify_laundering(result.equivalence, judgment_changed=True, roundtrip=result) is (
        LaunderingClassification.COMMUNICATION_FAILURE
    )


def test_extractor_error_is_separate_from_adequate_serialization() -> None:
    original = source()
    text = JsonTreeCodec().serialize(original)
    result = audit_roundtrip(
        original,
        JsonTreeCodec(),
        FixedExtractor(original.expression.children[0]),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(text, original.expression),
    )
    assert result is not None
    assert result.serialization_fault is FaultStatus.NO_FAULT
    assert result.extraction_fault is FaultStatus.FAULT


def test_end_to_end_mismatch_does_not_invent_stage_attribution() -> None:
    result = audit_roundtrip(
        source(),
        JsonTreeCodec(),
        FixedExtractor(Expression("atom", atom="rain")),
        PropositionalTreeVerifier(),
        enabled=True,
    )
    assert result is not None
    assert result.roundtrip_failure is True
    assert result.serialization_fault is FaultStatus.UNKNOWN
    assert result.extraction_fault is FaultStatus.UNKNOWN


def test_stage_evidence_cannot_be_replayed_against_other_serialized_text() -> None:
    with pytest.raises(ValueError, match="digest"):
        audit_roundtrip(
            source(),
            JsonTreeCodec(),
            JsonTreeCodec(),
            PropositionalTreeVerifier(),
            enabled=True,
            stage_evidence=stage_evidence("different text", source().expression),
        )


def test_surface_reordering_is_semantic_not_exact_equivalence() -> None:
    expression = source().expression
    reordered = Expression("and", tuple(reversed(expression.children)))
    result = PropositionalTreeVerifier().verify(expression, reordered)
    assert result.status is EquivalenceStatus.VERIFIED_SEMANTIC_EQUIVALENCE
    assert classify_laundering(result, judgment_changed=True) is (
        LaunderingClassification.PRESERVED_JUDGMENT_CHANGED
    )


def test_non_equivalent_structure_is_not_a_laundering_success() -> None:
    expression = source().expression
    changed = Expression("or", expression.children)
    result = PropositionalTreeVerifier().verify(expression, changed)
    assert result.status is EquivalenceStatus.NOT_EQUIVALENT
    assert classify_laundering(result, judgment_changed=True) is (
        LaunderingClassification.MEANING_CHANGED
    )


@pytest.mark.parametrize(
    "status", [EquivalenceStatus.UNKNOWN, EquivalenceStatus.HEURISTIC_SIMILARITY]
)
def test_unverified_similarity_cannot_count_as_preserved_semantics(status) -> None:
    result = EquivalenceResult(status, "external:judge", ("trace:judge",))
    assert classify_laundering(result, judgment_changed=True) is (
        LaunderingClassification.UNKNOWN_EQUIVALENCE
    )


def test_unknown_verifier_keeps_total_and_stage_faults_unknown() -> None:
    class UnknownVerifier:
        def verify(self, left: Expression, right: Expression) -> EquivalenceResult:
            return EquivalenceResult(EquivalenceStatus.UNKNOWN, "external:unknown", ())

    result = audit_roundtrip(
        source(), JsonTreeCodec(), JsonTreeCodec(), UnknownVerifier(), enabled=True
    )
    assert result is not None
    assert result.equivalence.status is EquivalenceStatus.UNKNOWN
    assert result.roundtrip_failure is None


def test_truth_table_budget_exhaustion_returns_unknown_but_exact_identity_is_cheap() -> None:
    original = source().expression
    verifier = PropositionalTreeVerifier(max_atoms=1)
    assert verifier.verify(original, original).status is EquivalenceStatus.EXACT_EQUIVALENCE
    other = Expression("and", tuple(reversed(original.children)))
    assert verifier.verify(original, other).status is EquivalenceStatus.UNKNOWN


@pytest.mark.parametrize("limit", [True, 0, 13, 1.5])
def test_invalid_truth_table_budgets_are_rejected(limit) -> None:
    with pytest.raises(ValueError, match="max_atoms"):
        PropositionalTreeVerifier(max_atoms=limit)


def test_expression_and_provenance_are_immutable_and_bounded() -> None:
    children = [Expression("atom", atom="rain"), Expression("atom", atom="cold")]
    refs = ["trace:1"]
    original = StructuredSource("source:1", Expression("and", children), refs)
    children.clear()
    refs.clear()
    assert len(original.expression.children) == 2
    assert original.evidence_refs == ("trace:1",)
    with pytest.raises(FrozenInstanceError):
        original.source_id = "changed"
    with pytest.raises(ValueError, match="children"):
        Expression("not", ())
    with pytest.raises(ValueError, match="operator"):
        Expression("execute", ())
    with pytest.raises(ValueError, match="bound"):
        Expression("and", (Expression("atom", atom="rain"),) * 257)


def test_invalid_extraction_is_a_visible_stage_failure() -> None:
    class MalformedSerializer:
        def serialize(self, original: StructuredSource) -> str:
            return "not a tree"

    result = audit_roundtrip(
        source(), MalformedSerializer(), JsonTreeCodec(), PropositionalTreeVerifier(), enabled=True
    )
    assert result is not None
    assert result.extraction_fault is FaultStatus.FAULT
    assert result.serialization_fault is FaultStatus.UNKNOWN
    assert result.roundtrip_failure is True
    assert result.equivalence.status is EquivalenceStatus.UNKNOWN


def test_codec_rejects_oversized_and_extra_field_payloads() -> None:
    with pytest.raises(ValueError):
        JsonTreeCodec().extract(" " * 32769)
    with pytest.raises(ValueError):
        JsonTreeCodec().extract('{"operator":"atom","atom":"x","children":[],"extra":1}')


def test_verification_of_implication_and_negation_uses_all_valuations() -> None:
    a, b = Expression("atom", atom="a"), Expression("atom", atom="b")
    implication = Expression("implies", (a, b))
    disjunction = Expression("or", (Expression("not", (a,)), b))
    assert PropositionalTreeVerifier().verify(implication, disjunction).status is (
        EquivalenceStatus.VERIFIED_SEMANTIC_EQUIVALENCE
    )


def test_preserved_meaning_without_changed_judgment_is_no_laundering_candidate() -> None:
    result = PropositionalTreeVerifier().verify(source().expression, source().expression)
    assert classify_laundering(result, judgment_changed=False) is (
        LaunderingClassification.PRESERVED_JUDGMENT_UNCHANGED
    )


def test_stage_verifier_results_retain_their_separate_evidence() -> None:
    class ReferencedVerifier:
        def verify(self, left: Expression, right: Expression) -> EquivalenceResult:
            result = PropositionalTreeVerifier().verify(left, right)
            return replace(result, evidence_refs=(f"proof:{left.operator}:{right.operator}",))

    original = source()
    changed = original.expression.children[0]
    serializer = FixedSerializer(changed)
    result = audit_roundtrip(
        original,
        serializer,
        JsonTreeCodec(),
        ReferencedVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(serializer.serialize(original), changed),
    )
    assert result is not None
    assert "proof:atom:atom" in result.evidence_refs
    assert result.serialization_equivalence.status is EquivalenceStatus.NOT_EQUIVALENT
    assert result.extraction_equivalence.status is EquivalenceStatus.EXACT_EQUIVALENCE


def test_result_rejects_mutable_or_untyped_evidence() -> None:
    with pytest.raises(ValueError, match="source"):
        RoundTripResult(
            {},
            None,
            None,
            EquivalenceResult(EquivalenceStatus.UNKNOWN, "fixture", ()),
            FaultStatus.UNKNOWN,
            FaultStatus.UNKNOWN,
        )


def test_serializer_exception_is_reported_without_calling_extractor() -> None:
    class BrokenSerializer:
        def serialize(self, original: StructuredSource) -> str:
            raise ValueError("synthetic serializer failure")

    class ForbiddenExtractor:
        def extract(self, text: str) -> Expression:
            raise AssertionError("must not extract a failed serialization")

    result = audit_roundtrip(
        source(),
        BrokenSerializer(),
        ForbiddenExtractor(),
        PropositionalTreeVerifier(),
        enabled=True,
    )
    assert result is not None
    assert result.serialization_fault is FaultStatus.FAULT
    assert result.extraction_fault is FaultStatus.UNKNOWN
    assert result.roundtrip_failure is True


def test_depth_bound_prevents_unbounded_recursive_verification() -> None:
    expression = Expression("atom", atom="x")
    for _ in range(31):
        expression = Expression("not", (expression,))
    with pytest.raises(ValueError, match="depth bound"):
        Expression("not", (expression,))


def test_communication_fault_classification_survives_copying_equivalence_records() -> None:
    original = source()
    serializer = FixedSerializer(original.expression.children[0])
    result = audit_roundtrip(
        original,
        serializer,
        JsonTreeCodec(),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(
            serializer.serialize(original), original.expression.children[0]
        ),
    )
    assert result is not None
    assert classify_laundering(
        replace(result.equivalence), judgment_changed=True, roundtrip=result
    ) is (LaunderingClassification.COMMUNICATION_FAILURE)


def test_total_channel_failure_blocks_laundering_even_with_unknown_stage_attribution() -> None:
    original = source()
    result = audit_roundtrip(
        original,
        JsonTreeCodec(),
        FixedExtractor(original.expression.children[0]),
        PropositionalTreeVerifier(),
        enabled=True,
    )
    assert result is not None
    assert result.serialization_fault is FaultStatus.UNKNOWN
    assert result.extraction_fault is FaultStatus.UNKNOWN
    preserved = PropositionalTreeVerifier().verify(original.expression, original.expression)
    assert classify_laundering(preserved, judgment_changed=True, roundtrip=result) is (
        LaunderingClassification.COMMUNICATION_FAILURE
    )


@pytest.mark.parametrize("field", ["serialization_fault", "extraction_fault"])
def test_result_rejects_stage_blame_without_independent_stage_evidence(field: str) -> None:
    result = audit_roundtrip(
        source(), JsonTreeCodec(), JsonTreeCodec(), PropositionalTreeVerifier(), enabled=True
    )
    assert result is not None
    with pytest.raises(ValueError):
        replace(result, **{field: FaultStatus.FAULT})


@pytest.mark.parametrize(
    "changes",
    [
        dict(serialization_fault=FaultStatus.FAULT),
        dict(extraction_fault=FaultStatus.FAULT),
        dict(serialized_text="unrelated serialized text"),
        dict(serialized_text=None),
        dict(reconstructed=None),
        dict(serialization_equivalence=None),
        dict(extraction_equivalence=None),
        dict(stage_evidence=None),
        dict(reconstructed=Expression("atom", atom="different")),
        dict(equivalence=EquivalenceResult(EquivalenceStatus.NOT_EQUIVALENT, "verifier", ())),
    ],
)
def test_result_rejects_inconsistent_roundtrip_artifacts(changes) -> None:
    original = source()
    result = audit_roundtrip(
        original,
        JsonTreeCodec(),
        JsonTreeCodec(),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(JsonTreeCodec().serialize(original), original.expression),
    )
    assert result is not None
    with pytest.raises(ValueError):
        replace(result, **changes)


def test_independently_verified_stage_errors_can_cancel_in_total_roundtrip() -> None:
    original = source()
    lost = original.expression.children[0]
    serializer = FixedSerializer(lost)
    result = audit_roundtrip(
        original,
        serializer,
        FixedExtractor(original.expression),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(serializer.serialize(original), lost),
    )
    assert result is not None
    assert result.roundtrip_failure is False
    assert result.serialization_fault is FaultStatus.FAULT
    assert result.extraction_fault is FaultStatus.FAULT
    assert classify_laundering(result.equivalence, judgment_changed=True, roundtrip=result) is (
        LaunderingClassification.COMMUNICATION_FAILURE
    )


def test_preserved_stages_cannot_claim_non_equivalent_total() -> None:
    original = source()
    reordered = Expression("and", tuple(reversed(original.expression.children)))
    result = audit_roundtrip(
        original,
        JsonTreeCodec(),
        FixedExtractor(reordered),
        PropositionalTreeVerifier(),
        enabled=True,
        stage_evidence=stage_evidence(JsonTreeCodec().serialize(original), original.expression),
    )
    assert result is not None
    with pytest.raises(ValueError, match="contradict"):
        replace(
            result,
            equivalence=EquivalenceResult(EquivalenceStatus.NOT_EQUIVALENT, "other-verifier", ()),
        )
