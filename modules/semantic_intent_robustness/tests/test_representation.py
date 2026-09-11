"""Tests for immutable, provenance-bound representation records."""

# Standard library
import math
from dataclasses import FrozenInstanceError, replace

# Third-party
import pytest

# Local
import semantic_intent_robustness
from semantic_intent_robustness.representation import (
    CandidateOutcome,
    RepresentationCandidate,
    RepresentationChannel,
    RepresentationLattice,
    SourceSpan,
)


def test_representation_types_are_exported_from_package() -> None:
    assert semantic_intent_robustness.CandidateOutcome is CandidateOutcome
    assert semantic_intent_robustness.RepresentationCandidate is RepresentationCandidate
    assert semantic_intent_robustness.RepresentationChannel is RepresentationChannel
    assert semantic_intent_robustness.RepresentationLattice is RepresentationLattice
    assert semantic_intent_robustness.SourceSpan is SourceSpan


def _candidate(
    *,
    start: int = 6,
    end: int = 11,
    raw_text: str = "wrold",
    candidate_text: str = "world",
    channel: RepresentationChannel = RepresentationChannel.ORTHOGRAPHIC,
    confidence: float = 0.8,
) -> RepresentationCandidate:
    return RepresentationCandidate(
        source_span=SourceSpan(
            source_id="source-1",
            start=start,
            end=end,
            raw_text=raw_text,
        ),
        candidate_text=candidate_text,
        transform_channel=channel,
        orthographic_score=0.9,
        phonetic_score=0.1,
        contextual_score=0.6,
        semantic_similarity=0.7,
        confidence=confidence,
        provenance=("adjacent-character-transposition",),
        generation_reason="One adjacent transposition matches the lexicon.",
    )


def test_source_span_preserves_exact_slice_and_is_immutable() -> None:
    source = "Hello wrold!"
    span = SourceSpan(source_id="source-1", start=6, end=11, raw_text=source[6:11])

    assert span.raw_text == "wrold"
    with pytest.raises(FrozenInstanceError):
        span.raw_text = "world"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("start", "end", "raw_text", "error_type"),
    [
        (-1, 1, "x", ValueError),
        (0, 0, "", ValueError),
        (2, 1, "x", ValueError),
        (0, 2, "x", ValueError),
        (False, 1, "x", TypeError),
        (0, 1.0, "x", TypeError),
    ],
)
def test_source_span_rejects_invalid_coordinates(
    start: object,
    end: object,
    raw_text: str,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        SourceSpan(  # type: ignore[arg-type]
            source_id="source-1",
            start=start,
            end=end,
            raw_text=raw_text,
        )


@pytest.mark.parametrize("source_id", ["", " source-1", "source-1 ", 1])
def test_source_span_requires_a_canonical_source_id(source_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceSpan(source_id=source_id, start=0, end=1, raw_text="x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_text": ""},
        {"candidate_text": 1},
        {"transform_channel": "orthographic"},
        {"outcome": "candidate"},
        {"provenance": ()},
        {"provenance": ("",)},
        {"provenance": (" untrimmed",)},
        {"provenance": ["derived"]},
        {"generation_reason": ""},
        {"generation_reason": " reason"},
        {"generation_reason": 1},
    ],
)
def test_candidate_requires_exact_typed_provenance_and_reason(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "source_span": SourceSpan("source-1", 0, 1, "x"),
        "candidate_text": "x",
        "transform_channel": RepresentationChannel.LITERAL,
        "orthographic_score": 1.0,
        "phonetic_score": 0.0,
        "contextual_score": 1.0,
        "semantic_similarity": 1.0,
        "confidence": 1.0,
        "provenance": ("literal-source",),
        "generation_reason": "Preserve the literal source.",
        "outcome": CandidateOutcome.CANDIDATE,
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        RepresentationCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("orthographic_score", -0.01, ValueError),
        ("phonetic_score", 1.01, ValueError),
        ("contextual_score", math.nan, ValueError),
        ("semantic_similarity", math.inf, ValueError),
        ("confidence", -math.inf, ValueError),
        ("confidence", 1, TypeError),
        ("confidence", True, TypeError),
    ],
)
def test_candidate_rejects_unbounded_or_non_float_scores(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        replace(_candidate(), **{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "orthographic_score",
        "phonetic_score",
        "contextual_score",
        "semantic_similarity",
        "confidence",
    ],
)
def test_candidate_rejects_negative_zero_scores(field_name: str) -> None:
    with pytest.raises(ValueError, match="negative zero"):
        replace(_candidate(), **{field_name: -0.0})


def test_lattice_binds_each_span_to_the_immutable_source_text() -> None:
    lattice = RepresentationLattice(
        source_id="source-1",
        raw_text="Hello wrold!",
        candidates=(_candidate(),),
        max_candidates=3,
    )

    assert lattice.raw_text == "Hello wrold!"
    assert lattice.candidates[0].candidate_text == "world"
    assert lattice.candidates[0].source_span.raw_text == lattice.raw_text[6:11]
    with pytest.raises(FrozenInstanceError):
        lattice.raw_text = "Hello world!"  # type: ignore[misc]


def test_frozen_records_expose_no_mutable_instance_dictionary() -> None:
    candidate = _candidate()
    lattice = RepresentationLattice(
        source_id="source-1",
        raw_text="Hello wrold!",
        candidates=(candidate,),
        max_candidates=1,
    )

    for record in (candidate.source_span, candidate, lattice):
        with pytest.raises(AttributeError):
            getattr(record, "__dict__")


def test_derived_candidate_round_trip_never_replaces_raw_text() -> None:
    original = "Hello wrold!"
    first = RepresentationLattice("source-1", original, (_candidate(),), 3)
    second = RepresentationLattice(
        source_id=first.source_id,
        raw_text=first.raw_text,
        candidates=(replace(first.candidates[0], candidate_text="world!"),),
        max_candidates=first.max_candidates,
    )

    assert second.raw_text == original
    assert second.candidates[0].candidate_text == "world!"
    assert second.candidates[0].source_span.raw_text == "wrold"


@pytest.mark.parametrize(
    "updates",
    [
        {"source_id": "other-source"},
        {"start": 5, "end": 10, "raw_text": "wrold"},
        {"end": 20, "raw_text": "x" * 14},
    ],
)
def test_lattice_rejects_candidates_not_bound_to_its_source(updates: dict[str, object]) -> None:
    candidate = _candidate()
    mismatched_span = replace(candidate.source_span, **updates)

    with pytest.raises(ValueError):
        RepresentationLattice(
            source_id="source-1",
            raw_text="Hello wrold!",
            candidates=(replace(candidate, source_span=mismatched_span),),
            max_candidates=3,
        )


@pytest.mark.parametrize("max_candidates", [0, -1, True, 2.0])
def test_lattice_requires_a_positive_exact_integer_maximum(max_candidates: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RepresentationLattice(
            source_id="source-1",
            raw_text="Hello wrold!",
            candidates=(_candidate(),),
            max_candidates=max_candidates,  # type: ignore[arg-type]
        )


def test_lattice_enforces_maximum_candidate_count() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        RepresentationLattice(
            source_id="source-1",
            raw_text="Hello wrold!",
            candidates=(_candidate(), _candidate(candidate_text="whirled")),
            max_candidates=1,
        )


def test_lattice_requires_an_immutable_candidate_tuple() -> None:
    with pytest.raises(TypeError):
        RepresentationLattice(
            source_id="source-1",
            raw_text="Hello wrold!",
            candidates=[_candidate()],  # type: ignore[arg-type]
            max_candidates=1,
        )


def test_lattice_orders_candidates_deterministically() -> None:
    contextual = _candidate(
        candidate_text="whirled",
        channel=RepresentationChannel.CONTEXTUAL,
        confidence=0.8,
    )
    literal = _candidate(
        candidate_text="wrold",
        channel=RepresentationChannel.LITERAL,
        confidence=1.0,
    )
    later_span = _candidate(
        start=11,
        end=12,
        raw_text="!",
        candidate_text="?",
        channel=RepresentationChannel.ORTHOGRAPHIC,
        confidence=0.8,
    )
    lexical_last = _candidate(candidate_text="worlds", confidence=0.8)
    lexical_first = _candidate(candidate_text="world", confidence=0.8)

    lattice = RepresentationLattice(
        source_id="source-1",
        raw_text="Hello wrold!",
        candidates=(later_span, lexical_last, contextual, literal, lexical_first),
        max_candidates=5,
    )

    assert lattice.candidates == (
        literal,
        contextual,
        lexical_first,
        lexical_last,
        later_span,
    )


def test_lattice_order_is_independent_of_candidate_input_order() -> None:
    lexical_last = _candidate(candidate_text="worlds", confidence=0.8)
    lexical_first = _candidate(candidate_text="world", confidence=0.8)

    forward = RepresentationLattice(
        source_id="source-1",
        raw_text="Hello wrold!",
        candidates=(lexical_last, lexical_first),
        max_candidates=2,
    )
    reversed_order = RepresentationLattice(
        source_id="source-1",
        raw_text="Hello wrold!",
        candidates=(lexical_first, lexical_last),
        max_candidates=2,
    )

    assert forward.candidates == reversed_order.candidates == (lexical_first, lexical_last)


@pytest.mark.parametrize(
    ("field_name", "corrupted"),
    [
        ("orthographic_score", math.nan),
        ("provenance", ["caller-owned"]),
    ],
)
def test_lattice_revalidates_corrupted_candidate_fields(
    field_name: str,
    corrupted: object,
) -> None:
    candidate = _candidate()
    object.__setattr__(candidate, field_name, corrupted)

    with pytest.raises((TypeError, ValueError)):
        RepresentationLattice(
            source_id="source-1",
            raw_text="Hello wrold!",
            candidates=(candidate,),
            max_candidates=1,
        )
