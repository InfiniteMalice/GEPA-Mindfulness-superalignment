"""Tests for bounded, provenance-preserving representation candidates."""

# Standard library
from dataclasses import FrozenInstanceError
from math import inf, nan
from types import MappingProxyType

# Third-party
import pytest

# Local
from semantic_intent_robustness.representation import (
    CandidateOutcome,
    RepresentationChannel,
)
from semantic_intent_robustness.representation_candidates import (
    CandidateBudget,
    build_candidate_lattice,
)


def _texts(raw_text: str, **kwargs: object) -> tuple[str, ...]:
    lattice = build_candidate_lattice("source-1", raw_text, **kwargs)  # type: ignore[arg-type]
    return tuple(candidate.candidate_text for candidate in lattice.candidates)


@pytest.mark.parametrize(
    ("observed", "intended"),
    [
        ("hellp", "hello"),  # adjacent-key substitution
        ("teh", "the"),  # adjacent transposition
        ("coool", "cool"),  # repeated character
        ("helo", "hello"),  # deleted character
        ("modem", "modern"),  # OCR-like rn/m confusion
        ("he11o", "hello"),  # OCR-like digit/letter confusion
    ],
)
def test_orthographic_errors_recover_injected_candidate_in_top_k(
    observed: str,
    intended: str,
) -> None:
    assert intended in _texts(observed, orthographic_lexicon=(intended,))[:4]


def test_all_caps_orthographic_error_preserves_the_source_case_pattern() -> None:
    assert "THE" in _texts("TEH", orthographic_lexicon=("the",))[:4]


@pytest.mark.parametrize(
    ("observed", "intended"),
    [
        ("their", "there"),  # homophone
        ("wreck a nice beach", "recognize speech"),  # ASR-like phrase
        ("fone", "phone"),  # phonetic spelling
    ],
)
def test_injected_phonetic_hypotheses_are_recovered_in_top_k(
    observed: str,
    intended: str,
) -> None:
    lexicon = MappingProxyType({observed: (intended,)})

    assert intended in _texts(observed, phonetic_lexicon=lexicon)[:4]


def test_bone_apple_tea_phrase_hypothesis_appears_without_replacing_source() -> None:
    raw_text = "bone apple tea"
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        context=("The speaker is ordering a French meal.",),
        phonetic_lexicon={raw_text: ("bon appétit",)},
    )

    assert lattice.raw_text == raw_text
    assert lattice.candidates[0].candidate_text == raw_text
    assert "bon appétit" in tuple(candidate.candidate_text for candidate in lattice.candidates[:4])
    alternate = next(
        candidate for candidate in lattice.candidates if candidate.candidate_text == "bon appétit"
    )
    assert alternate.transform_channel is RepresentationChannel.PHONOLOGICAL
    assert alternate.source_span.raw_text == raw_text
    assert alternate.provenance == ("phonetic-lexicon:bone apple tea",)
    assert "hypothesis" in alternate.generation_reason.lower()


def test_literal_context_control_keeps_source_as_the_highest_confidence_reading() -> None:
    raw_text = "Put the bone, apple, and tea on the table."
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        context=("Arrange groceries on a table.",),
        phonetic_lexicon={"bone apple tea": ("bon appétit",)},
    )

    assert lattice.candidates[0].candidate_text == raw_text
    assert lattice.candidates[0].confidence == 1.0
    assert all(candidate.candidate_text != "bon appétit" for candidate in lattice.candidates)
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


@pytest.mark.parametrize(
    "raw_text",
    [
        "I spoke with Zora yesterday.",
        "Syzygy is an unusual clean word.",
        "Keep 1,002.50 exactly.",
        "Do not send it.",
        "The therapist arrived.",
        "The rapist was arrested.",
    ],
)
def test_clean_controls_emit_no_automatic_repair(raw_text: str) -> None:
    lattice = build_candidate_lattice("source-1", raw_text)

    assert len(lattice.candidates) == 1
    assert lattice.candidates[0].candidate_text == raw_text
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


def test_title_cased_proper_noun_resists_close_dictionary_repair() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "Ask Evan today.",
        orthographic_lexicon=("even",),
    )

    assert all(candidate.candidate_text != "Even" for candidate in lattice.candidates)
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


def test_numbers_and_negation_are_not_orthographically_repaired() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "Do not send 1000.",
        orthographic_lexicon=("now", "1001"),
    )

    assert len(lattice.candidates) == 1
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


def test_conservative_views_are_retained_without_changing_raw_text() -> None:
    raw_text = "Cafe\u200b\u0301\r\n"
    lattice = build_candidate_lattice("source-1", raw_text)

    assert lattice.raw_text == raw_text
    assert {candidate.candidate_text for candidate in lattice.candidates} == {
        raw_text,
        "Café\n",
    }
    assert all(candidate.source_span.raw_text == raw_text for candidate in lattice.candidates)


def test_generated_evidence_scores_do_not_assert_semantic_truth() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "teh",
        orthographic_lexicon=("the",),
    )
    repair = next(
        candidate for candidate in lattice.candidates if candidate.candidate_text == "the"
    )

    assert 0.0 < repair.orthographic_score < 1.0
    assert repair.semantic_similarity == 0.5
    assert repair.confidence < lattice.candidates[0].confidence
    assert repair.outcome is CandidateOutcome.CANDIDATE


def test_duplicate_lexicon_text_does_not_duplicate_candidates() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "teh",
        orthographic_lexicon=("the", "the"),
        phonetic_lexicon={"teh": ("the", "the")},
    )

    generated = [candidate for candidate in lattice.candidates if candidate.candidate_text == "the"]
    assert len(generated) == 1
    assert generated[0].provenance == (
        "orthographic-lexicon:the",
        "phonetic-lexicon:teh",
    )


def test_contextual_score_records_that_context_supplied_evidence() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "fone",
        context=("Please answer the phone.",),
        phonetic_lexicon={"fone": ("phone",)},
    )
    candidate = next(item for item in lattice.candidates if item.candidate_text == "phone")

    assert candidate.contextual_score == 0.75
    assert "context-evidence:token-overlap" in candidate.provenance


def test_budget_suppression_does_not_claim_that_no_evidence_existed() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "teh",
        orthographic_lexicon=("the",),
        budget=CandidateBudget(1, 1, 1),
    )

    assert lattice.candidates[0].candidate_text == "teh"
    assert lattice.candidates[0].outcome is CandidateOutcome.CANDIDATE


def test_compute_cap_does_not_starve_explicit_phonetic_evidence() -> None:
    distractors = tuple(f"word{letter}{suffix}" for letter in "abcdef" for suffix in range(40))
    lattice = build_candidate_lattice(
        "source-1",
        "fone",
        orthographic_lexicon=distractors,
        phonetic_lexicon={"fone": ("phone",)},
        budget=CandidateBudget(1, 2, 2),
    )

    assert "phone" in tuple(candidate.candidate_text for candidate in lattice.candidates)


def test_candidate_order_is_independent_of_lexicon_insertion_order() -> None:
    first = build_candidate_lattice(
        "source-1",
        "teh",
        orthographic_lexicon=("ten", "the"),
        phonetic_lexicon={"teh": ("tea", "the")},
    )
    second = build_candidate_lattice(
        "source-1",
        "teh",
        orthographic_lexicon=("the", "ten"),
        phonetic_lexicon={"teh": ("the", "tea")},
    )

    assert first == second


def test_generator_never_exceeds_global_span_or_per_span_caps() -> None:
    budget = CandidateBudget(
        max_spans=2,
        max_candidates_per_span=2,
        max_candidates_total=3,
    )
    lattice = build_candidate_lattice(
        "source-1",
        "teh fone",
        orthographic_lexicon=("the", "ten", "phone", "tone"),
        phonetic_lexicon={"teh": ("tea",), "fone": ("phone",)},
        budget=budget,
    )
    span_counts: dict[tuple[int, int], int] = {}
    for candidate in lattice.candidates:
        key = (candidate.source_span.start, candidate.source_span.end)
        span_counts[key] = span_counts.get(key, 0) + 1

    assert len(lattice.candidates) <= budget.max_candidates_total
    assert len(span_counts) <= budget.max_spans
    assert max(span_counts.values()) <= budget.max_candidates_per_span


def test_one_slot_budget_still_returns_explicit_no_repair() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "clean",
        budget=CandidateBudget(1, 1, 1),
    )

    assert len(lattice.candidates) == 1
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


def test_one_slot_budget_never_discards_literal_for_a_normalized_view() -> None:
    raw_text = "Cafe\u200b\u0301"
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        budget=CandidateBudget(1, 1, 1),
    )

    assert lattice.candidates[0].candidate_text == raw_text
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR


@pytest.mark.parametrize("field", ["max_spans", "max_candidates_per_span", "max_candidates_total"])
@pytest.mark.parametrize("value", [0, -1])
def test_candidate_budget_requires_positive_values(field: str, value: int) -> None:
    kwargs = {
        "max_spans": 8,
        "max_candidates_per_span": 4,
        "max_candidates_total": 24,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="positive"):
        CandidateBudget(**kwargs)


@pytest.mark.parametrize("value", [True, 1.0, "1", type("IntSubclass", (int,), {})(1)])
def test_candidate_budget_requires_exact_integers(value: object) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        CandidateBudget(max_spans=value)  # type: ignore[arg-type]


def test_candidate_budget_is_frozen_and_slotted() -> None:
    budget = CandidateBudget()

    assert not hasattr(budget, "__dict__")
    with pytest.raises(FrozenInstanceError):
        budget.max_spans = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("source_id", " source-1", ValueError),
        ("source_id", "", ValueError),
        ("source_id", "source-e\u0301", ValueError),
        ("source_id", 1, TypeError),
        ("raw_text", "", ValueError),
        ("raw_text", type("TextSubclass", (str,), {})("text"), TypeError),
        ("context", ["context"], TypeError),
        ("orthographic_lexicon", ["word"], TypeError),
        ("budget", {"max_spans": 1}, TypeError),
    ],
)
def test_builder_rejects_noncanonical_top_level_inputs(
    argument: str,
    value: object,
    exception: type[Exception],
) -> None:
    kwargs: dict[str, object] = {"source_id": "source-1", "raw_text": "text"}
    kwargs[argument] = value

    with pytest.raises(exception):
        build_candidate_lattice(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " padded ", 1, True])
def test_builder_rejects_invalid_lexicon_entries(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_candidate_lattice(
            "source-1",
            "word",
            orthographic_lexicon=(value,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "lexicon",
    [
        {"word": ["alternative"]},
        {1: ("alternative",)},
        {"word": ("",)},
        {" word ": ("alternative",)},
    ],
)
def test_builder_rejects_invalid_phonetic_lexicon(lexicon: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_candidate_lattice(
            "source-1",
            "word",
            phonetic_lexicon=lexicon,  # type: ignore[arg-type]
        )


def test_builder_snapshots_a_mutable_phonetic_mapping() -> None:
    lexicon = {"teh": ("the",)}
    lattice = build_candidate_lattice("source-1", "teh", phonetic_lexicon=lexicon)
    lexicon["teh"] = ("ten",)

    assert "the" in tuple(candidate.candidate_text for candidate in lattice.candidates)
    assert "ten" not in tuple(candidate.candidate_text for candidate in lattice.candidates)


@pytest.mark.parametrize("value", [nan, inf, -inf, -0.0])
def test_budget_fields_cannot_be_noncanonical_numeric_values(value: float) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        CandidateBudget(max_spans=value)  # type: ignore[arg-type]


def test_pathologically_long_input_is_rejected_before_candidate_work() -> None:
    with pytest.raises(ValueError, match="too long"):
        build_candidate_lattice("source-1", "a" * 100_001)
