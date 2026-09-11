"""Tests for bounded, provenance-preserving representation candidates."""

# Standard library
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from itertools import islice, product
from math import inf, nan
from types import MappingProxyType

# Third-party
import pytest

# Local
from semantic_intent_robustness import representation_candidates as candidates_module
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


def test_decomposed_grapheme_candidate_span_includes_combining_mark() -> None:
    raw_text = "cafe\u0301"
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        orthographic_lexicon=("cafés",),
    )
    generated = next(
        candidate
        for candidate in lattice.candidates
        if candidate.transform_channel is RepresentationChannel.ORTHOGRAPHIC
    )

    assert generated.source_span.start == 0
    assert generated.source_span.end == len(raw_text)
    assert generated.source_span.raw_text == raw_text
    assert generated.candidate_text == "cafés"


def test_normalized_equivalent_does_not_emit_an_incompatible_grapheme_repair() -> None:
    raw_text = "cafe\u0301"
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        orthographic_lexicon=("café",),
    )

    assert all(
        candidate.transform_channel is not RepresentationChannel.ORTHOGRAPHIC
        for candidate in lattice.candidates
    )
    assert {candidate.candidate_text for candidate in lattice.candidates} == {
        raw_text,
        "café",
    }


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


def test_sorted_phonetic_distractors_cannot_starve_direct_exact_lookup() -> None:
    distractors = {
        "a" + "".join(chars): ("noise",)
        for chars in islice(product("bcdefghijklmnopqrstuvwxyz", repeat=3), 1_536)
    }
    forward = {**distractors, "fone": ("phone",)}
    reverse = dict(reversed(tuple(forward.items())))

    first = build_candidate_lattice("source-1", "fone", phonetic_lexicon=forward)
    second = build_candidate_lattice("source-1", "fone", phonetic_lexicon=reverse)

    assert first == second
    assert "phone" in tuple(candidate.candidate_text for candidate in first.candidates)
    assert first.candidates[0].outcome is CandidateOutcome.CANDIDATE


def test_context_boosted_late_alternative_is_pre_ranked_before_proposal_cap() -> None:
    earlier = tuple(f"aaa{letter}{suffix}" for letter in "abcdef" for suffix in range(16))
    alternatives = (*earlier, "zebra")

    lattice = build_candidate_lattice(
        "source-1",
        "fone",
        context=("The context explicitly mentions zebra.",),
        phonetic_lexicon={"fone": alternatives},
    )

    assert "zebra" in tuple(candidate.candidate_text for candidate in lattice.candidates)


def test_truncated_orthographic_search_marks_literal_unknown_not_no_repair() -> None:
    distractors = tuple(
        "".join(chars) for chars in islice(product("abcdefghijkl", repeat=4), 1_600)
    )
    lattice = build_candidate_lattice(
        "source-1",
        "zzzz",
        orthographic_lexicon=(*distractors, "zzzy"),
    )

    literal = next(
        candidate
        for candidate in lattice.candidates
        if candidate.transform_channel is RepresentationChannel.LITERAL
    )
    assert literal.outcome is CandidateOutcome.UNKNOWN
    assert "truncated" in literal.generation_reason.lower()
    assert "candidate-generation:search-truncated" in literal.provenance


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


def test_global_phonetic_alternative_cap_fails_before_traversing_excess_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = candidates_module._validate_canonical_lexicon_text

    def counting_validation(value: object, *, field_name: str) -> None:
        nonlocal calls
        calls += 1
        original(value, field_name=field_name)

    monkeypatch.setattr(
        candidates_module,
        "_validate_canonical_lexicon_text",
        counting_validation,
    )
    alternatives = tuple(f"term{index}" for index in range(64))
    lexicon = {f"key{index}": alternatives for index in range(65)}

    with pytest.raises(ValueError, match="global phonetic alternative cap"):
        build_candidate_lattice("source-1", "word", phonetic_lexicon=lexicon)

    assert calls <= 4_160


def test_context_tokens_are_snapshotted_once_for_many_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = candidates_module._context_word_set

    def counting_snapshot(context: tuple[str, ...]) -> frozenset[str]:
        nonlocal calls
        calls += 1
        return original(context)

    monkeypatch.setattr(candidates_module, "_context_word_set", counting_snapshot)
    alternatives = tuple(f"option{letter}" for letter in "abcdefghijkl")

    build_candidate_lattice(
        "source-1",
        "fone",
        context=("optiona is mentioned once",),
        phonetic_lexicon={"fone": alternatives},
    )

    assert calls == 1


def test_phonetic_candidate_materialization_is_bounded_by_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = candidates_module._candidate

    def counting_candidate(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(candidates_module, "_candidate", counting_candidate)
    alternatives = tuple(f"option{index}" for index in range(500))

    build_candidate_lattice(
        "source-1",
        " ".join("fone" for _ in range(10)),
        phonetic_lexicon={"fone": alternatives},
    )

    assert calls <= CandidateBudget().max_candidates_total * 4


def test_casefold_equivalent_phonetic_discovery_constructs_only_bounded_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = candidates_module._PhoneticMatch

    def counting_match(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(candidates_module, "_PhoneticMatch", counting_match)
    base = "abcdefgh"
    variants = tuple(
        "".join(
            character.upper() if mask & (1 << index) else character
            for index, character in enumerate(base)
        )
        for mask in range(256)
    )
    lexicon = {variant: ("result",) for variant in variants}
    budget = CandidateBudget(8, 4, 8)

    lattice = build_candidate_lattice(
        "source-1",
        " ".join(base for _ in range(500)),
        phonetic_lexicon=lexicon,
        budget=budget,
    )

    assert calls <= max(budget.max_spans * 4, budget.max_candidates_total * 4)
    assert "result" in tuple(candidate.candidate_text for candidate in lattice.candidates)
    assert lattice.candidates[0].outcome is CandidateOutcome.UNKNOWN


@pytest.mark.parametrize("value", [nan, inf, -inf, -0.0])
def test_budget_fields_cannot_be_noncanonical_numeric_values(value: float) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        CandidateBudget(max_spans=value)  # type: ignore[arg-type]


def test_pathologically_long_input_is_rejected_before_candidate_work() -> None:
    with pytest.raises(ValueError, match="too long"):
        build_candidate_lattice("source-1", "a" * 100_001)


def test_content_changing_conservative_view_prevents_no_repair_claim() -> None:
    lattice = build_candidate_lattice(
        "source-1",
        "Cafe\u0301",
        budget=CandidateBudget(1, 1, 1),
    )

    assert lattice.candidates[0].transform_channel is RepresentationChannel.LITERAL
    assert lattice.candidates[0].outcome is CandidateOutcome.CANDIDATE


def test_zero_width_hypothesis_below_evidence_floor_does_not_block_no_repair() -> None:
    lattice = build_candidate_lattice("source-1", "thera\u200bpist")

    assert lattice.candidates[0].candidate_text == "thera\u200bpist"
    assert lattice.candidates[0].outcome is CandidateOutcome.NO_REPAIR
    hypothesis = next(item for item in lattice.candidates if item.candidate_text == "therapist")
    assert hypothesis.confidence < 0.70


class _BoundedHostileMapping(Mapping[str, tuple[str, ...]]):
    def __init__(self) -> None:
        self.yield_count = 0

    def __getitem__(self, key: str) -> tuple[str, ...]:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for index in range(10_002):
            self.yield_count += 1
            if self.yield_count > 10_001:
                raise AssertionError("mapping snapshot consumed beyond cap + 1")
            yield f"key-{index}", ("repair",)


def test_phonetic_mapping_snapshot_never_consumes_beyond_cap_plus_one() -> None:
    mapping = _BoundedHostileMapping()

    with pytest.raises(ValueError, match="global phonetic alternative cap"):
        build_candidate_lattice("source-1", "word", phonetic_lexicon=mapping)

    assert mapping.yield_count == 4_097


def test_orthographic_budget_considers_a_tail_span_fairly() -> None:
    raw_text = " ".join((*(["alpha"] * 180), "teh"))

    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        orthographic_lexicon=("the",),
        budget=CandidateBudget(8, 4, 2),
    )

    repaired = next(item for item in lattice.candidates if item.candidate_text == "the")
    assert repaired.source_span.start == raw_text.rindex("teh")


def test_builder_revalidates_a_corrupted_budget_before_candidate_work() -> None:
    budget = CandidateBudget()
    object.__setattr__(budget, "max_candidates_total", 0)

    with pytest.raises(ValueError, match="max_candidates_total must be positive"):
        build_candidate_lattice("source-1", "teh", budget=budget)


def test_semantic_hinge_neighborhood_receives_orthographic_compute() -> None:
    tokens = ["alpha"] * 1_000
    tokens[500:502] = ["send", "teh"]
    raw_text = " ".join(tokens)

    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        orthographic_lexicon=("the",),
        budget=CandidateBudget(8, 4, 2),
    )

    repaired = next(item for item in lattice.candidates if item.candidate_text == "the")
    assert repaired.source_span.start == raw_text.index("teh")


@pytest.mark.parametrize(
    ("raw_text", "orthographic", "phonetic"),
    [
        ("Syzygy", ("sizing", "fizzy"), {"sissy gee": ("syzygy",)}),
        ("The therapist arrived.", ("the rapist", "therapists"), {}),
        ("The rapist was arrested.", ("therapist", "rapid"), {}),
        ("Ask Alice today.", ("Alicia", "Alise"), {}),
        ("thera\u200bpist", ("therapist", "the rapist"), {}),
    ],
)
def test_clean_controls_keep_literal_authoritative_with_competing_lexicons(
    raw_text: str,
    orthographic: tuple[str, ...],
    phonetic: Mapping[str, tuple[str, ...]],
) -> None:
    lattice = build_candidate_lattice(
        "source-1",
        raw_text,
        orthographic_lexicon=orthographic,
        phonetic_lexicon=phonetic,
    )

    assert lattice.raw_text == raw_text
    assert lattice.candidates[0].transform_channel is RepresentationChannel.LITERAL
    assert lattice.candidates[0].candidate_text == raw_text
    assert all(item.confidence <= 1.0 for item in lattice.candidates[1:])
