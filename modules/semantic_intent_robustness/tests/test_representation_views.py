"""Tests for literal and conservative representation views."""

# Standard library
from dataclasses import FrozenInstanceError

# Third-party
import pytest

# Local
from semantic_intent_robustness.representation import RepresentationChannel
from semantic_intent_robustness.representation_views import conservative_views, literal_view


def test_literal_view_preserves_the_complete_source_identity() -> None:
    text = "Cafe\u0301\u200b\r\nDo not send 1,002.50!"

    candidate = literal_view("source-1", text)

    assert candidate.source_span.source_id == "source-1"
    assert candidate.source_span.start == 0
    assert candidate.source_span.end == len(text)
    assert candidate.source_span.raw_text == text
    assert candidate.candidate_text == text
    assert candidate.transform_channel is RepresentationChannel.LITERAL
    assert candidate.provenance == ("literal-source",)


def test_literal_view_is_immutable() -> None:
    candidate = literal_view("source-1", "Do not send 42.")

    with pytest.raises(FrozenInstanceError):
        candidate.candidate_text = "Send 42."  # type: ignore[misc]


def test_conservative_views_apply_nfc_without_replacing_the_literal_source() -> None:
    text = "Cafe\u0301"

    literal, normalized = conservative_views("source-1", text)

    assert literal.candidate_text == text
    assert normalized.candidate_text == "Caf\u00e9"
    assert normalized.source_span.raw_text == text
    assert normalized.transform_channel is RepresentationChannel.CONSERVATIVE_NORMALIZATION
    assert normalized.provenance == ("unicode-normalization:NFC",)


def test_conservative_views_remove_only_known_zero_width_artifacts() -> None:
    text = "a\u200bb\ufeffc"

    literal, normalized = conservative_views("source-1", text)

    assert literal.candidate_text == text
    assert normalized.candidate_text == "abc"
    assert normalized.source_span.raw_text == text
    assert normalized.provenance == ("zero-width-removal:U+200B=1,U+FEFF=1",)


def test_conservative_views_normalize_crlf_and_bare_cr_with_exact_counts() -> None:
    text = "first\r\nsecond\rthird\nfourth"

    literal, normalized = conservative_views("source-1", text)

    assert literal.candidate_text == text
    assert normalized.candidate_text == "first\nsecond\nthird\nfourth"
    assert normalized.provenance == ("newline-normalization:CRLF->LF=1,CR->LF=1",)


def test_composed_conservative_view_records_each_applied_transform_in_order() -> None:
    text = "Cafe\u0301\u200b\r\nnot\ufeff now\r"

    views = conservative_views("source-1", text)

    assert type(views) is tuple
    assert len(views) == 2
    assert views[1].candidate_text == "Caf\u00e9\nnot now\n"
    assert views[1].provenance == (
        "unicode-normalization:NFC",
        "zero-width-removal:U+200B=1,U+FEFF=1",
        "newline-normalization:CRLF->LF=1,CR->LF=1",
    )


@pytest.mark.parametrize(
    "text",
    [
        "ordinary spaces stay here",
        "field one\tfield  two",
        "the rapist and therapist are distinct tokens",
        "Do not send 1,002.50!",
        "Keep punctuation: commas, periods... and (parentheses).",
        "join\u200cer and emoji \U0001f469\u200d\U0001f4bb stay joined",
    ],
)
def test_conservative_views_preserve_semantically_meaningful_text(text: str) -> None:
    assert conservative_views("source-1", text) == (literal_view("source-1", text),)


def test_conservative_views_do_not_emit_an_empty_derived_candidate() -> None:
    literal = literal_view("source-1", "\u200b\ufeff")

    assert conservative_views("source-1", "\u200b\ufeff") == (literal,)


@pytest.mark.parametrize("builder", [literal_view, conservative_views])
def test_view_builders_reject_empty_text_that_cannot_form_a_source_span(builder: object) -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        builder("source-1", "")  # type: ignore[operator]


@pytest.mark.parametrize("builder", [literal_view, conservative_views])
@pytest.mark.parametrize("value", [1, True, ["text"], type("TextSubclass", (str,), {})("text")])
def test_view_builders_require_exact_string_text(builder: object, value: object) -> None:
    with pytest.raises(TypeError, match="text must be an exact string"):
        builder("source-1", value)  # type: ignore[operator]
