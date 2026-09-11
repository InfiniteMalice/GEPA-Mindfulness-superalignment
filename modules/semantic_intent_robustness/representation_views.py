"""Deterministic literal and conservative-normalization representation views."""

# Standard library
from __future__ import annotations

import unicodedata

# Local
from .representation import RepresentationCandidate, RepresentationChannel, SourceSpan

_ZERO_WIDTH_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("\u200b", "U+200B"),
    ("\ufeff", "U+FEFF"),
)


def literal_view(source_id: str, text: str) -> RepresentationCandidate:
    """Return the complete source as an unchanged literal candidate."""

    _validate_text(text)
    span = SourceSpan(source_id=source_id, start=0, end=len(text), raw_text=text)
    return RepresentationCandidate(
        source_span=span,
        candidate_text=text,
        transform_channel=RepresentationChannel.LITERAL,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=("literal-source",),
        generation_reason="Preserve the literal source.",
    )


def conservative_views(
    source_id: str,
    text: str,
) -> tuple[RepresentationCandidate, ...]:
    """Return the literal view and one nonempty deterministic normalized view, if changed."""

    literal = literal_view(source_id, text)
    normalized = text
    provenance: list[str] = []

    normalized, zero_width_provenance = _remove_zero_width_artifacts(normalized)
    if zero_width_provenance is not None:
        provenance.append(zero_width_provenance)

    nfc_text = unicodedata.normalize("NFC", normalized)
    if nfc_text != normalized:
        normalized = nfc_text
        provenance.append("unicode-normalization:NFC")

    normalized, newline_provenance = _normalize_newlines(normalized)
    if newline_provenance is not None:
        provenance.append(newline_provenance)

    if not provenance or not normalized:
        return (literal,)

    normalized_view = RepresentationCandidate(
        source_span=literal.source_span,
        candidate_text=normalized,
        transform_channel=RepresentationChannel.CONSERVATIVE_NORMALIZATION,
        orthographic_score=1.0,
        phonetic_score=1.0,
        contextual_score=1.0,
        semantic_similarity=1.0,
        confidence=1.0,
        provenance=tuple(provenance),
        generation_reason="Apply deterministic conservative normalization.",
    )
    return (literal, normalized_view)


def _remove_zero_width_artifacts(text: str) -> tuple[str, str | None]:
    counts: list[str] = []
    normalized = text
    for character, code_point in _ZERO_WIDTH_ARTIFACTS:
        count = normalized.count(character)
        if count:
            counts.append(f"{code_point}={count}")
            normalized = normalized.replace(character, "")
    if not counts:
        return normalized, None
    return normalized, f"zero-width-removal:{','.join(counts)}"


def _normalize_newlines(text: str) -> tuple[str, str | None]:
    crlf_count = text.count("\r\n")
    normalized = text.replace("\r\n", "\n")
    cr_count = normalized.count("\r")
    normalized = normalized.replace("\r", "\n")
    if not crlf_count and not cr_count:
        return normalized, None
    provenance = f"newline-normalization:CRLF->LF={crlf_count},CR->LF={cr_count}"
    return normalized, provenance


def _validate_text(text: object) -> None:
    if type(text) is not str:
        raise TypeError("text must be an exact string")
    if not text:
        raise ValueError("text must not be empty")


__all__ = ["conservative_views", "literal_view"]
