"""Immutable, provenance-bound representation records."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import copysign, isfinite


class RepresentationChannel(str, Enum):
    """Channel that produced a representation candidate."""

    LITERAL = "literal"
    CONSERVATIVE_NORMALIZATION = "conservative_normalization"
    ORTHOGRAPHIC = "orthographic"
    PHONOLOGICAL = "phonological"
    CONTEXTUAL = "contextual"


class CandidateOutcome(str, Enum):
    """Outcome of bounded candidate generation."""

    CANDIDATE = "candidate"
    NO_REPAIR = "no_repair"
    UNKNOWN = "unknown"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Exact nonempty slice of an immutable source document."""

    source_id: str
    start: int
    end: int
    raw_text: str

    def __post_init__(self) -> None:
        _validate_identifier(self.source_id, field_name="source_id")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("start and end must be exact integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must satisfy 0 <= start < end")
        if type(self.raw_text) is not str:
            raise TypeError("raw_text must be an exact string")
        if len(self.raw_text) != self.end - self.start:
            raise ValueError("raw_text length must equal end - start")


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    """One bounded alternate reading with retained source provenance."""

    source_span: SourceSpan
    candidate_text: str
    transform_channel: RepresentationChannel
    orthographic_score: float
    phonetic_score: float
    contextual_score: float
    semantic_similarity: float
    confidence: float
    provenance: tuple[str, ...]
    generation_reason: str
    outcome: CandidateOutcome = CandidateOutcome.CANDIDATE

    def __post_init__(self) -> None:
        if type(self.source_span) is not SourceSpan:
            raise TypeError("source_span must be an exact SourceSpan")
        _validate_nonempty_string(self.candidate_text, field_name="candidate_text")
        if type(self.transform_channel) is not RepresentationChannel:
            raise TypeError("transform_channel must be a RepresentationChannel")
        if type(self.outcome) is not CandidateOutcome:
            raise TypeError("outcome must be a CandidateOutcome")
        for field_name in (
            "orthographic_score",
            "phonetic_score",
            "contextual_score",
            "semantic_similarity",
            "confidence",
        ):
            _validate_score(getattr(self, field_name), field_name=field_name)
        _validate_provenance(self.provenance)
        _validate_canonical_text(self.generation_reason, field_name="generation_reason")


@dataclass(frozen=True, slots=True)
class RepresentationLattice:
    """Deterministically ordered candidates bound to one immutable source."""

    source_id: str
    raw_text: str
    candidates: tuple[RepresentationCandidate, ...]
    max_candidates: int

    def __post_init__(self) -> None:
        _validate_identifier(self.source_id, field_name="source_id")
        if type(self.raw_text) is not str:
            raise TypeError("raw_text must be an exact string")
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be an exact tuple")
        if any(type(candidate) is not RepresentationCandidate for candidate in self.candidates):
            raise TypeError("candidates must contain exact RepresentationCandidate values")
        if type(self.max_candidates) is not int:
            raise TypeError("max_candidates must be an exact integer")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if len(self.candidates) > self.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")

        for candidate in self.candidates:
            span = candidate.source_span
            if span.source_id != self.source_id:
                raise ValueError("candidate source_id must match lattice source_id")
            if span.end > len(self.raw_text):
                raise ValueError("candidate source span exceeds lattice raw_text")
            if self.raw_text[span.start : span.end] != span.raw_text:
                raise ValueError("candidate source span must equal the exact raw_text slice")

        ordered = tuple(sorted(self.candidates, key=_candidate_sort_key))
        object.__setattr__(self, "candidates", ordered)


def _candidate_sort_key(candidate: RepresentationCandidate) -> tuple[object, ...]:
    span = candidate.source_span
    return (
        -candidate.confidence,
        candidate.transform_channel.value,
        span.start,
        span.end,
        candidate.candidate_text,
        candidate.orthographic_score,
        candidate.phonetic_score,
        candidate.contextual_score,
        candidate.semantic_similarity,
        candidate.outcome.value,
        candidate.provenance,
        candidate.generation_reason,
    )


def _validate_score(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or type(value) is not float:
        raise TypeError(f"{field_name} must be an exact float")
    if value == 0.0 and copysign(1.0, value) < 0.0:
        raise ValueError(f"{field_name} must not be negative zero")
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")


def _validate_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _validate_canonical_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be nonblank and have no surrounding whitespace")


def _validate_identifier(value: object, *, field_name: str) -> None:
    _validate_canonical_text(value, field_name=field_name)


def _validate_provenance(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("provenance must be an exact tuple")
    if not value:
        raise ValueError("provenance must not be empty")
    for item in value:
        _validate_canonical_text(item, field_name="provenance item")


__all__ = [
    "CandidateOutcome",
    "RepresentationCandidate",
    "RepresentationChannel",
    "RepresentationLattice",
    "SourceSpan",
]
