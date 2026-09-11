"""Bounded orthographic and phonological representation candidates."""

# Standard library
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace

# Local
from .representation import (
    CandidateOutcome,
    RepresentationCandidate,
    RepresentationChannel,
    RepresentationLattice,
    SourceSpan,
)
from .representation_views import conservative_views

_EVIDENCE_FLOOR = 0.70
_MAX_SOURCE_LENGTH = 100_000
_MAX_LEXICON_ENTRIES = 10_000
_MAX_LEXICON_TEXT_LENGTH = 256
_COMPARISONS_PER_OUTPUT_SLOT = 64
_NEGATION_HINGES = frozenset({"no", "not", "never", "neither", "nor", "without"})
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class CandidateBudget:
    """Hard caps applied to every returned candidate lattice."""

    max_spans: int = 8
    max_candidates_per_span: int = 4
    max_candidates_total: int = 24

    def __post_init__(self) -> None:
        for field_name in (
            "max_spans",
            "max_candidates_per_span",
            "max_candidates_total",
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an exact integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(slots=True)
class _GenerationLimiter:
    comparisons_remaining: int
    proposals_remaining: int

    def claim_comparison(self) -> bool:
        if self.comparisons_remaining <= 0:
            return False
        self.comparisons_remaining -= 1
        return True

    def claim_proposal(self) -> bool:
        if self.proposals_remaining <= 0:
            return False
        self.proposals_remaining -= 1
        return True


def build_candidate_lattice(
    source_id: str,
    raw_text: str,
    *,
    context: tuple[str, ...] = (),
    orthographic_lexicon: tuple[str, ...] = (),
    phonetic_lexicon: Mapping[str, tuple[str, ...]] | None = None,
    budget: CandidateBudget = CandidateBudget(),
) -> RepresentationLattice:
    """Build a deterministic lattice without treating any repair as intended truth."""

    _validate_source_id(source_id)
    _validate_raw_text(raw_text)
    context_snapshot = _snapshot_string_tuple(context, field_name="context")
    orthographic_snapshot = _snapshot_string_tuple(
        orthographic_lexicon,
        field_name="orthographic_lexicon",
        sort_and_deduplicate=True,
    )
    phonetic_snapshot = _snapshot_phonetic_lexicon(phonetic_lexicon)
    if type(budget) is not CandidateBudget:
        raise TypeError("budget must be an exact CandidateBudget")

    views = conservative_views(source_id, raw_text)
    generated = _generate_candidates(
        source_id,
        raw_text,
        context_snapshot,
        orthographic_snapshot,
        phonetic_snapshot,
        budget,
    )
    selected = _apply_budget(views, generated, budget)
    if not generated:
        selected = _mark_explicit_no_repair(selected)

    return RepresentationLattice(
        source_id=source_id,
        raw_text=raw_text,
        candidates=selected,
        max_candidates=budget.max_candidates_total,
    )


def _generate_candidates(
    source_id: str,
    raw_text: str,
    context: tuple[str, ...],
    orthographic_lexicon: tuple[str, ...],
    phonetic_lexicon: tuple[tuple[str, tuple[str, ...]], ...],
    budget: CandidateBudget,
) -> tuple[RepresentationCandidate, ...]:
    tokens = tuple(_WORD_PATTERN.finditer(raw_text))
    proposals: list[RepresentationCandidate] = []
    limiter = _GenerationLimiter(
        comparisons_remaining=(budget.max_candidates_total * _COMPARISONS_PER_OUTPUT_SLOT),
        proposals_remaining=budget.max_candidates_total * 4,
    )
    for observed, alternatives in phonetic_lexicon:
        if limiter.comparisons_remaining <= 0 or limiter.proposals_remaining <= 0:
            break
        for start, end in _exact_phrase_spans(raw_text, tokens, observed, limiter):
            source_text = raw_text[start:end]
            for alternative in alternatives:
                if not limiter.claim_proposal():
                    break
                candidate_text = _apply_case_pattern(source_text, alternative)
                if candidate_text == source_text:
                    continue
                contextual_score = _contextual_score(candidate_text, context)
                confidence = min(0.89, 0.83 + 0.06 * contextual_score)
                provenance = [f"phonetic-lexicon:{observed}"]
                if contextual_score > 0.5:
                    provenance.append("context-evidence:token-overlap")
                proposals.append(
                    _candidate(
                        source_id=source_id,
                        raw_text=raw_text,
                        start=start,
                        end=end,
                        candidate_text=candidate_text,
                        channel=RepresentationChannel.PHONOLOGICAL,
                        orthographic_score=0.0,
                        phonetic_score=0.9,
                        contextual_score=contextual_score,
                        confidence=confidence,
                        provenance=tuple(provenance),
                        reason="Retain an injected phonological hypothesis for evaluation.",
                    )
                )
    for lexicon_text in orthographic_lexicon:
        if limiter.comparisons_remaining <= 0 or limiter.proposals_remaining <= 0:
            break
        proposals.extend(
            _orthographic_candidates(
                source_id,
                raw_text,
                tokens,
                lexicon_text,
                context,
                limiter,
            )
        )
    return _deduplicate_candidates(proposals)


def _orthographic_candidates(
    source_id: str,
    raw_text: str,
    tokens: tuple[re.Match[str], ...],
    lexicon_text: str,
    context: tuple[str, ...],
    limiter: _GenerationLimiter,
) -> list[RepresentationCandidate]:
    lexicon_words = tuple(_WORD_PATTERN.finditer(lexicon_text))
    if not lexicon_words or "".join(match.group() for match in lexicon_words) != re.sub(
        r"\s+", "", lexicon_text
    ):
        return []

    word_count = len(lexicon_words)
    candidate_text = lexicon_text
    candidates: list[RepresentationCandidate] = []
    for index in range(max(0, len(tokens) - word_count + 1)):
        if not limiter.claim_comparison():
            break
        window = tokens[index : index + word_count]
        start = window[0].start()
        end = window[-1].end()
        observed = raw_text[start:end]
        if not _orthographic_span_is_eligible(observed, candidate_text, word_count):
            continue
        distance = _bounded_edit_distance(
            _comparison_form(observed),
            _comparison_form(candidate_text),
            _distance_limit(observed, candidate_text),
        )
        if distance is None or distance == 0:
            continue
        denominator = max(len(_comparison_form(observed)), len(_comparison_form(candidate_text)))
        score = 1.0 - distance / denominator
        contextual_score = _contextual_score(candidate_text, context)
        confidence = min(0.96, 0.70 + 0.24 * score + 0.02 * contextual_score)
        if confidence < _EVIDENCE_FLOOR:
            continue
        if not limiter.claim_proposal():
            break
        rendered = _apply_case_pattern(observed, candidate_text)
        provenance = [f"orthographic-lexicon:{lexicon_text}"]
        if contextual_score > 0.5:
            provenance.append("context-evidence:token-overlap")
        candidates.append(
            _candidate(
                source_id=source_id,
                raw_text=raw_text,
                start=start,
                end=end,
                candidate_text=rendered,
                channel=RepresentationChannel.ORTHOGRAPHIC,
                orthographic_score=score,
                phonetic_score=0.0,
                contextual_score=contextual_score,
                confidence=confidence,
                provenance=tuple(provenance),
                reason="Retain a bounded edit-distance hypothesis for evaluation.",
            )
        )
    return candidates


def _orthographic_span_is_eligible(observed: str, candidate: str, word_count: int) -> bool:
    observed_folded = observed.casefold()
    candidate_folded = candidate.casefold()
    if observed_folded == candidate_folded:
        return False
    if observed_folded.isdigit() or any(character.isdigit() for character in candidate):
        return False
    if observed_folded in _NEGATION_HINGES:
        return False
    if word_count == 1 and observed.istitle():
        return False
    if re.sub(r"\s+", "", observed_folded) == re.sub(r"\s+", "", candidate_folded):
        return False
    return True


def _candidate(
    *,
    source_id: str,
    raw_text: str,
    start: int,
    end: int,
    candidate_text: str,
    channel: RepresentationChannel,
    orthographic_score: float,
    phonetic_score: float,
    contextual_score: float,
    confidence: float,
    provenance: tuple[str, ...],
    reason: str,
) -> RepresentationCandidate:
    return RepresentationCandidate(
        source_span=SourceSpan(
            source_id=source_id,
            start=start,
            end=end,
            raw_text=raw_text[start:end],
        ),
        candidate_text=candidate_text,
        transform_channel=channel,
        orthographic_score=float(orthographic_score),
        phonetic_score=float(phonetic_score),
        contextual_score=float(contextual_score),
        semantic_similarity=0.5,
        confidence=float(confidence),
        provenance=provenance,
        generation_reason=reason,
    )


def _apply_budget(
    views: tuple[RepresentationCandidate, ...],
    generated: tuple[RepresentationCandidate, ...],
    budget: CandidateBudget,
) -> tuple[RepresentationCandidate, ...]:
    ordered = tuple(sorted((*views, *generated), key=_selection_key))
    selected: list[RepresentationCandidate] = []
    span_counts: dict[tuple[int, int], int] = {}
    selected_spans: set[tuple[int, int]] = set()
    for candidate in ordered:
        if len(selected) >= budget.max_candidates_total:
            break
        span_key = (candidate.source_span.start, candidate.source_span.end)
        if span_key not in selected_spans and len(selected_spans) >= budget.max_spans:
            continue
        if span_counts.get(span_key, 0) >= budget.max_candidates_per_span:
            continue
        selected.append(candidate)
        selected_spans.add(span_key)
        span_counts[span_key] = span_counts.get(span_key, 0) + 1
    return tuple(selected)


def _mark_explicit_no_repair(
    candidates: tuple[RepresentationCandidate, ...],
) -> tuple[RepresentationCandidate, ...]:
    marked: list[RepresentationCandidate] = []
    did_mark = False
    for candidate in candidates:
        if not did_mark and candidate.transform_channel is RepresentationChannel.LITERAL:
            marked.append(
                replace(
                    candidate,
                    provenance=(
                        *candidate.provenance,
                        "candidate-generation:no-evidence-above-floor",
                    ),
                    generation_reason=(
                        "Preserve the literal source; no alternate exceeded the evidence floor."
                    ),
                    outcome=CandidateOutcome.NO_REPAIR,
                )
            )
            did_mark = True
        else:
            marked.append(candidate)
    return tuple(marked)


def _deduplicate_candidates(
    candidates: list[RepresentationCandidate],
) -> tuple[RepresentationCandidate, ...]:
    selected: dict[tuple[int, int, str], RepresentationCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.source_span.start,
            candidate.source_span.end,
            candidate.candidate_text,
        )
        incumbent = selected.get(key)
        if incumbent is None:
            selected[key] = candidate
        else:
            selected[key] = _merge_candidate_evidence(incumbent, candidate)
    return tuple(sorted(selected.values(), key=_selection_key))


def _merge_candidate_evidence(
    first: RepresentationCandidate,
    second: RepresentationCandidate,
) -> RepresentationCandidate:
    preferred = min((first, second), key=_selection_key)
    return replace(
        preferred,
        orthographic_score=max(first.orthographic_score, second.orthographic_score),
        phonetic_score=max(first.phonetic_score, second.phonetic_score),
        contextual_score=max(first.contextual_score, second.contextual_score),
        semantic_similarity=max(first.semantic_similarity, second.semantic_similarity),
        confidence=max(first.confidence, second.confidence),
        provenance=tuple(sorted(set((*first.provenance, *second.provenance)))),
        generation_reason="Multiple bounded evidence channels produced this hypothesis.",
    )


def _selection_key(candidate: RepresentationCandidate) -> tuple[object, ...]:
    span = candidate.source_span
    return (
        candidate.transform_channel is not RepresentationChannel.LITERAL,
        -candidate.confidence,
        candidate.transform_channel.value,
        span.start,
        span.end,
        candidate.candidate_text,
        candidate.provenance,
    )


def _exact_phrase_spans(
    raw_text: str,
    tokens: tuple[re.Match[str], ...],
    observed: str,
    limiter: _GenerationLimiter,
) -> tuple[tuple[int, int], ...]:
    observed_tokens = tuple(_WORD_PATTERN.finditer(observed))
    if not observed_tokens:
        return ()
    count = len(observed_tokens)
    spans: list[tuple[int, int]] = []
    for index in range(max(0, len(tokens) - count + 1)):
        if not limiter.claim_comparison():
            break
        window = tokens[index : index + count]
        start = window[0].start()
        end = window[-1].end()
        if raw_text[start:end].casefold() == observed.casefold():
            spans.append((start, end))
    return tuple(spans)


def _bounded_edit_distance(left: str, right: str, limit: int) -> int | None:
    """Return optimal-string-alignment distance, stopping beyond ``limit``."""

    if abs(len(left) - len(right)) > limit:
        return None
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_character in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_character != right_character)
            value = min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                substitution,
            )
            if (
                previous_previous is not None
                and left_index > 1
                and right_index > 1
                and left_character == right[right_index - 2]
                and left[left_index - 2] == right_character
            ):
                value = min(value, previous_previous[right_index - 2] + 1)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > limit:
            return None
        previous_previous, previous = previous, current
    distance = previous[-1]
    return distance if distance <= limit else None


def _distance_limit(left: str, right: str) -> int:
    longest = max(len(left), len(right))
    if longest <= 4:
        return 1
    if longest <= 8:
        return 2
    return 3


def _comparison_form(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _apply_case_pattern(source: str, candidate: str) -> str:
    if source.isupper():
        return candidate.upper()
    if source.istitle():
        return candidate.title()
    return candidate


def _contextual_score(candidate: str, context: tuple[str, ...]) -> float:
    if not context:
        return 0.5
    candidate_words = {match.group().casefold() for match in _WORD_PATTERN.finditer(candidate)}
    context_words = {
        match.group().casefold() for item in context for match in _WORD_PATTERN.finditer(item)
    }
    return 0.75 if candidate_words & context_words else 0.5


def _snapshot_string_tuple(
    value: object,
    *,
    field_name: str,
    sort_and_deduplicate: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    if len(value) > _MAX_LEXICON_ENTRIES:
        raise ValueError(f"{field_name} has too many entries")
    snapshot: list[str] = []
    for item in value:
        _validate_canonical_lexicon_text(item, field_name=f"{field_name} item")
        snapshot.append(item)
    if sort_and_deduplicate:
        return tuple(sorted(set(snapshot), key=lambda item: (item.casefold(), item)))
    return tuple(snapshot)


def _snapshot_phonetic_lexicon(
    value: Mapping[str, tuple[str, ...]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("phonetic_lexicon must be a mapping or None")
    items = tuple(value.items())
    if len(items) > _MAX_LEXICON_ENTRIES:
        raise ValueError("phonetic_lexicon has too many entries")
    snapshot: list[tuple[str, tuple[str, ...]]] = []
    for observed, alternatives in items:
        _validate_canonical_lexicon_text(observed, field_name="phonetic key")
        candidates = _snapshot_string_tuple(
            alternatives,
            field_name="phonetic alternatives",
            sort_and_deduplicate=True,
        )
        if not candidates:
            raise ValueError("phonetic alternatives must not be empty")
        snapshot.append((observed, candidates))
    return tuple(sorted(snapshot, key=lambda item: (item[0], item[1])))


def _validate_source_id(value: object) -> None:
    if type(value) is not str:
        raise TypeError("source_id must be an exact string")
    if not value.strip() or value != value.strip():
        raise ValueError("source_id must be canonical and nonblank")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("source_id must use NFC Unicode")


def _validate_raw_text(value: object) -> None:
    if type(value) is not str:
        raise TypeError("raw_text must be an exact string")
    if not value:
        raise ValueError("raw_text must not be empty")
    if len(value) > _MAX_SOURCE_LENGTH:
        raise ValueError("raw_text is too long")


def _validate_canonical_lexicon_text(value: object, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be canonical and nonblank")
    if len(value) > _MAX_LEXICON_TEXT_LENGTH:
        raise ValueError(f"{field_name} is too long")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must use NFC Unicode")


__all__ = ["CandidateBudget", "build_candidate_lattice"]
