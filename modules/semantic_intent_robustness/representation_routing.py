"""Semantic-hinge detection and fail-closed representation disagreement routing."""

# Standard library
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

# Local
from .representation import RepresentationCandidate, SourceSpan
from .schemas import SemanticSafetyRecord
from .taxonomy import PolicyAction

_SOURCE_ID = "semantic-hinges"
_MAX_HINGE_TEXT_LENGTH = 100_000
_TERM_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "allow",
        "approve",
        "buy",
        "delete",
        "email",
        "execute",
        "install",
        "open",
        "proceed",
        "run",
        "send",
        "transfer",
        "write",
    ),
    (
        "account",
        "device",
        "draft",
        "file",
        "person",
        "record",
        "server",
        "system",
        "user",
    ),
    (
        "cannot",
        "can't",
        "deny",
        "do not",
        "don't",
        "never",
        "no",
        "not",
        "without",
    ),
    (
        "admin",
        "authorization",
        "authorized",
        "consent",
        "owner",
        "permission",
        "permitted",
    ),
    (
        "able",
        "access",
        "can",
        "capable",
        "command",
        "credential",
        "could",
        "may",
        "tool",
    ),
    (
        "after",
        "at least",
        "at most",
        "before",
        "except",
        "limit",
        "must",
        "only",
        "unless",
        "until",
        "within",
    ),
)
_TERM_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(term)
        for term in sorted(
            {term for group in _TERM_GROUPS for term in group},
            key=lambda value: (-len(value), value),
        )
    )
    + r")(?!\w)",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
_NAME_PATTERN = re.compile(r"(?<!\w)[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*(?!\w)")


@dataclass(frozen=True, slots=True)
class RepresentationDecision:
    """Policy routing result that keeps all assessed readings as hypotheses."""

    selected_candidate_ids: tuple[str, ...]
    disagreement: bool
    policy_action: PolicyAction
    explanation: str

    def __post_init__(self) -> None:
        if type(self.selected_candidate_ids) is not tuple or not self.selected_candidate_ids:
            raise ValueError("selected_candidate_ids must be a nonempty exact tuple")
        if any(type(item) is not str for item in self.selected_candidate_ids):
            raise TypeError("selected_candidate_ids must contain nonblank exact strings")
        if any(
            item != item.strip()
            or not item.removeprefix("representation-v1:").strip()
            or not item.startswith("representation-v1:")
            for item in self.selected_candidate_ids
        ):
            raise ValueError(
                "selected_candidate_ids must use the canonical representation-v1 namespace"
            )
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise ValueError("selected_candidate_ids must be unique")
        if type(self.disagreement) is not bool:
            raise TypeError("disagreement must be an exact bool")
        if type(self.policy_action) is not PolicyAction:
            raise TypeError("policy_action must be a PolicyAction")
        if type(self.explanation) is not str or not self.explanation.strip():
            raise TypeError("explanation must be a nonblank exact string")


def candidate_id_for(candidate: RepresentationCandidate) -> str:
    """Return a stable content identifier for a Task 1-3 candidate and its provenance."""

    if type(candidate) is not RepresentationCandidate:
        raise TypeError("candidate must be an exact RepresentationCandidate")
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


def source_digest_for(source_id: str, raw_text: str) -> str:
    """Bind a source identifier to the complete immutable lattice source text."""

    if type(source_id) is not str or not source_id.strip() or source_id != source_id.strip():
        raise ValueError("source_id must be a canonical nonblank exact string")
    if type(raw_text) is not str or not raw_text:
        raise ValueError("raw_text must be a nonempty exact string")
    payload = json.dumps(
        {"raw_text": raw_text, "source_id": source_id},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"representation-source-v1:{sha256(payload.encode('utf-8')).hexdigest()}"


def locate_semantic_hinges(text: str) -> tuple[SourceSpan, ...]:
    """Locate descriptive decision-sensitive spans without assigning harm labels."""

    if type(text) is not str:
        raise TypeError("text must be an exact string")
    if len(text) > _MAX_HINGE_TEXT_LENGTH:
        raise ValueError("text is too long for semantic-hinge routing")
    matches = [*_TERM_PATTERN.finditer(text), *_NUMBER_PATTERN.finditer(text)]
    matches.extend(_NAME_PATTERN.finditer(text))
    ordered = sorted(matches, key=lambda match: (match.start(), -(match.end() - match.start())))
    spans: list[SourceSpan] = []
    last_end = -1
    for match in ordered:
        if match.start() < last_end:
            continue
        spans.append(
            SourceSpan(
                source_id=_SOURCE_ID,
                start=match.start(),
                end=match.end(),
                raw_text=match.group(0),
            )
        )
        last_end = match.end()
    return tuple(spans)


def validate_representation_assessment(record: SemanticSafetyRecord) -> SemanticSafetyRecord:
    """Validate an optional representation binding before semantic decomposition."""

    if type(record) is not SemanticSafetyRecord:
        raise TypeError("record must be an exact SemanticSafetyRecord")
    values = _binding_values(record)
    if values is None:
        return record
    candidate_id, source_id, start, end, raw_text, source_document, source_digest = values
    if raw_text != record.representation_raw_text or end - start != len(raw_text):
        raise ValueError("representation assessment has inconsistent raw source span")
    if not candidate_id.startswith("representation-v1:") or not source_id:
        raise ValueError("representation assessment has malformed provenance")
    if source_document[start:end] != raw_text:
        raise ValueError("representation raw span conflicts with the immutable source document")
    if source_digest_for(source_id, source_document) != source_digest:
        raise ValueError(
            "representation source digest conflicts with the immutable source document"
        )
    return record


def route_representation_disagreement(
    assessments: Sequence[SemanticSafetyRecord],
    *,
    high_stakes: bool,
) -> RepresentationDecision:
    """Route policy disagreement without treating an alternate reading as established truth."""

    if type(high_stakes) is not bool:
        raise TypeError("high_stakes must be an exact bool")
    if isinstance(assessments, (str, bytes)) or not isinstance(assessments, Sequence):
        raise TypeError("assessments must be a sequence of SemanticSafetyRecord values")
    records = tuple(assessments)
    if not records:
        raise ValueError("assessments must not be empty")

    bindings: list[tuple[str, str, int, int, str, str, str]] = []
    policies: set[PolicyAction] = set()
    for record in records:
        validate_representation_assessment(record)
        binding = _binding_values(record)
        if binding is None:
            raise ValueError("assessment requires complete representation provenance")
        if type(record.policy_action) is not PolicyAction:
            raise TypeError("assessment policy_action must be a PolicyAction")
        bindings.append(binding)
        policies.add(record.policy_action)

    candidate_ids = [binding[0] for binding in bindings]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate representation candidate ID")
    contexts = {
        (
            record.semantic_cluster_id,
            record.parent_example_id,
            record.variant_type,
            record.language,
            record.turn_index,
            record.conversation_id,
        )
        for record in records
    }
    if len(contexts) != 1:
        raise ValueError("assessments must reference the same semantic assessment context")
    _validate_compatible_source_spans(bindings)

    selected_ids = tuple(sorted(candidate_ids))
    if len(policies) == 1:
        return RepresentationDecision(
            selected_candidate_ids=selected_ids,
            disagreement=False,
            policy_action=next(iter(policies)),
            explanation="Candidate assessments agree; preserve their common policy action.",
        )
    if high_stakes:
        return RepresentationDecision(
            selected_candidate_ids=selected_ids,
            disagreement=True,
            policy_action=PolicyAction.ABSTAIN,
            explanation=(
                "Candidate readings remain hypotheses; seek clarification or abstain because "
                "their policy assessments materially disagree in a high-stakes setting."
            ),
        )
    return RepresentationDecision(
        selected_candidate_ids=selected_ids,
        disagreement=True,
        policy_action=PolicyAction.ALLOW_WITH_BOUNDARIES,
        explanation=(
            "Candidate readings remain hypotheses; use bounded caution and invite clarification "
            "because their policy assessments materially disagree."
        ),
    )


def _binding_values(
    record: SemanticSafetyRecord,
) -> tuple[str, str, int, int, str, str, str] | None:
    candidate_id = record.representation_candidate_id
    source_id = record.representation_source_id
    start = record.representation_source_start
    end = record.representation_source_end
    raw_text = record.representation_raw_text
    source_document = record.representation_source_document
    source_digest = record.representation_source_digest
    values = (candidate_id, source_id, start, end, raw_text, source_document, source_digest)
    if all(value is None for value in values) and not record.representation_provenance:
        return None
    if (
        type(candidate_id) is not str
        or type(source_id) is not str
        or type(start) is not int
        or type(end) is not int
        or type(raw_text) is not str
        or type(source_document) is not str
        or type(source_digest) is not str
        or not record.representation_provenance
    ):
        raise ValueError("assessment requires complete representation provenance")
    return candidate_id, source_id, start, end, raw_text, source_document, source_digest


def _validate_compatible_source_spans(
    bindings: list[tuple[str, str, int, int, str, str, str]],
) -> None:
    source_documents = {(binding[1], binding[5], binding[6]) for binding in bindings}
    if len(source_documents) != 1:
        raise ValueError("assessments must reference the same immutable source document")
    for index, left in enumerate(bindings):
        _, _, left_start, left_end, left_text, _, _ = left
        for right in bindings[index + 1 :]:
            _, _, right_start, right_end, right_text, _, _ = right
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_start >= overlap_end:
                continue
            left_overlap = left_text[overlap_start - left_start : overlap_end - left_start]
            right_overlap = right_text[overlap_start - right_start : overlap_end - right_start]
            if left_overlap != right_overlap:
                raise ValueError("assessments must reference the same immutable source span")


__all__ = [
    "RepresentationDecision",
    "candidate_id_for",
    "locate_semantic_hinges",
    "route_representation_disagreement",
    "source_digest_for",
    "validate_representation_assessment",
]
