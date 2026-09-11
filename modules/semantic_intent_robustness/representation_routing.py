"""Semantic-hinge detection and fail-closed representation disagreement routing."""

# Standard library
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Local
from .representation import (
    RepresentationCandidate,
    SourceSpan,
    candidate_id_for,
    source_digest_for,
    validated_candidate_snapshot,
)
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
        "ask",
        "call",
        "review",
        "tell",
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
        "step",
    ),
    (
        "he",
        "her",
        "him",
        "it",
        "she",
        "that",
        "them",
        "they",
        "this",
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
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)" r"(?:[eE][+-]?\d+)?(?!\w)"
)
_APOSTROPHES = frozenset({"'", "’"})
_POLICY_SEVERITY = {
    PolicyAction.ALLOW: 0,
    PolicyAction.ALLOW_WITH_BOUNDARIES: 1,
    PolicyAction.REDIRECT: 2,
    PolicyAction.ABSTAIN: 3,
    PolicyAction.REFUSE: 4,
}


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
            or not item.startswith("representation-v1:")
            or not _is_lower_hex_digest(item.removeprefix("representation-v1:"))
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


def locate_semantic_hinges(text: str) -> tuple[SourceSpan, ...]:
    """Locate descriptive decision-sensitive spans without assigning harm labels."""

    if type(text) is not str:
        raise TypeError("text must be an exact string")
    if len(text) > _MAX_HINGE_TEXT_LENGTH:
        raise ValueError("text is too long for semantic-hinge routing")
    matches = [*_TERM_PATTERN.finditer(text), *_NUMBER_PATTERN.finditer(text)]
    raw_matches = [(match.start(), match.end()) for match in matches]
    raw_matches.extend(_unicode_name_spans(text))
    ordered = sorted(raw_matches, key=lambda item: (item[0], -(item[1] - item[0])))
    spans: list[SourceSpan] = []
    last_end = -1
    for start, end in ordered:
        if start < last_end:
            continue
        spans.append(
            SourceSpan(
                source_id=_SOURCE_ID,
                start=start,
                end=end,
                raw_text=text[start:end],
            )
        )
        last_end = end
    return tuple(spans)


def validate_representation_assessment(record: SemanticSafetyRecord) -> SemanticSafetyRecord:
    """Validate an optional representation binding before semantic decomposition."""

    if type(record) is not SemanticSafetyRecord:
        raise TypeError("record must be an exact SemanticSafetyRecord")
    values = _binding_values(record)
    if values is None:
        return record
    candidate, candidate_id, source_id, start, end, raw_text, source_document, source_digest = (
        values
    )
    candidate = validated_candidate_snapshot(candidate)
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
    if candidate_id_for(candidate) != candidate_id:
        raise ValueError("representation candidate identifier is not bound to its snapshot")
    span = candidate.source_span
    if (span.source_id, span.start, span.end, span.raw_text) != (
        source_id,
        start,
        end,
        raw_text,
    ):
        raise ValueError("representation candidate SourceSpan is not bound to the source record")
    if candidate.provenance != record.representation_provenance:
        raise ValueError("representation candidate provenance is not bound to the source record")
    assessed_text = source_document[:start] + candidate.candidate_text + source_document[end:]
    if record.prompt_text != assessed_text:
        raise ValueError("prompt_text must equal the actual assessed representation")
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

    bindings: list[tuple[RepresentationCandidate, str, str, int, int, str, str, str]] = []
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

    candidate_ids = [binding[1] for binding in bindings]
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
    permissive = {PolicyAction.ALLOW, PolicyAction.ALLOW_WITH_BOUNDARIES}
    if policies.isdisjoint(permissive):
        action = max(policies, key=_POLICY_SEVERITY.__getitem__)
        explanation = (
            "Candidate readings remain hypotheses; preserve the most restrictive assessed "
            "action because every reading is nonpermissive."
        )
    else:
        action = PolicyAction.ALLOW_WITH_BOUNDARIES
        explanation = (
            "Candidate readings remain hypotheses; use bounded caution and invite clarification "
            "because their policy assessments materially disagree."
        )
    return RepresentationDecision(
        selected_candidate_ids=selected_ids,
        disagreement=True,
        policy_action=action,
        explanation=explanation,
    )


def _binding_values(
    record: SemanticSafetyRecord,
) -> tuple[RepresentationCandidate, str, str, int, int, str, str, str] | None:
    candidate = record.representation_candidate
    candidate_id = record.representation_candidate_id
    source_id = record.representation_source_id
    start = record.representation_source_start
    end = record.representation_source_end
    raw_text = record.representation_raw_text
    source_document = record.representation_source_document
    source_digest = record.representation_source_digest
    values = (
        candidate,
        candidate_id,
        source_id,
        start,
        end,
        raw_text,
        source_document,
        source_digest,
    )
    if all(value is None for value in values) and not record.representation_provenance:
        return None
    if (
        type(candidate) is not RepresentationCandidate
        or type(candidate_id) is not str
        or type(source_id) is not str
        or type(start) is not int
        or type(end) is not int
        or type(raw_text) is not str
        or type(source_document) is not str
        or type(source_digest) is not str
        or not record.representation_provenance
    ):
        raise ValueError("assessment requires complete representation provenance")
    return candidate, candidate_id, source_id, start, end, raw_text, source_document, source_digest


def _validate_compatible_source_spans(
    bindings: list[tuple[RepresentationCandidate, str, str, int, int, str, str, str]],
) -> None:
    source_documents = {(binding[2], binding[6], binding[7]) for binding in bindings}
    if len(source_documents) != 1:
        raise ValueError("assessments must reference the same immutable source document")
    for index, left in enumerate(bindings):
        _, _, _, left_start, left_end, left_text, _, _ = left
        for right in bindings[index + 1 :]:
            _, _, _, right_start, right_end, right_text, _, _ = right
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_start >= overlap_end:
                continue
            left_overlap = left_text[overlap_start - left_start : overlap_end - left_start]
            right_overlap = right_text[overlap_start - right_start : overlap_end - right_start]
            if left_overlap != right_overlap:
                raise ValueError("assessments must reference the same immutable source span")


def _unicode_name_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not _is_letter(text[index]):
            index += 1
            continue
        start = index
        index += 1
        while index < len(text):
            character = text[index]
            if _is_letter(character) or unicodedata.category(character).startswith("M"):
                index += 1
                continue
            if (
                (character in _APOSTROPHES or character == "-")
                and index + 1 < len(text)
                and _is_letter(text[index + 1])
            ):
                index += 1
                continue
            break
        token = text[start:index]
        cased_letters = tuple(character for character in token if character.isalpha())
        is_cased_name = token[0].isupper() and (
            any(character.islower() for character in token)
            or (len(cased_letters) >= 2 and all(character.isupper() for character in cased_letters))
        )
        is_caseless_name = len(cased_letters) >= 2 and all(
            not character.islower() and not character.isupper() for character in cased_letters
        )
        if is_cased_name or is_caseless_name:
            spans.append((start, index))
    return spans


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_lower_hex_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "RepresentationDecision",
    "candidate_id_for",
    "locate_semantic_hinges",
    "route_representation_disagreement",
    "source_digest_for",
    "validate_representation_assessment",
]
