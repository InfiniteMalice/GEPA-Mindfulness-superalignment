"""Typed destinations and immutable proposals for controlled learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

EnumT = TypeVar("EnumT", bound=Enum)


class LearningSurface(str, Enum):
    """The single primary destination selected for a lesson."""

    TRACE_ONLY = "trace_only"
    MEMORY = "memory"
    HARNESS = "harness"
    SKILL_GRAPH = "skill_graph"
    MODEL = "model"
    HUMAN = "human"


class LessonKind(str, Enum):
    """Evidence characteristics that determine an automatic learning destination."""

    ONE_OFF_OBSERVATION = "one_off_observation"
    EPISODE_FACT = "episode_fact"
    STABLE_PROCEDURAL_CONVENTION = "stable_procedural_convention"
    REUSABLE_DEPENDENCY = "reusable_dependency"
    PERSISTENT_INTRINSIC_BEHAVIOR = "persistent_intrinsic_behavior"


class LessonReviewStatus(str, Enum):
    """The explicit human review decision attached to a proposal.

    Pending proposals have no review decision, approved proposals may be considered for their
    declared destination, and rejected proposals must not be applied.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class LessonCharacteristics:
    """Typed facts used by the routing decision table instead of lesson prose."""

    kind: LessonKind
    normative: bool = False
    ambiguous: bool = False
    difficult_to_reverse: bool = False

    def __post_init__(self) -> None:
        _validate_characteristics_fields(self)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible characteristics record."""

        snapshot = _snapshot_characteristics(self)
        return {
            "kind": snapshot.kind.value,
            "normative": snapshot.normative,
            "ambiguous": snapshot.ambiguous,
            "difficult_to_reverse": snapshot.difficult_to_reverse,
        }

    @classmethod
    def from_dict(cls, data: object) -> LessonCharacteristics:
        """Restore characteristics from their exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {"kind", "normative", "ambiguous", "difficult_to_reverse"},
            "LessonCharacteristics",
        )
        return cls(
            kind=_parse_exact_enum(values["kind"], LessonKind, "kind"),
            normative=_require_exact_bool(values["normative"], "normative"),
            ambiguous=_require_exact_bool(values["ambiguous"], "ambiguous"),
            difficult_to_reverse=_require_exact_bool(
                values["difficult_to_reverse"],
                "difficult_to_reverse",
            ),
        )


_DESTINATION_BY_KIND = {
    LessonKind.ONE_OFF_OBSERVATION: LearningSurface.TRACE_ONLY,
    LessonKind.EPISODE_FACT: LearningSurface.MEMORY,
    LessonKind.STABLE_PROCEDURAL_CONVENTION: LearningSurface.HARNESS,
    LessonKind.REUSABLE_DEPENDENCY: LearningSurface.SKILL_GRAPH,
    LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR: LearningSurface.MODEL,
}


def classify_learning_surface(characteristics: LessonCharacteristics) -> LearningSurface:
    """Select one destination from explicit typed characteristics."""

    snapshot = _snapshot_characteristics(characteristics)
    if snapshot.normative or snapshot.ambiguous or snapshot.difficult_to_reverse:
        return LearningSurface.HUMAN
    return _DESTINATION_BY_KIND[snapshot.kind]


@dataclass(frozen=True, slots=True)
class LessonProposal:
    """An immutable, evidence-bound proposal with one primary destination."""

    lesson_id: str
    summary: str
    primary_destination: LearningSurface
    evidence_refs: tuple[EvidenceReference, ...]
    rationale: str
    reversible: bool
    review_status: LessonReviewStatus

    def __post_init__(self) -> None:
        _require_nonblank_string(self.lesson_id, "lesson_id")
        _require_nonblank_string(self.summary, "summary")
        if type(self.primary_destination) is not LearningSurface:
            raise ValueError("primary_destination must be exactly one LearningSurface")
        references = _snapshot_evidence_refs(self.evidence_refs)
        if not any(reference.is_observable for reference in references):
            raise ValueError("LessonProposal requires observable evidence_refs")
        reference_ids = {reference.reference_id for reference in references}
        if len(reference_ids) != len(references):
            raise ValueError("evidence_refs must have unique reference_id values")
        _require_nonblank_string(self.rationale, "rationale")
        _require_exact_bool(self.reversible, "reversible")
        if type(self.review_status) is not LessonReviewStatus:
            raise ValueError("review_status must be an exact LessonReviewStatus")
        object.__setattr__(self, "evidence_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible proposal snapshot."""

        snapshot = _snapshot_proposal(self)
        return {
            "lesson_id": snapshot.lesson_id,
            "summary": snapshot.summary,
            "primary_destination": snapshot.primary_destination.value,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "rationale": snapshot.rationale,
            "reversible": snapshot.reversible,
            "review_status": snapshot.review_status.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> LessonProposal:
        """Restore a proposal from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "lesson_id",
                "summary",
                "primary_destination",
                "evidence_refs",
                "rationale",
                "reversible",
                "review_status",
            },
            "LessonProposal",
        )
        raw_references = values["evidence_refs"]
        if type(raw_references) is not list:
            raise ValueError("LessonProposal evidence_refs must be an array")
        reference_items = cast(list[object], raw_references)
        return cls(
            lesson_id=cast(str, values["lesson_id"]),
            summary=cast(str, values["summary"]),
            primary_destination=_parse_exact_enum(
                values["primary_destination"],
                LearningSurface,
                "primary_destination",
            ),
            evidence_refs=tuple(EvidenceReference.from_dict(item) for item in reference_items),
            rationale=cast(str, values["rationale"]),
            reversible=_require_exact_bool(values["reversible"], "reversible"),
            review_status=_parse_exact_enum(
                values["review_status"],
                LessonReviewStatus,
                "review_status",
            ),
        )


def _validate_characteristics_fields(value: LessonCharacteristics) -> None:
    if type(value.kind) is not LessonKind:
        raise ValueError("kind must be an exact LessonKind")
    _require_exact_bool(value.normative, "normative")
    _require_exact_bool(value.ambiguous, "ambiguous")
    _require_exact_bool(value.difficult_to_reverse, "difficult_to_reverse")


def _snapshot_characteristics(value: object) -> LessonCharacteristics:
    if type(value) is not LessonCharacteristics:
        raise ValueError("characteristics must be an exact LessonCharacteristics")
    _validate_characteristics_fields(value)
    return LessonCharacteristics(
        kind=value.kind,
        normative=value.normative,
        ambiguous=value.ambiguous,
        difficult_to_reverse=value.difficult_to_reverse,
    )


def _snapshot_proposal(value: object) -> LessonProposal:
    if type(value) is not LessonProposal:
        raise ValueError("proposal must be an exact LessonProposal")
    if type(value.evidence_refs) is not tuple:
        raise ValueError("LessonProposal evidence_refs must remain an exact tuple")
    return LessonProposal(
        lesson_id=value.lesson_id,
        summary=value.summary,
        primary_destination=value.primary_destination,
        evidence_refs=value.evidence_refs,
        rationale=value.rationale,
        reversible=value.reversible,
        review_status=value.review_status,
    )


def _snapshot_evidence_refs(values: object) -> tuple[EvidenceReference, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError("evidence_refs must be an exact tuple or list")
    items = cast(tuple[object, ...] | list[object], values)
    references: list[EvidenceReference] = []
    for reference in items:
        if type(reference) is not EvidenceReference:
            raise ValueError("evidence_refs must contain exact EvidenceReference values")
        checked_reference = cast(EvidenceReference, reference)
        _require_nonblank_string(checked_reference.reference_id, "evidence_refs reference_id")
        if type(checked_reference.source_kind) is not EvidenceSourceKind:
            raise ValueError("evidence_refs source_kind must be an exact EvidenceSourceKind")
        references.append(
            EvidenceReference(checked_reference.reference_id, checked_reference.source_kind)
        )
    return tuple(references)


def _require_nonblank_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank built-in string")
    return value


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be an exact bool")
    return value


def _parse_exact_enum(
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a built-in string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} has unknown value {value!r}") from exc


def _require_exact_mapping(
    data: object,
    expected_fields: set[str],
    record_name: str,
) -> Mapping[str, object]:
    if type(data) is not dict:
        raise ValueError(f"{record_name} must be an exact object")
    if set(data) != expected_fields:
        raise ValueError(f"{record_name} requires exactly {sorted(expected_fields)!r}")
    return cast(Mapping[str, object], data)


__all__ = [
    "LearningSurface",
    "LessonCharacteristics",
    "LessonKind",
    "LessonProposal",
    "LessonReviewStatus",
    "classify_learning_surface",
]
