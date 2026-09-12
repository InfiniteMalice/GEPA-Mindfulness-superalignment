"""Typed diagnostic outputs for explicitly enabled experimental overlays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, TypeAlias, cast

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

from .cases.registry import CANONICAL_CASE_IDS


class ExperimentalMaturity(str, Enum):
    EXPERIMENTAL = "experimental"


class DiagnosticStatus(str, Enum):
    DIAGNOSTIC = "diagnostic"


class OrchestrationScope(str, Enum):
    GLOBAL = "global"
    FOCUS = "focus"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class _DiagnosticRecord:
    record_id: str
    source_case_id: int
    uncertainty: float
    provenance_refs: tuple[EvidenceReference, ...]
    feature_flag: str
    maturity: ExperimentalMaturity
    diagnostic_status: DiagnosticStatus

    RECORD_TYPE: ClassVar[str]
    FEATURE_FLAG: ClassVar[str]

    def __post_init__(self) -> None:
        _token(self.record_id, "record_id")
        if type(self.source_case_id) is not int or self.source_case_id not in CANONICAL_CASE_IDS:
            raise ValueError("source_case_id must name one canonical V5 case")
        _probability(self.uncertainty, "uncertainty")
        object.__setattr__(
            self,
            "provenance_refs",
            _refs(self.provenance_refs, "provenance_refs"),
        )
        if self.feature_flag != self.FEATURE_FLAG:
            raise ValueError(f"feature_flag must be {self.FEATURE_FLAG!r}")
        if self.maturity is not ExperimentalMaturity.EXPERIMENTAL:
            raise ValueError("maturity must be experimental")
        if self.diagnostic_status is not DiagnosticStatus.DIAGNOSTIC:
            raise ValueError("diagnostic_status must be diagnostic")

    def _common_dict(self) -> dict[str, object]:
        return {
            "record_type": self.RECORD_TYPE,
            "record_id": self.record_id,
            "source_case_id": self.source_case_id,
            "uncertainty": self.uncertainty,
            "provenance_refs": [item.to_dict() for item in self.provenance_refs],
            "feature_flag": self.feature_flag,
            "maturity": self.maturity.value,
            "diagnostic_status": self.diagnostic_status.value,
        }


@dataclass(frozen=True, slots=True)
class HypothesisSet(_DiagnosticRecord):
    hypotheses: tuple[str, ...]

    RECORD_TYPE = "hypothesis_set"
    FEATURE_FLAG = "competing_hypotheses"

    def __post_init__(self) -> None:
        super(HypothesisSet, self).__post_init__()
        values = _tokens(self.hypotheses, "hypotheses")
        if len(values) < 2:
            raise ValueError("hypotheses must contain at least two alternatives")
        object.__setattr__(self, "hypotheses", values)

    def to_dict(self) -> dict[str, object]:
        return {**self._common_dict(), "hypotheses": list(self.hypotheses)}


@dataclass(frozen=True, slots=True)
class InformationGainQuestion(_DiagnosticRecord):
    question: str
    expected_information_gain: float

    RECORD_TYPE = "information_gain_question"
    FEATURE_FLAG = "expected_information_gain_inquiry"

    def __post_init__(self) -> None:
        super(InformationGainQuestion, self).__post_init__()
        _token(self.question, "question")
        _finite(self.expected_information_gain, "expected_information_gain")
        if self.expected_information_gain < 0.0:
            raise ValueError("expected_information_gain must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            **self._common_dict(),
            "question": self.question,
            "expected_information_gain": self.expected_information_gain,
        }


@dataclass(frozen=True, slots=True)
class TopologyProposal(_DiagnosticRecord):
    agent_roles: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]

    RECORD_TYPE = "topology_proposal"
    FEATURE_FLAG = "adaptive_small_multi_agent_topology"

    def __post_init__(self) -> None:
        super(TopologyProposal, self).__post_init__()
        roles = _tokens(self.agent_roles, "agent_roles")
        if not 2 <= len(roles) <= 5:
            raise ValueError("agent_roles must contain two to five unique roles")
        if type(self.directed_edges) is not tuple or not self.directed_edges:
            raise ValueError("directed_edges must be a nonempty exact tuple")
        edges: list[tuple[str, str]] = []
        for value in cast(tuple[object, ...], self.directed_edges):
            if type(value) is not tuple or len(value) != 2:
                raise ValueError("directed_edges must contain exact two-item tuples")
            source = _token(value[0], "edge source")
            target = _token(value[1], "edge target")
            if source not in roles or target not in roles:
                raise ValueError("directed_edges must reference known roles")
            if source == target:
                raise ValueError("directed_edges must not contain self edges")
            edges.append((source, target))
        if len(set(edges)) != len(edges):
            raise ValueError("directed_edges must be unique")
        object.__setattr__(self, "agent_roles", roles)
        object.__setattr__(self, "directed_edges", tuple(edges))

    def to_dict(self) -> dict[str, object]:
        return {
            **self._common_dict(),
            "agent_roles": list(self.agent_roles),
            "directed_edges": [list(item) for item in self.directed_edges],
        }


@dataclass(frozen=True, slots=True)
class OrchestrationScopeDeclaration(_DiagnosticRecord):
    scope: OrchestrationScope
    declaration: str

    RECORD_TYPE = "orchestration_scope_declaration"
    FEATURE_FLAG = "declarative_orchestration_scope"

    def __post_init__(self) -> None:
        super(OrchestrationScopeDeclaration, self).__post_init__()
        if type(self.scope) is not OrchestrationScope:
            raise ValueError("scope must be an exact OrchestrationScope")
        _token(self.declaration, "declaration")

    def to_dict(self) -> dict[str, object]:
        return {
            **self._common_dict(),
            "scope": self.scope.value,
            "declaration": self.declaration,
        }


@dataclass(frozen=True, slots=True)
class MechanisticAuditReference(_DiagnosticRecord):
    audit_refs: tuple[EvidenceReference, ...]
    observation: str

    RECORD_TYPE = "mechanistic_audit_reference"
    FEATURE_FLAG = "mechanistic_circuit_audit"

    def __post_init__(self) -> None:
        super(MechanisticAuditReference, self).__post_init__()
        object.__setattr__(self, "audit_refs", _refs(self.audit_refs, "audit_refs"))
        _token(self.observation, "observation")

    def to_dict(self) -> dict[str, object]:
        return {
            **self._common_dict(),
            "audit_refs": [item.to_dict() for item in self.audit_refs],
            "observation": self.observation,
        }


ExperimentalRecord: TypeAlias = (
    HypothesisSet
    | InformationGainQuestion
    | TopologyProposal
    | OrchestrationScopeDeclaration
    | MechanisticAuditReference
)

_COMMON_FIELDS = {
    "record_type",
    "record_id",
    "source_case_id",
    "uncertainty",
    "provenance_refs",
    "feature_flag",
    "maturity",
    "diagnostic_status",
}
_TYPE_FIELDS = {
    HypothesisSet.RECORD_TYPE: {"hypotheses"},
    InformationGainQuestion.RECORD_TYPE: {"question", "expected_information_gain"},
    TopologyProposal.RECORD_TYPE: {"agent_roles", "directed_edges"},
    OrchestrationScopeDeclaration.RECORD_TYPE: {"scope", "declaration"},
    MechanisticAuditReference.RECORD_TYPE: {"audit_refs", "observation"},
}


def experimental_record_from_dict(value: object) -> ExperimentalRecord:
    """Restore one exact diagnostic record without accepting effect-bearing fields."""

    fields = _mapping(value, "experimental record")
    record_type = _token(fields.get("record_type"), "record_type")
    specific = _TYPE_FIELDS.get(record_type)
    if specific is None:
        raise ValueError(f"unknown experimental record_type {record_type!r}")
    expected = _COMMON_FIELDS | specific
    missing = sorted(expected - set(fields))
    unknown = sorted(set(fields) - expected)
    if missing or unknown:
        raise ValueError(
            f"experimental record fields are invalid; missing {missing}; unknown fields {unknown}"
        )
    common = _parse_common(fields)
    if record_type == HypothesisSet.RECORD_TYPE:
        return HypothesisSet(
            hypotheses=_restore_tokens(fields["hypotheses"], "hypotheses"), **common
        )
    if record_type == InformationGainQuestion.RECORD_TYPE:
        return InformationGainQuestion(
            question=cast(str, fields["question"]),
            expected_information_gain=cast(float, fields["expected_information_gain"]),
            **common,
        )
    if record_type == TopologyProposal.RECORD_TYPE:
        raw_edges = _list(fields["directed_edges"], "directed_edges")
        return TopologyProposal(
            agent_roles=_restore_tokens(fields["agent_roles"], "agent_roles"),
            directed_edges=cast(
                tuple[tuple[str, str], ...],
                tuple(
                    tuple(cast(str, item) for item in _list(edge, "directed edge"))
                    for edge in raw_edges
                ),
            ),
            **common,
        )
    if record_type == OrchestrationScopeDeclaration.RECORD_TYPE:
        return OrchestrationScopeDeclaration(
            scope=_enum(fields["scope"], OrchestrationScope, "scope"),
            declaration=cast(str, fields["declaration"]),
            **common,
        )
    return MechanisticAuditReference(
        audit_refs=_restore_refs(fields["audit_refs"], "audit_refs"),
        observation=cast(str, fields["observation"]),
        **common,
    )


def _parse_common(value: Mapping[str, object]) -> dict[str, Any]:
    return {
        "record_id": value["record_id"],
        "source_case_id": value["source_case_id"],
        "uncertainty": value["uncertainty"],
        "provenance_refs": _restore_refs(value["provenance_refs"], "provenance_refs"),
        "feature_flag": value["feature_flag"],
        "maturity": _enum(value["maturity"], ExperimentalMaturity, "maturity"),
        "diagnostic_status": _enum(
            value["diagnostic_status"], DiagnosticStatus, "diagnostic_status"
        ),
    }


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise ValueError(f"{context} must be an exact string-keyed mapping")
    return value


def _list(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{context} must be an exact list")
    return value


def _token(value: object, context: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{context} must be a nonblank exact string")
    return value


def _tokens(value: object, context: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{context} must be an exact tuple")
    values = tuple(_token(item, context) for item in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{context} must be nonempty and unique")
    return values


def _restore_tokens(value: object, context: str) -> tuple[str, ...]:
    return tuple(_token(item, context) for item in _list(value, context))


def _refs(value: object, context: str) -> tuple[EvidenceReference, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{context} must be an exact tuple")
    references: list[EvidenceReference] = []
    reference_type = cast(Any, EvidenceReference)
    for item in cast(tuple[object, ...], value):
        if type(item) is not EvidenceReference:
            raise ValueError(f"{context} must contain exact EvidenceReference values")
        reference = cast(EvidenceReference, item)
        if type(reference.source_kind) is not EvidenceSourceKind:
            raise ValueError(f"{context} source kinds must be exact")
        references.append(reference_type(reference.reference_id, reference.source_kind))
    if not references or not any(item.is_observable for item in references):
        raise ValueError(f"{context} requires observable provenance")
    identities = tuple((item.reference_id, item.source_kind) for item in references)
    if len(set(identities)) != len(identities):
        raise ValueError(f"{context} must be unique")
    return tuple(references)


def _restore_refs(value: object, context: str) -> tuple[EvidenceReference, ...]:
    return tuple(EvidenceReference.from_dict(item) for item in _list(value, context))


def _finite(value: object, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite built-in float")
    return value


def _probability(value: object, context: str) -> float:
    result = _finite(value, context)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be between 0 and 1")
    return result


EnumType = type[Enum]


def _enum(value: object, enum_type: EnumType, context: str) -> Any:
    if type(value) is not str:
        raise ValueError(f"{context} must be an exact string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{context} is invalid") from exc


__all__ = [
    "DiagnosticStatus",
    "ExperimentalMaturity",
    "ExperimentalRecord",
    "HypothesisSet",
    "InformationGainQuestion",
    "MechanisticAuditReference",
    "OrchestrationScope",
    "OrchestrationScopeDeclaration",
    "TopologyProposal",
    "experimental_record_from_dict",
]
