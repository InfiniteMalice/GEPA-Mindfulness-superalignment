"""Derived metacognitive diagnostics; no evidence, reward, execution or storage authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from math import isfinite
from typing import Any, TypeVar, cast

from evaluation.experimental_records import DiagnosticStatus
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

from .epistemic_state import EpistemicContext, EpistemicStateEstimate
from .hypothesis_records import HypothesisState
from .state import EvidenceState

SCHEMA_VERSION = "metacognitive-v1"
_T = TypeVar("_T")


def text(value: object, name: str, maximum: int = 512) -> str:
    """Validate bounded public metadata without coercion."""
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} requires nonblank text of at most {maximum} UTF-8 bytes")
    return value


def strings(value: object, name: str, maximum: int = 1024) -> tuple[str, ...]:
    """Require a bounded immutable sequence of unique IDs."""
    if type(value) is not tuple or len(value) > maximum:
        raise ValueError(f"{name} requires an exact tuple of at most {maximum} items")
    result = tuple(text(v, name) for v in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} requires unique values")
    return result


def unit(value: object, name: str) -> float | None:
    """Preserve unknowns and reject nonfinite, boolean or out-of-range measurements."""
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{name} requires a finite unit measurement or null")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def snapshot(value: _T, cls: type[_T]) -> _T:
    """Reconstruct exact dataclasses, revalidating rather than trusting frozen state."""
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls(**{f.name: getattr(value, f.name) for f in fields(cast(Any, cls))})


def records(value: object, cls: type[_T], maximum: int = 512) -> tuple[_T, ...]:
    """Detach a bounded immutable record sequence."""
    if type(value) is not tuple or len(value) > maximum:
        raise ValueError(f"{cls.__name__} records require a bounded exact tuple")
    return tuple(snapshot(v, cls) for v in value)


def refs(value: object) -> tuple[EvidenceReference, ...]:
    """Require observable provenance, preserving unique source identities and kinds."""
    result = records(value, EvidenceReference, 4096)
    if any(type(r.source_kind) is not EvidenceSourceKind or not r.is_observable for r in result):
        raise ValueError("diagnostic support requires observable evidence references")
    if len({r.reference_id for r in result}) != len(result):
        raise ValueError("evidence reference IDs must be unique")
    return result


def exact(value: object, names: set[str]) -> dict[str, Any]:
    """Parse an exact JSON object without silently dropping unknown fields."""
    if type(value) is not dict or set(value) != names:
        raise ValueError("record fields must match the supported schema exactly")
    return dict(value)


def array(value: object) -> tuple[Any, ...]:
    """Parse a bounded JSON array at the wire boundary."""
    if type(value) is not list or len(value) > 4096:
        raise ValueError("expected a bounded JSON array")
    return tuple(value)


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _Record):
        return value.to_dict()
    if type(value) is V5EvaluationCell:
        return {f.name: getattr(value, f.name) for f in fields(V5EvaluationCell)}
    if type(value) is tuple:
        return [_encode(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return value.to_dict()


@dataclass(frozen=True, slots=True)
class _Record:
    """Versioned diagnostic serialization; construction confers no verification."""

    def to_dict(self) -> dict[str, Any]:
        """Export detached JSON after recursive revalidation."""
        checked = snapshot(self, type(self))
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": type(self).__name__,
            **{f.name: _encode(getattr(checked, f.name)) for f in fields(checked)},
        }

    @classmethod
    def from_dict(cls: type[_T], value: object) -> _T:
        """Restore only the exact supported schema and nested record types."""
        names = {f.name for f in fields(cast(Any, cls))} | {"schema_version", "record_type"}
        data = exact(value, names)
        if data.pop("schema_version") != SCHEMA_VERSION or data.pop("record_type") != cls.__name__:
            raise ValueError("unsupported metacognitive schema or record type")
        for name, decoder in _DECODERS.get(cls.__name__, {}).items():
            data[name] = decoder(data[name])
        return cls(**data)


class AssessmentStatus(str, Enum):
    """Monitoring lifecycle, separate from existing evidence truth statuses."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNDER_REVIEW = "under_review"
    SUPERSEDED = "superseded"


class FindingKind(str, Enum):
    """Inspective findings used to propose diagnostic methods, never rewards."""

    UNVERIFIED_ASSUMPTION = "unverified_assumption"
    CONTRADICTED_ASSUMPTION = "contradicted_assumption"
    OUT_OF_SCOPE_ASSUMPTION = "out_of_scope_assumption"
    SHARED_DEPENDENCY = "shared_dependency"
    CONTRADICTION = "contradiction"
    EVIDENCE_CONTINUITY = "evidence_continuity"
    CONFIDENCE_GAP = "confidence_gap"
    REPEATED_FAILURE = "repeated_failure"
    STRATEGY_OUTCOME_UNVERIFIED = "strategy_outcome_unverified"
    AMBIGUOUS_INSTRUCTION = "ambiguous_instruction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNNECESSARY_ABSTENTION = "unnecessary_abstention"


@dataclass(frozen=True, slots=True)
class UncertaintyDimensions(_Record):
    """Independent declared diagnostics; missingness is never zero uncertainty."""

    missing_evidence: float | None = None
    conflicting_evidence: float | None = None
    model: float | None = None
    environmental: float | None = None
    ambiguous_instructions: float | None = None
    unverified_assumptions: float | None = None
    unreliable_verification_sources: float | None = None
    monitor: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, unit(getattr(self, field.name), field.name))


@dataclass(frozen=True, slots=True)
class AssumptionBinding(_Record):
    """A hypothesis depends on an existing claim under existing condition claims."""

    hypothesis_id: str
    claim_id: str
    applicability_claim_ids: tuple[str, ...] = ()
    alternative_hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        text(self.hypothesis_id, "hypothesis_id")
        text(self.claim_id, "claim_id")
        for name in ("applicability_claim_ids", "alternative_hypothesis_ids"):
            object.__setattr__(self, name, strings(getattr(self, name), name, 64))
        if self.hypothesis_id in self.alternative_hypothesis_ids:
            raise ValueError("an alternative must identify another hypothesis")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceSnapshot(_Record):
    """Detached original records supplied by the host; not another evidence store.

    Event ID existence, causal chronology and stripe association require the PEO adapter and
    authenticated host episode. These inert contracts validate declared identities only.
    """

    context: EpistemicContext
    cell: V5EvaluationCell
    evidence_state: EvidenceState
    hypotheses: HypothesisState | None
    epistemic: EpistemicStateEstimate | None
    assumptions: tuple[AssumptionBinding, ...]
    source_event_ids: tuple[str, ...]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", snapshot(self.context, EpistemicContext))
        object.__setattr__(self, "cell", snapshot(self.cell, V5EvaluationCell))
        if (
            self.context.repeat_id,
            self.context.system.model_version,
            self.context.system.harness_version,
        ) != (self.cell.repeat_id, self.cell.model_version, self.cell.harness_version):
            raise ValueError("source context and evaluation cell must match")
        if type(self.evidence_state) is not EvidenceState or len(self.evidence_state.claims) > 512:
            raise ValueError("source evidence requires an exact bounded EvidenceState")
        object.__setattr__(
            self, "evidence_state", EvidenceState.from_dict(self.evidence_state.to_dict())
        )
        if self.hypotheses is not None:
            if type(self.hypotheses) is not HypothesisState:
                raise ValueError("expected exact HypothesisState")
            h = HypothesisState.from_dict(self.hypotheses.to_dict())
            if h.context != self.context or h.source_case_id != self.cell.case_id:
                raise ValueError("hypothesis evaluation identity must match")
            if len(h.hypotheses) > 64 or len(h.assessments) > 512 or len(h.triggers) > 64:
                raise ValueError("hypothesis source history exceeds monitoring bounds")
            object.__setattr__(self, "hypotheses", h)
        if self.epistemic is not None:
            e = snapshot(self.epistemic, EpistemicStateEstimate)
            if e.context != self.context:
                raise ValueError("epistemic evaluation identity must match")
            object.__setattr__(self, "epistemic", e)
        object.__setattr__(self, "assumptions", records(self.assumptions, AssumptionBinding, 256))
        claims = {c.claim_id for c in self.evidence_state.claims}
        hypotheses = {h.id for h in self.hypotheses.hypotheses} if self.hypotheses else set()
        seen = set()
        for binding in self.assumptions:
            if (
                binding.hypothesis_id not in hypotheses
                or binding.claim_id not in claims
                or not set(binding.applicability_claim_ids) <= claims
                or not set(binding.alternative_hypothesis_ids) <= hypotheses
            ):
                raise ValueError("assumption links must resolve in retained sources")
            key = (binding.hypothesis_id, binding.claim_id)
            if key in seen:
                raise ValueError("duplicate assumption binding")
            seen.add(key)
        for name in ("source_event_ids", "provenance"):
            values = strings(getattr(self, name), name, 1024 if name == "source_event_ids" else 32)
            if not values:
                raise ValueError(f"{name} must retain at least one source")
            object.__setattr__(self, name, values)
        refs(self.evidence_refs)

    @property
    def evidence_refs(self) -> tuple[EvidenceReference, ...]:
        """All retained references; conflicting source-kind relabels are rejected."""
        sources = list(self.evidence_state.claims)
        if self.hypotheses:
            sources.extend(self.hypotheses.hypotheses)
            sources.extend(self.hypotheses.assessments)
            sources.extend(self.hypotheses.triggers)
        if self.epistemic:
            sources.append(self.epistemic)
        by_id: dict[str, EvidenceReference] = {}
        for source in sources:
            for ref in source.evidence_refs:
                if ref.reference_id in by_id and ref != by_id[ref.reference_id]:
                    raise ValueError("ambiguous evidence source kind")
                by_id[ref.reference_id] = ref
        return tuple(by_id.values())


@dataclass(frozen=True, slots=True)
class SharedDependency(_Record):
    """Common premise or source; sharing does not prove falsehood or independence."""

    dependency_id: str
    kind: str
    hypothesis_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        text(self.dependency_id, "dependency_id")
        if type(self.kind) is not str or self.kind not in {"assumption", "evidence"}:
            raise ValueError("unsupported dependency kind")
        object.__setattr__(self, "hypothesis_ids", strings(self.hypothesis_ids, "hypotheses", 64))
        if len(self.hypothesis_ids) < 2:
            raise ValueError("shared dependencies require at least two alternatives")
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitoringFinding(_Record):
    """A diagnostic conclusion with inspectable support and an unverified next step."""

    finding_id: str
    kind: FindingKind
    claim_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    source_event_ids: tuple[str, ...]
    uncertainty: float | None
    next_step: str
    verification_status: str = "unverified"

    def __post_init__(self) -> None:
        text(self.finding_id, "finding_id")
        text(self.next_step, "next_step")
        if type(self.kind) is not FindingKind:
            raise ValueError("finding kind must be an exact FindingKind")
        if type(self.verification_status) is not str or self.verification_status not in {
            "unverified",
            "unavailable",
        }:
            raise ValueError("a diagnostic finding cannot self-certify verification")
        for name in ("claim_ids", "hypothesis_ids", "source_event_ids"):
            object.__setattr__(self, name, strings(getattr(self, name), name))
        if not self.source_event_ids:
            raise ValueError("a finding must identify source events")
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))
        object.__setattr__(self, "uncertainty", unit(self.uncertainty, "uncertainty"))


@dataclass(frozen=True, slots=True, kw_only=True)
class MetacognitiveAssessment(_Record):
    """Derived immutable view; never an accepted evidence update or authority token."""

    assessment_id: str
    sources: SourceSnapshot
    method_version: str
    status: AssessmentStatus
    uncertainty: UncertaintyDimensions
    findings: tuple[MonitoringFinding, ...]
    shared_dependencies: tuple[SharedDependency, ...]
    diagnostic_status: DiagnosticStatus = DiagnosticStatus.DIAGNOSTIC
    unassessed_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        text(self.assessment_id, "assessment_id")
        text(self.method_version, "method_version")
        if type(self.status) is not AssessmentStatus:
            raise ValueError("expected exact monitoring lifecycle status")
        if self.diagnostic_status is not DiagnosticStatus.DIAGNOSTIC:
            raise ValueError("metacognitive records must remain diagnostic")
        domains = strings(self.unassessed_domains, "unassessed_domains", 8)
        if not set(domains) <= {
            "assumptions",
            "dependencies",
            "contradictions",
            "continuity",
            "calibration",
            "repetition",
            "strategy",
            "evidence_use",
        }:
            raise ValueError("unknown monitoring domain")
        if domains and self.status is AssessmentStatus.COMPLETE:
            raise ValueError("a complete assessment cannot contain unassessed domains")
        object.__setattr__(self, "unassessed_domains", domains)
        object.__setattr__(self, "sources", snapshot(self.sources, SourceSnapshot))
        object.__setattr__(self, "uncertainty", snapshot(self.uncertainty, UncertaintyDimensions))
        object.__setattr__(self, "findings", records(self.findings, MonitoringFinding, 512))
        object.__setattr__(
            self, "shared_dependencies", records(self.shared_dependencies, SharedDependency)
        )
        if len({f.finding_id for f in self.findings}) != len(self.findings):
            raise ValueError("finding IDs must be unique")
        s = self.sources
        known_claims = {c.claim_id for c in s.evidence_state.claims}
        known_hypotheses = {h.id for h in s.hypotheses.hypotheses} if s.hypotheses else set()
        known_refs = set(s.evidence_refs)
        for finding in self.findings:
            if (
                not set(finding.claim_ids) <= known_claims
                or not set(finding.hypothesis_ids) <= known_hypotheses
                or not set(finding.source_event_ids) <= set(s.source_event_ids)
                or not set(finding.evidence_refs) <= known_refs
            ):
                raise ValueError("finding references must resolve in retained sources")
        keys = set()
        bindings = {(b.hypothesis_id, b.claim_id) for b in s.assumptions}
        by_hypothesis = {h.id: h for h in s.hypotheses.hypotheses} if s.hypotheses else {}
        for dependency in self.shared_dependencies:
            key = (dependency.kind, dependency.dependency_id)
            if key in keys:
                raise ValueError("duplicate shared dependency")
            keys.add(key)
            if (
                not set(dependency.hypothesis_ids) <= known_hypotheses
                or not set(dependency.evidence_refs) <= known_refs
            ):
                raise ValueError("shared dependency references must resolve")
            if dependency.kind == "assumption":
                if not all(
                    (h, dependency.dependency_id) in bindings for h in dependency.hypothesis_ids
                ):
                    raise ValueError("shared assumption must exist for every named hypothesis")
            elif not all(
                any(
                    r.reference_id == dependency.dependency_id
                    for r in by_hypothesis[h].evidence_refs
                )
                for h in dependency.hypothesis_ids
            ):
                raise ValueError("shared evidence must support every named hypothesis")


def _sequence(cls: type[Any]):
    return lambda value: tuple(cls.from_dict(v) for v in array(value))


_DECODERS = {
    "AssumptionBinding": {
        "applicability_claim_ids": array,
        "alternative_hypothesis_ids": array,
    },
    "SourceSnapshot": {
        "context": EpistemicContext.from_dict,
        "cell": lambda v: V5EvaluationCell(**exact(v, {f.name for f in fields(V5EvaluationCell)})),
        "evidence_state": EvidenceState.from_dict,
        "hypotheses": lambda v: None if v is None else HypothesisState.from_dict(v),
        "epistemic": lambda v: None if v is None else EpistemicStateEstimate.from_dict(v),
        "assumptions": _sequence(AssumptionBinding),
        "source_event_ids": array,
        "provenance": array,
    },
    "SharedDependency": {"hypothesis_ids": array, "evidence_refs": _sequence(EvidenceReference)},
    "MonitoringFinding": {
        "kind": FindingKind,
        "claim_ids": array,
        "hypothesis_ids": array,
        "evidence_refs": _sequence(EvidenceReference),
        "source_event_ids": array,
    },
    "MetacognitiveAssessment": {
        "sources": SourceSnapshot.from_dict,
        "status": AssessmentStatus,
        "uncertainty": UncertaintyDimensions.from_dict,
        "findings": _sequence(MonitoringFinding),
        "shared_dependencies": _sequence(SharedDependency),
        "diagnostic_status": DiagnosticStatus,
        "unassessed_domains": array,
    },
}
