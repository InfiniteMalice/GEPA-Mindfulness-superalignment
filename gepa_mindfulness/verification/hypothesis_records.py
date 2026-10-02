"""Inert external hypothesis history; records confer no truth, reward or authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, TypeVar

from evaluation.cases.registry import CANONICAL_CASE_IDS
from gepa_mindfulness.core.evidence import (
    OBSERVABLE_EVIDENCE_SOURCE_KINDS,
    EvidenceReference,
    EvidenceSourceKind,
)
from gepa_mindfulness.training.eligibility import TrainingEligibility
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

from .epistemic_state import EpistemicContext

SCHEMA_VERSION = "multi-hypothesis-state-v1"
SCORE_NAMES = (
    "evidence_fit",
    "uncertainty",
    "complexity",
    "risk",
    "reversibility",
    "compute_cost",
    "transfer",
)
MAXIMIZE = frozenset({"evidence_fit", "reversibility", "transfer"})
STATUSES = ("supported", "challenged", "context_limited", "unresolved")
TRIGGERS = (
    "persistent_innovation",
    "multimodal_evidence",
    "verifier_conflict",
    "regime_shift",
    "structural_alternatives",
)


def _text(value: str, name: str, maximum: int = 128) -> None:
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be a nonblank exact string of at most {maximum} UTF-8 bytes")


def _integer(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
        raise ValueError(f"{name} must be a nonnegative JSON-safe exact integer")


def _number(value: float | None, name: str, *, unit: bool) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric or None")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(numeric) or numeric < 0 or (unit and numeric > 1):
        raise ValueError(f"{name} must be finite, nonnegative and within its declared range")
    if type(value) is int and int(numeric) != value:
        raise ValueError(f"{name} integer must convert to float exactly")


def _refs(values: tuple[EvidenceReference, ...]) -> tuple[EvidenceReference, ...]:
    if type(values) is not tuple or not values:
        raise ValueError("evidence_refs must be a nonempty exact tuple")
    copied = []
    for value in values:
        if type(value) is not EvidenceReference:
            raise ValueError("evidence_refs must contain exact EvidenceReference records")
        _text(value.reference_id, "reference_id")
        if type(value.source_kind) is not EvidenceSourceKind:
            raise ValueError("source_kind must be an exact EvidenceSourceKind")
        if value.source_kind not in OBSERVABLE_EVIDENCE_SOURCE_KINDS:
            raise ValueError("hypothesis evidence must be observable")
        copied.append(EvidenceReference(value.reference_id, value.source_kind))
    if len({r.reference_id for r in copied}) != len(copied):
        raise ValueError("evidence reference IDs must be unique within a record")
    return tuple(copied)


def _context(value: EpistemicContext) -> EpistemicContext:
    if type(value) is not EpistemicContext or type(value.system) is not EvaluatedSystemVersion:
        raise ValueError("context and system must be exact typed records")
    _text(value.run_id, "run_id")
    _text(value.system.model_version, "model_version")
    _text(value.system.harness_version, "harness_version")
    return EpistemicContext(
        value.run_id,
        value.repeat_id,
        EvaluatedSystemVersion(value.system.model_version, value.system.harness_version),
    )


@dataclass(frozen=True, slots=True)
class HypothesisScores:
    """Separate host measurements; None means unavailable, never a zero or posterior weight."""

    evidence_fit: float | None = None
    uncertainty: float | None = None
    complexity: float | None = None
    risk: float | None = None
    reversibility: float | None = None
    compute_cost: float | None = None
    transfer: float | None = None

    def __post_init__(self) -> None:
        for name in SCORE_NAMES:
            value = getattr(self, name)
            _number(value, name, unit=name not in ("complexity", "compute_cost"))
            if value is not None:
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """An immutable public claim with host-retained observable provenance."""

    id: str
    statement: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.statement, "statement", 512)
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class HypothesisAssessment:
    """One assessor's diagnostic verdict; supersession preserves the original record."""

    id: str
    hypothesis_id: str
    assessor_id: str
    status: str
    scores: HypothesisScores
    evidence_refs: tuple[EvidenceReference, ...]
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "hypothesis_id", "assessor_id", "status"):
            _text(getattr(self, name), name)
        if self.status not in STATUSES:
            raise ValueError("unknown assessment status")
        if self.supersedes is not None:
            _text(self.supersedes, "supersedes")
        object.__setattr__(self, "scores", _snapshot(self.scores, HypothesisScores))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class HypothesisTrigger:
    """An evidence-linked host declaration of why alternatives warrant retention."""

    id: str
    kind: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.kind, "kind")
        if self.kind not in TRIGGERS:
            raise ValueError("unknown hypothesis trigger")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class HypothesisState:
    """Full immutable host history; storing this record requires separate host authorization."""

    state_id: str
    context: EpistemicContext
    source_case_id: int
    protocol_id: str
    complexity_unit: str
    compute_unit: str
    hypotheses: tuple[Hypothesis, ...]
    assessments: tuple[HypothesisAssessment, ...]
    triggers: tuple[HypothesisTrigger, ...]
    training_eligibility: TrainingEligibility = TrainingEligibility.DEVELOPMENT
    revision: int = 0

    def __post_init__(self) -> None:
        for name in ("state_id", "protocol_id", "complexity_unit", "compute_unit"):
            _text(getattr(self, name), name)
        _integer(self.revision, "revision")
        if type(self.source_case_id) is not int or self.source_case_id not in CANONICAL_CASE_IDS:
            raise ValueError("source_case_id must name an existing canonical case")
        if type(self.training_eligibility) is not TrainingEligibility:
            raise ValueError("training_eligibility must be an exact TrainingEligibility")
        if self.training_eligibility is TrainingEligibility.TRAIN:
            raise ValueError("hypothesis histories are not training data")
        object.__setattr__(self, "context", _context(self.context))
        for name, cls in (
            ("hypotheses", Hypothesis),
            ("assessments", HypothesisAssessment),
            ("triggers", HypothesisTrigger),
        ):
            values = _records(getattr(self, name), cls)
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"{name} IDs must be unique")
            object.__setattr__(self, name, values)
        if len(self.hypotheses) < 2 or not self.triggers:
            raise ValueError("state requires at least two hypotheses and one trigger")
        known = {h.id for h in self.hypotheses}
        active: dict[str, HypothesisAssessment] = {}
        for assessment in self.assessments:
            if assessment.hypothesis_id not in known:
                raise ValueError("assessment refers to unknown hypothesis")
            if assessment.supersedes is not None:
                old = active.get(assessment.supersedes)
                if old is None or (old.hypothesis_id, old.assessor_id) != (
                    assessment.hypothesis_id,
                    assessment.assessor_id,
                ):
                    raise ValueError(
                        "supersedes must name an earlier live same-assessor assessment"
                    )
                del active[assessment.supersedes]
            active[assessment.id] = assessment

    def to_dict(self) -> dict[str, Any]:
        """Return a detached full external history after revalidating every field.

        Returns:
            JSON-safe state, including external provenance; never an actor payload.

        Raises:
            ValueError: A record was mutated into an invalid state.
        """
        checked = _snapshot(self, HypothesisState)
        return {"schema_version": SCHEMA_VERSION, **_encode(checked)}

    @classmethod
    def from_dict(cls, value: object) -> HypothesisState:
        """Restore a strict full snapshot; this does not authenticate its history.

        Args:
            value: JSON object with exactly the exported fields and schema version.

        Returns:
            Detached validated state. Hosts validate extensions against their stored prior.

        Raises:
            ValueError: Unknown fields, invalid primitives, graph links or evidence.
        """
        if type(value) is not dict:
            raise ValueError("state must be an exact JSON object")
        _plain_containers(value, set())
        data = thaw_json_mapping(freeze_json_mapping(value, field_name="hypothesis state"))
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("unsupported hypothesis schema_version")
        return _restore(HypothesisState, data)


_T = TypeVar(
    "_T", HypothesisScores, Hypothesis, HypothesisAssessment, HypothesisTrigger, HypothesisState
)


def _snapshot(value: _T, cls: type[_T]) -> _T:
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls(**{field.name: getattr(value, field.name) for field in fields(cls)})


def _records(values: tuple[_T, ...], cls: type[_T]) -> tuple[_T, ...]:
    if type(values) is not tuple:
        raise ValueError("history records must be an exact tuple")
    return tuple(_snapshot(value, cls) for value in values)


def _encode(value: Any) -> Any:
    if type(value) in (TrainingEligibility, EvidenceSourceKind):
        return value.value
    if type(value) is tuple:
        return [_encode(v) for v in value]
    if type(value) in (
        HypothesisState,
        Hypothesis,
        HypothesisAssessment,
        HypothesisScores,
        HypothesisTrigger,
        EpistemicContext,
        EvaluatedSystemVersion,
        EvidenceReference,
    ):
        # Read the class field inventory and values directly: legacy evidence has a
        # mutable instance dictionary that can shadow serialization methods/metadata.
        return {f.name: _encode(getattr(value, f.name)) for f in fields(type(value))}
    return value


def _object(value: Any, cls: type[Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    return dict(value)


def _plain_containers(value: Any, active: set[int]) -> None:
    if type(value) in (str, int, float, bool) or value is None:
        return
    if type(value) not in (dict, list) or id(value) in active:
        raise ValueError("state requires acyclic exact JSON containers")
    active.add(id(value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("state object keys must be exact strings")
        children = value.values()
    else:
        children = value
    for child in children:
        _plain_containers(child, active)
    active.remove(id(value))


def _restore(cls: type[_T], value: Any) -> _T:
    data = _object(value, cls)
    for name, raw in tuple(data.items()):
        if name == "evidence_refs":
            if type(raw) is not list:
                raise ValueError("evidence_refs must be a JSON array")
            refs = []
            for ref in raw:
                ref = _object(ref, EvidenceReference)
                refs.append(
                    EvidenceReference(ref["reference_id"], EvidenceSourceKind(ref["source_kind"]))
                )
            data[name] = tuple(refs)
        elif name == "context":
            context = _object(raw, EpistemicContext)
            system = EvaluatedSystemVersion(**_object(context["system"], EvaluatedSystemVersion))
            data[name] = EpistemicContext(context["run_id"], context["repeat_id"], system)
        elif name == "scores":
            data[name] = _restore(HypothesisScores, raw)
        elif name == "training_eligibility":
            data[name] = TrainingEligibility(raw)
        elif name in ("hypotheses", "assessments", "triggers"):
            if type(raw) is not list:
                raise ValueError(f"{name} must be a JSON array")
            record = {
                "hypotheses": Hypothesis,
                "assessments": HypothesisAssessment,
                "triggers": HypothesisTrigger,
            }[name]
            data[name] = tuple(_restore(record, v) for v in raw)
    return cls(**data)
