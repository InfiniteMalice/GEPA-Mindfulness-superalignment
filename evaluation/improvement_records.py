"""Immutable, strictly serialized host inputs for offline improvement audits."""

# Standard library
from __future__ import annotations

import json
import math
import types
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract

from .causal_records import canonical_json, content_digest

PURPOSES = (
    "synthetic_training",
    "optimizer_selection",
    "independent_audit",
    "final_test",
    "ood_combinations",
)


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if type(value) is tuple:
        return [_encode(item) for item in value]
    return value


def _typed(value: Any, kind: Any, *, restore: bool = False) -> Any:
    """Only interpret type declarations from this module, never input expressions."""
    origin, args = get_origin(kind), get_args(kind)
    if origin is types.UnionType:
        for option in args:
            try:
                return _typed(value, option, restore=restore)
            except ValueError:
                pass
        raise ValueError("invalid optional value")
    if origin is tuple:
        expected = list if restore else tuple
        if type(value) is not expected:
            raise ValueError(f"expected exact {expected.__name__}")
        return tuple(_typed(item, args[0], restore=restore) for item in value)
    if kind is type(None):
        if value is not None:
            raise ValueError("expected null")
        return value
    if kind in (str, int, float, bool):
        allowed = (int, float) if kind is float else (kind,)
        if type(value) not in allowed:
            raise ValueError(f"expected exact {kind.__name__}")
        if kind is str and not value.strip():
            raise ValueError("text must be nonblank")
        if kind in (int, float) and not math.isfinite(value):
            raise ValueError("number must be finite")
        if kind is int and abs(value) > 2**53 - 1:
            raise ValueError("integer exceeds serialization-safe range")
        return value
    if isinstance(kind, type) and issubclass(kind, Enum):
        if restore:
            if type(value) is not str:
                raise ValueError("enum requires string")
            return kind(value)
        if type(value) is not kind:
            raise ValueError("wrong enum type")
        return value
    if is_dataclass(kind):
        if restore:
            return _construct(kind, value)
        if type(value) is not kind:
            raise ValueError(f"expected exact {kind.__name__}")
        value.__post_init__()
        if kind is EvidenceReference and not value.is_observable:
            raise ValueError("private evidence is not permitted")
        return value
    raise ValueError("unsupported record field type")


def _construct(cls: type, value: object) -> Any:
    names = {f.name for f in fields(cls)}
    if type(value) is not dict or set(value) != names:
        raise ValueError(f"{cls.__name__} requires exact fields")
    hints = get_type_hints(cls)
    try:
        return cls(**{name: _typed(value[name], hints[name], restore=True) for name in names})
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"invalid {cls.__name__}") from error


class ImprovementRecord:
    """Strict frozen subclasses contain only primitives, tuples and trusted records."""

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            _typed(value, hints[item.name])
            if item.name.endswith("digest") and value is not None:
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"invalid {item.name}")
        self._validate()

    def _validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return _encode(self)

    @classmethod
    def from_dict(cls, value: object) -> ImprovementRecord:
        return _construct(cls, value)


def record_digest(record: object) -> str:
    if not isinstance(record, ImprovementRecord):
        raise ValueError("expected improvement record")
    return content_digest(record.to_dict())


def _choice(value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"unsupported choice: {value}")


def _nonnegative(*values: int) -> None:
    if any(value < 0 for value in values):
        raise ValueError("ordinal and repeat values must be nonnegative")


@dataclass(frozen=True)
class SystemConfig(ImprovementRecord):
    model_version: str
    harness_version: str
    config_digest: str


@dataclass(frozen=True)
class DatasetCase(ImprovementRecord):
    case_id: str
    content_digest: str
    family_id: str
    purpose: str
    parent_ids: tuple[str, ...] = ()
    transformation_ids: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    evaluated: bool = True
    combination_digest: str | None = None

    def _validate(self) -> None:
        _choice(self.purpose, PURPOSES)
        for items in (self.parent_ids, self.transformation_ids, self.dependency_ids):
            if len(set(items)) != len(items):
                raise ValueError("duplicate lineage identifiers")


@dataclass(frozen=True)
class DatasetManifest(ImprovementRecord):
    dataset_id: str
    cases: tuple[DatasetCase, ...]
    evidence_refs: tuple[EvidenceReference, ...] = ()
    ood_definitions: tuple[str, ...] = ()
    dependencies_known: bool = False

    def _validate(self) -> None:
        for definition in self.ood_definitions:
            if canonical_json(json.loads(definition)) != definition:
                raise ValueError("OOD definitions require canonical JSON")


@dataclass(frozen=True)
class CandidateSpec(ImprovementRecord):
    candidate_id: str
    baseline: SystemConfig
    candidate: SystemConfig
    proposal_digest: str
    freeze_ordinal: int
    parent_candidate_id: str | None = None

    def _validate(self) -> None:
        _nonnegative(self.freeze_ordinal)


@dataclass(frozen=True)
class MetricSpec(ImprovementRecord):
    metric_id: str
    source_schema: str
    source_metric: str
    rubric_digest: str
    unit: str = "rate"
    direction: str = "higher"
    transform: str = "identity"
    calibration_digest: str | None = None

    def _validate(self) -> None:
        _choice(self.direction, ("higher", "lower"))
        _choice(self.transform, ("identity", "one_minus"))


@dataclass(frozen=True)
class EvaluationSlot(ImprovementRecord):
    slot_id: str
    candidate_id: str
    case_id: str
    metric_id: str
    arm: str
    repeat_id: int
    budget_digest: str
    condition_id: str
    seed: int | None = None
    seed_policy: str = "unknown"

    def _validate(self) -> None:
        _choice(self.arm, ("baseline", "candidate"))
        _choice(self.seed_policy, ("shared", "per_arm", "unknown"))
        _nonnegative(self.repeat_id)
        if self.seed_policy != "unknown" and self.seed is None:
            raise ValueError("declared seed policy requires known seed")


@dataclass(frozen=True)
class ResamplingPolicy(ImprovementRecord):
    seed: int
    resamples: int = 10000
    confidence_level: float = 0.95
    minimum_clusters: int = 20

    def _validate(self) -> None:
        if self.resamples < 1 or self.minimum_clusters < 2 or not 0 < self.confidence_level < 1:
            raise ValueError("invalid resampling policy")


@dataclass(frozen=True)
class ImprovementProtocol(ImprovementRecord):
    protocol_id: str
    manifest_digest: str
    candidates: tuple[CandidateSpec, ...]
    metrics: tuple[MetricSpec, ...]
    slots: tuple[EvaluationSlot, ...]
    evaluator: TrustedEvaluatorContract
    evidence_refs: tuple[EvidenceReference, ...]
    policy: ResamplingPolicy


@dataclass(frozen=True)
class CostEntry(ImprovementRecord):
    value: float | None
    unit: str
    basis: str
    currency: str | None = None

    def _validate(self) -> None:
        _choice(self.basis, ("estimated", "billed"))
        if self.value is not None and self.value < 0:
            raise ValueError("cost must be nonnegative")


@dataclass(frozen=True)
class AttemptEvent(ImprovementRecord):
    event_id: str
    ordinal: int
    candidate_id: str
    round_id: str
    kind: str
    slot_id: str | None = None
    terminal_status: str | None = None
    costs: tuple[CostEntry, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()

    def _validate(self) -> None:
        _nonnegative(self.ordinal)
        _choice(self.kind, ("proposal", "evaluation_started", "evaluation_finished", "decision"))
        if self.kind.startswith("evaluation_") != (self.slot_id is not None):
            raise ValueError("only evaluation events require a slot")
        if (self.kind == "decision") != (self.terminal_status is not None):
            raise ValueError("only decisions require a terminal status")
        if self.terminal_status is not None:
            _choice(self.terminal_status, ("selected", "rejected", "failed", "cancelled"))


@dataclass(frozen=True)
class AttemptJournal(ImprovementRecord):
    journal_id: str
    protocol_digest: str
    events: tuple[AttemptEvent, ...]
    evidence_refs: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class ExposureRecord(ImprovementRecord):
    ordinal: int
    partition_digest: str
    purpose: str
    candidate_ids: tuple[str, ...]
    round_ids: tuple[str, ...]
    use: str
    evidence_refs: tuple[EvidenceReference, ...] = ()

    def _validate(self) -> None:
        _nonnegative(self.ordinal)
        _choice(self.purpose, PURPOSES)
        _choice(self.use, ("evaluation_only", "proposal", "selection"))


@dataclass(frozen=True)
class FinalTestAuthorization(ImprovementRecord):
    candidate_id: str
    candidate_digest: str
    protocol_digest: str
    partition_digest: str
    evaluation_event_id: str
    evidence_refs: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class DiagnosticEvidence(ImprovementRecord):
    slot_id: str
    source_json: str
    row_list: str
    row_id_key: str
    row_id: str
    evidence_refs: tuple[EvidenceReference, ...] = ()

    def _validate(self) -> None:
        source = json.loads(self.source_json)
        if type(source) is not dict or canonical_json(source) != self.source_json:
            raise ValueError("source requires a canonical JSON object")
        if source.get("training_eligibility") not in ("DEVELOPMENT", "HIDDEN_EVAL", "REGRESSION"):
            raise ValueError("source requires non-training eligibility")
        _choice(self.row_list, ("rows", "metric_rows"))
        _choice(self.row_id_key, ("probe_id", "opportunity_id"))


@dataclass(frozen=True)
class AuthenticationRequest(ImprovementRecord):
    purpose: str
    subject_digest: str
    evaluator: TrustedEvaluatorContract
    evidence_refs: tuple[EvidenceReference, ...]
