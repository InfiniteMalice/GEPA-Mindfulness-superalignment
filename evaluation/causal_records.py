"""Immutable public records for opt-in causal diagnostics, never reward authorization."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from hashlib import sha256
from typing import Any, ClassVar, TypeVar

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility

from .ladder import Severity
from .v5_records import CaseIdentity, RobustnessIdentity, SystemIdentity


class InterventionKind(str, Enum):
    """Declared factor count, independent of semantic relevance."""

    SINGLE = "single_variable"
    COMPOUND = "compound"


class SeedPolicy(str, Enum):
    """Host-declared relationship between the recorded arm seeds."""

    SHARED = "shared"
    PER_ARM = "per_arm"


class CaptureStatus(str, Enum):
    """Missing captures are absent records, not an observed status."""

    OBSERVED = "observed"
    CENSORED = "censored"


class JudgmentStatus(str, Enum):
    """Declared status still requires independent host authentication."""

    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    DISPUTED = "disputed"


class Relevance(str, Enum):
    """Semantic relevance is independently adjudicated, not inferred from wording."""

    IRRELEVANT = "irrelevant"
    RELEVANT = "relevant"
    UNKNOWN = "unknown"


class CausalMetric(str, Enum):
    """Diagnostic opportunities; these are not new robustness stripes."""

    FLIP = "spurious_decision_flip_rate"
    UPDATE = "required_update_success_rate"
    PRESSURE = "inappropriate_pressure_compliance"
    ABSTENTION = "unjustified_abstention_stability"
    RESUMPTION = "clarification_resumption_correctness"
    LAUNDERING = "semantic_laundering_susceptibility"
    RECOVERY = "post_error_recovery"
    DISAGREEMENT = "paired_outcome_disagreement"
    COVERAGE = "verification_coverage"
    SEVERE = "severe_event_frequency"


def canonical_json(value: object) -> str:
    """Encode complete public inputs without nonfinite numbers or locale dependence."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_digest(value: object) -> str:
    """Bind the exact canonical JSON payload; this is integrity, not authentication."""
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonblank text")


def _bool(value: object, name: str, *, nullable: bool = False) -> None:
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"{name} requires a boolean" + (" or None" if nullable else ""))


def _choice(value: object, enum: type[Enum]) -> None:
    if type(value) is not str:
        raise ValueError(f"{enum.__name__} requires a serialized string value")
    enum(value)


def _digest(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("digest must be lowercase SHA-256")


def _tuple(value: object, kind: type, *, nonempty: bool = False) -> None:
    if (
        type(value) is not tuple
        or (nonempty and not value)
        or any(type(v) is not kind for v in value)
    ):
        raise ValueError(f"expected {'nonempty ' if nonempty else ''}tuple of {kind.__name__}")


def _refs(value: object, *, nonempty: bool = True) -> None:
    _tuple(value, EvidenceReference, nonempty=nonempty)
    refs: Any = value
    if any(not r.is_observable for r in refs):
        raise ValueError("evidence must be observable")
    if len({r.reference_id for r in refs}) != len(refs):
        raise ValueError("evidence references must be unique")


def _value(value: object) -> None:
    _text(value, "factor value")
    try:
        if canonical_json(json.loads(value)) != value:  # type: ignore[arg-type]
            raise ValueError("factor value must be canonical JSON")
    except (TypeError, ValueError) as error:
        raise ValueError("factor value must be canonical finite JSON") from error


def _object(kind: type, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(kind)}:
        raise ValueError(f"invalid {kind.__name__} fields")
    return dict(value)


def _identity(kind: type, value: object) -> Any:
    return kind(**_object(kind, value))


def _array(value: object, convert: Callable[[Any], Any]) -> tuple[Any, ...]:
    if type(value) is not list:
        raise ValueError("serialized tuple must be an array")
    return tuple(convert(v) for v in value)


R = TypeVar("R", bound="_Record")


@dataclass(frozen=True)
class _Record:
    """Exact fields and explicit nested decoders shared by these diagnostic records only."""

    _decoders: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON values, with no mutable references to the source record."""
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls: type[R], value: object) -> R:
        """Restore nested records through their constructors and reject unknown fields."""
        values = _object(cls, value)
        for name, decoder in cls._decoders.items():
            values[name] = decoder(values[name])
        return cls(**values)


@dataclass(frozen=True)
class PromptTurn(_Record):
    """One public actor-visible message, in sequence order."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unknown public message role")
        _text(self.content, "content")


@dataclass(frozen=True)
class FactorChange(_Record):
    """One declared structured change; semantic relevance requires a separate judgment."""

    factor: str
    before: str
    after: str

    def __post_init__(self) -> None:
        _text(self.factor, "factor")
        _value(self.before)
        _value(self.after)
        if self.before == self.after:
            raise ValueError("declared factor must change")


@dataclass(frozen=True)
class CausalVariant(_Record):
    """Manifest-bound arm; expected actions are evaluator-only data."""

    variant_id: str
    case: CaseIdentity
    robustness: RobustnessIdentity
    system: SystemIdentity
    turns: tuple[PromptTurn, ...]
    factors: tuple[tuple[str, str], ...]
    expected_actions: tuple[str, ...]

    _decoders = {
        "case": lambda v: _identity(CaseIdentity, v),
        "robustness": lambda v: _identity(RobustnessIdentity, v),
        "system": lambda v: _identity(SystemIdentity, v),
        "turns": lambda v: _array(v, PromptTurn.from_dict),
        "factors": lambda v: _array(v, lambda row: _array(row, lambda item: item)),
        "expected_actions": lambda v: _array(v, lambda item: item),
    }

    def __post_init__(self) -> None:
        _text(self.variant_id, "variant_id")
        for value, kind in (
            (self.case, CaseIdentity),
            (self.robustness, RobustnessIdentity),
            (self.system, SystemIdentity),
        ):
            if type(value) is not kind:
                raise ValueError(f"expected exact {kind.__name__}")
            value.__post_init__()
        _tuple(self.turns, PromptTurn, nonempty=True)
        _tuple(self.factors, tuple, nonempty=True)
        for factor in self.factors:
            if len(factor) != 2:
                raise ValueError("factor must contain a name and JSON value")
            _text(factor[0], "factor name")
            _value(factor[1])
        if len(dict(self.factors)) != len(self.factors):
            raise ValueError("factor names must be unique")
        _tuple(self.expected_actions, str, nonempty=True)
        for action in self.expected_actions:
            _text(action, "expected action")
        if len(set(self.expected_actions)) != len(self.expected_actions):
            raise ValueError("expected action classes must be unique")

    @property
    def prompt_digest(self) -> str:
        """Hash all ordered public turns, never the evaluator's expected actions."""
        return content_digest([t.to_dict() for t in self.turns])


@dataclass(frozen=True)
class CausalPair(_Record):
    """One declared intervention with independently classified V5 arms."""

    pair_id: str
    family_id: str
    before: CausalVariant
    after: CausalVariant
    intervention_kind: str
    changes: tuple[FactorChange, ...]
    claimed_equivalence: bool
    seed_policy: str
    source_refs: tuple[EvidenceReference, ...]
    training_eligibility: TrainingEligibility

    _decoders = {
        "before": CausalVariant.from_dict,
        "after": CausalVariant.from_dict,
        "changes": lambda v: _array(v, FactorChange.from_dict),
        "source_refs": lambda v: _array(v, EvidenceReference.from_dict),
        "training_eligibility": TrainingEligibility,
    }

    def __post_init__(self) -> None:
        _text(self.pair_id, "pair_id")
        _text(self.family_id, "family_id")
        if type(self.before) is not CausalVariant or type(self.after) is not CausalVariant:
            raise ValueError("pair arms require exact CausalVariant records")
        if self.before.variant_id == self.after.variant_id:
            raise ValueError("variant IDs must differ")
        _choice(self.intervention_kind, InterventionKind)
        _choice(self.seed_policy, SeedPolicy)
        _bool(self.claimed_equivalence, "claimed_equivalence")
        _tuple(self.changes, FactorChange, nonempty=True)
        if len({c.factor for c in self.changes}) != len(self.changes):
            raise ValueError("changed factors must be unique")
        if (len(self.changes) == 1) != (self.intervention_kind == "single_variable"):
            raise ValueError("single-variable and compound change counts must remain distinct")
        before, after = dict(self.before.factors), dict(self.after.factors)
        if before.keys() != after.keys():
            raise ValueError("both arms must declare the same factor roster; use null for absence")
        actual = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if actual != {c.factor: (c.before, c.after) for c in self.changes}:
            raise ValueError("pair must contain exactly its declared structured changes")
        for field in ("model_version", "harness_version", "repeat_id"):
            if getattr(self.before.system, field) != getattr(self.after.system, field):
                raise ValueError("arms must use the same model, harness and repeat")
        if self.seed_policy == "shared" and self.before.system.seed != self.after.system.seed:
            raise ValueError("shared seed policy requires identical arm seeds")
        _refs(self.source_refs)
        if type(self.training_eligibility) is not TrainingEligibility or (
            self.training_eligibility is TrainingEligibility.TRAIN
        ):
            raise ValueError("causal diagnostics require non-TRAIN eligibility")

    @property
    def digest(self) -> str:
        """Bind complete evaluator-side provenance and both arm identities."""
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class CausalCapture(_Record):
    """Host-captured public action sequence or explicitly censored arm window."""

    pair_digest: str
    variant_id: str
    prompt_digest: str
    system: SystemIdentity
    status: str
    actions: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str

    _decoders = {
        "system": lambda v: _identity(SystemIdentity, v),
        "actions": lambda v: _array(v, lambda item: item),
        "evidence_refs": lambda v: _array(v, EvidenceReference.from_dict),
    }

    def __post_init__(self) -> None:
        _digest(self.pair_digest)
        _digest(self.prompt_digest)
        _text(self.variant_id, "variant_id")
        if type(self.system) is not SystemIdentity:
            raise ValueError("capture requires exact SystemIdentity")
        self.system.__post_init__()
        _choice(self.status, CaptureStatus)
        _tuple(self.actions, str, nonempty=self.status == "observed")
        if self.status == "censored" and self.actions:
            raise ValueError("censored captures cannot assert complete actions")
        for action in self.actions:
            _text(action, "action")
        _refs(self.evidence_refs)
        _text(self.reason, "capture reason")


def capture_digest(captures: tuple[CausalCapture, ...]) -> str:
    """Bind an order-independent snapshot, including explicit capture status."""
    _tuple(captures, CausalCapture)
    return content_digest(
        [c.to_dict() for c in sorted(captures, key=lambda c: (c.pair_digest, c.variant_id))]
    )


@dataclass(frozen=True)
class MetricOpportunity(_Record):
    """A predeclared measurement window; no opportunity implies no denominator."""

    opportunity_id: str
    pair_id: str
    metric: str
    severity: Severity
    window_id: str
    cohort: str

    _decoders = {"severity": Severity}

    def __post_init__(self) -> None:
        for name in ("opportunity_id", "pair_id", "window_id", "cohort"):
            _text(getattr(self, name), name)
        _choice(self.metric, CausalMetric)
        if self.metric == CausalMetric.COVERAGE.value:
            raise ValueError("coverage is derived over the pair roster, not supplied opportunities")
        if type(self.severity) is not Severity:
            raise ValueError("severity must be Severity")


@dataclass(frozen=True)
class MetricVerdict(_Record):
    """Independent host assertion; unknown applicability is not a negative outcome."""

    opportunity_id: str
    eligible: bool | None
    value: bool | None
    reason: str
    evidence_refs: tuple[EvidenceReference, ...]

    _decoders = {"evidence_refs": lambda v: _array(v, EvidenceReference.from_dict)}

    def __post_init__(self) -> None:
        _text(self.opportunity_id, "opportunity_id")
        _text(self.reason, "verdict reason")
        _bool(self.eligible, "eligible", nullable=True)
        _bool(self.value, "value", nullable=True)
        if self.eligible is not True and self.value is not None:
            raise ValueError("unknown or ineligible opportunity cannot assert a value")
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class PairAdjudication(_Record):
    """Digest-bound independent judgment; a serialized status never authenticates itself."""

    pair_digest: str
    capture_digest: str
    protocol_digest: str
    evaluator: TrustedEvaluatorContract
    status: str
    relevance: str
    before_correct: bool | None
    after_correct: bool | None
    action_changed: bool | None
    required_update: bool | None
    update_satisfied: bool | None
    change_justified: bool | None
    human_required: bool
    reason: str
    evidence_refs: tuple[EvidenceReference, ...]
    metric_verdicts: tuple[MetricVerdict, ...]

    _decoders = {
        "evaluator": lambda v: _identity(TrustedEvaluatorContract, v),
        "evidence_refs": lambda v: _array(v, EvidenceReference.from_dict),
        "metric_verdicts": lambda v: _array(v, MetricVerdict.from_dict),
    }

    def __post_init__(self) -> None:
        for value in (self.pair_digest, self.capture_digest, self.protocol_digest):
            _digest(value)
        if type(self.evaluator) is not TrustedEvaluatorContract:
            raise ValueError("evaluator requires exact TrustedEvaluatorContract")
        self.evaluator.__post_init__()
        _choice(self.status, JudgmentStatus)
        _choice(self.relevance, Relevance)
        for name in (
            "before_correct",
            "after_correct",
            "action_changed",
            "required_update",
            "update_satisfied",
            "change_justified",
        ):
            _bool(getattr(self, name), name, nullable=True)
        _bool(self.human_required, "human_required")
        _text(self.reason, "adjudication reason")
        _refs(self.evidence_refs)
        _tuple(self.metric_verdicts, MetricVerdict)
        if len({v.opportunity_id for v in self.metric_verdicts}) != len(self.metric_verdicts):
            raise ValueError("metric verdicts require unique opportunity IDs")
