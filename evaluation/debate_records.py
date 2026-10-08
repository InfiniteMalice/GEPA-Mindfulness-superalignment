"""V5-bound debate protocols and independent authentication envelopes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, cast

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.debate_records import (
    BoundCheckResult,
    DebateRound,
    DebateSession,
    _digest,
    _integer,
    _record,
)
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    _text,
    choice,
    encode,
    public_refs,
    records,
    restore_records,
    restore_refs,
)

from .causal_records import CausalVariant, MetricVerdict, PromptTurn, content_digest
from .ladder import Severity
from .v5_records import SystemIdentity

DEBATE_METRICS = (
    "transition_detection",
    "premise_fault_localization",
    "correct_recovery",
    "justified_position_change",
    "unsupported_authority",
    "contradiction_introduced",
    "contradiction_resolved",
)


def _restore_identity(value: Any, cls: type) -> Any:
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    return cls(**value)


def _identity(value: Any, cls: type) -> Any:
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return _restore_identity(asdict(cast(Any, value)), cls)


class DebateRecord(DiagnosticRecord):
    """Diagnostic serialization with an explicit existing evaluator-contract encoding."""

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "training_eligibility": "DEVELOPMENT",
            **{
                f.name: (asdict(v) if type(v) is TrustedEvaluatorContract else encode(v))
                for f in fields(self)
                for v in (getattr(self, f.name),)
            },
        }


@dataclass(frozen=True)
class DebateActor(DebateRecord):
    """Declared role and system identity; the host independently establishes separation."""

    actor_id: str
    role: str
    system: SystemIdentity
    schema_version = "debate-actor-v1"
    restorers = {"system": lambda v: _restore_identity(v, SystemIdentity)}

    def __post_init__(self) -> None:
        _text(self.actor_id, "actor_id")
        choice(self.role, "role", ("defender", "challenger", "verifier"))
        object.__setattr__(self, "system", _identity(self.system, SystemIdentity))


@dataclass(frozen=True)
class CheckSlot(DebateRecord):
    """One planned check retained in coverage even if no request arrives."""

    check_id: str
    round_index: int
    schema_version = "debate-check-slot-v1"

    def __post_init__(self) -> None:
        _text(self.check_id, "check_id")
        _integer(self.round_index, "round_index")


@dataclass(frozen=True)
class DebateOpportunity(DebateRecord):
    """One predeclared semantic measurement with visible severity and cohort."""

    opportunity_id: str
    round_index: int
    metric: str
    severity: Severity
    cohort: str
    schema_version = "debate-opportunity-v1"
    restorers = {"severity": Severity}

    def __post_init__(self) -> None:
        _text(self.opportunity_id, "opportunity_id")
        _text(self.cohort, "cohort")
        _integer(self.round_index, "round_index")
        choice(self.metric, "metric", DEBATE_METRICS)
        if type(self.severity) is not Severity:
            raise ValueError("severity requires Severity")


@dataclass(frozen=True)
class DebateProtocol(DebateRecord):
    """Predeclared experiment, including planned denominators and host-only outcomes."""

    session_id: str
    rubric_id: str
    subject: CausalVariant
    actors: tuple[DebateActor, ...]
    evaluator: TrustedEvaluatorContract
    max_rounds: int
    check_slots: tuple[CheckSlot, ...]
    opportunities: tuple[DebateOpportunity, ...]
    schema_version = "debate-protocol-v1"
    restorers = {
        "subject": CausalVariant.from_dict,
        "actors": lambda v: restore_records(v, DebateActor),
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
        "check_slots": lambda v: restore_records(v, CheckSlot),
        "opportunities": lambda v: restore_records(v, DebateOpportunity),
    }

    def __post_init__(self) -> None:
        for name in ("session_id", "rubric_id"):
            _text(getattr(self, name), name)
        _integer(self.max_rounds, "max_rounds", 8)
        if self.max_rounds == 0:
            raise ValueError("max_rounds must be positive")
        if type(self.subject) is not CausalVariant:
            raise ValueError("subject requires CausalVariant")
        object.__setattr__(self, "subject", CausalVariant.from_dict(self.subject.to_dict()))
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        for name, cls in (
            ("actors", DebateActor),
            ("check_slots", CheckSlot),
            ("opportunities", DebateOpportunity),
        ):
            object.__setattr__(self, name, records(getattr(self, name), cls))
        if (
            len(self.actors) != 3
            or {a.role for a in self.actors} != {"defender", "challenger", "verifier"}
            or len({a.actor_id for a in self.actors}) != 3
        ):
            raise ValueError("protocol requires three distinct actor roles")
        if self.evaluator.evaluator_id != next(
            a.actor_id for a in self.actors if a.role == "verifier"
        ):
            raise ValueError("evaluator must match verifier actor")
        if len(self.check_slots) > 1024 or len({s.check_id for s in self.check_slots}) != len(
            self.check_slots
        ):
            raise ValueError("requires at most 1024 unique check slots")
        if any(s.round_index >= self.max_rounds for s in self.check_slots) or any(
            o.round_index >= self.max_rounds for o in self.opportunities
        ):
            raise ValueError("slot/opportunity exceeds round budget")
        if len({o.opportunity_id for o in self.opportunities}) != len(self.opportunities) or len(
            {(o.round_index, o.metric) for o in self.opportunities}
        ) != len(self.opportunities):
            raise ValueError("duplicate metric opportunity")


@dataclass(frozen=True)
class DebateContext(DebateRecord):
    """Detached actor-visible context excluding host expectations and assessment receipts."""

    session_id: str
    round_index: int
    prior_rounds: tuple[DebateRound, ...]
    check_slots: tuple[CheckSlot, ...]
    task_turns: tuple[PromptTurn, ...]
    schema_version = "debate-context-v1"
    restorers = {
        "prior_rounds": lambda v: restore_records(v, DebateRound),
        "check_slots": lambda v: restore_records(v, CheckSlot),
        "task_turns": lambda v: tuple(PromptTurn.from_dict(t) for t in v),
    }

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _integer(self.round_index, "round_index")
        object.__setattr__(self, "prior_rounds", records(self.prior_rounds, DebateRound))
        object.__setattr__(self, "check_slots", records(self.check_slots, CheckSlot))
        if type(self.task_turns) is not tuple or any(
            type(t) is not PromptTurn for t in self.task_turns
        ):
            raise ValueError("task_turns requires public turns")
        object.__setattr__(
            self, "task_turns", tuple(PromptTurn.from_dict(t.to_dict()) for t in self.task_turns)
        )


@dataclass(frozen=True)
class DebateVerification(DebateRecord):
    """Exact authentication candidate; accepting a label alone violates the host contract."""

    protocol_digest: str
    round_index: int
    before_digest: str
    challenge_digest: str
    check: BoundCheckResult
    evaluator: TrustedEvaluatorContract
    schema_version = "debate-verification-v1"
    restorers = {
        "check": BoundCheckResult.from_dict,
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
    }

    def __post_init__(self) -> None:
        for name in ("protocol_digest", "before_digest", "challenge_digest"):
            _digest(getattr(self, name))
        _integer(self.round_index, "round_index")
        object.__setattr__(self, "check", _record(self.check, BoundCheckResult))
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        if self.check.snapshot_digest != self.before_digest:
            raise ValueError("check snapshot binding mismatch")


@dataclass(frozen=True)
class DebateAssessment(DebateRecord):
    """Independent semantic labels bound to a captured session and planned opportunities."""

    session_digest: str
    opportunities_digest: str
    evaluator: TrustedEvaluatorContract
    status: str
    human_required: bool
    verdicts: tuple[MetricVerdict, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str
    schema_version = "debate-assessment-v1"
    restorers = {
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
        "verdicts": lambda v: tuple(MetricVerdict.from_dict(x) for x in v),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        _digest(self.session_digest)
        _digest(self.opportunities_digest)
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        choice(self.status, "status", ("verified", "unresolved", "disputed"))
        if type(self.human_required) is not bool:
            raise ValueError("human_required must be boolean")
        if type(self.verdicts) is not tuple or any(
            type(v) is not MetricVerdict for v in self.verdicts
        ):
            raise ValueError("verdicts require exact metric verdicts")
        object.__setattr__(
            self, "verdicts", tuple(MetricVerdict.from_dict(v.to_dict()) for v in self.verdicts)
        )
        if len({v.opportunity_id for v in self.verdicts}) != len(self.verdicts):
            raise ValueError("duplicate verdict")
        object.__setattr__(
            self,
            "evidence_refs",
            public_refs(self.evidence_refs, required=self.status == "verified"),
        )
        _text(self.reason, "reason")


def debate_protocol_digest(protocol: DebateProtocol) -> str:
    """Bind the complete predeclared experiment, including host-only information."""
    return content_digest(protocol.to_dict())


def debate_session_digest(session: DebateSession) -> str:
    """Bind all captured public phases and incomplete-state reasons."""
    return content_digest(session.to_dict())


def debate_opportunities_digest(opportunities: tuple[DebateOpportunity, ...]) -> str:
    """Bind the exact measurement roster independently of input ordering."""
    return content_digest(
        [o.to_dict() for o in sorted(opportunities, key=lambda o: o.opportunity_id)]
    )
