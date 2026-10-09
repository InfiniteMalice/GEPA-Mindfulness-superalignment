"""Inert public debate transcripts; authored assertions carry no verification authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

# Third-party
# Local
from ..core.evidence import EvidenceReference
from .check_records import CheckRequest, CheckResult
from .claim_graph import ClaimGraph
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    choice,
    public_refs,
    records,
    restore_records,
    restore_refs,
    strings,
)

R = TypeVar("R", bound=DiagnosticRecord)
ROUND_STATUSES = ("observed", "missing", "callback_error", "human_pending", "undecomposable")
STOP_REASONS = ("host_stopped", "budget_exhausted") + ROUND_STATUSES[1:]


def _integer(value: object, name: str, maximum: int = 7) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} requires an integer in 0..{maximum}")


def _digest(value: object) -> None:
    if type(value) is not str or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("expected lowercase SHA-256 digest")


def _record(value: object, cls: type[R]) -> R:
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls.from_dict(value.to_dict())


def _optional(value: Any, cls: type[R]) -> R | None:
    return None if value is None else cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class ArgumentSnapshot(DiagnosticRecord):
    """One public proposal; graph correctness still requires independent checking."""

    actor_id: str
    graph: ClaimGraph | None
    conclusion_claim_id: str | None
    proposed_action: str
    constraint_claim_ids: tuple[str, ...]
    public_statement: str
    decomposition_status: str
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "debate-argument-v1"
    restorers = {"graph": lambda v: _optional(v, ClaimGraph), "evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        for name in ("actor_id", "proposed_action", "public_statement"):
            _text(getattr(self, name), name)
        choice(self.decomposition_status, "decomposition_status", ("proposed", "undecomposable"))
        object.__setattr__(
            self, "constraint_claim_ids", strings(self.constraint_claim_ids, "constraints")
        )
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        if self.decomposition_status == "undecomposable":
            if (
                self.graph is not None
                or self.conclusion_claim_id is not None
                or self.constraint_claim_ids
            ):
                raise ValueError("undecomposable snapshot cannot assert a graph or conclusion")
            return
        graph = _record(self.graph, ClaimGraph)
        ids = {n.claim.claim_id for n in graph.nodes}
        if self.conclusion_claim_id not in ids or not set(self.constraint_claim_ids) <= ids:
            raise ValueError("conclusion or constraint references unknown claim")
        for node in graph.nodes:
            public_refs(node.claim.evidence_refs, required=True)
        object.__setattr__(self, "graph", graph)


@dataclass(frozen=True, slots=True)
class DebateChallenge(DiagnosticRecord):
    """A public challenge and proposed checks, never executable procedures."""

    challenge_id: str
    actor_id: str
    snapshot_digest: str
    target_claim_ids: tuple[str, ...]
    requests: tuple[CheckRequest, ...]
    public_statement: str
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "debate-challenge-v1"
    restorers = {
        "requests": lambda v: restore_records(v, CheckRequest),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        for name in ("challenge_id", "actor_id", "public_statement"):
            _text(getattr(self, name), name)
        _digest(self.snapshot_digest)
        object.__setattr__(self, "target_claim_ids", strings(self.target_claim_ids, "targets"))
        object.__setattr__(self, "requests", records(self.requests, CheckRequest))
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        if len({r.check_id for r in self.requests}) != len(self.requests):
            raise ValueError("duplicate check request")
        if any(r.claim_id not in self.target_claim_ids for r in self.requests):
            raise ValueError("check request outside challenge targets")


@dataclass(frozen=True, slots=True)
class BoundCheckResult(DiagnosticRecord):
    """A claimed check result bound to the exact public inputs it purports to verify."""

    snapshot_digest: str
    request_digest: str
    result: CheckResult
    human_required: bool

    schema_version = "debate-bound-check-v1"
    restorers = {"result": CheckResult.from_dict}

    def __post_init__(self) -> None:
        _digest(self.snapshot_digest)
        _digest(self.request_digest)
        object.__setattr__(self, "result", _record(self.result, CheckResult))
        if type(self.human_required) is not bool:
            raise ValueError("human_required must be boolean")


@dataclass(frozen=True, slots=True)
class DebateRound(DiagnosticRecord):
    """An ordered public round, including partial capture at an explicit stopping point."""

    round_index: int
    before: ArgumentSnapshot | None
    challenge: DebateChallenge | None
    results: tuple[BoundCheckResult, ...]
    after: ArgumentSnapshot | None
    status: str
    reason: str

    schema_version = "debate-round-v1"
    restorers = {
        "before": lambda v: _optional(v, ArgumentSnapshot),
        "challenge": lambda v: _optional(v, DebateChallenge),
        "results": lambda v: restore_records(v, BoundCheckResult),
        "after": lambda v: _optional(v, ArgumentSnapshot),
    }

    def __post_init__(self) -> None:
        _integer(self.round_index, "round_index")
        choice(self.status, "round status", ROUND_STATUSES)
        _text(self.reason, "reason")
        for name, cls in (
            ("before", ArgumentSnapshot),
            ("challenge", DebateChallenge),
            ("after", ArgumentSnapshot),
        ):
            obj = getattr(self, name)
            if obj is not None:
                object.__setattr__(self, name, _record(obj, cls))
        object.__setattr__(self, "results", records(self.results, BoundCheckResult))
        if len({r.result.check_id for r in self.results}) != len(self.results):
            raise ValueError("duplicate check result")
        if self.before is None and (
            self.challenge is not None or self.results or self.after is not None
        ):
            raise ValueError("later phases require before snapshot")
        if self.challenge is None and (self.results or self.after is not None):
            raise ValueError("verification/revision requires challenge")
        if self.status == "observed" and (
            self.before is None or self.challenge is None or self.after is None
        ):
            raise ValueError("observed round requires complete public phases")


@dataclass(frozen=True, slots=True)
class DebateSession(DiagnosticRecord):
    """Finite transcript; stopping never proves the proposal true or authorized."""

    session_id: str
    protocol_digest: str
    rounds: tuple[DebateRound, ...]
    stop_reason: str

    schema_version = "debate-session-v1"
    restorers = {"rounds": lambda v: restore_records(v, DebateRound)}

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _digest(self.protocol_digest)
        choice(self.stop_reason, "stop_reason", STOP_REASONS)
        object.__setattr__(self, "rounds", records(self.rounds, DebateRound))
        if not 1 <= len(self.rounds) <= 8 or tuple(r.round_index for r in self.rounds) != tuple(
            range(len(self.rounds))
        ):
            raise ValueError("session requires 1..8 contiguous rounds")
        if any(r.status != "observed" for r in self.rounds[:-1]):
            raise ValueError("partial round must end the session")
