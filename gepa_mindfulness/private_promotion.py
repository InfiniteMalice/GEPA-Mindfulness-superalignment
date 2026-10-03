"""Opt-in private promotion requests; the host owns evaluator isolation and execution."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from evaluation import V5EvaluationRecord
from gepa_mindfulness.coevolution import CandidateSystem, CoevolutionStore, ValidationBundle
from gepa_mindfulness.learning_surfaces import (
    EvaluationEpochStore,
    ValidationReceipt,
    ValidationSplit,
    evaluation_record_id,
)


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, 2**53-1]")


def _token(value: object, name: str) -> None:
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 128:
        raise ValueError(f"{name} must be a nonblank string of at most 128 UTF-8 bytes")


def _digest_value(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("commitment must be a lowercase SHA256 digest")


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    """Positive per-seed caps over the combined held-out and protected evaluation."""

    tokens: int
    tool_calls: int
    wall_time_ms: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _integer(getattr(self, item.name), item.name, 1)


@dataclass(frozen=True, slots=True)
class SeedUsage:
    """Host-metered combined evaluation usage for one seed, including failed attempts."""

    seed: int
    tokens: int
    tool_calls: int
    wall_time_ms: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _integer(getattr(self, item.name), item.name)


@dataclass(frozen=True, slots=True)
class PrivateProtocol:
    """Host-owned evaluation commitments; digests do not establish secrecy or authenticity."""

    protocol_id: str
    metric_policy_id: str
    protected_suite_id: str
    seeds: tuple[int, ...]
    budget: ComputeBudget
    generator_digest: str
    worlds_digest: str
    renderings_digest: str

    def __post_init__(self) -> None:
        for name in ("protocol_id", "metric_policy_id", "protected_suite_id"):
            _token(getattr(self, name), name)
        if type(self.seeds) is not tuple or not 2 <= len(self.seeds) <= 128:
            raise ValueError("seeds must be an exact tuple with 2..128 entries")
        for seed in self.seeds:
            _integer(seed, "seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if type(self.budget) is not ComputeBudget:
            raise ValueError("budget must be an exact ComputeBudget")
        object.__setattr__(self, "budget", ComputeBudget(**asdict(self.budget)))
        for name in ("generator_digest", "worlds_digest", "renderings_digest"):
            _digest_value(getattr(self, name))


class ExperimentOperation(str, Enum):
    """The complete improver-facing operation allowlist."""

    SUBMIT_CANDIDATE = "submit_candidate"
    REQUEST_PRIVATE_EVALUATION = "request_private_evaluation"
    READ_PROMOTION_STATUS = "read_promotion_status"
    REQUEST_REVIEW = "request_review"


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    """An improver request names an existing immutable candidate; it carries no code or paths."""

    operation: ExperimentOperation
    candidate_id: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if type(self.operation) is not ExperimentOperation:
            raise ValueError("operation must be an exact ExperimentOperation")
        _token(self.candidate_id, "candidate_id")
        _digest_value(self.artifact_digest)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _response(candidate_id: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": "private-promotion-v1",
        "candidate_id": candidate_id,
        "status": status,
        "training_eligibility": "HIDDEN_EVAL",
        "execute_candidate": False,
    }


class PrivatePromotionStore:
    """Host-side promotion adapter with append-only provenance.

    Only dispatch responses may cross to an improver. Host methods, instances, catalogs,
    protocol, exceptions and audit events must stay behind the host's access controls.
    """

    def __init__(
        self,
        database_path: str | Path,
        coevolution: CoevolutionStore,
        evaluation: EvaluationEpochStore,
        protocol: PrivateProtocol,
        baseline_receipt: ValidationReceipt,
        source_usage: tuple[SeedUsage, ...],
        *,
        enabled: bool = False,
    ) -> None:
        """Create or reopen a host-private catalog with one immutable protocol.

        Args:
            database_path: Host-owned SQLite path, distinct from evaluation catalogs.
            coevolution: Existing candidate and acceptance authority.
            evaluation: The same evaluation authority pinned by coevolution.
            protocol: Fixed seeds, resource caps and withheld artifact commitments.
            baseline_receipt: Canonical source held-out receipt.
            source_usage: Measured source usage across both splits for every seed.
            enabled: Must be exactly True before any catalog access.

        Raises:
            ValueError: Disabled, malformed, mismatched or changed configuration.
        """
        if enabled is not True:
            raise ValueError("private promotion requires enabled=True")
        if (
            type(coevolution) is not CoevolutionStore
            or type(evaluation) is not EvaluationEpochStore
        ):
            raise ValueError("exact coevolution and evaluation stores are required")
        if type(protocol) is not PrivateProtocol:
            raise ValueError("protocol must be an exact PrivateProtocol")
        self._protocol = PrivateProtocol(
            **{item.name: getattr(protocol, item.name) for item in fields(PrivateProtocol)}
        )
        self._coevolution = coevolution
        self._evaluation = evaluation
        self._baseline = evaluation.validate_validation_receipt(baseline_receipt)
        self._source_usage = self._usage(source_usage)
        self._path = str(Path(database_path).resolve())
        authority = coevolution.authority()
        if self._path in (authority.catalog_path, authority.evaluation_authority.catalog_path):
            raise ValueError("promotion catalog must be separate from evaluation catalogs")
        configuration = self._configuration()
        with self._connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS promotion_events "
                "(sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            for action in ("UPDATE", "DELETE"):
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS promotion_events_no_{action.lower()} "
                    f"BEFORE {action} ON promotion_events BEGIN "
                    "SELECT RAISE(ABORT, 'promotion audit is append-only'); END"
                )
            events = self._events(db)
            if not events:
                self._append(db, events, "protocol", "", configuration, "configured")
            elif events[0]["payload"] != configuration:
                raise ValueError("promotion configuration differs from pinned catalog")
        self._pinned = configuration

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=30)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _usage(self, usage: tuple[SeedUsage, ...]) -> list[dict[str, int]]:
        if type(usage) is not tuple or any(type(item) is not SeedUsage for item in usage):
            raise ValueError("usage must be an exact tuple of SeedUsage")
        snapshots = tuple(SeedUsage(**asdict(item)) for item in usage)
        if len(snapshots) != len(self._protocol.seeds) or {item.seed for item in snapshots} != set(
            self._protocol.seeds
        ):
            raise ValueError("usage must cover each protocol seed exactly once")
        for item in snapshots:
            for name in ("tokens", "tool_calls", "wall_time_ms"):
                if getattr(item, name) > getattr(self._protocol.budget, name):
                    raise ValueError("measured usage exceeds the pinned budget")
        return [asdict(item) for item in sorted(snapshots, key=lambda item: item.seed)]

    def _records(self, receipt: ValidationReceipt) -> tuple[V5EvaluationRecord, ...]:
        receipt = self._evaluation.validate_validation_receipt(receipt)
        authority = self._coevolution.authority()
        if receipt.lineage_id != authority.lineage_id:
            raise ValueError("receipt belongs to a different lineage")
        _, _, records = self._evaluation.resolve_epoch(receipt.lineage_id, receipt.epoch_id)
        by_id = {evaluation_record_id(record): record for record in records}
        selected = tuple(by_id[key] for key in receipt.record_ids)
        if {item.system.seed for item in selected} != set(self._protocol.seeds):
            raise ValueError("receipt must cover exactly the protocol seeds")
        return selected

    def _configuration(self) -> dict[str, Any]:
        authority = self._coevolution.authority()
        if authority.evaluation_authority != self._evaluation.authority():
            raise ValueError("evaluation authority must match coevolution")
        baseline = self._coevolution.read_source_validation_receipt(self._baseline.receipt_id)
        if baseline != self._baseline or baseline.split is not ValidationSplit.HELD_OUT:
            raise ValueError("baseline must be the pinned source held-out receipt")
        manifest = self._coevolution.read_protected_suite(self._protocol.protected_suite_id)
        protected = self._coevolution.read_source_validation_receipt(manifest.source_receipt_id)
        if protected.split is not ValidationSplit.PROTECTED or (
            protected.epoch_id != baseline.epoch_id
            or manifest.source_epoch_id != baseline.epoch_id
            or manifest.source_record_ids != protected.record_ids
        ):
            raise ValueError("protected suite must match the baseline epoch and records")
        self._records(baseline)
        self._records(protected)
        if set(baseline.record_ids) & set(protected.record_ids):
            raise ValueError("held-out and protected records must be disjoint")
        policy = self._coevolution.read_metric_policy(self._protocol.metric_policy_id)
        # JSON roundtrip normalizes tuples once for durable equality across restarts.
        return json.loads(
            _json(
                {
                    "authority": authority.to_dict(),
                    "protocol": asdict(self._protocol),
                    "baseline": baseline.to_dict(),
                    "manifest": manifest.to_dict(),
                    "policy": policy.to_dict(),
                    "source_usage": self._source_usage,
                }
            )
        )

    def _events(self, db: sqlite3.Connection) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous = ""
        for sequence, payload, digest in db.execute(
            "SELECT sequence, payload, digest FROM promotion_events ORDER BY sequence"
        ):
            event = json.loads(payload)
            if (
                sequence != len(events) + 1
                or event.get("sequence") != sequence
                or event.get("previous") != previous
                or _hash(event) != digest
            ):
                raise ValueError("promotion audit chain is invalid")
            events.append(event)
            previous = digest
        return events

    def _check(self, db: sqlite3.Connection) -> list[dict[str, Any]]:
        events = self._events(db)
        if not events or events[0]["payload"] != self._pinned:
            raise ValueError("promotion audit configuration is missing or changed")
        if self._configuration() != self._pinned:
            raise ValueError("live promotion configuration changed")
        return events

    @staticmethod
    def _append(
        db: sqlite3.Connection,
        events: list[dict[str, Any]],
        operation: str,
        candidate_id: str,
        payload: dict[str, Any],
        status: str,
    ) -> None:
        event = {
            "sequence": len(events) + 1,
            "previous": _hash(events[-1]) if events else "",
            "operation": operation,
            "candidate_id": candidate_id,
            "payload": payload,
            "status": status,
        }
        db.execute(
            "INSERT INTO promotion_events VALUES (?, ?, ?)",
            (
                event["sequence"],
                _json(event),
                _hash(event),
            ),
        )

    def _candidate(self, candidate_id: str, digest: str | None = None) -> CandidateSystem:
        candidate = self._coevolution.read_candidate(candidate_id)
        if candidate.source_epoch_id != self._baseline.epoch_id or (
            digest is not None and candidate.artifact_digest != digest
        ):
            raise ValueError("candidate differs from pinned source or artifact")
        return candidate

    def _require_empty(self, candidate: CandidateSystem) -> None:
        _, epoch, records = self._evaluation.resolve_epoch(
            candidate.lineage_id,
            candidate.candidate_epoch_id,
        )
        if epoch.closed or records:
            raise ValueError("private evaluation must be requested before candidate evaluation")

    def dispatch(self, request: ExperimentRequest) -> dict[str, Any]:
        """Validate one improver request and return only its allowlisted status.

        Args:
            request: Exact typed operation, candidate ID and artifact digest.

        Returns:
            A non-training response; invalid requests receive a fixed status without details.
            Storage and validation errors are deliberately hidden from this public projection.
        """
        candidate_id = ""
        try:
            if type(request) is not ExperimentRequest:
                raise ValueError("exact ExperimentRequest required")
            snapshot = ExperimentRequest(
                request.operation, request.candidate_id, request.artifact_digest
            )
            candidate_id = snapshot.candidate_id
            with self._connection() as db:
                events = self._check(db)
                candidate = self._candidate(candidate_id, snapshot.artifact_digest)
                prior = [e for e in events if e["candidate_id"] == candidate_id]
                operation = snapshot.operation.value
                completed = next(
                    (e for e in prior if e["operation"] == "complete_evaluation"), None
                )
                if completed is not None:
                    self._coevolution.read_decision(completed["payload"]["decision_id"])
                if snapshot.operation is ExperimentOperation.READ_PROMOTION_STATUS:
                    return _response(candidate_id, prior[-1]["status"] if prior else "unsubmitted")
                existing = next((e for e in prior if e["operation"] == operation), None)
                if existing is not None:
                    return _response(candidate_id, existing["status"])
                if snapshot.operation is ExperimentOperation.SUBMIT_CANDIDATE:
                    self._require_empty(candidate)
                    status = "submitted"
                elif snapshot.operation is ExperimentOperation.REQUEST_PRIVATE_EVALUATION:
                    if not prior or prior[-1]["status"] != "submitted":
                        raise ValueError("candidate must be submitted first")
                    self._require_empty(candidate)
                    status = "pending"
                else:
                    if not prior or prior[-1]["status"] not in ("accepted", "rejected", "failed"):
                        raise ValueError("review requires a completed evaluation")
                    status = "review_requested"
                self._append(
                    db,
                    events,
                    operation,
                    candidate_id,
                    {
                        "artifact_digest": candidate.artifact_digest,
                        "candidate_epoch_id": candidate.candidate_epoch_id,
                        "source_epoch_id": candidate.source_epoch_id,
                        "correction_proposal_id": candidate.correction.proposal_id,
                    },
                    status,
                )
                return _response(candidate_id, status)
        except (ValueError, TypeError, KeyError, AttributeError, sqlite3.Error, OSError):
            return _response(candidate_id, "invalid_request")

    def complete_evaluation(
        self,
        candidate_id: str,
        bundle: ValidationBundle,
        usage: tuple[SeedUsage, ...],
    ) -> dict[str, Any]:
        """Host-only: validate private evidence, persist a decision and append its provenance.

        Args:
            candidate_id: Candidate with a pending private request.
            bundle: Existing canonical candidate validation bundle.
            usage: Host-metered totals for both splits, including failed attempts, per seed.

        Returns:
            An allowlisted accepted/rejected status without evaluation evidence.

        Raises:
            ValueError: Invalid state, substituted evidence, budget or seed mismatch.
        """
        _token(candidate_id, "candidate_id")
        measured = self._usage(usage)
        if type(bundle) is not ValidationBundle:
            raise ValueError("exact ValidationBundle required")
        snapshot = ValidationBundle.from_dict(bundle.to_dict())
        with self._connection() as db:
            events = self._check(db)
            prior = [e for e in events if e["candidate_id"] == candidate_id]
            if not prior or prior[-1]["status"] != "pending":
                raise ValueError("completion requires a pending private request")
            candidate = self._candidate(candidate_id)
            if snapshot.candidate != candidate:
                raise ValueError("completion candidate differs from request")
            metric = snapshot.metric_receipt
            if (
                metric.policy_id != self._protocol.metric_policy_id
                or metric.source_receipt_id != self._baseline.receipt_id
                or snapshot.protected_suite_id != self._protocol.protected_suite_id
            ):
                raise ValueError("completion differs from pinned private protocol")
            self._records(snapshot.held_out_receipt)
            self._records(snapshot.protected_receipt)
            if set(snapshot.held_out_receipt.record_ids) & set(
                snapshot.protected_receipt.record_ids
            ):
                raise ValueError("held-out and protected records must be disjoint")
            decision = self._coevolution.decide(snapshot)
            status = "accepted" if decision.accepted else "rejected"
            self._append(
                db,
                events,
                "complete_evaluation",
                candidate_id,
                {
                    "decision_id": decision.decision_id,
                    "decision_digest": decision.decision_digest,
                    "usage": measured,
                },
                status,
            )
            return _response(candidate_id, status)

    def record_failure(self, candidate_id: str) -> dict[str, Any]:
        """Host-only: terminate a pending request when evaluation cannot produce valid evidence.

        Args:
            candidate_id: Candidate with a pending request, or an already recorded failure.

        Returns:
            The allowlisted failed status. No private error text is accepted or emitted.

        Raises:
            ValueError: Candidate has no pending request or has already completed otherwise.
        """
        _token(candidate_id, "candidate_id")
        with self._connection() as db:
            events = self._check(db)
            self._candidate(candidate_id)
            prior = [e for e in events if e["candidate_id"] == candidate_id]
            if any(e["operation"] == "record_failure" for e in prior):
                return _response(candidate_id, "failed")
            if not prior or prior[-1]["status"] != "pending":
                raise ValueError("failure requires a pending private request")
            self._append(db, events, "record_failure", candidate_id, {}, "failed")
            return _response(candidate_id, "failed")

    def audit_events(self) -> list[dict[str, Any]]:
        """Host-only: return detached, validated private events for independent review.

        Returns:
            Full private provenance, including protocol commitments and decision identities.

        Raises:
            ValueError: Audit chain or pinned configuration fails validation.
        """
        with self._connection() as db:
            return self._check(db)
