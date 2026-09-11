"""Authoritative, evidence-bound, and revisioned recovery consumption.

``select_recovery`` reserves one pending proposal against runtime-owned state. It does not spend a
retry or replan count. ``consume_recovery`` revalidates that proposal against the current store
revision and atomically records the counter transition. The runtime owner must authenticate
verifier references before it constructs classification or repeated-route bindings; this module
does not authenticate verifier identities or dereference evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Protocol, cast
from weakref import WeakKeyDictionary

from gepa_mindfulness.core.evidence import EvidenceReference
from mindful_trace_gepa.action_bound_events import ActionRecord

from .failure_graph import FailureNode
from .runtime_governance import _snapshot_action, action_record_digest
from .state import _require_exact_mapping, _require_nonblank_string, _require_sha256
from .state import _snapshot_evidence_refs as _snapshot_canonical_evidence

MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991


class FailureCategory(str, Enum):
    """A verifier-backed failure class that selects one intervention level."""

    TRANSIENT_RUNTIME_ERROR = "transient_runtime_error"
    ARGUMENT_ERROR = "argument_error"
    SCHEMA_ERROR = "schema_error"
    INPUT_ERROR = "input_error"
    STRATEGY_FAILURE = "strategy_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    MISSING_USER_FACT = "missing_user_fact"
    CONSEQUENTIAL_AMBIGUITY = "consequential_ambiguity"
    UNSAFE_UNCERTAINTY = "unsafe_uncertainty"


class RecoveryAction(str, Enum):
    """The selected intervention or terminal recovery status."""

    RETRY = "retry"
    REPAIR_ARGUMENTS = "repair_arguments"
    REPLAN = "replan"
    REQUEST_CLARIFICATION = "request_clarification"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    REJECT_ROUTE = "reject_route"
    EXHAUSTED = "exhausted"


_DIRECT_ACTIONS = {
    FailureCategory.ARGUMENT_ERROR: RecoveryAction.REPAIR_ARGUMENTS,
    FailureCategory.SCHEMA_ERROR: RecoveryAction.REPAIR_ARGUMENTS,
    FailureCategory.INPUT_ERROR: RecoveryAction.REPAIR_ARGUMENTS,
    FailureCategory.MISSING_USER_FACT: RecoveryAction.REQUEST_CLARIFICATION,
    FailureCategory.CONSEQUENTIAL_AMBIGUITY: RecoveryAction.ESCALATE,
    FailureCategory.UNSAFE_UNCERTAINTY: RecoveryAction.ABSTAIN,
}
_REPLAN_CATEGORIES = frozenset(
    {
        FailureCategory.STRATEGY_FAILURE,
        FailureCategory.DEPENDENCY_FAILURE,
    }
)


class _ActionFields(Protocol):
    action_id: str


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    """Finite retry and replan maxima plus their consumed counts."""

    max_retries: int
    retries_used: int
    max_replans: int
    replans_used: int

    def __post_init__(self) -> None:
        """Reject nonportable, negative, or inconsistent integer counts."""

        for field_name, value in (
            ("max_retries", self.max_retries),
            ("retries_used", self.retries_used),
            ("max_replans", self.max_replans),
            ("replans_used", self.replans_used),
        ):
            _require_json_safe_integer(value, field_name)
        if self.retries_used > self.max_retries or self.replans_used > self.max_replans:
            raise ValueError("recovery counts used cannot exceed their declared maximum")

    def to_dict(self) -> dict[str, int]:
        """Return an exact JSON-compatible finite-budget snapshot."""

        snapshot = _snapshot_budget(self, "budget")
        return {
            "max_retries": snapshot.max_retries,
            "retries_used": snapshot.retries_used,
            "max_replans": snapshot.max_replans,
            "replans_used": snapshot.replans_used,
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryBudget:
        """Restore and validate an exact JSON-compatible budget."""

        values = _require_exact_mapping(
            data,
            {"max_retries", "retries_used", "max_replans", "replans_used"},
            "RecoveryBudget",
        )
        return cls(
            cast(int, values["max_retries"]),
            cast(int, values["retries_used"]),
            cast(int, values["max_replans"]),
            cast(int, values["replans_used"]),
        )


@dataclass(frozen=True, slots=True)
class FailureClassificationBinding:
    """One verifier-backed failure category bound to an action and failure event."""

    failure: FailureNode
    action_id: str
    action_digest: str
    category: FailureCategory
    verification_event_id: str
    evidence_refs: tuple[EvidenceReference, ...]
    verifier_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate exact identities and evidence drawn from the bound failure."""

        failure = _snapshot_failure(self.failure)
        _require_nonblank_string(self.action_id, "action_id")
        _require_sha256(self.action_digest, "action_digest")
        if type(self.category) is not FailureCategory:
            raise ValueError("category must be an exact FailureCategory")
        _require_nonblank_string(self.verification_event_id, "verification_event_id")
        evidence = _snapshot_evidence(self.evidence_refs, "evidence_refs")
        _require_observable_subset(evidence, failure.evidence_refs, "classification evidence")
        verifier_refs = _snapshot_identifiers(self.verifier_refs, "verifier_refs", required=True)
        object.__setattr__(self, "failure", failure)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "verifier_refs", verifier_refs)

    @property
    def classification_id(self) -> str:
        """Return the digest identity of every classification field."""

        snapshot = _snapshot_classification(self)
        return _classification_digest(snapshot)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible classification binding."""

        snapshot = _snapshot_classification(self)
        return {
            "classification_id": _classification_digest(snapshot),
            "failure": snapshot.failure.to_dict(),
            "action_id": snapshot.action_id,
            "action_digest": snapshot.action_digest,
            "category": snapshot.category.value,
            "verification_event_id": snapshot.verification_event_id,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "verifier_refs": list(snapshot.verifier_refs),
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureClassificationBinding:
        """Restore and verify the digest identity of a serialized classification."""

        values = _require_exact_mapping(
            data,
            {
                "classification_id",
                "failure",
                "action_id",
                "action_digest",
                "category",
                "verification_event_id",
                "evidence_refs",
                "verifier_refs",
            },
            "FailureClassificationBinding",
        )
        serialized_id = _require_sha256(values["classification_id"], "classification_id")
        binding = cls(
            failure=FailureNode.from_dict(values["failure"]),
            action_id=cast(str, values["action_id"]),
            action_digest=cast(str, values["action_digest"]),
            category=_restore_category(values["category"]),
            verification_event_id=cast(str, values["verification_event_id"]),
            evidence_refs=_restore_evidence(values["evidence_refs"], "evidence_refs"),
            verifier_refs=_snapshot_identifiers(
                values["verifier_refs"],
                "verifier_refs",
                required=True,
            ),
        )
        if serialized_id != binding.classification_id:
            raise ValueError("classification_id does not match the classification fields")
        return binding


@dataclass(frozen=True, slots=True)
class RepeatedRouteFinding:
    """A verifier-backed finding that one exact consumed route is being repeated."""

    route_id: str
    prior_action_id: str
    prior_action_digest: str
    prior_decision_id: str
    prior_revision: int
    current_failure_id: str
    current_failure_event_id: str
    classification_id: str
    classification_verification_event_id: str
    evidence_refs: tuple[EvidenceReference, ...]
    verifier_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate complete prior and current identities plus verifier provenance."""

        for field_name, value in (
            ("route_id", self.route_id),
            ("prior_action_id", self.prior_action_id),
            ("current_failure_id", self.current_failure_id),
            ("current_failure_event_id", self.current_failure_event_id),
            ("classification_verification_event_id", self.classification_verification_event_id),
        ):
            _require_nonblank_string(value, field_name)
        _require_sha256(self.prior_action_digest, "prior_action_digest")
        _require_sha256(self.prior_decision_id, "prior_decision_id")
        _require_json_safe_integer(self.prior_revision, "prior_revision")
        _require_sha256(self.classification_id, "classification_id")
        evidence = _snapshot_evidence(self.evidence_refs, "evidence_refs")
        if not evidence or not any(reference.is_observable for reference in evidence):
            raise ValueError("repeated route finding requires observable evidence_refs")
        verifier_refs = _snapshot_identifiers(self.verifier_refs, "verifier_refs", required=True)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "verifier_refs", verifier_refs)

    @property
    def finding_id(self) -> str:
        """Return the digest identity of every repeated-route field."""

        snapshot = _snapshot_route_finding(self)
        return _route_finding_digest(snapshot)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible repeated-route finding."""

        snapshot = _snapshot_route_finding(self)
        return {
            "finding_id": _route_finding_digest(snapshot),
            "route_id": snapshot.route_id,
            "prior_action_id": snapshot.prior_action_id,
            "prior_action_digest": snapshot.prior_action_digest,
            "prior_decision_id": snapshot.prior_decision_id,
            "prior_revision": snapshot.prior_revision,
            "current_failure_id": snapshot.current_failure_id,
            "current_failure_event_id": snapshot.current_failure_event_id,
            "classification_id": snapshot.classification_id,
            "classification_verification_event_id": (snapshot.classification_verification_event_id),
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "verifier_refs": list(snapshot.verifier_refs),
        }

    @classmethod
    def from_dict(cls, data: object) -> RepeatedRouteFinding:
        """Restore and verify the digest identity of a repeated-route finding."""

        values = _require_exact_mapping(
            data,
            {
                "finding_id",
                "route_id",
                "prior_action_id",
                "prior_action_digest",
                "prior_decision_id",
                "prior_revision",
                "current_failure_id",
                "current_failure_event_id",
                "classification_id",
                "classification_verification_event_id",
                "evidence_refs",
                "verifier_refs",
            },
            "RepeatedRouteFinding",
        )
        serialized_id = _require_sha256(values["finding_id"], "finding_id")
        finding = cls(
            route_id=cast(str, values["route_id"]),
            prior_action_id=cast(str, values["prior_action_id"]),
            prior_action_digest=cast(str, values["prior_action_digest"]),
            prior_decision_id=cast(str, values["prior_decision_id"]),
            prior_revision=cast(int, values["prior_revision"]),
            current_failure_id=cast(str, values["current_failure_id"]),
            current_failure_event_id=cast(str, values["current_failure_event_id"]),
            classification_id=cast(str, values["classification_id"]),
            classification_verification_event_id=cast(
                str,
                values["classification_verification_event_id"],
            ),
            evidence_refs=_restore_evidence(values["evidence_refs"], "evidence_refs"),
            verifier_refs=_snapshot_identifiers(
                values["verifier_refs"],
                "verifier_refs",
                required=True,
            ),
        )
        if serialized_id != finding.finding_id:
            raise ValueError("finding_id does not match the repeated route fields")
        return finding


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """A pending recovery proposal that must be consumed through its originating store."""

    plan_id: str
    revision: int
    action_id: str
    action_digest: str
    route_id: str
    classification: FailureClassificationBinding
    action: RecoveryAction
    budget_before: RecoveryBudget
    budget_after: RecoveryBudget
    repeated_route_finding: RepeatedRouteFinding | None = None

    def __post_init__(self) -> None:
        """Validate the proposal identity and deterministic budget transition."""

        _require_nonblank_string(self.plan_id, "plan_id")
        _require_json_safe_integer(self.revision, "revision")
        _require_nonblank_string(self.action_id, "action_id")
        _require_sha256(self.action_digest, "action_digest")
        _require_nonblank_string(self.route_id, "route_id")
        classification = _snapshot_classification(self.classification)
        if classification.action_id != self.action_id:
            raise ValueError("classification action_id does not match decision action_id")
        if classification.action_digest != self.action_digest:
            raise ValueError("classification action_digest does not match decision action_digest")
        if type(self.action) is not RecoveryAction:
            raise ValueError("action must be an exact RecoveryAction")
        before = _snapshot_budget(self.budget_before, "budget_before")
        after = _snapshot_budget(self.budget_after, "budget_after")
        finding = _snapshot_optional_route_finding(self.repeated_route_finding)
        _validate_decision_route_finding(
            self.revision,
            self.action_id,
            self.action_digest,
            self.route_id,
            classification,
            finding,
        )
        _validate_transition(
            classification.category, self.action, before, after, finding is not None
        )
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "budget_before", before)
        object.__setattr__(self, "budget_after", after)
        object.__setattr__(self, "repeated_route_finding", finding)

    @property
    def decision_id(self) -> str:
        """Return the digest identity of every pending decision field."""

        snapshot = _snapshot_decision(self)
        return _decision_digest(snapshot)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible pending decision."""

        snapshot = _snapshot_decision(self)
        return {
            "decision_id": _decision_digest(snapshot),
            "plan_id": snapshot.plan_id,
            "revision": snapshot.revision,
            "action_id": snapshot.action_id,
            "action_digest": snapshot.action_digest,
            "route_id": snapshot.route_id,
            "classification": snapshot.classification.to_dict(),
            "action": snapshot.action.value,
            "budget_before": snapshot.budget_before.to_dict(),
            "budget_after": snapshot.budget_after.to_dict(),
            "repeated_route_finding": (
                None
                if snapshot.repeated_route_finding is None
                else snapshot.repeated_route_finding.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryDecision:
        """Restore and verify the digest identity of a pending decision."""

        values = _require_exact_mapping(
            data,
            {
                "decision_id",
                "plan_id",
                "revision",
                "action_id",
                "action_digest",
                "route_id",
                "classification",
                "action",
                "budget_before",
                "budget_after",
                "repeated_route_finding",
            },
            "RecoveryDecision",
        )
        serialized_id = _require_sha256(values["decision_id"], "decision_id")
        decision = cls(
            plan_id=cast(str, values["plan_id"]),
            revision=cast(int, values["revision"]),
            action_id=cast(str, values["action_id"]),
            action_digest=cast(str, values["action_digest"]),
            route_id=cast(str, values["route_id"]),
            classification=FailureClassificationBinding.from_dict(values["classification"]),
            action=_restore_action(values["action"]),
            budget_before=RecoveryBudget.from_dict(values["budget_before"]),
            budget_after=RecoveryBudget.from_dict(values["budget_after"]),
            repeated_route_finding=_restore_optional_route_finding(
                values["repeated_route_finding"]
            ),
        )
        if serialized_id != decision.decision_id:
            raise ValueError("decision_id does not match the recovery decision fields")
        return decision


@dataclass(frozen=True, slots=True)
class RecoveryStateSnapshot:
    """A read-only view of one authoritative recovery store revision."""

    plan_id: str
    revision: int
    action_id: str
    action_digest: str
    failure_id: str
    failure_event_id: str
    budget: RecoveryBudget
    pending_decision_id: str | None

    def __post_init__(self) -> None:
        """Validate the store identities, revision, budget, and optional pending ID."""

        _require_nonblank_string(self.plan_id, "plan_id")
        _require_json_safe_integer(self.revision, "revision")
        _require_nonblank_string(self.action_id, "action_id")
        _require_sha256(self.action_digest, "action_digest")
        _require_nonblank_string(self.failure_id, "failure_id")
        _require_nonblank_string(self.failure_event_id, "failure_event_id")
        budget = _snapshot_budget(self.budget, "budget")
        if self.pending_decision_id is not None:
            _require_sha256(self.pending_decision_id, "pending_decision_id")
        object.__setattr__(self, "budget", budget)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible state snapshot."""

        snapshot = _snapshot_state_snapshot(self)
        return {
            "plan_id": snapshot.plan_id,
            "revision": snapshot.revision,
            "action_id": snapshot.action_id,
            "action_digest": snapshot.action_digest,
            "failure_id": snapshot.failure_id,
            "failure_event_id": snapshot.failure_event_id,
            "budget": snapshot.budget.to_dict(),
            "pending_decision_id": snapshot.pending_decision_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryStateSnapshot:
        """Restore an exact JSON-compatible state snapshot."""

        values = _require_exact_mapping(
            data,
            {
                "plan_id",
                "revision",
                "action_id",
                "action_digest",
                "failure_id",
                "failure_event_id",
                "budget",
                "pending_decision_id",
            },
            "RecoveryStateSnapshot",
        )
        return cls(
            plan_id=cast(str, values["plan_id"]),
            revision=cast(int, values["revision"]),
            action_id=cast(str, values["action_id"]),
            action_digest=cast(str, values["action_digest"]),
            failure_id=cast(str, values["failure_id"]),
            failure_event_id=cast(str, values["failure_event_id"]),
            budget=RecoveryBudget.from_dict(values["budget"]),
            pending_decision_id=cast(str | None, values["pending_decision_id"]),
        )


@dataclass(frozen=True, slots=True)
class RecoveryConsumption:
    """The observable result of one recovery proposal consumed by its store."""

    plan_id: str
    decision_id: str
    consumed_revision: int
    current_revision: int
    action: RecoveryAction
    budget: RecoveryBudget

    def __post_init__(self) -> None:
        """Validate an exact one-revision authoritative transition."""

        _require_nonblank_string(self.plan_id, "plan_id")
        _require_sha256(self.decision_id, "decision_id")
        _require_json_safe_integer(self.consumed_revision, "consumed_revision")
        _require_json_safe_integer(self.current_revision, "current_revision")
        if self.consumed_revision >= MAX_JSON_SAFE_INTEGER:
            raise ValueError("consumed_revision cannot advance beyond the JSON-safe maximum")
        if self.current_revision != self.consumed_revision + 1:
            raise ValueError("current_revision must be exactly one after consumed_revision")
        if type(self.action) is not RecoveryAction:
            raise ValueError("action must be an exact RecoveryAction")
        object.__setattr__(self, "budget", _snapshot_budget(self.budget, "budget"))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible consumption record."""

        snapshot = _snapshot_consumption(self)
        return {
            "plan_id": snapshot.plan_id,
            "decision_id": snapshot.decision_id,
            "consumed_revision": snapshot.consumed_revision,
            "current_revision": snapshot.current_revision,
            "action": snapshot.action.value,
            "budget": snapshot.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryConsumption:
        """Restore an exact JSON-compatible consumption record."""

        values = _require_exact_mapping(
            data,
            {
                "plan_id",
                "decision_id",
                "consumed_revision",
                "current_revision",
                "action",
                "budget",
            },
            "RecoveryConsumption",
        )
        return cls(
            plan_id=cast(str, values["plan_id"]),
            decision_id=cast(str, values["decision_id"]),
            consumed_revision=cast(int, values["consumed_revision"]),
            current_revision=cast(int, values["current_revision"]),
            action=_restore_action(values["action"]),
            budget=RecoveryBudget.from_dict(values["budget"]),
        )


@dataclass(frozen=True, slots=True)
class _RecoveryStoreEntry:
    plan_id: str
    revision: int
    action: ActionRecord
    failure: FailureNode
    budget: RecoveryBudget
    pending: RecoveryDecision | None
    consumed: tuple[RecoveryDecision, ...]


class RecoveryStateStore:
    """A runtime-owned recovery ledger whose public handle contains no authoritative fields.

    ``enroll`` is the trust boundary. The runtime owner must authenticate the initial action,
    failure observation, and budget before enrollment. Module-private storage uses weak object
    identity; snapshots returned to callers cannot modify the authoritative state.
    """

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        """Prevent construction that bypasses explicit enrollment."""

        raise TypeError("use RecoveryStateStore.enroll()")

    def __setattr__(self, name: str, value: object) -> None:
        """Reject public attempts to add or replace authority-bearing fields."""

        del name, value
        raise AttributeError("RecoveryStateStore is read-only")

    @classmethod
    def enroll(
        cls,
        *,
        plan_id: str,
        action: ActionRecord,
        failure: FailureNode,
        budget: RecoveryBudget,
    ) -> RecoveryStateStore:
        """Enroll one exact plan, action, failure, and initial finite budget."""

        if cls is not RecoveryStateStore:
            raise ValueError("enrollment requires the exact RecoveryStateStore type")
        entry = _snapshot_store_entry(
            _RecoveryStoreEntry(
                plan_id=_require_nonblank_string(plan_id, "plan_id"),
                revision=0,
                action=_snapshot_action(action),
                failure=_snapshot_failure(failure),
                budget=_snapshot_budget(budget, "budget"),
                pending=None,
                consumed=(),
            )
        )
        store = object.__new__(RecoveryStateStore)
        with _RECOVERY_STORE_LOCK:
            _RECOVERY_STORE_STATE[store] = entry
        return store

    def snapshot(self) -> RecoveryStateSnapshot:
        """Return a detached view of the current authoritative revision."""

        with _RECOVERY_STORE_LOCK:
            entry = _validated_store_entry(self)
            return _entry_snapshot(entry)


_RECOVERY_STORE_LOCK = RLock()
_RECOVERY_STORE_STATE: WeakKeyDictionary[RecoveryStateStore, _RecoveryStoreEntry] = (
    WeakKeyDictionary()
)


def select_recovery(
    store: RecoveryStateStore,
    *,
    expected_revision: int,
    action: ActionRecord,
    classification: FailureClassificationBinding,
    route_id: str,
    repeated_route_finding: RepeatedRouteFinding | None = None,
) -> RecoveryDecision:
    """Reserve one pending proposal against the current authoritative store revision.

    Selection does not consume a retry or replan count. Call ``consume_recovery`` with the returned
    decision to atomically record the transition.
    """

    _require_exact_store(store)
    revision = _require_json_safe_integer(expected_revision, "expected_revision")
    action_snapshot = _snapshot_action(action)
    classification_snapshot = _snapshot_classification(classification)
    route_snapshot = _require_nonblank_string(route_id, "route_id")
    finding_snapshot = _snapshot_optional_route_finding(repeated_route_finding)
    with _RECOVERY_STORE_LOCK:
        entry = _validated_store_entry(store)
        _require_current_revision(entry, revision)
        if entry.pending is not None:
            raise RuntimeError("recovery store already has a pending decision")
        _validate_action_binding(entry, action_snapshot)
        _validate_classification_binding(entry, action_snapshot, classification_snapshot)
        _validate_route_finding(
            entry,
            action_snapshot,
            classification_snapshot,
            route_snapshot,
            finding_snapshot,
        )
        selected_action, budget_after = _select_action(
            classification_snapshot.category,
            entry.budget,
            finding_snapshot is not None,
        )
        decision = RecoveryDecision(
            plan_id=entry.plan_id,
            revision=entry.revision,
            action_id=cast(_ActionFields, action_snapshot).action_id,
            action_digest=action_record_digest(action_snapshot),
            route_id=route_snapshot,
            classification=classification_snapshot,
            action=selected_action,
            budget_before=entry.budget,
            budget_after=budget_after,
            repeated_route_finding=finding_snapshot,
        )
        updated = _RecoveryStoreEntry(
            entry.plan_id,
            entry.revision,
            entry.action,
            entry.failure,
            entry.budget,
            decision,
            entry.consumed,
        )
        _RECOVERY_STORE_STATE[store] = _snapshot_store_entry(updated)
        return _snapshot_decision(decision)


def consume_recovery(
    store: RecoveryStateStore,
    decision: RecoveryDecision,
    *,
    expected_revision: int,
    action: ActionRecord,
) -> RecoveryConsumption:
    """Atomically consume the exact pending decision at the current store revision."""

    _require_exact_store(store)
    revision = _require_json_safe_integer(expected_revision, "expected_revision")
    action_snapshot = _snapshot_action(action)
    decision_snapshot = _snapshot_decision(decision)
    with _RECOVERY_STORE_LOCK:
        entry = _validated_store_entry(store)
        _require_current_revision(entry, revision)
        _validate_action_binding(entry, action_snapshot)
        if entry.pending is None:
            raise RuntimeError("recovery store has no pending decision to consume")
        pending = _snapshot_decision(entry.pending)
        if _canonical_json(pending.to_dict()) != _canonical_json(decision_snapshot.to_dict()):
            raise RuntimeError("decision does not match the authoritative pending decision")
        if entry.revision >= MAX_JSON_SAFE_INTEGER:
            raise RuntimeError("recovery store revision is exhausted")
        new_revision = entry.revision + 1
        updated = _RecoveryStoreEntry(
            entry.plan_id,
            new_revision,
            entry.action,
            entry.failure,
            pending.budget_after,
            None,
            (*entry.consumed, pending),
        )
        _RECOVERY_STORE_STATE[store] = _snapshot_store_entry(updated)
        return RecoveryConsumption(
            plan_id=entry.plan_id,
            decision_id=pending.decision_id,
            consumed_revision=entry.revision,
            current_revision=new_revision,
            action=pending.action,
            budget=pending.budget_after,
        )


def _select_action(
    category: FailureCategory,
    budget: RecoveryBudget,
    repeated_route: bool,
) -> tuple[RecoveryAction, RecoveryBudget]:
    if repeated_route:
        return RecoveryAction.REJECT_ROUTE, budget
    if category is FailureCategory.TRANSIENT_RUNTIME_ERROR:
        if budget.retries_used >= budget.max_retries:
            return RecoveryAction.EXHAUSTED, budget
        return RecoveryAction.RETRY, RecoveryBudget(
            budget.max_retries,
            budget.retries_used + 1,
            budget.max_replans,
            budget.replans_used,
        )
    if category in _REPLAN_CATEGORIES:
        if budget.replans_used >= budget.max_replans:
            return RecoveryAction.EXHAUSTED, budget
        return RecoveryAction.REPLAN, RecoveryBudget(
            budget.max_retries,
            budget.retries_used,
            budget.max_replans,
            budget.replans_used + 1,
        )
    return _DIRECT_ACTIONS[category], budget


def _validate_transition(
    category: FailureCategory,
    action: RecoveryAction,
    before: RecoveryBudget,
    after: RecoveryBudget,
    repeated_route: bool,
) -> None:
    expected_action, expected_after = _select_action(category, before, repeated_route)
    if action is not expected_action or after != expected_after:
        raise ValueError("recovery action and budget transition do not match the policy")


def _validate_action_binding(entry: _RecoveryStoreEntry, action: ActionRecord) -> None:
    if action_record_digest(entry.action) != action_record_digest(action):
        raise ValueError("action does not match the action enrolled in the recovery store")


def _validate_classification_binding(
    entry: _RecoveryStoreEntry,
    action: ActionRecord,
    classification: FailureClassificationBinding,
) -> None:
    action_id = cast(_ActionFields, action).action_id
    if classification.action_id != action_id:
        raise ValueError("classification action_id does not match the current action")
    if classification.action_digest != action_record_digest(action):
        raise ValueError("classification action_digest does not match the current action digest")
    if _canonical_json(classification.failure.to_dict()) != _canonical_json(
        entry.failure.to_dict()
    ):
        raise ValueError("classification failure does not match the enrolled failure")


def _validate_route_finding(
    entry: _RecoveryStoreEntry,
    action: ActionRecord,
    classification: FailureClassificationBinding,
    route_id: str,
    finding: RepeatedRouteFinding | None,
) -> None:
    if finding is None:
        return
    action_id = cast(_ActionFields, action).action_id
    expected = (
        finding.route_id == route_id
        and finding.prior_action_id == action_id
        and finding.prior_action_digest == action_record_digest(action)
        and finding.prior_revision < entry.revision
        and finding.current_failure_id == classification.failure.failure_id
        and finding.current_failure_event_id == classification.failure.event_id
        and finding.classification_id == classification.classification_id
        and finding.classification_verification_event_id == classification.verification_event_id
    )
    prior = [item for item in entry.consumed if item.decision_id == finding.prior_decision_id]
    if len(prior) != 1:
        expected = False
    else:
        prior_decision = prior[0]
        expected = expected and (
            prior_decision.route_id == finding.route_id
            and prior_decision.action_id == finding.prior_action_id
            and prior_decision.action_digest == finding.prior_action_digest
            and prior_decision.revision == finding.prior_revision
        )
    if not expected:
        raise ValueError("repeated route finding does not match authoritative recovery history")


def _validate_decision_route_finding(
    revision: int,
    action_id: str,
    action_digest: str,
    route_id: str,
    classification: FailureClassificationBinding,
    finding: RepeatedRouteFinding | None,
) -> None:
    if finding is None:
        return
    matches = (
        finding.route_id == route_id
        and finding.prior_action_id == action_id
        and finding.prior_action_digest == action_digest
        and finding.prior_revision < revision
        and finding.current_failure_id == classification.failure.failure_id
        and finding.current_failure_event_id == classification.failure.event_id
        and finding.classification_id == classification.classification_id
        and finding.classification_verification_event_id == classification.verification_event_id
    )
    if not matches:
        raise ValueError("decision does not match its repeated route finding")


def _require_current_revision(entry: _RecoveryStoreEntry, expected_revision: int) -> None:
    if expected_revision != entry.revision:
        raise RuntimeError(
            f"stale recovery revision {expected_revision}; current revision is {entry.revision}"
        )


def _require_exact_store(store: object) -> None:
    if type(store) is not RecoveryStateStore:
        raise ValueError("store must be an exact RecoveryStateStore")


def _validated_store_entry(store: RecoveryStateStore) -> _RecoveryStoreEntry:
    entry = _RECOVERY_STORE_STATE.get(store)
    if entry is None:
        raise ValueError("store is not enrolled in authoritative recovery storage")
    return _snapshot_store_entry(entry)


def _snapshot_store_entry(entry: object) -> _RecoveryStoreEntry:
    if type(entry) is not _RecoveryStoreEntry:
        raise ValueError("authoritative recovery storage contains an invalid entry")
    _require_nonblank_string(entry.plan_id, "plan_id")
    revision = _require_json_safe_integer(entry.revision, "revision")
    action = _snapshot_action(entry.action)
    failure = _snapshot_failure(entry.failure)
    budget = _snapshot_budget(entry.budget, "budget")
    pending = _snapshot_optional_decision(entry.pending)
    if type(entry.consumed) is not tuple:
        raise ValueError("authoritative consumed decisions must be an exact tuple")
    consumed = tuple(_snapshot_decision(item) for item in entry.consumed)
    if revision != len(consumed):
        raise ValueError("authoritative revision must equal consumed decision count")
    decision_ids = tuple(item.decision_id for item in consumed)
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("authoritative consumed decision IDs must be unique")
    for index, decision in enumerate(consumed):
        _validate_stored_decision(entry.plan_id, action, failure, decision, index)
    if pending is not None:
        _validate_stored_decision(entry.plan_id, action, failure, pending, revision)
        if pending.budget_before != budget:
            raise ValueError("pending decision budget does not match authoritative budget")
    return _RecoveryStoreEntry(
        entry.plan_id,
        revision,
        action,
        failure,
        budget,
        pending,
        consumed,
    )


def _validate_stored_decision(
    plan_id: str,
    action: ActionRecord,
    failure: FailureNode,
    decision: RecoveryDecision,
    revision: int,
) -> None:
    if decision.plan_id != plan_id or decision.revision != revision:
        raise ValueError("stored decision does not match its plan revision")
    if decision.action_id != cast(_ActionFields, action).action_id:
        raise ValueError("stored decision does not match its enrolled action")
    if decision.action_digest != action_record_digest(action):
        raise ValueError("stored decision does not match its enrolled action digest")
    if _canonical_json(decision.classification.failure.to_dict()) != _canonical_json(
        failure.to_dict()
    ):
        raise ValueError("stored decision does not match its enrolled failure")


def _entry_snapshot(entry: _RecoveryStoreEntry) -> RecoveryStateSnapshot:
    action = _snapshot_action(entry.action)
    failure = _snapshot_failure(entry.failure)
    return RecoveryStateSnapshot(
        plan_id=entry.plan_id,
        revision=entry.revision,
        action_id=cast(_ActionFields, action).action_id,
        action_digest=action_record_digest(action),
        failure_id=failure.failure_id,
        failure_event_id=failure.event_id,
        budget=entry.budget,
        pending_decision_id=None if entry.pending is None else entry.pending.decision_id,
    )


def _snapshot_budget(value: object, field_name: str) -> RecoveryBudget:
    if type(value) is not RecoveryBudget:
        raise ValueError(f"{field_name} must be an exact RecoveryBudget")
    try:
        return RecoveryBudget(
            value.max_retries,
            value.retries_used,
            value.max_replans,
            value.replans_used,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains an invalid RecoveryBudget: {exc}") from exc


def _snapshot_failure(value: object) -> FailureNode:
    if not isinstance(value, FailureNode) or type(value) is not FailureNode:
        raise ValueError("failure must be an exact FailureNode")
    try:
        return FailureNode(
            value.failure_id,
            value.event_id,
            value.summary,
            value.observed_at,
            value.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"failure contains an invalid FailureNode: {exc}") from exc


def _snapshot_classification(value: object) -> FailureClassificationBinding:
    if type(value) is not FailureClassificationBinding:
        raise ValueError("classification must be an exact FailureClassificationBinding")
    if type(value.evidence_refs) is not tuple or type(value.verifier_refs) is not tuple:
        raise ValueError("classification reference fields must remain exact tuples")
    try:
        return FailureClassificationBinding(
            value.failure,
            value.action_id,
            value.action_digest,
            value.category,
            value.verification_event_id,
            value.evidence_refs,
            value.verifier_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"classification contains invalid binding state: {exc}") from exc


def _snapshot_route_finding(value: object) -> RepeatedRouteFinding:
    if type(value) is not RepeatedRouteFinding:
        raise ValueError("repeated route finding must be an exact RepeatedRouteFinding")
    if type(value.evidence_refs) is not tuple or type(value.verifier_refs) is not tuple:
        raise ValueError("repeated route finding references must remain exact tuples")
    try:
        return RepeatedRouteFinding(
            value.route_id,
            value.prior_action_id,
            value.prior_action_digest,
            value.prior_decision_id,
            value.prior_revision,
            value.current_failure_id,
            value.current_failure_event_id,
            value.classification_id,
            value.classification_verification_event_id,
            value.evidence_refs,
            value.verifier_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"repeated route finding contains invalid state: {exc}") from exc


def _snapshot_optional_route_finding(value: object) -> RepeatedRouteFinding | None:
    if value is None:
        return None
    return _snapshot_route_finding(value)


def _snapshot_decision(value: object) -> RecoveryDecision:
    if type(value) is not RecoveryDecision:
        raise ValueError("decision must be an exact RecoveryDecision")
    try:
        return RecoveryDecision(
            value.plan_id,
            value.revision,
            value.action_id,
            value.action_digest,
            value.route_id,
            value.classification,
            value.action,
            value.budget_before,
            value.budget_after,
            value.repeated_route_finding,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decision contains invalid recovery state: {exc}") from exc


def _snapshot_optional_decision(value: object) -> RecoveryDecision | None:
    if value is None:
        return None
    return _snapshot_decision(value)


def _snapshot_state_snapshot(value: object) -> RecoveryStateSnapshot:
    if type(value) is not RecoveryStateSnapshot:
        raise ValueError("state snapshot must be an exact RecoveryStateSnapshot")
    return RecoveryStateSnapshot(
        value.plan_id,
        value.revision,
        value.action_id,
        value.action_digest,
        value.failure_id,
        value.failure_event_id,
        value.budget,
        value.pending_decision_id,
    )


def _snapshot_consumption(value: object) -> RecoveryConsumption:
    if type(value) is not RecoveryConsumption:
        raise ValueError("consumption must be an exact RecoveryConsumption")
    return RecoveryConsumption(
        value.plan_id,
        value.decision_id,
        value.consumed_revision,
        value.current_revision,
        value.action,
        value.budget,
    )


def _snapshot_evidence(values: object, field_name: str) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be an ordered array")
    references = _snapshot_canonical_evidence(values)
    identities = {(reference.reference_id, reference.source_kind) for reference in references}
    if len(identities) != len(references):
        raise ValueError(f"{field_name} must be unique")
    return references


def _restore_evidence(values: object, field_name: str) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return tuple(EvidenceReference.from_dict(value) for value in values)


def _require_observable_subset(
    selected: tuple[EvidenceReference, ...],
    available: tuple[EvidenceReference, ...],
    field_name: str,
) -> None:
    if not selected or not any(reference.is_observable for reference in selected):
        raise ValueError(f"{field_name} requires at least one observable reference")
    available_ids = {(reference.reference_id, reference.source_kind) for reference in available}
    if any(
        (reference.reference_id, reference.source_kind) not in available_ids
        for reference in selected
    ):
        raise ValueError(f"{field_name} must be a subset of the bound failure evidence")


def _snapshot_identifiers(
    values: object,
    field_name: str,
    *,
    required: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be an ordered array")
    identifiers = tuple(_require_nonblank_string(value, field_name) for value in values)
    if required and not identifiers:
        raise ValueError(f"{field_name} must contain at least one reference")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field_name} must be unique")
    return identifiers


def _require_json_safe_integer(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_JSON_SAFE_INTEGER:
        raise ValueError(
            f"{field_name} must be an exact integer from 0 through {MAX_JSON_SAFE_INTEGER}"
        )
    return value


def _restore_category(value: object) -> FailureCategory:
    if type(value) is not str:
        raise ValueError("category must be a built-in string")
    try:
        return FailureCategory(value)
    except ValueError as exc:
        raise ValueError(f"unknown category {value!r}") from exc


def _restore_action(value: object) -> RecoveryAction:
    if type(value) is not str:
        raise ValueError("action must be a built-in string")
    try:
        return RecoveryAction(value)
    except ValueError as exc:
        raise ValueError(f"unknown action {value!r}") from exc


def _restore_optional_route_finding(value: object) -> RepeatedRouteFinding | None:
    if value is None:
        return None
    return RepeatedRouteFinding.from_dict(value)


def _classification_digest(binding: FailureClassificationBinding) -> str:
    return _digest(
        {
            "failure": binding.failure.to_dict(),
            "action_id": binding.action_id,
            "action_digest": binding.action_digest,
            "category": binding.category.value,
            "verification_event_id": binding.verification_event_id,
            "evidence_refs": [reference.to_dict() for reference in binding.evidence_refs],
            "verifier_refs": list(binding.verifier_refs),
        }
    )


def _route_finding_digest(finding: RepeatedRouteFinding) -> str:
    return _digest(
        {
            "route_id": finding.route_id,
            "prior_action_id": finding.prior_action_id,
            "prior_action_digest": finding.prior_action_digest,
            "prior_decision_id": finding.prior_decision_id,
            "prior_revision": finding.prior_revision,
            "current_failure_id": finding.current_failure_id,
            "current_failure_event_id": finding.current_failure_event_id,
            "classification_id": finding.classification_id,
            "classification_verification_event_id": (finding.classification_verification_event_id),
            "evidence_refs": [reference.to_dict() for reference in finding.evidence_refs],
            "verifier_refs": list(finding.verifier_refs),
        }
    )


def _decision_digest(decision: RecoveryDecision) -> str:
    return _digest(
        {
            "plan_id": decision.plan_id,
            "revision": decision.revision,
            "action_id": decision.action_id,
            "action_digest": decision.action_digest,
            "route_id": decision.route_id,
            "classification": decision.classification.to_dict(),
            "action": decision.action.value,
            "budget_before": decision.budget_before.to_dict(),
            "budget_after": decision.budget_after.to_dict(),
            "repeated_route_finding": (
                None
                if decision.repeated_route_finding is None
                else decision.repeated_route_finding.to_dict()
            ),
        }
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "MAX_JSON_SAFE_INTEGER",
    "FailureCategory",
    "FailureClassificationBinding",
    "RepeatedRouteFinding",
    "RecoveryAction",
    "RecoveryBudget",
    "RecoveryConsumption",
    "RecoveryDecision",
    "RecoveryStateSnapshot",
    "RecoveryStateStore",
    "consume_recovery",
    "select_recovery",
]
