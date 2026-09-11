"""Immutable, deterministic, and bounded recovery policy records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

from gepa_mindfulness.core.evidence import EvidenceReference

from .failure_graph import FailureNode
from .state import _require_exact_mapping, _require_nonblank_string, _snapshot_evidence_refs


class FailureCategory(str, Enum):
    """A verified failure class that selects one recovery intervention level."""

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
    """The exact next recovery operation or terminal recovery status."""

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


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    """Finite retry and replan maxima plus their already-consumed counts."""

    max_retries: int
    retries_used: int
    max_replans: int
    replans_used: int

    def __post_init__(self) -> None:
        """Reject non-exact, negative, or internally inconsistent counts."""

        for field_name, value in (
            ("max_retries", self.max_retries),
            ("retries_used", self.retries_used),
            ("max_replans", self.max_replans),
            ("replans_used", self.replans_used),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be an exact nonnegative integer")
        if self.retries_used > self.max_retries or self.replans_used > self.max_replans:
            raise ValueError("recovery counts used cannot exceed their declared maximum")

    def to_dict(self) -> dict[str, int]:
        """Return the exact JSON-compatible finite budget snapshot."""

        snapshot = _snapshot_budget(self, "budget")
        return {
            "max_retries": snapshot.max_retries,
            "retries_used": snapshot.retries_used,
            "max_replans": snapshot.max_replans,
            "replans_used": snapshot.replans_used,
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryBudget:
        """Restore a budget from its exact JSON-compatible representation."""

        values = _require_exact_mapping(
            data,
            {"max_retries", "retries_used", "max_replans", "replans_used"},
            "RecoveryBudget",
        )
        return cls(
            max_retries=cast(int, values["max_retries"]),
            retries_used=cast(int, values["retries_used"]),
            max_replans=cast(int, values["max_replans"]),
            replans_used=cast(int, values["replans_used"]),
        )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """One evidence-bound recovery choice and its exact budget transition."""

    failure: FailureNode
    action_id: str
    route_id: str
    failure_category: FailureCategory
    action: RecoveryAction
    budget_before: RecoveryBudget
    budget_after: RecoveryBudget
    category_evidence_refs: tuple[EvidenceReference, ...]
    repeated_failed_route: bool = False
    repeated_route_evidence_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot nested records and validate the complete recovery transition."""

        failure = _snapshot_failure(self.failure)
        action_id = _require_nonblank_string(self.action_id, "action_id")
        route_id = _require_nonblank_string(self.route_id, "route_id")
        if type(self.failure_category) is not FailureCategory:
            raise ValueError("failure_category must be an exact FailureCategory")
        if type(self.action) is not RecoveryAction:
            raise ValueError("action must be an exact RecoveryAction")
        budget_before = _snapshot_budget(self.budget_before, "budget_before")
        budget_after = _snapshot_budget(self.budget_after, "budget_after")
        category_evidence = _snapshot_ordered_evidence(
            self.category_evidence_refs,
            "category_evidence_refs",
        )
        _validate_category_evidence(failure, category_evidence)
        repeated = _require_exact_bool(self.repeated_failed_route, "repeated_failed_route")
        repeated_evidence = _snapshot_ordered_evidence(
            self.repeated_route_evidence_refs,
            "repeated_route_evidence_refs",
        )
        _validate_repeated_route_evidence(repeated, repeated_evidence)
        _validate_transition(
            self.failure_category,
            self.action,
            budget_before,
            budget_after,
            repeated,
        )
        object.__setattr__(self, "failure", failure)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "route_id", route_id)
        object.__setattr__(self, "budget_before", budget_before)
        object.__setattr__(self, "budget_after", budget_after)
        object.__setattr__(self, "category_evidence_refs", category_evidence)
        object.__setattr__(self, "repeated_failed_route", repeated)
        object.__setattr__(self, "repeated_route_evidence_refs", repeated_evidence)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible recovery decision snapshot."""

        snapshot = _snapshot_decision(self)
        return {
            "failure": snapshot.failure.to_dict(),
            "action_id": snapshot.action_id,
            "route_id": snapshot.route_id,
            "failure_category": snapshot.failure_category.value,
            "action": snapshot.action.value,
            "budget_before": snapshot.budget_before.to_dict(),
            "budget_after": snapshot.budget_after.to_dict(),
            "category_evidence_refs": [
                reference.to_dict() for reference in snapshot.category_evidence_refs
            ],
            "repeated_failed_route": snapshot.repeated_failed_route,
            "repeated_route_evidence_refs": [
                reference.to_dict() for reference in snapshot.repeated_route_evidence_refs
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> RecoveryDecision:
        """Restore and revalidate an exact JSON-compatible recovery decision."""

        values = _require_exact_mapping(
            data,
            {
                "failure",
                "action_id",
                "route_id",
                "failure_category",
                "action",
                "budget_before",
                "budget_after",
                "category_evidence_refs",
                "repeated_failed_route",
                "repeated_route_evidence_refs",
            },
            "RecoveryDecision",
        )
        return cls(
            failure=FailureNode.from_dict(values["failure"]),
            action_id=cast(str, values["action_id"]),
            route_id=cast(str, values["route_id"]),
            failure_category=_restore_category(values["failure_category"]),
            action=_restore_action(values["action"]),
            budget_before=RecoveryBudget.from_dict(values["budget_before"]),
            budget_after=RecoveryBudget.from_dict(values["budget_after"]),
            category_evidence_refs=_restore_evidence_refs(
                values["category_evidence_refs"],
                "category_evidence_refs",
            ),
            repeated_failed_route=cast(bool, values["repeated_failed_route"]),
            repeated_route_evidence_refs=_restore_evidence_refs(
                values["repeated_route_evidence_refs"],
                "repeated_route_evidence_refs",
            ),
        )


def select_recovery(
    *,
    failure: FailureNode,
    action_id: str,
    route_id: str,
    category: FailureCategory,
    budget: RecoveryBudget,
    category_evidence_refs: tuple[EvidenceReference, ...],
    repeated_failed_route: bool = False,
    repeated_route_evidence_refs: tuple[EvidenceReference, ...] = (),
) -> RecoveryDecision:
    """Select one deterministic recovery action and consume only its bounded counter."""

    failure_snapshot = _snapshot_failure(failure)
    action_snapshot = _require_nonblank_string(action_id, "action_id")
    route_snapshot = _require_nonblank_string(route_id, "route_id")
    if type(category) is not FailureCategory:
        raise ValueError("category must be an exact FailureCategory")
    budget_snapshot = _snapshot_budget(budget, "budget")
    category_evidence = _snapshot_ordered_evidence(
        category_evidence_refs,
        "category_evidence_refs",
    )
    _validate_category_evidence(failure_snapshot, category_evidence)
    repeated = _require_exact_bool(repeated_failed_route, "repeated_failed_route")
    repeated_evidence = _snapshot_ordered_evidence(
        repeated_route_evidence_refs,
        "repeated_route_evidence_refs",
    )
    _validate_repeated_route_evidence(repeated, repeated_evidence)

    action, budget_after = _select_action(category, budget_snapshot, repeated)
    return RecoveryDecision(
        failure=failure_snapshot,
        action_id=action_snapshot,
        route_id=route_snapshot,
        failure_category=category,
        action=action,
        budget_before=budget_snapshot,
        budget_after=budget_after,
        category_evidence_refs=category_evidence,
        repeated_failed_route=repeated,
        repeated_route_evidence_refs=repeated_evidence,
    )


def _select_action(
    category: FailureCategory,
    budget: RecoveryBudget,
    repeated_failed_route: bool,
) -> tuple[RecoveryAction, RecoveryBudget]:
    if repeated_failed_route:
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
    budget_before: RecoveryBudget,
    budget_after: RecoveryBudget,
    repeated_failed_route: bool,
) -> None:
    expected_action, expected_budget = _select_action(
        category,
        budget_before,
        repeated_failed_route,
    )
    if action is not expected_action or budget_after != expected_budget:
        raise ValueError("recovery action and budget transition do not match the policy")


def _validate_repeated_route_evidence(
    repeated_failed_route: bool,
    evidence_refs: tuple[EvidenceReference, ...],
) -> None:
    if repeated_failed_route:
        if not evidence_refs or not any(reference.is_observable for reference in evidence_refs):
            raise ValueError("observable evidence is required for a repeated failed route")
    elif evidence_refs:
        raise ValueError("false repeated_failed_route status cannot carry repeated-route evidence")


def _validate_category_evidence(
    failure: FailureNode,
    evidence_refs: tuple[EvidenceReference, ...],
) -> None:
    if not evidence_refs or not any(reference.is_observable for reference in evidence_refs):
        raise ValueError("category requires at least one observable evidence reference")
    failure_evidence = {
        (reference.reference_id, reference.source_kind) for reference in failure.evidence_refs
    }
    if any(
        (reference.reference_id, reference.source_kind) not in failure_evidence
        for reference in evidence_refs
    ):
        raise ValueError("category evidence must belong to the bound failure")


def _snapshot_budget(value: object, field_name: str) -> RecoveryBudget:
    if type(value) is not RecoveryBudget:
        raise ValueError(f"{field_name} must be an exact RecoveryBudget")
    try:
        return RecoveryBudget(
            max_retries=value.max_retries,
            retries_used=value.retries_used,
            max_replans=value.max_replans,
            replans_used=value.replans_used,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains an invalid recovery budget: {exc}") from exc


def _snapshot_failure(value: object) -> FailureNode:
    if not isinstance(value, FailureNode) or type(value) is not FailureNode:
        raise ValueError("failure must be an exact FailureNode")
    try:
        return FailureNode(
            failure_id=value.failure_id,
            event_id=value.event_id,
            summary=value.summary,
            observed_at=value.observed_at,
            evidence_refs=value.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"failure contains an invalid FailureNode: {exc}") from exc


def _snapshot_ordered_evidence(
    values: object,
    field_name: str,
) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be an ordered array")
    references = _snapshot_evidence_refs(values)
    identities = {(reference.reference_id, reference.source_kind) for reference in references}
    if len(identities) != len(references):
        raise ValueError(f"{field_name} must be unique")
    return references


def _restore_evidence_refs(
    values: object,
    field_name: str,
) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"RecoveryDecision {field_name} must be an array")
    return tuple(EvidenceReference.from_dict(value) for value in values)


def _snapshot_decision(value: object) -> RecoveryDecision:
    if type(value) is not RecoveryDecision:
        raise ValueError("decision must be an exact RecoveryDecision")
    try:
        return RecoveryDecision(
            failure=value.failure,
            action_id=value.action_id,
            route_id=value.route_id,
            failure_category=value.failure_category,
            action=value.action,
            budget_before=value.budget_before,
            budget_after=value.budget_after,
            category_evidence_refs=value.category_evidence_refs,
            repeated_failed_route=value.repeated_failed_route,
            repeated_route_evidence_refs=value.repeated_route_evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decision contains invalid recovery state: {exc}") from exc


def _restore_category(value: object) -> FailureCategory:
    if type(value) is not str:
        raise ValueError("RecoveryDecision failure_category must be a built-in string")
    try:
        return FailureCategory(value)
    except ValueError as exc:
        raise ValueError(f"unknown RecoveryDecision failure_category {value!r}") from exc


def _restore_action(value: object) -> RecoveryAction:
    if type(value) is not str:
        raise ValueError("RecoveryDecision action must be a built-in string")
    try:
        return RecoveryAction(value)
    except ValueError as exc:
        raise ValueError(f"unknown RecoveryDecision action {value!r}") from exc


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be an exact boolean")
    return value


__all__ = [
    "FailureCategory",
    "RecoveryAction",
    "RecoveryBudget",
    "RecoveryDecision",
    "select_recovery",
]
