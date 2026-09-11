"""Tests for deterministic, evidence-bound, bounded recovery decisions."""

import json
from dataclasses import FrozenInstanceError
from enum import Enum
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.failure_graph import FailureNode
from gepa_mindfulness.verification.recovery import (
    FailureCategory,
    RecoveryAction,
    RecoveryBudget,
    RecoveryDecision,
    select_recovery,
)


def _observable(reference_id: str = "observation:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _private(reference_id: str = "reasoning:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.PRIVATE_REASONING)


def _failure(
    failure_id: str = "failure-1",
    *,
    observed_at: str = "2026-09-10T12:00:00Z",
) -> FailureNode:
    return FailureNode(
        failure_id=failure_id,
        event_id=f"event:{failure_id}",
        summary=f"Observed {failure_id}",
        observed_at=observed_at,
        evidence_refs=(_observable(f"observation:{failure_id}"),),
    )


def _select(
    category: FailureCategory,
    budget: RecoveryBudget | None = None,
    **changes: object,
) -> RecoveryDecision:
    values: dict[str, object] = {
        "failure": _failure(),
        "action_id": "action-1",
        "route_id": "route-1",
        "category": category,
        "budget": RecoveryBudget(2, 0, 2, 0),
    }
    if budget is not None:
        values["budget"] = budget
    values.update(changes)
    failure = values["failure"]
    if "category_evidence_refs" not in values and isinstance(failure, FailureNode):
        values["category_evidence_refs"] = (
            failure.evidence_refs if type(failure) is FailureNode else ()
        )
    elif "category_evidence_refs" not in values:
        values["category_evidence_refs"] = ()
    return select_recovery(**cast(Any, values))


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (FailureCategory.TRANSIENT_RUNTIME_ERROR, RecoveryAction.RETRY),
        (FailureCategory.ARGUMENT_ERROR, RecoveryAction.REPAIR_ARGUMENTS),
        (FailureCategory.SCHEMA_ERROR, RecoveryAction.REPAIR_ARGUMENTS),
        (FailureCategory.INPUT_ERROR, RecoveryAction.REPAIR_ARGUMENTS),
        (FailureCategory.STRATEGY_FAILURE, RecoveryAction.REPLAN),
        (FailureCategory.DEPENDENCY_FAILURE, RecoveryAction.REPLAN),
        (FailureCategory.MISSING_USER_FACT, RecoveryAction.REQUEST_CLARIFICATION),
        (FailureCategory.CONSEQUENTIAL_AMBIGUITY, RecoveryAction.ESCALATE),
        (FailureCategory.UNSAFE_UNCERTAINTY, RecoveryAction.ABSTAIN),
    ],
)
def test_failure_categories_map_to_distinct_recovery_actions(
    category: FailureCategory,
    expected: RecoveryAction,
) -> None:
    """Catch a failure category selecting the wrong intervention level."""

    assert _select(category).action is expected


def test_retry_and_replan_consume_only_their_own_budget() -> None:
    """Catch retry and replan sharing a counter or mutating the caller's budget."""

    budget = RecoveryBudget(max_retries=2, retries_used=1, max_replans=4, replans_used=2)

    retry = _select(FailureCategory.TRANSIENT_RUNTIME_ERROR, budget)
    replan = _select(FailureCategory.STRATEGY_FAILURE, budget)

    assert retry.budget_before == budget
    assert retry.budget_after == RecoveryBudget(2, 2, 4, 2)
    assert replan.budget_before == budget
    assert replan.budget_after == RecoveryBudget(2, 1, 4, 3)
    assert budget == RecoveryBudget(2, 1, 4, 2)


@pytest.mark.parametrize(
    ("category", "budget"),
    [
        (FailureCategory.TRANSIENT_RUNTIME_ERROR, RecoveryBudget(0, 0, 1, 0)),
        (FailureCategory.TRANSIENT_RUNTIME_ERROR, RecoveryBudget(2, 2, 1, 0)),
        (FailureCategory.STRATEGY_FAILURE, RecoveryBudget(1, 0, 0, 0)),
        (FailureCategory.DEPENDENCY_FAILURE, RecoveryBudget(1, 0, 3, 3)),
    ],
)
def test_exact_budget_boundary_returns_exhausted_without_consuming(
    category: FailureCategory,
    budget: RecoveryBudget,
) -> None:
    """Catch retry or replan being permitted at or beyond its explicit maximum."""

    decision = _select(category, budget)

    assert decision.action is RecoveryAction.EXHAUSTED
    assert decision.budget_before == budget
    assert decision.budget_after == budget


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_retries", -1),
        ("retries_used", -1),
        ("max_replans", -1),
        ("replans_used", -1),
        ("max_retries", True),
        ("retries_used", False),
        ("max_replans", 1.0),
        ("replans_used", float("nan")),
        ("max_retries", float("inf")),
        ("max_replans", None),
        ("max_retries", "unlimited"),
    ],
)
def test_budget_rejects_negative_noninteger_and_unlimited_values(
    field_name: str,
    value: object,
) -> None:
    """Catch an invalid count creating an unbounded or ambiguous recovery budget."""

    values: dict[str, object] = {
        "max_retries": 1,
        "retries_used": 0,
        "max_replans": 1,
        "replans_used": 0,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        RecoveryBudget(**cast(Any, values))


@pytest.mark.parametrize(
    ("retries_used", "max_retries", "replans_used", "max_replans"),
    [(2, 1, 0, 1), (0, 1, 2, 1)],
)
def test_budget_rejects_a_counter_beyond_its_maximum(
    retries_used: int,
    max_retries: int,
    replans_used: int,
    max_replans: int,
) -> None:
    """Catch persisted usage exceeding the declared finite budget."""

    with pytest.raises(ValueError, match="used.*maximum"):
        RecoveryBudget(max_retries, retries_used, max_replans, replans_used)


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.TRANSIENT_RUNTIME_ERROR,
        FailureCategory.ARGUMENT_ERROR,
        FailureCategory.STRATEGY_FAILURE,
    ],
)
def test_verified_repeated_route_is_rejected_before_any_recovery(category: FailureCategory) -> None:
    """Catch a known-failed route being retried, repaired, or replanned unchanged."""

    budget = RecoveryBudget(3, 1, 3, 1)
    decision = _select(
        category,
        budget,
        repeated_failed_route=True,
        repeated_route_evidence_refs=(_observable("verification:route-repeat"),),
    )

    assert decision.action is RecoveryAction.REJECT_ROUTE
    assert decision.repeated_failed_route is True
    assert decision.budget_after == budget
    assert decision.repeated_route_evidence_refs == (_observable("verification:route-repeat"),)


def test_repeated_route_requires_observable_evidence_and_false_status_rejects_evidence() -> None:
    """Catch a route-repetition veto based only on private or contradictory provenance."""

    with pytest.raises(ValueError, match="observable.*repeated"):
        _select(
            FailureCategory.TRANSIENT_RUNTIME_ERROR,
            repeated_failed_route=True,
            repeated_route_evidence_refs=(_private(),),
        )
    with pytest.raises(ValueError, match="false.*evidence"):
        _select(
            FailureCategory.TRANSIENT_RUNTIME_ERROR,
            repeated_failed_route=False,
            repeated_route_evidence_refs=(_observable("verification:not-repeated"),),
        )


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_repeated_route_status_requires_an_exact_boolean(value: object) -> None:
    """Catch truthy or falsy coercion changing whether a route is rejected."""

    with pytest.raises(ValueError, match="repeated_failed_route"):
        _select(
            FailureCategory.TRANSIENT_RUNTIME_ERROR,
            repeated_failed_route=value,
        )


def test_decision_binds_exact_failure_action_and_route_identity() -> None:
    """Catch a recovery decision losing the observation or operation it governs."""

    failure = _failure("failure-bound")
    decision = _select(
        FailureCategory.DEPENDENCY_FAILURE,
        failure=failure,
        action_id="action-bound",
        route_id="route-bound",
    )

    assert decision.failure == failure
    assert decision.failure is not failure
    assert decision.action_id == "action-bound"
    assert decision.route_id == "route-bound"
    assert decision.failure.evidence_refs == (_observable("observation:failure-bound"),)
    assert decision.category_evidence_refs == (_observable("observation:failure-bound"),)


def test_failure_category_requires_observable_evidence_from_the_bound_failure() -> None:
    """Catch a categorical recovery branch using absent, private, or unrelated evidence."""

    with pytest.raises(ValueError, match="category.*observable"):
        _select(FailureCategory.ARGUMENT_ERROR, category_evidence_refs=())
    with pytest.raises(ValueError, match="category.*observable"):
        _select(
            FailureCategory.ARGUMENT_ERROR,
            category_evidence_refs=(_private(),),
        )
    with pytest.raises(ValueError, match="category.*failure"):
        _select(
            FailureCategory.ARGUMENT_ERROR,
            category_evidence_refs=(_observable("observation:unrelated"),),
        )


@pytest.mark.parametrize("field_name", ["action_id", "route_id"])
@pytest.mark.parametrize("value", ["", "   ", 1, None])
def test_selection_rejects_noncanonical_action_and_route_ids(
    field_name: str,
    value: object,
) -> None:
    """Catch a recovery decision detached from a canonical action or route identity."""

    with pytest.raises(ValueError, match=field_name):
        _select(FailureCategory.ARGUMENT_ERROR, **cast(Any, {field_name: value}))


def test_selection_rejects_noncanonical_failure_category_and_failure_node() -> None:
    """Catch string coercion or a lookalike failure record selecting a recovery branch."""

    with pytest.raises(ValueError, match="category"):
        _select(cast(Any, FailureCategory.TRANSIENT_RUNTIME_ERROR.value))
    with pytest.raises(ValueError, match="failure"):
        _select(FailureCategory.ARGUMENT_ERROR, failure=object())


def test_selection_ignores_chronology_when_category_and_identity_are_unchanged() -> None:
    """Catch event timestamps being treated as failure category or causal support."""

    early = _select(
        FailureCategory.ARGUMENT_ERROR,
        failure=_failure("same-failure", observed_at="2026-09-10T01:00:00Z"),
    )
    late = _select(
        FailureCategory.ARGUMENT_ERROR,
        failure=_failure("same-failure", observed_at="2026-09-10T23:00:00Z"),
    )

    assert early.action is RecoveryAction.REPAIR_ARGUMENTS
    assert late.action is RecoveryAction.REPAIR_ARGUMENTS


def test_select_recovery_revalidates_corrupted_inputs_at_use_time() -> None:
    """Catch frozen-record corruption bypassing validation at the decision boundary."""

    budget = RecoveryBudget(2, 0, 2, 0)
    failure = _failure()
    object.__setattr__(budget, "max_retries", -1)
    object.__setattr__(failure, "failure_id", "")

    with pytest.raises(ValueError, match="budget"):
        _select(FailureCategory.TRANSIENT_RUNTIME_ERROR, budget)
    with pytest.raises(ValueError, match="failure"):
        _select(FailureCategory.ARGUMENT_ERROR, failure=failure)


def test_recovery_records_are_frozen_and_slotted() -> None:
    """Catch recovery inputs or decisions acquiring mutable state or aliases."""

    budget = RecoveryBudget(1, 0, 1, 0)
    decision = _select(FailureCategory.ARGUMENT_ERROR, budget)

    with pytest.raises(FrozenInstanceError):
        budget.retries_used = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.route_id = "other"  # type: ignore[misc]
    assert not hasattr(budget, "__dict__")
    assert not hasattr(decision, "__dict__")


def test_recovery_json_round_trip_is_exact_and_revalidates_nested_records() -> None:
    """Catch serialization losing bindings or accepting extra and mutated state."""

    decision = _select(
        FailureCategory.TRANSIENT_RUNTIME_ERROR,
        repeated_failed_route=True,
        repeated_route_evidence_refs=(_observable("verification:repeated"),),
    )
    payload = json.loads(json.dumps(decision.to_dict()))

    assert RecoveryDecision.from_dict(payload) == decision
    assert RecoveryDecision.from_dict(payload).to_dict() == payload
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="exactly"):
        RecoveryDecision.from_dict(payload)

    object.__setattr__(decision.failure, "failure_id", "")
    with pytest.raises(ValueError, match="failure"):
        decision.to_dict()


def test_budget_json_round_trip_rejects_hostile_mapping_values() -> None:
    """Catch deserialization accepting aliases, booleans, or non-finite sentinels."""

    budget = RecoveryBudget(2, 1, 3, 2)
    assert RecoveryBudget.from_dict(json.loads(json.dumps(budget.to_dict()))) == budget

    payload = budget.to_dict()
    payload["max_retries"] = True
    with pytest.raises(ValueError, match="max_retries"):
        RecoveryBudget.from_dict(payload)


def test_recovery_enums_reject_hostile_string_and_enum_subclasses() -> None:
    """Catch overridden equality making a foreign category select a privileged route."""

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    class ForeignCategory(str, Enum):
        TRANSIENT_RUNTIME_ERROR = "transient_runtime_error"

    with pytest.raises(ValueError, match="failure_category"):
        RecoveryDecision.from_dict(
            {
                **_select(FailureCategory.ARGUMENT_ERROR).to_dict(),
                "failure_category": HostileString("argument_error"),
            }
        )
    with pytest.raises(ValueError, match="category"):
        _select(cast(Any, ForeignCategory.TRANSIENT_RUNTIME_ERROR))


def test_public_package_exports_recovery_contract_without_breaking_prior_exports() -> None:
    """Catch recovery exports hiding established verification package contracts."""

    from gepa_mindfulness import verification

    assert verification.RecoveryBudget is RecoveryBudget
    assert verification.FailureCategory is FailureCategory
    assert verification.RecoveryAction is RecoveryAction
    assert verification.RecoveryDecision is RecoveryDecision
    assert verification.select_recovery is select_recovery
    assert hasattr(verification, "EvidenceState")
    assert hasattr(verification, "FailureGraph")
    assert hasattr(verification, "AuthorityGrant")
