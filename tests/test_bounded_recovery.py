"""Tests for authoritative, evidence-bound, bounded recovery consumption."""

import gc
import json
import weakref
from dataclasses import FrozenInstanceError
from enum import Enum
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.failure_graph import FailureNode
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)
from gepa_mindfulness.verification.recovery import (
    MAX_JSON_SAFE_INTEGER,
    FailureCategory,
    FailureClassificationBinding,
    RecoveryAction,
    RecoveryBudget,
    RecoveryConsumption,
    RecoveryDecision,
    RecoveryStateSnapshot,
    RecoveryStateStore,
    RepeatedRouteFinding,
    consume_recovery,
    select_recovery,
)
from gepa_mindfulness.verification.runtime_governance import action_record_digest
from mindful_trace_gepa.action_bound_events import ActionRecord


def _observable(reference_id: str = "observation:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _private(reference_id: str = "reasoning:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.PRIVATE_REASONING)


def _action(action_id: str = "action-1", *, action_class: str = "tool") -> ActionRecord:
    return ActionRecord(action_id, action_class, True, "workspace:repo", "prediction-1")


def _failure(failure_id: str = "failure-1") -> FailureNode:
    return FailureNode(
        failure_id=failure_id,
        event_id=f"event:{failure_id}",
        summary=f"Observed {failure_id}",
        observed_at="2026-09-10T12:00:00Z",
        evidence_refs=(_observable(f"observation:{failure_id}"),),
    )


def _classification(
    category: FailureCategory,
    *,
    action: ActionRecord | None = None,
    failure: FailureNode | None = None,
    evidence_refs: tuple[EvidenceReference, ...] | None = None,
    verifier_refs: tuple[str, ...] = ("verifier:failure-classifier",),
    store_id: str = "a" * 64,
    plan_id: str = "plan-1",
    verification_result: RelationalVerificationResult | None = None,
) -> FailureClassificationBinding:
    action = _action() if action is None else action
    failure = _failure() if failure is None else failure
    references = failure.evidence_refs if evidence_refs is None else evidence_refs
    verification_result = (
        _relational(action.action_id, references)
        if verification_result is None
        else verification_result
    )
    return FailureClassificationBinding(
        store_id=store_id,
        plan_id=plan_id,
        failure=failure,
        action_id=action.action_id,
        action_digest=action_record_digest(action),
        category=category,
        verification_event_id="verification-event:classification-1",
        evidence_refs=references,
        verifier_refs=verifier_refs,
        verification_result=verification_result,
    )


def _relational(
    action_id: str,
    evidence_refs: tuple[EvidenceReference, ...],
    *,
    repeated_failed_route: bool = False,
) -> RelationalVerificationResult:
    bindings = (
        (VerificationEvidenceBinding("repeated_failed_route", evidence_refs),)
        if repeated_failed_route
        else ()
    )
    return RelationalVerificationResult(
        action_id=action_id,
        task_fit=False,
        dependencies_satisfied=False,
        contradiction_status="unknown",
        provenance_intact=False,
        authorization_scope_valid=False,
        claimed_outcome_supported=False,
        repeated_failed_route=repeated_failed_route,
        evidence_refs=evidence_refs,
        evidence_bindings=bindings,
    )


def _store(
    *,
    action: ActionRecord | None = None,
    failure: FailureNode | None = None,
    budget: RecoveryBudget | None = None,
) -> RecoveryStateStore:
    return RecoveryStateStore.enroll(
        plan_id="plan-1",
        action=_action() if action is None else action,
        failure=_failure() if failure is None else failure,
        budget=RecoveryBudget(2, 0, 2, 0) if budget is None else budget,
    )


def _select(
    category: FailureCategory,
    *,
    action: ActionRecord | None = None,
    failure: FailureNode | None = None,
    budget: RecoveryBudget | None = None,
    store: RecoveryStateStore | None = None,
    expected_revision: int = 0,
    route_id: str = "route-1",
    classification: FailureClassificationBinding | None = None,
    repeated_route_finding: RepeatedRouteFinding | None = None,
) -> tuple[RecoveryStateStore, ActionRecord, RecoveryDecision]:
    action = _action() if action is None else action
    failure = _failure() if failure is None else failure
    store = _store(action=action, failure=failure, budget=budget) if store is None else store
    store_snapshot = store.snapshot()
    classification = (
        _classification(
            category,
            action=action,
            failure=failure,
            store_id=store_snapshot.store_id,
            plan_id=store_snapshot.plan_id,
        )
        if classification is None
        else classification
    )
    classification_id = store.enroll_classification(classification)
    finding_id = (
        None
        if repeated_route_finding is None
        else store.enroll_route_finding(repeated_route_finding)
    )
    decision = select_recovery(
        store,
        expected_revision=expected_revision,
        action=action,
        classification_id=classification_id,
        route_id=route_id,
        repeated_route_finding_id=finding_id,
    )
    return store, action, decision


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
def test_verified_categories_select_distinct_pending_actions(
    category: FailureCategory,
    expected: RecoveryAction,
) -> None:
    """Catch a verifier-backed category selecting the wrong intervention level."""

    store, _, decision = _select(category)

    assert decision.action is expected
    assert store.snapshot().pending_decision_id == decision.decision_id
    assert store.snapshot().budget == RecoveryBudget(2, 0, 2, 0)


def test_consumption_atomically_advances_revision_and_only_the_selected_counter() -> None:
    """Catch selection spending quota early or consumption changing both counters."""

    budget = RecoveryBudget(3, 1, 4, 2)
    store, action, decision = _select(FailureCategory.TRANSIENT_RUNTIME_ERROR, budget=budget)

    assert decision.budget_after == RecoveryBudget(3, 2, 4, 2)
    assert store.snapshot().budget == budget

    consumed = consume_recovery(store, decision, expected_revision=0, action=action)

    assert consumed == RecoveryConsumption(
        plan_id="plan-1",
        decision_id=decision.decision_id,
        consumed_revision=0,
        current_revision=1,
        action=RecoveryAction.RETRY,
        budget=RecoveryBudget(3, 2, 4, 2),
    )
    assert store.snapshot().revision == 1
    assert store.snapshot().budget == RecoveryBudget(3, 2, 4, 2)
    assert store.snapshot().pending_decision_id is None


def test_replan_consumption_does_not_spend_retry_budget() -> None:
    """Catch bounded replanning being charged to the retry counter."""

    budget = RecoveryBudget(3, 1, 4, 2)
    store, action, decision = _select(FailureCategory.DEPENDENCY_FAILURE, budget=budget)

    consume_recovery(store, decision, expected_revision=0, action=action)

    assert store.snapshot().budget == RecoveryBudget(3, 1, 4, 3)


@pytest.mark.parametrize(
    ("category", "budget"),
    [
        (FailureCategory.TRANSIENT_RUNTIME_ERROR, RecoveryBudget(0, 0, 1, 0)),
        (FailureCategory.TRANSIENT_RUNTIME_ERROR, RecoveryBudget(2, 2, 1, 0)),
        (FailureCategory.STRATEGY_FAILURE, RecoveryBudget(1, 0, 0, 0)),
        (FailureCategory.DEPENDENCY_FAILURE, RecoveryBudget(1, 0, 3, 3)),
    ],
)
def test_zero_and_exact_max_select_exhausted_without_counter_change(
    category: FailureCategory,
    budget: RecoveryBudget,
) -> None:
    """Catch retry or replan being permitted at its authoritative maximum."""

    store, action, decision = _select(category, budget=budget)
    consumed = consume_recovery(store, decision, expected_revision=0, action=action)

    assert decision.action is RecoveryAction.EXHAUSTED
    assert consumed.budget == budget
    assert store.snapshot().budget == budget


def test_stale_selection_concurrent_pending_and_replayed_consumption_fail_closed() -> None:
    """Catch two callers spending one revision or replaying a consumed decision."""

    action = _action()
    failure = _failure()
    store = _store(action=action, failure=failure)
    snapshot = store.snapshot()
    classification = _classification(
        FailureCategory.TRANSIENT_RUNTIME_ERROR,
        action=action,
        failure=failure,
        store_id=snapshot.store_id,
    )
    classification_id = store.enroll_classification(classification)
    first = select_recovery(
        store,
        expected_revision=0,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
    )

    with pytest.raises(RuntimeError, match="pending"):
        select_recovery(
            store,
            expected_revision=0,
            action=action,
            classification_id=classification_id,
            route_id="route-2",
        )

    consume_recovery(store, first, expected_revision=0, action=action)

    with pytest.raises(RuntimeError, match="stale.*revision"):
        select_recovery(
            store,
            expected_revision=0,
            action=action,
            classification_id=classification_id,
            route_id="route-2",
        )
    with pytest.raises(RuntimeError, match="stale.*revision"):
        consume_recovery(store, first, expected_revision=0, action=action)


def test_coherent_mutation_of_decision_and_budget_cannot_spend_another_counter() -> None:
    """Catch caller mutation creating a second recovery from one pending store revision."""

    store, action, decision = _select(FailureCategory.TRANSIENT_RUNTIME_ERROR)
    forged_budget = RecoveryBudget(2, 0, 2, 1)
    object.__setattr__(
        decision.classification,
        "category",
        FailureCategory.STRATEGY_FAILURE,
    )
    object.__setattr__(decision, "action", RecoveryAction.REPLAN)
    object.__setattr__(decision, "budget_after", forged_budget)

    with pytest.raises(RuntimeError, match="authoritative pending decision"):
        consume_recovery(store, decision, expected_revision=0, action=action)
    assert store.snapshot().budget == RecoveryBudget(2, 0, 2, 0)
    assert store.snapshot().revision == 0


def test_enrollment_detaches_action_failure_and_budget_from_callers() -> None:
    """Catch post-enrollment object mutation retargeting authoritative recovery state."""

    action = _action()
    failure = _failure()
    budget = RecoveryBudget(2, 0, 2, 0)
    store = _store(action=action, failure=failure, budget=budget)
    object.__setattr__(action, "action_id", "attacker-action")
    object.__setattr__(failure, "failure_id", "attacker-failure")
    object.__setattr__(budget, "retries_used", 2)

    snapshot = store.snapshot()
    assert snapshot.action_id == "action-1"
    assert snapshot.failure_id == "failure-1"
    assert snapshot.budget == RecoveryBudget(2, 0, 2, 0)

    object.__setattr__(snapshot.budget, "retries_used", 2)
    assert store.snapshot().budget == RecoveryBudget(2, 0, 2, 0)


def test_store_handle_has_no_retargetable_fields_and_uses_weak_lifecycle() -> None:
    """Catch public fields or reusable object IDs becoming recovery authority."""

    store = _store()
    store_ref = weakref.ref(store)

    assert not hasattr(store, "__dict__")
    for field_name in ("state", "budget", "revision", "pending", "_entries", "_seal"):
        with pytest.raises((AttributeError, TypeError)):
            setattr(store, field_name, object())
        with pytest.raises(AttributeError):
            object.__setattr__(store, field_name, object())

    del store
    gc.collect()
    assert store_ref() is None


def test_classification_binds_failure_action_digest_evidence_and_verifier() -> None:
    """Catch failure classification becoming a free-form category label."""

    action = _action()
    failure = _failure()
    binding = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
    )

    assert binding.failure is not failure
    assert binding.failure.failure_id == failure.failure_id
    assert binding.failure.event_id == failure.event_id
    assert binding.action_id == action.action_id
    assert binding.action_digest == action_record_digest(action)
    assert binding.evidence_refs == failure.evidence_refs
    assert binding.verifier_refs == ("verifier:failure-classifier",)
    assert binding.verification_result.action_id == action.action_id


def test_enrolled_classification_resists_coherent_category_and_id_mutation() -> None:
    """Catch a caller-recomputed digest being mistaken for authenticated classification."""

    action = _action()
    failure = _failure()
    store = _store(action=action, failure=failure)
    binding = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=store.snapshot().store_id,
    )
    enrolled_id = store.enroll_classification(binding)
    object.__setattr__(binding, "category", FailureCategory.STRATEGY_FAILURE)
    attacker_id = binding.classification_id

    with pytest.raises(KeyError, match="classification"):
        select_recovery(
            store,
            expected_revision=0,
            action=action,
            classification_id=attacker_id,
            route_id="route-1",
        )

    decision = select_recovery(
        store,
        expected_revision=0,
        action=action,
        classification_id=enrolled_id,
        route_id="route-1",
    )
    assert decision.action is RecoveryAction.REPAIR_ARGUMENTS
    assert decision.classification.category is FailureCategory.ARGUMENT_ERROR


def test_classification_enrollment_rejects_cross_store_duplicate_and_alias_mutation() -> None:
    """Catch a binding being replaced, re-enrolled, or moved to another store."""

    action = _action()
    failure = _failure()
    first_store = _store(action=action, failure=failure)
    second_store = _store(action=action, failure=failure)
    binding = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=first_store.snapshot().store_id,
    )
    binding_id = first_store.enroll_classification(binding)

    with pytest.raises(ValueError, match="already enrolled"):
        first_store.enroll_classification(binding)
    with pytest.raises(ValueError, match="store_id"):
        second_store.enroll_classification(binding)

    object.__setattr__(binding, "action_id", "attacker-action")
    selected = select_recovery(
        first_store,
        expected_revision=0,
        action=action,
        classification_id=binding_id,
        route_id="route-1",
    )
    assert selected.action_id == "action-1"


@pytest.mark.parametrize(
    "changes",
    [
        {"action_id": "other-action"},
        {"action_digest": "0" * 64},
        {"failure": _failure("other-failure")},
        {"category": "transient_runtime_error"},
    ],
)
def test_classification_digest_rejects_conflicting_serialized_identity(
    changes: dict[str, object],
) -> None:
    """Catch category, action, digest, or failure tampering under an old binding ID."""

    action = _action()
    failure = _failure()
    values = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
    ).to_dict()
    values.update(changes)
    if type(values["failure"]) is FailureNode:
        values["failure"] = cast(FailureNode, values["failure"]).to_dict()
    with pytest.raises(
        ValueError,
        match="classification_id|classification evidence|verification_result",
    ):
        FailureClassificationBinding.from_dict(values)


def test_classification_requires_observable_failure_evidence_and_verifier_refs() -> None:
    """Catch an unverified or unrelated category controlling recovery."""

    action = _action()
    failure = _failure()
    for evidence_refs in ((), (_private(),), (_observable("unrelated"),)):
        with pytest.raises(ValueError, match="evidence"):
            _classification(
                FailureCategory.ARGUMENT_ERROR,
                action=action,
                failure=failure,
                evidence_refs=evidence_refs,
            )
    with pytest.raises(ValueError, match="verifier_refs"):
        _classification(
            FailureCategory.ARGUMENT_ERROR,
            action=action,
            failure=failure,
            verifier_refs=(),
        )


def test_select_requires_exact_action_and_typed_classification_not_scalar_labels() -> None:
    """Catch the old caller-owned category and action-ID interface bypassing bindings."""

    action = _action()
    store = _store(action=action)
    binding = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        store_id=store.snapshot().store_id,
    )
    classification_id = store.enroll_classification(binding)

    with pytest.raises(TypeError, match="unexpected keyword argument 'classification'"):
        cast(Any, select_recovery)(
            store,
            expected_revision=0,
            action=action,
            classification=binding,
            route_id="route-1",
            category=FailureCategory.ARGUMENT_ERROR,
        )
    with pytest.raises(TypeError, match="exact ActionRecord"):
        select_recovery(
            store,
            expected_revision=0,
            action=cast(Any, action.action_id),
            classification_id=classification_id,
            route_id="route-1",
        )


def _repeat_finding(
    prior: RecoveryDecision,
    classification: FailureClassificationBinding,
    *,
    route_id: str | None = None,
) -> RepeatedRouteFinding:
    return RepeatedRouteFinding(
        store_id=classification.store_id,
        plan_id=prior.plan_id,
        current_revision=prior.revision + 1,
        route_id=prior.route_id if route_id is None else route_id,
        prior_action_id=prior.action_id,
        prior_action_digest=prior.action_digest,
        prior_decision_id=prior.decision_id,
        prior_revision=prior.revision,
        current_failure_id=classification.failure.failure_id,
        current_failure_event_id=classification.failure.event_id,
        classification_id=classification.classification_id,
        classification_verification_event_id=classification.verification_event_id,
        evidence_refs=(_observable("verification:route-repeat"),),
        verifier_refs=("verifier:route-repeat",),
        verification_result=_relational(
            prior.action_id,
            (_observable("verification:route-repeat"),),
            repeated_failed_route=True,
        ),
    )


def test_typed_repeated_route_finding_rejects_only_the_exact_consumed_route() -> None:
    """Catch evidence label text alone rejecting a route without prior consumed identity."""

    action = _action()
    failure = _failure()
    store = _store(action=action, failure=failure)
    snapshot = store.snapshot()
    classification = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=snapshot.store_id,
    )
    classification_id = store.enroll_classification(classification)
    first = select_recovery(
        store,
        expected_revision=0,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
    )
    consume_recovery(store, first, expected_revision=0, action=action)
    finding = _repeat_finding(first, classification)
    finding_id = store.enroll_route_finding(finding)

    rejected = select_recovery(
        store,
        expected_revision=1,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
        repeated_route_finding_id=finding_id,
    )

    assert rejected.action is RecoveryAction.REJECT_ROUTE
    assert rejected.repeated_route_finding == finding


def test_enrolled_route_finding_resists_coherent_identity_and_id_mutation() -> None:
    """Catch recomputed finding digests replacing runtime-enrolled route attestations."""

    action = _action()
    failure = _failure()
    store = _store(action=action, failure=failure)
    classification = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=store.snapshot().store_id,
    )
    classification_id = store.enroll_classification(classification)
    prior = select_recovery(
        store,
        expected_revision=0,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
    )
    consume_recovery(store, prior, expected_revision=0, action=action)
    finding = _repeat_finding(prior, classification)
    enrolled_id = store.enroll_route_finding(finding)
    object.__setattr__(finding, "route_id", "route-attacker")
    object.__setattr__(finding, "prior_decision_id", "0" * 64)
    object.__setattr__(finding, "prior_revision", 1)
    attacker_id = finding.finding_id

    with pytest.raises(KeyError, match="route finding"):
        select_recovery(
            store,
            expected_revision=1,
            action=action,
            classification_id=classification_id,
            route_id="route-attacker",
            repeated_route_finding_id=attacker_id,
        )

    rejected = select_recovery(
        store,
        expected_revision=1,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
        repeated_route_finding_id=enrolled_id,
    )
    assert rejected.action is RecoveryAction.REJECT_ROUTE


def test_route_finding_enrollment_rejects_cross_store_and_reenrollment() -> None:
    """Catch a repeated-route attestation crossing its store or replacing its ID."""

    action = _action()
    failure = _failure()
    first_store = _store(action=action, failure=failure)
    classification = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=first_store.snapshot().store_id,
    )
    classification_id = first_store.enroll_classification(classification)
    prior = select_recovery(
        first_store,
        expected_revision=0,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
    )
    consume_recovery(first_store, prior, expected_revision=0, action=action)
    finding = _repeat_finding(prior, classification)
    first_store.enroll_route_finding(finding)

    with pytest.raises(ValueError, match="already enrolled"):
        first_store.enroll_route_finding(finding)
    second_store = _store(action=action, failure=failure)
    with pytest.raises(ValueError, match="store_id"):
        second_store.enroll_route_finding(finding)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("route_id", "other-route"),
        ("prior_action_id", "other-action"),
        ("prior_action_digest", "0" * 64),
        ("prior_decision_id", "0" * 64),
        ("prior_revision", 1),
        ("current_failure_id", "other-failure"),
        ("current_failure_event_id", "event:other"),
        ("classification_id", "0" * 64),
        ("classification_verification_event_id", "verification-event:other"),
    ],
)
def test_repeated_route_finding_rejects_each_identity_mismatch(
    field_name: str,
    value: object,
) -> None:
    """Catch a forged route-repeat claim matching only evidence text."""

    action = _action()
    failure = _failure()
    store = _store(action=action, failure=failure)
    snapshot = store.snapshot()
    classification = _classification(
        FailureCategory.ARGUMENT_ERROR,
        action=action,
        failure=failure,
        store_id=snapshot.store_id,
    )
    classification_id = store.enroll_classification(classification)
    first = select_recovery(
        store,
        expected_revision=0,
        action=action,
        classification_id=classification_id,
        route_id="route-1",
    )
    consume_recovery(store, first, expected_revision=0, action=action)
    original = _repeat_finding(first, classification)
    values: dict[str, object] = {
        "store_id": original.store_id,
        "plan_id": original.plan_id,
        "current_revision": original.current_revision,
        "route_id": original.route_id,
        "prior_action_id": original.prior_action_id,
        "prior_action_digest": original.prior_action_digest,
        "prior_decision_id": original.prior_decision_id,
        "prior_revision": original.prior_revision,
        "current_failure_id": original.current_failure_id,
        "current_failure_event_id": original.current_failure_event_id,
        "classification_id": original.classification_id,
        "classification_verification_event_id": (original.classification_verification_event_id),
        "evidence_refs": original.evidence_refs,
        "verifier_refs": original.verifier_refs,
        "verification_result": original.verification_result,
    }
    values[field_name] = value
    with pytest.raises(ValueError, match="repeated route finding|verification_result"):
        finding = RepeatedRouteFinding(**cast(Any, values))
        store.enroll_route_finding(finding)


def test_repeated_route_finding_requires_observable_evidence_and_verifier_refs() -> None:
    """Catch an unsupported route-repeat claim controlling policy."""

    prior = _select(FailureCategory.ARGUMENT_ERROR)[2]
    classification = prior.classification
    values = _repeat_finding(prior, classification).to_dict()
    values["evidence_refs"] = [_private().to_dict()]
    with pytest.raises(ValueError, match="observable"):
        RepeatedRouteFinding.from_dict(values)
    values = _repeat_finding(prior, classification).to_dict()
    values["verifier_refs"] = []
    with pytest.raises(ValueError, match="verifier_refs"):
        RepeatedRouteFinding.from_dict(values)


def test_decision_rejects_a_route_finding_bound_to_another_current_route() -> None:
    """Catch direct construction bypassing the finding-to-decision identity boundary."""

    _, _, prior = _select(FailureCategory.ARGUMENT_ERROR)
    classification = prior.classification
    wrong_route = _repeat_finding(prior, classification, route_id="other-route")

    with pytest.raises(ValueError, match="decision.*repeated route finding"):
        RecoveryDecision(
            plan_id=prior.plan_id,
            revision=1,
            action_id=prior.action_id,
            action_digest=prior.action_digest,
            route_id="route-1",
            classification=classification,
            action=RecoveryAction.REJECT_ROUTE,
            budget_before=prior.budget_before,
            budget_after=prior.budget_before,
            repeated_route_finding=wrong_route,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_retries", -1),
        ("retries_used", True),
        ("max_replans", 1.0),
        ("replans_used", float("nan")),
        ("max_retries", float("inf")),
        ("max_replans", None),
        ("max_retries", "unlimited"),
        ("max_retries", MAX_JSON_SAFE_INTEGER + 1),
        pytest.param("replans_used", 10**5000, id="huge-integer"),
    ],
)
def test_budget_rejects_negative_noninteger_unlimited_and_non_json_safe_values(
    field_name: str,
    value: object,
) -> None:
    """Catch an invalid integer creating an unbounded or nonportable budget."""

    values: dict[str, object] = {
        "max_retries": 1,
        "retries_used": 0,
        "max_replans": 1,
        "replans_used": 0,
    }
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        RecoveryBudget(**cast(Any, values))
    with pytest.raises(ValueError, match=field_name):
        RecoveryBudget.from_dict(values)


def test_json_safe_integer_boundary_passes_for_counts_and_revisions() -> None:
    """Catch rejecting the maximum interoperable JSON integer or accepting one above it."""

    boundary = RecoveryBudget(
        MAX_JSON_SAFE_INTEGER,
        MAX_JSON_SAFE_INTEGER,
        MAX_JSON_SAFE_INTEGER,
        MAX_JSON_SAFE_INTEGER,
    )
    assert RecoveryBudget.from_dict(boundary.to_dict()) == boundary

    snapshot_payload = {
        "store_id": "a" * 64,
        "plan_id": "plan-1",
        "revision": MAX_JSON_SAFE_INTEGER,
        "action_id": "action-1",
        "action_digest": "a" * 64,
        "failure_id": "failure-1",
        "failure_event_id": "event:failure-1",
        "budget": boundary.to_dict(),
        "pending_decision_id": None,
    }
    assert RecoveryStateSnapshot.from_dict(snapshot_payload).revision == MAX_JSON_SAFE_INTEGER
    snapshot_payload["revision"] = MAX_JSON_SAFE_INTEGER + 1
    with pytest.raises(ValueError, match="revision"):
        RecoveryStateSnapshot.from_dict(snapshot_payload)

    _, _, decision = _select(FailureCategory.ARGUMENT_ERROR)
    finding_values = {
        "store_id": decision.classification.store_id,
        "plan_id": decision.plan_id,
        "current_revision": MAX_JSON_SAFE_INTEGER,
        "route_id": decision.route_id,
        "prior_action_id": decision.action_id,
        "prior_action_digest": decision.action_digest,
        "prior_decision_id": decision.decision_id,
        "prior_revision": MAX_JSON_SAFE_INTEGER,
        "current_failure_id": decision.classification.failure.failure_id,
        "current_failure_event_id": decision.classification.failure.event_id,
        "classification_id": decision.classification.classification_id,
        "classification_verification_event_id": (decision.classification.verification_event_id),
        "evidence_refs": (_observable("verification:repeat"),),
        "verifier_refs": ("verifier:repeat",),
        "verification_result": _relational(
            decision.action_id,
            (_observable("verification:repeat"),),
            repeated_failed_route=True,
        ),
    }
    finding = RepeatedRouteFinding(**finding_values)
    assert RepeatedRouteFinding.from_dict(finding.to_dict()) == finding
    finding_values["prior_revision"] = MAX_JSON_SAFE_INTEGER + 1
    with pytest.raises(ValueError, match="prior_revision"):
        RepeatedRouteFinding(**finding_values)
    finding_values["prior_revision"] = MAX_JSON_SAFE_INTEGER
    finding_values["current_revision"] = MAX_JSON_SAFE_INTEGER + 1
    with pytest.raises(ValueError, match="current_revision"):
        RepeatedRouteFinding(**finding_values)


def test_use_time_rejects_corrupted_large_budget_and_revision_values() -> None:
    """Catch frozen-object corruption bypassing JSON-safe integer checks at use time."""

    budget = RecoveryBudget(2, 0, 2, 0)
    object.__setattr__(budget, "max_retries", 10**5000)
    with pytest.raises(ValueError, match="budget"):
        _store(budget=budget)

    store, action, decision = _select(FailureCategory.ARGUMENT_ERROR)
    object.__setattr__(decision, "revision", 10**5000)
    with pytest.raises(ValueError, match="revision"):
        consume_recovery(store, decision, expected_revision=0, action=action)

    fresh_store = _store()
    fresh_classification = _classification(
        FailureCategory.ARGUMENT_ERROR,
        store_id=fresh_store.snapshot().store_id,
    )
    fresh_classification_id = fresh_store.enroll_classification(fresh_classification)
    with pytest.raises(ValueError, match="expected_revision"):
        select_recovery(
            fresh_store,
            expected_revision=10**5000,
            action=_action(),
            classification_id=fresh_classification_id,
            route_id="route-1",
        )


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda: RecoveryBudget(1, 0, 1, 0),
        lambda: _classification(FailureCategory.ARGUMENT_ERROR),
        lambda: _select(FailureCategory.ARGUMENT_ERROR)[2],
        lambda: _store().snapshot(),
    ],
)
def test_public_recovery_records_are_frozen_and_slotted(record_factory: Any) -> None:
    """Catch public recovery records acquiring mutable state or aliases."""

    record = record_factory()
    assert not hasattr(record, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        record.unexpected = True


def test_public_records_have_exact_json_round_trips_and_use_time_validation() -> None:
    """Catch serialization dropping authority bindings or accepting extra fields."""

    store, action, decision = _select(FailureCategory.TRANSIENT_RUNTIME_ERROR)
    consumed = consume_recovery(store, decision, expected_revision=0, action=action)
    records = (
        RecoveryBudget(2, 1, 3, 2),
        decision.classification,
        decision,
        consumed,
        store.snapshot(),
    )
    for record in records:
        payload = json.loads(json.dumps(record.to_dict()))
        restored = type(record).from_dict(payload)
        assert restored == record
        payload["unexpected"] = True
        with pytest.raises(ValueError, match="exactly"):
            type(record).from_dict(payload)


def test_hostile_string_enum_and_integer_subclasses_fail_closed() -> None:
    """Catch overridden equality or numeric behavior crossing recovery boundaries."""

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    class HostileInt(int):
        def __eq__(self, other: object) -> bool:
            return True

    class ForeignCategory(str, Enum):
        ARGUMENT_ERROR = "argument_error"

    with pytest.raises(ValueError, match="max_retries"):
        RecoveryBudget(cast(Any, HostileInt(1)), 0, 1, 0)
    with pytest.raises(ValueError, match="category"):
        FailureClassificationBinding(
            store_id="a" * 64,
            plan_id="plan-1",
            failure=_failure(),
            action_id="action-1",
            action_digest="a" * 64,
            category=cast(Any, ForeignCategory.ARGUMENT_ERROR),
            verification_event_id="verification-event-1",
            evidence_refs=(_observable("observation:failure-1"),),
            verifier_refs=("verifier-1",),
            verification_result=_relational("action-1", (_observable("observation:failure-1"),)),
        )
    payload = _classification(FailureCategory.ARGUMENT_ERROR).to_dict()
    payload["category"] = HostileString("argument_error")
    with pytest.raises(ValueError, match="category"):
        FailureClassificationBinding.from_dict(payload)


def test_public_package_exports_authoritative_recovery_contract() -> None:
    """Catch package exports preserving only the rejected caller-owned interface."""

    from gepa_mindfulness import verification

    for name in (
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
    ):
        assert hasattr(verification, name)
    assert hasattr(verification, "EvidenceState")
    assert hasattr(verification, "FailureGraph")
    assert hasattr(verification, "AuthorityGrant")
