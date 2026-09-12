"""Regression tests for the PR-5 whole-stage authority and state review."""

from __future__ import annotations

import json
import multiprocessing
from typing import Any, cast

import pytest
from test_bounded_recovery import _store as _recovery_store
from test_runtime_authority import _clock

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    ActionAuthorityPolicy,
    ArtifactObservation,
    AuthorityGrant,
    AuthorityGrantRegistry,
    EvidenceClaim,
    EvidenceState,
    FailureEdge,
    FailureGraph,
    FailureLocalization,
    FailureNode,
    FailureRelation,
    FailureRole,
    FailureRoleEvidence,
    IrreversibleApprovalBinding,
    RuntimeCapability,
    RuntimeRole,
    WorldStateChange,
    action_record_digest,
    authorize_action,
    consume_authorization,
)
from mindful_trace_gepa import ActionRecord


def _observable(name: str = "observed") -> EvidenceReference:
    return EvidenceReference(name, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _action(*, reversible: bool, action_class: str = "edit") -> ActionRecord:
    return ActionRecord("action-1", action_class, reversible, "repo:src", "prediction-1")


def _registry(
    action: ActionRecord,
    capability: RuntimeCapability,
) -> tuple[AuthorityGrantRegistry, ActionAuthorityPolicy, AuthorityGrant]:
    policy = ActionAuthorityPolicy(
        "policy-1",
        action.action_id,
        action_record_digest(action),
        action.authorization_scope,
        capability,
    )
    grant = AuthorityGrant(
        "executor-grant",
        "executor-1",
        RuntimeRole.EXECUTOR,
        (capability,),
        action.action_id,
        action_record_digest(action),
        action.authorization_scope,
    )
    return AuthorityGrantRegistry.enroll((grant,), (policy,)), policy, grant


def _approval(
    action: ActionRecord,
    policy: ActionAuthorityPolicy,
) -> tuple[AuthorityGrant, IrreversibleApprovalBinding]:
    evidence = (_observable("human-approval"),)
    grant = AuthorityGrant(
        "human-grant",
        "human-1",
        RuntimeRole.HUMAN,
        (RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
        action.action_id,
        action_record_digest(action),
        action.authorization_scope,
        evidence_refs=evidence,
    )
    approval = IrreversibleApprovalBinding(
        "human-1",
        action.action_id,
        action_record_digest(action),
        action.authorization_scope,
        "human-grant",
        policy.policy_id,
        policy.required_capability,
        evidence,
    )
    return grant, approval


@pytest.mark.parametrize("capability", [RuntimeCapability.WRITE, RuntimeCapability.EXECUTE])
def test_irreversible_write_and_execute_require_policy_bound_human_approval(
    capability: RuntimeCapability,
) -> None:
    action = _action(reversible=False)
    registry, policy, grant = _registry(action, capability)

    denied = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=capability,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(grant.grant_id,),
        clock=_clock(),
    )
    human, approval = _approval(action, policy)
    approved_registry = AuthorityGrantRegistry.enroll((grant, human), (policy,))
    approved = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=capability,
        policy_id=policy.policy_id,
        grant_registry=approved_registry,
        grant_ids=(grant.grant_id, human.grant_id),
        clock=_clock(),
        irreversible_approval=approval,
    )

    assert denied.authorized is False
    assert approved.authorized is True


def test_enrolled_policy_prevents_capability_downgrade_and_approval_relabelling() -> None:
    action = _action(reversible=False)
    registry, policy, grant = _registry(action, RuntimeCapability.EXECUTE)
    human, approval = _approval(action, policy)
    registry = AuthorityGrantRegistry.enroll((grant, human), (policy,))

    downgraded = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(grant.grant_id, human.grant_id),
        clock=_clock(),
        irreversible_approval=approval,
    )
    object.__setattr__(approval, "required_capability", RuntimeCapability.WRITE)
    mutated = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.EXECUTE,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(grant.grant_id, human.grant_id),
        clock=_clock(),
        irreversible_approval=approval,
    )

    assert downgraded.authorized is False
    assert mutated.authorized is False


def test_authorization_decision_is_issued_once_by_exact_registry_and_not_reconstructable() -> None:
    action = _action(reversible=True)
    registry, policy, grant = _registry(action, RuntimeCapability.WRITE)
    issued = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(grant.grant_id,),
        clock=_clock(),
    )
    reconstructed = type(issued).from_dict(json.loads(json.dumps(issued.to_dict())))
    other_registry = AuthorityGrantRegistry.enroll((grant,), (policy,))

    with pytest.raises(PermissionError, match="issued"):
        consume_authorization(reconstructed, action, grant_registry=registry, clock=_clock())
    with pytest.raises(PermissionError, match="registry"):
        consume_authorization(issued, action, grant_registry=other_registry, clock=_clock())
    assert consume_authorization(issued, action, grant_registry=registry, clock=_clock())
    with pytest.raises(PermissionError, match="consumed"):
        consume_authorization(issued, action, grant_registry=registry, clock=_clock())


def test_state_records_revalidate_coherent_use_time_corruption() -> None:
    observation = ArtifactObservation(
        "observation-1",
        "artifact:fix.patch",
        "a" * 64,
        "2026-09-10T12:00:00Z",
        (_observable(),),
    )
    change = WorldStateChange("change-1", "action-1", None, observation)
    claim = EvidenceClaim("claim-1", "fixed", (_observable(),), "supported")
    state = EvidenceState((claim,))

    object.__setattr__(change.after_observation, "observation_id", " ")
    object.__setattr__(state.claims[0], "status", "invented")
    with pytest.raises(ValueError, match="observation_id"):
        change.to_dict()
    with pytest.raises(ValueError, match="status"):
        state.resolve("claim-1")
    with pytest.raises(ValueError, match="status"):
        state.to_dict()


def test_world_change_binds_exact_canonical_artifact_observation() -> None:
    after = ArtifactObservation(
        "observation-after",
        "artifact:fix.patch",
        "a" * 64,
        "2026-09-10T12:00:00Z",
        (_observable("tool-output"),),
    )
    change = WorldStateChange("change-1", "action-1", None, after)

    assert change.after_observation == after
    assert change.artifact_ref == "artifact:fix.patch"
    assert WorldStateChange.from_dict(change.to_dict()) == change


def test_failure_localization_rejects_disconnected_arbitrary_first_anomaly() -> None:
    nodes = tuple(
        FailureNode(
            name,
            f"event:{name}",
            name,
            "2026-09-10T12:00:00Z",
            (_observable(name),),
        )
        for name in ("root", "decisive", "unrelated")
    )
    edge = FailureEdge("root", "decisive", FailureRelation.CAUSAL, ("verifier-1",))
    bindings = tuple(
        FailureRoleEvidence(role, name, (f"verifier:{name}",))
        for role, name in (
            (FailureRole.FIRST_ANOMALY, "unrelated"),
            (FailureRole.ROOT_CAUSE, "root"),
            (FailureRole.DECISIVE_FAILURE, "decisive"),
        )
    )
    localization = FailureLocalization("unrelated", "root", "decisive", (), None, bindings)

    with pytest.raises(ValueError, match="first_anomaly"):
        FailureGraph(nodes, (edge,), localization)


def test_failure_localization_requires_typed_verifier_binding_for_every_role() -> None:
    with pytest.raises(ValueError, match="role evidence"):
        FailureLocalization("anomaly", None, None, (), None, ())


def _consume_in_child(
    registry: AuthorityGrantRegistry,
    decision: object,
    action: ActionRecord,
    queue: Any,
) -> None:
    try:
        consume_authorization(
            cast(Any, decision),
            action,
            grant_registry=registry,
            clock=_clock(),
        )
    except Exception as exc:  # pragma: no cover - assertion is in the parent process
        queue.put((type(exc).__name__, str(exc)))
    else:  # pragma: no cover - this is the security failure path
        queue.put(("AUTHORIZED", ""))


def _snapshot_recovery_in_child(store: object, queue: Any) -> None:
    try:
        cast(Any, store).snapshot()
    except Exception as exc:  # pragma: no cover - assertion is in the parent process
        queue.put((type(exc).__name__, str(exc)))
    else:  # pragma: no cover - this is the security failure path
        queue.put(("SNAPSHOT", ""))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no POSIX fork")
def test_forked_authority_handle_and_decision_fail_closed_without_deadlock() -> None:
    action = _action(reversible=True)
    registry, policy, grant = _registry(action, RuntimeCapability.WRITE)
    decision = authorize_action(
        action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(grant.grant_id,),
        clock=_clock(),
    )
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_consume_in_child, args=(registry, decision, action, queue))
    process.start()
    process.join(5)

    assert process.exitcode == 0
    assert queue.get(timeout=1)[0] == "ValueError"


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no POSIX fork")
def test_forked_recovery_store_fails_closed_without_deadlock() -> None:
    store = _recovery_store()
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_snapshot_recovery_in_child, args=(store, queue))
    process.start()
    process.join(5)

    assert process.exitcode == 0
    assert queue.get(timeout=1)[0] == "ValueError"


def test_pr5_recommendations_are_implemented_with_concrete_references() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (root / "docs/recommendations/registry.yaml").read_text(encoding="utf-8")
    )
    by_id = {item["id"]: item for item in registry["recommendations"]}

    for recommendation_id in ("REC-006", "REC-007", "REC-008"):
        item = by_id[recommendation_id]
        assert item["status"] == "implemented"
        assert item["implementation_refs"]
        assert "src/mindful_trace_gepa/logging_schema.py" not in item["repo_refs"]


def test_duplicate_grant_reason_is_not_a_public_unreachable_enum_member() -> None:
    from gepa_mindfulness.verification import AuthorizationReason

    assert "duplicate_grant_ids" not in {reason.value for reason in AuthorizationReason}
