"""Contract tests for least-authority runtime governance."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    AuthorityGrant,
    AuthorityGrantRegistry,
    AuthorizationDecision,
    AuthorizationReason,
    IrreversibleApprovalBinding,
    RuntimeCapability,
    RuntimeRole,
    TrustedClock,
    action_record_digest,
    authorize_action,
    consume_authorization,
)
from mindful_trace_gepa import ActionRecord


@dataclass
class _FixedClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


def _clock(timestamp: str = "2026-09-10T11:00:00+00:00") -> _FixedClock:
    return _FixedClock(datetime.fromisoformat(timestamp))


def _action(
    *,
    action_id: str = "action-1",
    action_class: str = "edit",
    reversible: bool = True,
    scope: str = "repo:src",
    prediction_id: str = "prediction-1",
) -> ActionRecord:
    return ActionRecord(action_id, action_class, reversible, scope, prediction_id)


def _observable(reference_id: str = "approval-1") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.EXTERNAL_RECORD)


def _grant(
    principal_id: str = "executor-1",
    role: RuntimeRole = RuntimeRole.EXECUTOR,
    capabilities: tuple[RuntimeCapability, ...] = (RuntimeCapability.EXECUTE,),
    *,
    grant_id: str = "grant-1",
    action_id: str = "action-1",
    scope: str = "repo:src",
    action: ActionRecord | None = None,
    expires_at: str | None = None,
    evidence_refs: tuple[EvidenceReference, ...] = (),
) -> AuthorityGrant:
    bound_action = _action(action_id=action_id, scope=scope) if action is None else action
    return AuthorityGrant(
        grant_id=grant_id,
        principal_id=principal_id,
        role=role,
        capabilities=capabilities,
        action_id=bound_action.action_id,
        action_digest=action_record_digest(bound_action),
        authorization_scope=bound_action.authorization_scope,
        expires_at=expires_at,
        evidence_refs=evidence_refs,
    )


def _authorize(
    capability: RuntimeCapability,
    grants: tuple[AuthorityGrant, ...] = (),
    *,
    action: ActionRecord | None = None,
    principal_id: str = "executor-1",
    role: RuntimeRole = RuntimeRole.EXECUTOR,
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
    clock: TrustedClock | None = None,
    irreversible_approval: IrreversibleApprovalBinding | None = None,
) -> AuthorizationDecision:
    registry = AuthorityGrantRegistry.enroll(grants)
    return authorize_action(
        _action() if action is None else action,
        principal_id=principal_id,
        role=role,
        capability=capability,
        grant_registry=registry,
        grant_ids=tuple(grant.grant_id for grant in grants),
        clock=_clock() if clock is None else clock,
        action_author_id=action_author_id,
        action_executor_id=action_executor_id,
        irreversible_approval=irreversible_approval,
    )


def _approval(
    action: ActionRecord,
    *,
    human_principal_id: str = "human-1",
    grant_id: str = "human-grant-1",
    reference_id: str = "approval-1",
    expires_at: str | None = None,
) -> tuple[AuthorityGrant, IrreversibleApprovalBinding]:
    references = (_observable(reference_id),)
    grant = _grant(
        principal_id=human_principal_id,
        role=RuntimeRole.HUMAN,
        capabilities=(RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
        grant_id=grant_id,
        action=action,
        expires_at=expires_at,
        evidence_refs=references,
    )
    binding = IrreversibleApprovalBinding(
        human_principal_id=human_principal_id,
        action_id=action.action_id,
        action_digest=action_record_digest(action),
        authorization_scope=action.authorization_scope,
        authorization_grant_id=grant_id,
        evidence_refs=references,
    )
    return grant, binding


def test_runtime_roles_and_capabilities_have_exact_stable_values() -> None:
    assert [role.value for role in RuntimeRole] == [
        "planner",
        "executor",
        "verifier",
        "auditor",
        "human",
    ]
    assert [capability.value for capability in RuntimeCapability] == [
        "read",
        "propose",
        "write",
        "execute",
        "verify",
        "audit",
        "authorize_irreversible",
    ]


@pytest.mark.parametrize(
    ("role", "principal_id", "capability"),
    [
        (RuntimeRole.PLANNER, "planner-1", RuntimeCapability.WRITE),
        (RuntimeRole.PLANNER, "planner-1", RuntimeCapability.EXECUTE),
        (RuntimeRole.AUDITOR, "auditor-1", RuntimeCapability.EXECUTE),
        (RuntimeRole.EXECUTOR, "executor-1", RuntimeCapability.EXECUTE),
    ],
)
def test_roles_confer_no_capabilities_without_an_explicit_grant(
    role: RuntimeRole,
    principal_id: str,
    capability: RuntimeCapability,
) -> None:
    decision = _authorize(capability, principal_id=principal_id, role=role)

    assert decision.authorized is False
    assert decision.grant_id is None
    assert decision.reason == "no_matching_grant"


@pytest.mark.parametrize(
    ("role", "capability"),
    [
        (RuntimeRole.PLANNER, RuntimeCapability.WRITE),
        (RuntimeRole.PLANNER, RuntimeCapability.EXECUTE),
        (RuntimeRole.AUDITOR, RuntimeCapability.EXECUTE),
        (RuntimeRole.EXECUTOR, RuntimeCapability.VERIFY),
        (RuntimeRole.VERIFIER, RuntimeCapability.EXECUTE),
    ],
)
def test_grants_reject_role_incompatible_capabilities(
    role: RuntimeRole,
    capability: RuntimeCapability,
) -> None:
    with pytest.raises(ValueError, match="not permitted for role"):
        _grant(role=role, capabilities=(capability,))


def test_executor_can_execute_only_the_exact_explicitly_granted_action() -> None:
    grant = _grant()

    decision = _authorize(RuntimeCapability.EXECUTE, (grant,))

    assert decision.authorized is True
    assert decision.grant_id == "grant-1"
    assert decision.reason == "authorized"


@pytest.mark.parametrize(
    "grant",
    [
        _grant(principal_id="executor-2"),
        _grant(action_id="action-2"),
        _grant(scope="repo:tests"),
        _grant(capabilities=(RuntimeCapability.WRITE,)),
    ],
)
def test_executor_grants_match_principal_role_action_scope_and_capability_exactly(
    grant: AuthorityGrant,
) -> None:
    decision = _authorize(RuntimeCapability.EXECUTE, (grant,))

    assert decision.authorized is False
    assert decision.reason == "no_matching_grant"


def test_multiple_matching_grants_fail_closed_as_ambiguous() -> None:
    first = _grant(grant_id="grant-1")
    second = _grant(grant_id="grant-2")

    decision = _authorize(RuntimeCapability.EXECUTE, (first, second))

    assert decision.authorized is False
    assert decision.reason == "ambiguous_matching_grants"


def test_duplicate_grant_ids_fail_closed() -> None:
    duplicated = (_grant(), _grant())

    with pytest.raises(ValueError, match="unique"):
        AuthorityGrantRegistry.enroll(duplicated)


def test_coordinator_label_has_only_capabilities_listed_in_typed_grant() -> None:
    coordinator = _grant(
        principal_id="prime-coordinator",
        role=RuntimeRole.PLANNER,
        capabilities=(RuntimeCapability.PROPOSE,),
    )

    proposed = _authorize(
        RuntimeCapability.PROPOSE,
        (coordinator,),
        principal_id="prime-coordinator",
        role=RuntimeRole.PLANNER,
    )
    executed = _authorize(
        RuntimeCapability.EXECUTE,
        (coordinator,),
        principal_id="prime-coordinator",
        role=RuntimeRole.PLANNER,
    )

    assert proposed.authorized is True
    assert executed.authorized is False


def test_verifier_must_be_independent_of_action_author_and_executor() -> None:
    grant = _grant(
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        capabilities=(RuntimeCapability.VERIFY,),
    )

    independent = _authorize(
        RuntimeCapability.VERIFY,
        (grant,),
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        action_author_id="planner-1",
        action_executor_id="executor-1",
    )
    author = _authorize(
        RuntimeCapability.VERIFY,
        (grant,),
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        action_author_id="verifier-1",
        action_executor_id="executor-1",
    )
    executor = _authorize(
        RuntimeCapability.VERIFY,
        (grant,),
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        action_author_id="planner-1",
        action_executor_id="verifier-1",
    )

    assert independent.authorized is True
    assert author.reason == "verifier_authored_action"
    assert executor.reason == "verifier_executed_action"


@pytest.mark.parametrize(
    ("action_author_id", "action_executor_id"),
    [(None, "executor-1"), ("planner-1", None), (None, None)],
)
def test_verification_fails_closed_without_complete_independence_context(
    action_author_id: str | None,
    action_executor_id: str | None,
) -> None:
    grant = _grant(
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        capabilities=(RuntimeCapability.VERIFY,),
    )

    decision = _authorize(
        RuntimeCapability.VERIFY,
        (grant,),
        principal_id="verifier-1",
        role=RuntimeRole.VERIFIER,
        action_author_id=action_author_id,
        action_executor_id=action_executor_id,
    )

    assert decision.authorized is False
    assert decision.reason == "missing_verifier_independence_context"


def test_irreversible_execution_requires_separate_observable_human_authorization() -> None:
    action = _action(reversible=False)
    executor_grant = _grant(action=action)
    human_grant, approval = _approval(action)

    denied = _authorize(RuntimeCapability.EXECUTE, (executor_grant,), action=action)
    authorized = _authorize(
        RuntimeCapability.EXECUTE,
        (executor_grant, human_grant),
        action=action,
        irreversible_approval=approval,
    )

    assert denied.reason == "missing_human_irreversible_authorization"
    assert authorized.authorized is True
    assert authorized.authorization_scope == "repo:src"
    assert authorized.authorization_refs == (_observable(),)
    assert authorized.irreversible_approval == approval


def test_irreversible_authorization_is_bound_to_exact_action_and_scope() -> None:
    action = _action(reversible=False)
    executor_grant = _grant(action=action)
    wrong_action = _grant(
        principal_id="human-1",
        role=RuntimeRole.HUMAN,
        capabilities=(RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
        grant_id="human-action",
        action_id="action-2",
        evidence_refs=(_observable("approval-action"),),
    )
    wrong_scope = _grant(
        principal_id="human-1",
        role=RuntimeRole.HUMAN,
        capabilities=(RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
        grant_id="human-scope",
        scope="repo:tests",
        evidence_refs=(_observable("approval-scope"),),
    )

    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (executor_grant, wrong_action, wrong_scope),
        action=action,
    )

    assert decision.authorized is False
    assert decision.reason == "missing_human_irreversible_authorization"


def test_human_irreversible_grant_requires_observable_evidence() -> None:
    private = EvidenceReference("thought-1", EvidenceSourceKind.PRIVATE_REASONING)

    with pytest.raises(ValueError, match="observable evidence"):
        _grant(
            principal_id="human-1",
            role=RuntimeRole.HUMAN,
            capabilities=(RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
            evidence_refs=(private,),
        )


def test_authorized_decision_cannot_claim_role_incompatible_authority() -> None:
    with pytest.raises(ValueError, match="not permitted for role"):
        AuthorizationDecision(
            action_id="action-1",
            action_digest="0" * 64,
            authorization_scope="repo:src",
            principal_id="planner-1",
            role=RuntimeRole.PLANNER,
            capability=RuntimeCapability.EXECUTE,
            authorized=True,
            grant_id="grant-1",
            authorization_refs=(),
            irreversible_approval=None,
            reason=AuthorizationReason.AUTHORIZED,
            observed_at="2026-09-10T11:00:00Z",
            effective_expires_at=None,
        )


def test_irreversible_authorization_decision_requires_observable_evidence() -> None:
    private = EvidenceReference("thought-1", EvidenceSourceKind.PRIVATE_REASONING)

    with pytest.raises(ValueError, match="observable evidence"):
        AuthorizationDecision(
            action_id="action-1",
            action_digest="0" * 64,
            authorization_scope="repo:src",
            principal_id="human-1",
            role=RuntimeRole.HUMAN,
            capability=RuntimeCapability.AUTHORIZE_IRREVERSIBLE,
            authorized=True,
            grant_id="human-grant-1",
            authorization_refs=(private,),
            irreversible_approval=None,
            reason=AuthorizationReason.AUTHORIZED,
            observed_at="2026-09-10T11:00:00Z",
            effective_expires_at=None,
        )


def test_ordinary_execution_does_not_claim_irreversible_authorization_evidence() -> None:
    decision = _authorize(RuntimeCapability.EXECUTE, (_grant(),))

    assert decision.authorized is True
    assert decision.authorization_refs == ()


def test_expired_grants_fail_closed_at_trusted_time() -> None:
    grant = _grant(expires_at="2026-09-10T12:00:00Z")

    expired = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        clock=_clock("2026-09-10T12:00:01+00:00"),
    )
    current = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        clock=_clock("2026-09-10T11:59:59+00:00"),
    )
    boundary = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        clock=_clock("2026-09-10T12:00:00+00:00"),
    )

    assert expired.reason is AuthorizationReason.GRANT_EXPIRED
    assert current.authorized is True
    assert boundary.reason is AuthorizationReason.GRANT_EXPIRED


def test_authority_grant_snapshots_collections_and_is_frozen() -> None:
    capabilities = [RuntimeCapability.EXECUTE]
    references = [_observable()]
    grant = AuthorityGrant(
        "grant-1",
        "executor-1",
        RuntimeRole.EXECUTOR,
        cast(Any, capabilities),
        "action-1",
        action_record_digest(_action()),
        "repo:src",
        evidence_refs=cast(Any, references),
    )
    capabilities.append(RuntimeCapability.WRITE)
    references[0] = _observable("rewritten")

    assert grant.capabilities == (RuntimeCapability.EXECUTE,)
    assert grant.evidence_refs == (_observable(),)
    with pytest.raises(FrozenInstanceError):
        cast(Any, grant).principal_id = "attacker"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("role", "executor", "role"),
        ("capabilities", ("execute",), "capabilities"),
        ("capabilities", (RuntimeCapability.EXECUTE, RuntimeCapability.EXECUTE), "unique"),
        ("action_id", "*", "wildcard"),
        ("authorization_scope", "*", "wildcard"),
        ("expires_at", "tomorrow", "expires_at"),
    ],
)
def test_authority_grant_rejects_noncanonical_or_ambient_values(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs: dict[str, object] = {
        "grant_id": "grant-1",
        "principal_id": "executor-1",
        "role": RuntimeRole.EXECUTOR,
        "capabilities": (RuntimeCapability.EXECUTE,),
        "action_id": "action-1",
        "action_digest": action_record_digest(_action()),
        "authorization_scope": "repo:src",
        "expires_at": None,
        "evidence_refs": (),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        AuthorityGrant(**cast(Any, kwargs))


def test_authorize_action_rejects_subclasses_and_string_enum_standins() -> None:
    class ActionSubclass(ActionRecord):
        pass

    class GrantSubclass(AuthorityGrant):
        pass

    grant = _grant()

    with pytest.raises(TypeError, match="exact ActionRecord"):
        _authorize(
            RuntimeCapability.EXECUTE,
            (grant,),
            action=ActionSubclass(
                "action-1",
                "edit",
                True,
                "repo:src",
                "prediction-1",
            ),
        )
    with pytest.raises(ValueError, match="role"):
        authorize_action(
            _action(),
            principal_id="executor-1",
            role=cast(Any, "executor"),
            capability=RuntimeCapability.EXECUTE,
            grant_registry=AuthorityGrantRegistry.enroll((grant,)),
            grant_ids=(grant.grant_id,),
            clock=_clock(),
        )
    with pytest.raises(ValueError, match="capability"):
        authorize_action(
            _action(),
            principal_id="executor-1",
            role=RuntimeRole.EXECUTOR,
            capability=cast(Any, "execute"),
            grant_registry=AuthorityGrantRegistry.enroll((grant,)),
            grant_ids=(grant.grant_id,),
            clock=_clock(),
        )
    with pytest.raises(ValueError, match="exact AuthorityGrant"):
        _authorize(
            RuntimeCapability.EXECUTE,
            (
                GrantSubclass(
                    grant.grant_id,
                    grant.principal_id,
                    grant.role,
                    grant.capabilities,
                    grant.action_id,
                    grant.action_digest,
                    grant.authorization_scope,
                    grant.expires_at,
                    grant.evidence_refs,
                ),
            ),
        )


def test_authority_records_have_exact_json_round_trips_and_public_exports() -> None:
    grant = _grant(
        expires_at="2026-09-10T12:00:00Z",
        evidence_refs=(_observable(),),
    )
    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        clock=_clock("2026-09-10T11:59:59+00:00"),
    )

    assert AuthorityGrant.from_dict(json.loads(json.dumps(grant.to_dict()))) == grant
    assert AuthorizationDecision.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision
    with pytest.raises(ValueError, match="exactly"):
        AuthorityGrant.from_dict({**grant.to_dict(), "ambient": True})
    with pytest.raises(ValueError, match="exactly"):
        AuthorizationDecision.from_dict({**decision.to_dict(), "verified": True})


@pytest.mark.parametrize(
    "changed_action",
    [
        _action(action_class="delete"),
        _action(reversible=False),
        _action(prediction_id="prediction-2"),
    ],
)
def test_same_action_id_cannot_reuse_grant_after_canonical_action_changes(
    changed_action: ActionRecord,
) -> None:
    original = _action()
    grant = _grant(action=original)

    decision = _authorize(RuntimeCapability.EXECUTE, (grant,), action=changed_action)

    assert action_record_digest(changed_action) != action_record_digest(original)
    assert decision.authorized is False
    assert decision.reason is AuthorizationReason.NO_MATCHING_GRANT


@pytest.mark.parametrize("changed_field", ["action", "scope", "principal", "evidence"])
def test_irreversible_approval_binding_cannot_be_replayed_or_relabelled(
    changed_field: str,
) -> None:
    original = _action(reversible=False)
    human_grant, approval = _approval(original)
    action = original
    if changed_field == "action":
        action = _action(action_class="delete", reversible=False)
    elif changed_field == "scope":
        action = _action(reversible=False, scope="repo:tests")
    executor_grant = _grant(action=action)
    if changed_field == "principal":
        approval = IrreversibleApprovalBinding(
            human_principal_id="human-2",
            action_id=approval.action_id,
            action_digest=approval.action_digest,
            authorization_scope=approval.authorization_scope,
            authorization_grant_id=approval.authorization_grant_id,
            evidence_refs=approval.evidence_refs,
        )
    elif changed_field == "evidence":
        approval = IrreversibleApprovalBinding(
            human_principal_id=approval.human_principal_id,
            action_id=approval.action_id,
            action_digest=approval.action_digest,
            authorization_scope=approval.authorization_scope,
            authorization_grant_id=approval.authorization_grant_id,
            evidence_refs=(_observable("approval-for-something-else"),),
        )

    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (executor_grant, human_grant),
        action=action,
        irreversible_approval=approval,
    )

    assert decision.authorized is False
    assert decision.reason is AuthorizationReason.MISSING_IRREVERSIBLE_APPROVAL


def test_irreversible_approval_binding_is_exact_immutable_and_json_round_trippable() -> None:
    action = _action(reversible=False)
    _, approval = _approval(action)

    restored = IrreversibleApprovalBinding.from_dict(json.loads(json.dumps(approval.to_dict())))

    assert restored == approval
    with pytest.raises(FrozenInstanceError):
        cast(Any, approval).human_principal_id = "attacker"
    with pytest.raises(ValueError, match="observable evidence"):
        IrreversibleApprovalBinding(
            human_principal_id="human-1",
            action_id=action.action_id,
            action_digest=action_record_digest(action),
            authorization_scope=action.authorization_scope,
            authorization_grant_id="human-grant-1",
            evidence_refs=(EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),),
        )


def test_authorized_decision_is_consumed_only_after_authoritative_revalidation() -> None:
    action = _action()
    grant = _grant(action=action)
    issued = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        action=action,
        clock=_clock("2026-09-10T11:00:00+00:00"),
    )
    serialized = AuthorizationDecision.from_dict(json.loads(json.dumps(issued.to_dict())))

    current = consume_authorization(
        serialized,
        action,
        grant_registry=AuthorityGrantRegistry.enroll((grant,)),
        clock=_clock("2026-09-10T11:30:00+00:00"),
    )

    assert current.authorized is True
    assert current.observed_at == "2026-09-10T11:30:00Z"
    assert current.action_digest == action_record_digest(action)


def test_consumption_revalidates_structured_irreversible_approval() -> None:
    action = _action(reversible=False)
    executor = _grant(action=action)
    human, approval = _approval(action)
    issued = _authorize(
        RuntimeCapability.EXECUTE,
        (executor, human),
        action=action,
        irreversible_approval=approval,
    )

    current = consume_authorization(
        issued,
        action,
        grant_registry=AuthorityGrantRegistry.enroll((executor, human)),
        clock=_clock("2026-09-10T11:30:00+00:00"),
    )
    forged_data = issued.to_dict()
    forged_approval = cast(dict[str, object], forged_data["irreversible_approval"])
    forged_approval["human_principal_id"] = "human-2"
    forged = AuthorizationDecision.from_dict(forged_data)

    assert current.irreversible_approval == approval
    with pytest.raises(PermissionError, match="current authoritative authorization"):
        consume_authorization(
            forged,
            action,
            grant_registry=AuthorityGrantRegistry.enroll((executor, human)),
            clock=_clock("2026-09-10T11:30:00+00:00"),
        )


@pytest.mark.parametrize("forged_field", ["grant_id", "action_digest", "principal_id"])
def test_consumption_rejects_forged_or_deserialized_decisions(forged_field: str) -> None:
    action = _action()
    grant = _grant(action=action)
    issued = _authorize(RuntimeCapability.EXECUTE, (grant,), action=action)
    data = issued.to_dict()
    replacements = {
        "grant_id": "forged-grant",
        "action_digest": "f" * 64,
        "principal_id": "attacker",
    }
    data[forged_field] = replacements[forged_field]
    forged = AuthorizationDecision.from_dict(json.loads(json.dumps(data)))

    with pytest.raises(PermissionError, match="current authoritative authorization"):
        consume_authorization(
            forged,
            action,
            grant_registry=AuthorityGrantRegistry.enroll((grant,)),
            clock=_clock(),
        )


def test_to_dict_revalidates_mutated_decision_and_nested_approval() -> None:
    action = _action(reversible=False)
    executor = _grant(action=action)
    human, approval = _approval(action)
    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (executor, human),
        action=action,
        irreversible_approval=approval,
    )
    object.__setattr__(decision, "reason", "authorized")

    with pytest.raises(ValueError, match="AuthorizationReason"):
        decision.to_dict()

    fresh = _authorize(
        RuntimeCapability.EXECUTE,
        (executor, human),
        action=action,
        irreversible_approval=approval,
    )
    nested = cast(IrreversibleApprovalBinding, fresh.irreversible_approval)
    object.__setattr__(nested, "evidence_refs", list(nested.evidence_refs))
    with pytest.raises(ValueError, match="tuple"):
        fresh.to_dict()


def test_consumption_rejects_decision_replay_at_effective_expiry() -> None:
    action = _action()
    grant = _grant(action=action, expires_at="2026-09-10T12:00:00Z")
    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        action=action,
        clock=_clock("2026-09-10T11:59:59+00:00"),
    )

    assert decision.effective_expires_at == "2026-09-10T12:00:00Z"
    with pytest.raises(PermissionError, match="current authoritative authorization"):
        consume_authorization(
            decision,
            action,
            grant_registry=AuthorityGrantRegistry.enroll((grant,)),
            clock=_clock("2026-09-10T12:00:00+00:00"),
        )


def test_authorization_time_comes_only_from_injected_trusted_clock() -> None:
    action = _action()
    grant = _grant(action=action, expires_at="2026-09-10T12:00:00Z")

    decision = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        action=action,
        clock=_clock("2026-09-10T12:00:01+00:00"),
    )

    assert decision.reason is AuthorizationReason.GRANT_EXPIRED
    with pytest.raises(TypeError, match="observed_at"):
        cast(Any, authorize_action)(
            action,
            principal_id="executor-1",
            role=RuntimeRole.EXECUTOR,
            capability=RuntimeCapability.EXECUTE,
            grant_registry=AuthorityGrantRegistry.enroll((grant,)),
            grant_ids=(grant.grant_id,),
            clock=_clock("2026-09-10T12:00:01+00:00"),
            observed_at="2026-09-10T11:00:00Z",
        )


@pytest.mark.parametrize(
    "clock_value",
    [
        datetime(2026, 9, 10, 11, 0, 0),
        cast(Any, "2026-09-10T11:00:00Z"),
        type("HostileDatetime", (datetime,), {})(
            2026,
            9,
            10,
            11,
            tzinfo=_clock().current.tzinfo,
        ),
    ],
)
def test_trusted_clock_must_return_an_exact_aware_datetime(clock_value: object) -> None:
    clock = _FixedClock(cast(Any, clock_value))

    with pytest.raises(ValueError, match="clock"):
        _authorize(RuntimeCapability.EXECUTE, (_grant(),), clock=clock)


def test_authorization_reason_and_decision_invariants_fail_closed() -> None:
    action = _action()
    decision = _authorize(RuntimeCapability.EXECUTE, (), action=action)
    data = decision.to_dict()

    assert type(decision.reason) is AuthorizationReason
    with pytest.raises(ValueError, match="denied.*authorized"):
        AuthorizationDecision.from_dict({**data, "reason": "authorized"})
    with pytest.raises(ValueError, match="denied.*references"):
        AuthorizationDecision.from_dict({**data, "authorization_refs": [_observable().to_dict()]})
    irreversible_action = _action(reversible=False)
    _, approval = _approval(irreversible_action)
    with pytest.raises(ValueError, match="denied.*irreversible approval"):
        AuthorizationDecision.from_dict({**data, "irreversible_approval": approval.to_dict()})


def test_top_level_runtime_governance_module_preserves_public_compatibility() -> None:
    from gepa_mindfulness import runtime_governance as public_runtime
    from gepa_mindfulness.verification import runtime_governance as verification_runtime

    assert public_runtime.AuthorityGrant is verification_runtime.AuthorityGrant
    assert public_runtime.AuthorityGrantRegistry is verification_runtime.AuthorityGrantRegistry
    assert public_runtime.AuthorizationDecision is verification_runtime.AuthorizationDecision
    assert public_runtime.authorize_action is verification_runtime.authorize_action
    assert public_runtime.consume_authorization is verification_runtime.consume_authorization


def test_enrollment_snapshots_prevent_coherent_grant_retargeting() -> None:
    original_action = _action()
    retargeted_action = _action(
        action_class="delete",
        reversible=False,
        scope="repo:tests",
        prediction_id="prediction-2",
    )
    original_grant = _grant(action=original_action)
    registry = AuthorityGrantRegistry.enroll((original_grant,))
    object.__setattr__(original_grant, "action_id", retargeted_action.action_id)
    object.__setattr__(
        original_grant,
        "action_digest",
        action_record_digest(retargeted_action),
    )
    object.__setattr__(
        original_grant,
        "authorization_scope",
        retargeted_action.authorization_scope,
    )
    object.__setattr__(original_grant, "principal_id", "attacker")
    object.__setattr__(original_grant, "capabilities", (RuntimeCapability.WRITE,))
    object.__setattr__(original_grant, "expires_at", "2026-09-11T12:00:00Z")

    original = authorize_action(
        original_action,
        principal_id="executor-1",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.EXECUTE,
        grant_registry=registry,
        grant_ids=("grant-1",),
        clock=_clock(),
    )
    retargeted = authorize_action(
        retargeted_action,
        principal_id="attacker",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        grant_registry=registry,
        grant_ids=("grant-1",),
        clock=_clock(),
    )

    assert original.authorized is True
    assert retargeted.reason is AuthorizationReason.NO_MATCHING_GRANT


def test_registry_resolution_returns_defensive_snapshots_without_aliases() -> None:
    grant = _grant()
    registry = AuthorityGrantRegistry.enroll((grant,))
    resolved = registry.resolve((grant.grant_id,))[0]
    object.__setattr__(resolved, "principal_id", "attacker")

    fresh = registry.resolve((grant.grant_id,))[0]

    assert fresh.principal_id == "executor-1"
    assert fresh is not grant
    assert fresh is not resolved


def test_registry_rejects_duplicate_enrollment_and_ambiguous_or_unknown_resolution() -> None:
    grant = _grant()
    with pytest.raises(ValueError, match="unique"):
        AuthorityGrantRegistry.enroll((grant, grant))
    registry = AuthorityGrantRegistry.enroll((grant,))

    with pytest.raises(ValueError, match="unique"):
        registry.resolve((grant.grant_id, grant.grant_id))
    with pytest.raises(KeyError, match="unknown authority grant"):
        registry.resolve(("unknown-grant",))


def test_authorization_accepts_only_an_enrolled_exact_registry() -> None:
    action = _action()
    grant = _grant(action=action)

    with pytest.raises(TypeError, match="unexpected keyword argument 'grants'"):
        cast(Any, authorize_action)(
            action,
            principal_id="executor-1",
            role=RuntimeRole.EXECUTOR,
            capability=RuntimeCapability.EXECUTE,
            grants=(grant,),
            clock=_clock(),
        )
    with pytest.raises(ValueError, match="exact AuthorityGrantRegistry"):
        authorize_action(
            action,
            principal_id="executor-1",
            role=RuntimeRole.EXECUTOR,
            capability=RuntimeCapability.EXECUTE,
            grant_registry=cast(Any, object()),
            grant_ids=(grant.grant_id,),
            clock=_clock(),
        )


def test_mutating_authoritative_registry_storage_fails_closed() -> None:
    grant = _grant()
    registry = AuthorityGrantRegistry.enroll((grant,))
    forged = _grant(principal_id="attacker")
    object.__setattr__(registry, "_entries", (forged,))

    with pytest.raises(ValueError, match="registry integrity"):
        registry.resolve((grant.grant_id,))
