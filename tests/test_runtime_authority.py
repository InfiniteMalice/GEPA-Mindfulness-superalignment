"""Contract tests for least-authority runtime governance."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    AuthorityGrant,
    AuthorizationDecision,
    RuntimeCapability,
    RuntimeRole,
    authorize_action,
)
from mindful_trace_gepa import ActionRecord


def _action(*, reversible: bool = True, scope: str = "repo:src") -> ActionRecord:
    return ActionRecord("action-1", "edit", reversible, scope, "prediction-1")


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
    expires_at: str | None = None,
    evidence_refs: tuple[EvidenceReference, ...] = (),
) -> AuthorityGrant:
    return AuthorityGrant(
        grant_id=grant_id,
        principal_id=principal_id,
        role=role,
        capabilities=capabilities,
        action_id=action_id,
        authorization_scope=scope,
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
    observed_at: str | None = None,
) -> AuthorizationDecision:
    return authorize_action(
        _action() if action is None else action,
        principal_id=principal_id,
        role=role,
        capability=capability,
        grants=grants,
        action_author_id=action_author_id,
        action_executor_id=action_executor_id,
        observed_at=observed_at,
    )


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

    decision = _authorize(RuntimeCapability.EXECUTE, duplicated)

    assert decision.authorized is False
    assert decision.reason == "duplicate_grant_ids"


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
    executor_grant = _grant()
    human_grant = _grant(
        principal_id="human-1",
        role=RuntimeRole.HUMAN,
        capabilities=(RuntimeCapability.AUTHORIZE_IRREVERSIBLE,),
        grant_id="human-grant-1",
        evidence_refs=(_observable(),),
    )

    denied = _authorize(RuntimeCapability.EXECUTE, (executor_grant,), action=action)
    authorized = _authorize(
        RuntimeCapability.EXECUTE,
        (executor_grant, human_grant),
        action=action,
    )

    assert denied.reason == "missing_human_irreversible_authorization"
    assert authorized.authorized is True
    assert authorized.authorization_scope == "repo:src"
    assert authorized.authorization_refs == (_observable(),)


def test_irreversible_authorization_is_bound_to_exact_action_and_scope() -> None:
    executor_grant = _grant()
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
        action=_action(reversible=False),
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
            authorization_scope="repo:src",
            principal_id="planner-1",
            role=RuntimeRole.PLANNER,
            capability=RuntimeCapability.EXECUTE,
            authorized=True,
            grant_id="grant-1",
            authorization_refs=(),
            reason="authorized",
        )


def test_irreversible_authorization_decision_requires_observable_evidence() -> None:
    private = EvidenceReference("thought-1", EvidenceSourceKind.PRIVATE_REASONING)

    with pytest.raises(ValueError, match="observable evidence"):
        AuthorizationDecision(
            action_id="action-1",
            authorization_scope="repo:src",
            principal_id="human-1",
            role=RuntimeRole.HUMAN,
            capability=RuntimeCapability.AUTHORIZE_IRREVERSIBLE,
            authorized=True,
            grant_id="human-grant-1",
            authorization_refs=(private,),
            reason="authorized",
        )


def test_ordinary_execution_does_not_claim_irreversible_authorization_evidence() -> None:
    decision = _authorize(RuntimeCapability.EXECUTE, (_grant(),))

    assert decision.authorized is True
    assert decision.authorization_refs == ()


def test_expired_grants_fail_closed_at_an_explicit_observation_time() -> None:
    grant = _grant(expires_at="2026-09-10T12:00:00Z")

    expired = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        observed_at="2026-09-10T12:00:01Z",
    )
    current = _authorize(
        RuntimeCapability.EXECUTE,
        (grant,),
        observed_at="2026-09-10T12:00:00Z",
    )
    unknown_time = _authorize(RuntimeCapability.EXECUTE, (grant,))

    assert expired.reason == "no_matching_grant"
    assert current.authorized is True
    assert unknown_time.reason == "missing_observation_time"


def test_authority_grant_snapshots_collections_and_is_frozen() -> None:
    capabilities = [RuntimeCapability.EXECUTE]
    references = [_observable()]
    grant = AuthorityGrant(
        "grant-1",
        "executor-1",
        RuntimeRole.EXECUTOR,
        cast(Any, capabilities),
        "action-1",
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
            grants=(grant,),
        )
    with pytest.raises(ValueError, match="capability"):
        authorize_action(
            _action(),
            principal_id="executor-1",
            role=RuntimeRole.EXECUTOR,
            capability=cast(Any, "execute"),
            grants=(grant,),
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
        observed_at="2026-09-10T11:59:59Z",
    )

    assert AuthorityGrant.from_dict(json.loads(json.dumps(grant.to_dict()))) == grant
    assert AuthorizationDecision.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision
    with pytest.raises(ValueError, match="exactly"):
        AuthorityGrant.from_dict({**grant.to_dict(), "ambient": True})
    with pytest.raises(ValueError, match="exactly"):
        AuthorizationDecision.from_dict({**decision.to_dict(), "verified": True})
