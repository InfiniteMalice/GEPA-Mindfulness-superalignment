"""Least-authority grants for runtime actions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, cast

from gepa_mindfulness.core.evidence import EvidenceReference
from mindful_trace_gepa.action_bound_events import ActionRecord

from .state import (
    _require_exact_mapping,
    _require_nonblank_string,
    _require_rfc3339,
    _snapshot_evidence_refs,
)


class RuntimeRole(str, Enum):
    """A runtime actor category without ambient capabilities."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    AUDITOR = "auditor"
    HUMAN = "human"


class RuntimeCapability(str, Enum):
    """One operation that an action-scoped grant may authorize."""

    READ = "read"
    PROPOSE = "propose"
    WRITE = "write"
    EXECUTE = "execute"
    VERIFY = "verify"
    AUDIT = "audit"
    AUTHORIZE_IRREVERSIBLE = "authorize_irreversible"


_ROLE_CAPABILITIES = {
    RuntimeRole.PLANNER: frozenset(
        {
            RuntimeCapability.READ,
            RuntimeCapability.PROPOSE,
        }
    ),
    RuntimeRole.EXECUTOR: frozenset(
        {
            RuntimeCapability.READ,
            RuntimeCapability.WRITE,
            RuntimeCapability.EXECUTE,
        }
    ),
    RuntimeRole.VERIFIER: frozenset(
        {
            RuntimeCapability.READ,
            RuntimeCapability.VERIFY,
        }
    ),
    RuntimeRole.AUDITOR: frozenset(
        {
            RuntimeCapability.READ,
            RuntimeCapability.AUDIT,
        }
    ),
    RuntimeRole.HUMAN: frozenset(
        {
            RuntimeCapability.READ,
            RuntimeCapability.PROPOSE,
            RuntimeCapability.AUTHORIZE_IRREVERSIBLE,
        }
    ),
}


class _ActionRecordFields(Protocol):
    action_id: str
    action_class: str
    reversible: bool
    authorization_scope: str
    prediction_commit_id: str


@dataclass(frozen=True, slots=True)
class AuthorityGrant:
    """An immutable capability grant bound to one principal, action, and scope."""

    grant_id: str
    principal_id: str
    role: RuntimeRole
    capabilities: tuple[RuntimeCapability, ...]
    action_id: str
    authorization_scope: str
    expires_at: str | None = None
    evidence_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambient, role-incompatible, or unobservable authority."""

        _require_nonblank_string(self.grant_id, "grant_id")
        _require_nonblank_string(self.principal_id, "principal_id")
        if type(self.role) is not RuntimeRole:
            raise ValueError("role must be an exact RuntimeRole")
        capabilities = _snapshot_capabilities(self.capabilities)
        role_capabilities = _ROLE_CAPABILITIES[self.role]
        incompatible = [item.value for item in capabilities if item not in role_capabilities]
        if incompatible:
            raise ValueError(
                f"capabilities {incompatible!r} are not permitted for role {self.role.value!r}"
            )
        object.__setattr__(self, "capabilities", capabilities)
        _require_scoped_string(self.action_id, "action_id")
        _require_scoped_string(self.authorization_scope, "authorization_scope")
        if self.expires_at is not None:
            _require_rfc3339(self.expires_at, "expires_at")
        references = _snapshot_evidence_refs(self.evidence_refs)
        object.__setattr__(self, "evidence_refs", references)
        if RuntimeCapability.AUTHORIZE_IRREVERSIBLE in capabilities and not any(
            reference.is_observable for reference in references
        ):
            raise ValueError("irreversible authorization requires observable evidence")

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible grant snapshot."""

        snapshot = _snapshot_grant(self)
        return {
            "grant_id": snapshot.grant_id,
            "principal_id": snapshot.principal_id,
            "role": snapshot.role.value,
            "capabilities": [capability.value for capability in snapshot.capabilities],
            "action_id": snapshot.action_id,
            "authorization_scope": snapshot.authorization_scope,
            "expires_at": snapshot.expires_at,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> AuthorityGrant:
        """Restore a grant from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "grant_id",
                "principal_id",
                "role",
                "capabilities",
                "action_id",
                "authorization_scope",
                "expires_at",
                "evidence_refs",
            },
            "AuthorityGrant",
        )
        return cls(
            grant_id=cast(str, values["grant_id"]),
            principal_id=cast(str, values["principal_id"]),
            role=_restore_role(values["role"], "AuthorityGrant"),
            capabilities=_restore_capabilities(values["capabilities"]),
            action_id=cast(str, values["action_id"]),
            authorization_scope=cast(str, values["authorization_scope"]),
            expires_at=cast(str | None, values["expires_at"]),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"]),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """An auditable authorization finding that does not claim execution or verification."""

    action_id: str
    authorization_scope: str
    principal_id: str
    role: RuntimeRole
    capability: RuntimeCapability
    authorized: bool
    grant_id: str | None
    authorization_refs: tuple[EvidenceReference, ...]
    reason: str

    def __post_init__(self) -> None:
        """Validate the finding and detach its authorization evidence."""

        _require_nonblank_string(self.action_id, "action_id")
        _require_scoped_string(self.authorization_scope, "authorization_scope")
        _require_nonblank_string(self.principal_id, "principal_id")
        if type(self.role) is not RuntimeRole:
            raise ValueError("role must be an exact RuntimeRole")
        if type(self.capability) is not RuntimeCapability:
            raise ValueError("capability must be an exact RuntimeCapability")
        if type(self.authorized) is not bool:
            raise ValueError("authorized must be a built-in bool")
        if self.grant_id is not None:
            _require_nonblank_string(self.grant_id, "grant_id")
        _require_nonblank_string(self.reason, "reason")
        references = _snapshot_evidence_refs(self.authorization_refs)
        object.__setattr__(self, "authorization_refs", references)
        if self.authorized:
            if self.grant_id is None or self.reason != "authorized":
                raise ValueError("authorized decisions require a grant_id and authorized reason")
            if self.capability not in _ROLE_CAPABILITIES[self.role]:
                raise ValueError(
                    f"capability {self.capability.value!r} is not permitted for role "
                    f"{self.role.value!r}"
                )
            if self.capability is RuntimeCapability.AUTHORIZE_IRREVERSIBLE and not any(
                reference.is_observable for reference in references
            ):
                raise ValueError("irreversible authorization requires observable evidence")
        elif self.grant_id is not None:
            raise ValueError("denied decisions cannot claim a grant_id")

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible authorization finding."""

        snapshot = _snapshot_decision(self)
        return {
            "action_id": snapshot.action_id,
            "authorization_scope": snapshot.authorization_scope,
            "principal_id": snapshot.principal_id,
            "role": snapshot.role.value,
            "capability": snapshot.capability.value,
            "authorized": snapshot.authorized,
            "grant_id": snapshot.grant_id,
            "authorization_refs": [
                reference.to_dict() for reference in snapshot.authorization_refs
            ],
            "reason": snapshot.reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> AuthorizationDecision:
        """Restore an authorization finding from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "action_id",
                "authorization_scope",
                "principal_id",
                "role",
                "capability",
                "authorized",
                "grant_id",
                "authorization_refs",
                "reason",
            },
            "AuthorizationDecision",
        )
        return cls(
            action_id=cast(str, values["action_id"]),
            authorization_scope=cast(str, values["authorization_scope"]),
            principal_id=cast(str, values["principal_id"]),
            role=_restore_role(values["role"], "AuthorizationDecision"),
            capability=_restore_capability(values["capability"], "AuthorizationDecision"),
            authorized=cast(bool, values["authorized"]),
            grant_id=cast(str | None, values["grant_id"]),
            authorization_refs=_restore_evidence_refs(values["authorization_refs"]),
            reason=cast(str, values["reason"]),
        )


def authorize_action(
    action: ActionRecord,
    *,
    principal_id: str,
    role: RuntimeRole,
    capability: RuntimeCapability,
    grants: Sequence[AuthorityGrant],
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
    observed_at: str | None = None,
) -> AuthorizationDecision:
    """Authorize one exact action capability without inferring authority from a role label."""

    action = _snapshot_action(action)
    _require_nonblank_string(principal_id, "principal_id")
    if type(role) is not RuntimeRole:
        raise ValueError("role must be an exact RuntimeRole")
    if type(capability) is not RuntimeCapability:
        raise ValueError("capability must be an exact RuntimeCapability")
    _require_optional_identifier(action_author_id, "action_author_id")
    _require_optional_identifier(action_executor_id, "action_executor_id")
    if observed_at is not None:
        _require_rfc3339(observed_at, "observed_at")
    snapshots = _snapshot_grants(grants)
    if len({grant.grant_id for grant in snapshots}) != len(snapshots):
        return _deny(action, principal_id, role, capability, "duplicate_grant_ids")

    candidates = _matching_grants(
        action,
        principal_id,
        role,
        capability,
        snapshots,
    )
    candidates, time_failure = _filter_current_grants(candidates, observed_at)
    if time_failure is not None:
        return _deny(action, principal_id, role, capability, time_failure)
    if not candidates:
        return _deny(action, principal_id, role, capability, "no_matching_grant")
    if len(candidates) != 1:
        return _deny(action, principal_id, role, capability, "ambiguous_matching_grants")
    grant = candidates[0]

    independence_failure = _verify_independence(
        principal_id,
        capability,
        action_author_id,
        action_executor_id,
    )
    if independence_failure is not None:
        return _deny(action, principal_id, role, capability, independence_failure)

    authorization_refs: tuple[EvidenceReference, ...] = ()
    if capability is RuntimeCapability.AUTHORIZE_IRREVERSIBLE:
        authorization_refs = grant.evidence_refs
    if capability is RuntimeCapability.EXECUTE and not action.reversible:
        human_grants = _matching_irreversible_grants(action, snapshots)
        human_grants, time_failure = _filter_current_grants(human_grants, observed_at)
        if time_failure is not None:
            return _deny(action, principal_id, role, capability, time_failure)
        if not human_grants:
            return _deny(
                action,
                principal_id,
                role,
                capability,
                "missing_human_irreversible_authorization",
            )
        if len(human_grants) != 1:
            return _deny(
                action,
                principal_id,
                role,
                capability,
                "ambiguous_human_irreversible_authorization",
            )
        authorization_refs = human_grants[0].evidence_refs

    return AuthorizationDecision(
        action_id=action.action_id,
        authorization_scope=action.authorization_scope,
        principal_id=principal_id,
        role=role,
        capability=capability,
        authorized=True,
        grant_id=grant.grant_id,
        authorization_refs=authorization_refs,
        reason="authorized",
    )


def _snapshot_action(action: object) -> ActionRecord:
    if type(action) is not ActionRecord:
        raise TypeError("action must be an exact ActionRecord")
    typed_action = cast(_ActionRecordFields, action)
    action_type = cast(Callable[[str, str, bool, str, str], ActionRecord], ActionRecord)
    try:
        return cast(
            ActionRecord,
            action_type(
                typed_action.action_id,
                typed_action.action_class,
                typed_action.reversible,
                typed_action.authorization_scope,
                typed_action.prediction_commit_id,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action contains invalid canonical fields: {exc}") from exc


def _snapshot_capabilities(value: object) -> tuple[RuntimeCapability, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ValueError("capabilities must be an iterable of exact RuntimeCapability values")
    capabilities: list[RuntimeCapability] = []
    for capability in value:
        if type(capability) is not RuntimeCapability:
            raise ValueError("capabilities must contain exact RuntimeCapability values")
        capabilities.append(capability)
    if not capabilities:
        raise ValueError("capabilities must contain at least one capability")
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("capabilities must be unique")
    return tuple(capabilities)


def _snapshot_grant(grant: object) -> AuthorityGrant:
    if type(grant) is not AuthorityGrant:
        raise ValueError("grants must contain exact AuthorityGrant values")
    try:
        return AuthorityGrant(
            grant.grant_id,
            grant.principal_id,
            grant.role,
            grant.capabilities,
            grant.action_id,
            grant.authorization_scope,
            grant.expires_at,
            grant.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grants contains an invalid AuthorityGrant: {exc}") from exc


def _snapshot_grants(value: object) -> tuple[AuthorityGrant, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("grants must be an ordered array of AuthorityGrant values")
    return tuple(_snapshot_grant(grant) for grant in value)


def _snapshot_decision(decision: object) -> AuthorizationDecision:
    if type(decision) is not AuthorizationDecision:
        raise ValueError("decision must be an exact AuthorizationDecision")
    return AuthorizationDecision(
        decision.action_id,
        decision.authorization_scope,
        decision.principal_id,
        decision.role,
        decision.capability,
        decision.authorized,
        decision.grant_id,
        decision.authorization_refs,
        decision.reason,
    )


def _matching_grants(
    action: ActionRecord,
    principal_id: str,
    role: RuntimeRole,
    capability: RuntimeCapability,
    grants: tuple[AuthorityGrant, ...],
) -> tuple[AuthorityGrant, ...]:
    return tuple(
        grant
        for grant in grants
        if grant.principal_id == principal_id
        and grant.role is role
        and grant.action_id == action.action_id
        and grant.authorization_scope == action.authorization_scope
        and capability in grant.capabilities
    )


def _matching_irreversible_grants(
    action: ActionRecord,
    grants: tuple[AuthorityGrant, ...],
) -> tuple[AuthorityGrant, ...]:
    return tuple(
        grant
        for grant in grants
        if grant.role is RuntimeRole.HUMAN
        and grant.action_id == action.action_id
        and grant.authorization_scope == action.authorization_scope
        and RuntimeCapability.AUTHORIZE_IRREVERSIBLE in grant.capabilities
        and any(reference.is_observable for reference in grant.evidence_refs)
    )


def _filter_current_grants(
    grants: tuple[AuthorityGrant, ...],
    observed_at: str | None,
) -> tuple[tuple[AuthorityGrant, ...], str | None]:
    if any(grant.expires_at is not None for grant in grants) and observed_at is None:
        return (), "missing_observation_time"
    if observed_at is None:
        return grants, None
    observed = _parse_rfc3339(observed_at)
    current = tuple(
        grant
        for grant in grants
        if grant.expires_at is None or observed <= _parse_rfc3339(grant.expires_at)
    )
    return current, None


def _verify_independence(
    principal_id: str,
    capability: RuntimeCapability,
    action_author_id: str | None,
    action_executor_id: str | None,
) -> str | None:
    if capability is not RuntimeCapability.VERIFY:
        return None
    if action_author_id is None or action_executor_id is None:
        return "missing_verifier_independence_context"
    if principal_id == action_author_id:
        return "verifier_authored_action"
    if principal_id == action_executor_id:
        return "verifier_executed_action"
    return None


def _deny(
    action: ActionRecord,
    principal_id: str,
    role: RuntimeRole,
    capability: RuntimeCapability,
    reason: str,
) -> AuthorizationDecision:
    return AuthorizationDecision(
        action_id=action.action_id,
        authorization_scope=action.authorization_scope,
        principal_id=principal_id,
        role=role,
        capability=capability,
        authorized=False,
        grant_id=None,
        authorization_refs=(),
        reason=reason,
    )


def _require_scoped_string(value: object, field_name: str) -> str:
    result = _require_nonblank_string(value, field_name)
    if result == "*":
        raise ValueError(f"{field_name} cannot be a wildcard")
    return result


def _require_optional_identifier(value: object, field_name: str) -> None:
    if value is not None:
        _require_nonblank_string(value, field_name)


def _parse_rfc3339(value: str) -> datetime:
    parsed_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(parsed_value)


def _restore_role(value: object, record_name: str) -> RuntimeRole:
    if type(value) is not str:
        raise ValueError(f"{record_name} role must be a built-in string")
    try:
        return RuntimeRole(value)
    except ValueError as exc:
        raise ValueError(f"unknown {record_name} role {value!r}") from exc


def _restore_capability(value: object, record_name: str) -> RuntimeCapability:
    if type(value) is not str:
        raise ValueError(f"{record_name} capability must be a built-in string")
    try:
        return RuntimeCapability(value)
    except ValueError as exc:
        raise ValueError(f"unknown {record_name} capability {value!r}") from exc


def _restore_capabilities(value: object) -> tuple[RuntimeCapability, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("AuthorityGrant capabilities must be an array")
    return tuple(_restore_capability(item, "AuthorityGrant") for item in value)


def _restore_evidence_refs(value: object) -> tuple[EvidenceReference, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("evidence references must be an array")
    return tuple(EvidenceReference.from_dict(item) for item in value)


__all__ = [
    "AuthorityGrant",
    "AuthorizationDecision",
    "RuntimeCapability",
    "RuntimeRole",
    "authorize_action",
]
