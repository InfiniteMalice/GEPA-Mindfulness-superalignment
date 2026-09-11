"""Least-authority grants for runtime actions.

Before a runtime consumer uses a stored ``AuthorizationDecision``, the consumer must call
``consume_authorization`` with the current action, authoritative grant registry, and trusted
clock.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, cast

from gepa_mindfulness.core.evidence import EvidenceReference
from mindful_trace_gepa.action_bound_events import ActionRecord

from .state import (
    _require_exact_mapping,
    _require_nonblank_string,
    _require_rfc3339,
    _require_sha256,
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


class AuthorizationReason(str, Enum):
    """The exact reason for one authorization decision."""

    AUTHORIZED = "authorized"
    NO_MATCHING_GRANT = "no_matching_grant"
    DUPLICATE_GRANT_IDS = "duplicate_grant_ids"
    AMBIGUOUS_MATCHING_GRANTS = "ambiguous_matching_grants"
    GRANT_EXPIRED = "grant_expired"
    MISSING_VERIFIER_INDEPENDENCE_CONTEXT = "missing_verifier_independence_context"
    VERIFIER_AUTHORED_ACTION = "verifier_authored_action"
    VERIFIER_EXECUTED_ACTION = "verifier_executed_action"
    MISSING_IRREVERSIBLE_APPROVAL = "missing_human_irreversible_authorization"
    IRREVERSIBLE_APPROVAL_EXPIRED = "irreversible_approval_expired"


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


class TrustedClock(Protocol):
    """A runtime-owned source of an aware current datetime."""

    def now(self) -> datetime:
        """Return the trusted current time as an exact aware datetime."""
        ...


@dataclass(frozen=True, slots=True)
class AuthorityGrant:
    """An immutable capability grant bound to one principal, action, and scope."""

    grant_id: str
    principal_id: str
    role: RuntimeRole
    capabilities: tuple[RuntimeCapability, ...]
    action_id: str
    action_digest: str
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
        _require_sha256(self.action_digest, "action_digest")
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
            "action_digest": snapshot.action_digest,
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
                "action_digest",
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
            action_digest=cast(str, values["action_digest"]),
            authorization_scope=cast(str, values["authorization_scope"]),
            expires_at=cast(str | None, values["expires_at"]),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"]),
        )


class AuthorityGrantRegistry:
    """An authoritative local store of defensively enrolled grant snapshots.

    ``enroll`` is the trust boundary: the runtime owner must authenticate issuers before
    enrollment. This local registry detects later mutation but does not authenticate issuers or
    provide cryptographic signatures. Authorization accepts only an exact enrolled registry and
    never accepts raw grants.
    """

    __slots__ = ("_entries", "_seal")
    _entries: tuple[AuthorityGrant, ...]
    _seal: str

    def __init__(self) -> None:
        """Prevent construction that bypasses explicit enrollment."""

        raise TypeError("use AuthorityGrantRegistry.enroll()")

    def __setattr__(self, name: str, value: object) -> None:
        """Keep enrolled storage read-only through the public object interface."""

        del name, value
        raise AttributeError("AuthorityGrantRegistry is read-only")

    @classmethod
    def enroll(cls, grants: Sequence[AuthorityGrant]) -> AuthorityGrantRegistry:
        """Create an authoritative registry after the runtime owner authenticates issuers."""

        snapshots = _snapshot_grants(grants)
        grant_ids = tuple(grant.grant_id for grant in snapshots)
        if len(set(grant_ids)) != len(grant_ids):
            raise ValueError("enrolled AuthorityGrant IDs must be unique")
        registry = object.__new__(cls)
        object.__setattr__(registry, "_entries", snapshots)
        object.__setattr__(registry, "_seal", _registry_digest(snapshots))
        return registry

    def resolve(self, grant_ids: Sequence[str]) -> tuple[AuthorityGrant, ...]:
        """Return defensive snapshots for exact, unique enrolled grant IDs."""

        entries = _validated_registry_entries(self)
        requested_ids = _snapshot_grant_ids(grant_ids)
        grants_by_id = {grant.grant_id: grant for grant in entries}
        missing = [grant_id for grant_id in requested_ids if grant_id not in grants_by_id]
        if missing:
            raise KeyError(f"unknown authority grant IDs {missing!r}")
        return tuple(_snapshot_grant(grants_by_id[grant_id]) for grant_id in requested_ids)


@dataclass(frozen=True, slots=True)
class IrreversibleApprovalBinding:
    """Observable human approval bound to one complete irreversible action identity."""

    human_principal_id: str
    action_id: str
    action_digest: str
    authorization_scope: str
    authorization_grant_id: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Require exact action, human-grant, scope, and observable evidence bindings."""

        _require_nonblank_string(self.human_principal_id, "human_principal_id")
        _require_scoped_string(self.action_id, "action_id")
        _require_sha256(self.action_digest, "action_digest")
        _require_scoped_string(self.authorization_scope, "authorization_scope")
        _require_nonblank_string(self.authorization_grant_id, "authorization_grant_id")
        references = _snapshot_evidence_refs(self.evidence_refs)
        if not references or not any(reference.is_observable for reference in references):
            raise ValueError("irreversible approval requires observable evidence")
        object.__setattr__(self, "evidence_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible approval binding."""

        snapshot = _snapshot_approval(self)
        return {
            "human_principal_id": snapshot.human_principal_id,
            "action_id": snapshot.action_id,
            "action_digest": snapshot.action_digest,
            "authorization_scope": snapshot.authorization_scope,
            "authorization_grant_id": snapshot.authorization_grant_id,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> IrreversibleApprovalBinding:
        """Restore an approval binding from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "human_principal_id",
                "action_id",
                "action_digest",
                "authorization_scope",
                "authorization_grant_id",
                "evidence_refs",
            },
            "IrreversibleApprovalBinding",
        )
        return cls(
            human_principal_id=cast(str, values["human_principal_id"]),
            action_id=cast(str, values["action_id"]),
            action_digest=cast(str, values["action_digest"]),
            authorization_scope=cast(str, values["authorization_scope"]),
            authorization_grant_id=cast(str, values["authorization_grant_id"]),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"]),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Authorization evidence that consumers must revalidate before use."""

    action_id: str
    action_digest: str
    authorization_scope: str
    principal_id: str
    role: RuntimeRole
    capability: RuntimeCapability
    authorized: bool
    grant_id: str | None
    authorization_refs: tuple[EvidenceReference, ...]
    irreversible_approval: IrreversibleApprovalBinding | None
    reason: AuthorizationReason
    observed_at: str
    effective_expires_at: str | None

    def __post_init__(self) -> None:
        """Validate the finding and detach its authorization evidence."""

        _require_nonblank_string(self.action_id, "action_id")
        _require_sha256(self.action_digest, "action_digest")
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
        if type(self.reason) is not AuthorizationReason:
            raise ValueError("reason must be an exact AuthorizationReason")
        _require_rfc3339(self.observed_at, "observed_at")
        if self.effective_expires_at is not None:
            _require_rfc3339(self.effective_expires_at, "effective_expires_at")
        references = _snapshot_evidence_refs(self.authorization_refs)
        object.__setattr__(self, "authorization_refs", references)
        approval = _snapshot_optional_approval(self.irreversible_approval)
        object.__setattr__(self, "irreversible_approval", approval)
        if self.authorized:
            if self.grant_id is None or self.reason is not AuthorizationReason.AUTHORIZED:
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
            if approval is not None:
                if self.capability is not RuntimeCapability.EXECUTE:
                    raise ValueError("only execution decisions can contain irreversible approval")
                if references != approval.evidence_refs:
                    raise ValueError("authorization_refs must match irreversible approval evidence")
            elif references and self.capability is not RuntimeCapability.AUTHORIZE_IRREVERSIBLE:
                raise ValueError("authorized decisions cannot carry unrelated authorization refs")
            if self.effective_expires_at is not None and not (
                _parse_rfc3339(self.observed_at) < _parse_rfc3339(self.effective_expires_at)
            ):
                raise ValueError("authorized decision must precede its effective expiry")
        else:
            if self.reason is AuthorizationReason.AUTHORIZED:
                raise ValueError("denied decisions cannot use the authorized reason")
            if self.grant_id is not None:
                raise ValueError("denied decisions cannot claim a grant_id")
            if references:
                raise ValueError("denied decisions cannot carry authorization references")
            if approval is not None:
                raise ValueError("denied decisions cannot carry an irreversible approval")

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible authorization finding."""

        snapshot = _snapshot_decision(self)
        return {
            "action_id": snapshot.action_id,
            "action_digest": snapshot.action_digest,
            "authorization_scope": snapshot.authorization_scope,
            "principal_id": snapshot.principal_id,
            "role": snapshot.role.value,
            "capability": snapshot.capability.value,
            "authorized": snapshot.authorized,
            "grant_id": snapshot.grant_id,
            "authorization_refs": [
                reference.to_dict() for reference in snapshot.authorization_refs
            ],
            "irreversible_approval": (
                None
                if snapshot.irreversible_approval is None
                else snapshot.irreversible_approval.to_dict()
            ),
            "reason": snapshot.reason.value,
            "observed_at": snapshot.observed_at,
            "effective_expires_at": snapshot.effective_expires_at,
        }

    @classmethod
    def from_dict(cls, data: object) -> AuthorizationDecision:
        """Restore an authorization finding from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "action_id",
                "action_digest",
                "authorization_scope",
                "principal_id",
                "role",
                "capability",
                "authorized",
                "grant_id",
                "authorization_refs",
                "irreversible_approval",
                "reason",
                "observed_at",
                "effective_expires_at",
            },
            "AuthorizationDecision",
        )
        return cls(
            action_id=cast(str, values["action_id"]),
            action_digest=cast(str, values["action_digest"]),
            authorization_scope=cast(str, values["authorization_scope"]),
            principal_id=cast(str, values["principal_id"]),
            role=_restore_role(values["role"], "AuthorizationDecision"),
            capability=_restore_capability(values["capability"], "AuthorizationDecision"),
            authorized=cast(bool, values["authorized"]),
            grant_id=cast(str | None, values["grant_id"]),
            authorization_refs=_restore_evidence_refs(values["authorization_refs"]),
            irreversible_approval=_restore_optional_approval(values["irreversible_approval"]),
            reason=_restore_reason(values["reason"]),
            observed_at=cast(str, values["observed_at"]),
            effective_expires_at=cast(str | None, values["effective_expires_at"]),
        )


def authorize_action(
    action: ActionRecord,
    *,
    principal_id: str,
    role: RuntimeRole,
    capability: RuntimeCapability,
    grant_registry: AuthorityGrantRegistry,
    grant_ids: Sequence[str],
    clock: TrustedClock,
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
    irreversible_approval: IrreversibleApprovalBinding | None = None,
) -> AuthorizationDecision:
    """Evaluate current authority using the injected trusted clock.

    Consumers must call :func:`consume_authorization` before using a stored decision as authority.
    """

    action = _snapshot_action(action)
    action_digest = action_record_digest(action)
    _require_nonblank_string(principal_id, "principal_id")
    if type(role) is not RuntimeRole:
        raise ValueError("role must be an exact RuntimeRole")
    if type(capability) is not RuntimeCapability:
        raise ValueError("capability must be an exact RuntimeCapability")
    _require_optional_identifier(action_author_id, "action_author_id")
    _require_optional_identifier(action_executor_id, "action_executor_id")
    observed = _read_trusted_time(clock)
    observed_at = _format_datetime(observed)
    approval = _snapshot_optional_approval(irreversible_approval)
    snapshots = _resolve_registry_grants(grant_registry, grant_ids)

    candidates = _matching_grants(
        action,
        action_digest,
        principal_id,
        role,
        capability,
        snapshots,
    )
    if not candidates:
        return _deny(
            action,
            action_digest,
            principal_id,
            role,
            capability,
            AuthorizationReason.NO_MATCHING_GRANT,
            observed_at,
        )
    current_candidates = _current_grants(candidates, observed)
    if not current_candidates:
        return _deny(
            action,
            action_digest,
            principal_id,
            role,
            capability,
            AuthorizationReason.GRANT_EXPIRED,
            observed_at,
            _effective_expiry(candidates),
        )
    if len(current_candidates) != 1:
        return _deny(
            action,
            action_digest,
            principal_id,
            role,
            capability,
            AuthorizationReason.AMBIGUOUS_MATCHING_GRANTS,
            observed_at,
        )
    grant = current_candidates[0]

    independence_failure = _verify_independence(
        principal_id,
        capability,
        action_author_id,
        action_executor_id,
    )
    if independence_failure is not None:
        return _deny(
            action,
            action_digest,
            principal_id,
            role,
            capability,
            independence_failure,
            observed_at,
        )

    authorization_refs: tuple[EvidenceReference, ...] = ()
    approval_for_decision: IrreversibleApprovalBinding | None = None
    effective_expiry = grant.expires_at
    if capability is RuntimeCapability.AUTHORIZE_IRREVERSIBLE:
        authorization_refs = grant.evidence_refs
    if capability is RuntimeCapability.EXECUTE and not action.reversible:
        human_grants = _matching_irreversible_grants(
            action,
            action_digest,
            approval,
            snapshots,
        )
        if not human_grants:
            return _deny(
                action,
                action_digest,
                principal_id,
                role,
                capability,
                AuthorizationReason.MISSING_IRREVERSIBLE_APPROVAL,
                observed_at,
            )
        current_human_grants = _current_grants(human_grants, observed)
        if not current_human_grants:
            return _deny(
                action,
                action_digest,
                principal_id,
                role,
                capability,
                AuthorizationReason.IRREVERSIBLE_APPROVAL_EXPIRED,
                observed_at,
                _minimum_expiry(effective_expiry, _effective_expiry(human_grants)),
            )
        human_grant = current_human_grants[0]
        authorization_refs = cast(IrreversibleApprovalBinding, approval).evidence_refs
        approval_for_decision = approval
        effective_expiry = _minimum_expiry(effective_expiry, human_grant.expires_at)

    return AuthorizationDecision(
        action_id=action.action_id,
        action_digest=action_digest,
        authorization_scope=action.authorization_scope,
        principal_id=principal_id,
        role=role,
        capability=capability,
        authorized=True,
        grant_id=grant.grant_id,
        authorization_refs=authorization_refs,
        irreversible_approval=approval_for_decision,
        reason=AuthorizationReason.AUTHORIZED,
        observed_at=observed_at,
        effective_expires_at=_normalize_optional_timestamp(effective_expiry),
    )


def consume_authorization(
    decision: AuthorizationDecision,
    action: ActionRecord,
    *,
    grant_registry: AuthorityGrantRegistry,
    clock: TrustedClock,
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
) -> AuthorizationDecision:
    """Return a fresh current decision or reject a stored decision that no longer authorizes."""

    snapshot = _snapshot_decision(decision)
    if not snapshot.authorized:
        raise PermissionError("decision has no authority to consume")
    grant_ids = [cast(str, snapshot.grant_id)]
    if snapshot.irreversible_approval is not None:
        grant_ids.append(snapshot.irreversible_approval.authorization_grant_id)
    try:
        current = authorize_action(
            action,
            principal_id=snapshot.principal_id,
            role=snapshot.role,
            capability=snapshot.capability,
            grant_registry=grant_registry,
            grant_ids=tuple(grant_ids),
            clock=clock,
            action_author_id=action_author_id,
            action_executor_id=action_executor_id,
            irreversible_approval=snapshot.irreversible_approval,
        )
    except KeyError as exc:
        raise PermissionError(
            "decision does not match current authoritative authorization"
        ) from exc
    if not current.authorized or not _same_authority(snapshot, current):
        raise PermissionError("decision does not match current authoritative authorization")
    if _parse_rfc3339(snapshot.observed_at) > _parse_rfc3339(current.observed_at):
        raise PermissionError("decision does not match current authoritative authorization")
    return current


def action_record_digest(action: ActionRecord) -> str:
    """Digest every canonical serialized field of one validated action record."""

    snapshot = _snapshot_action(action)
    payload = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    if type(grant.capabilities) is not tuple or type(grant.evidence_refs) is not tuple:
        raise ValueError("AuthorityGrant tuple fields must remain exact tuples")
    try:
        return AuthorityGrant(
            grant.grant_id,
            grant.principal_id,
            grant.role,
            grant.capabilities,
            grant.action_id,
            grant.action_digest,
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


def _snapshot_grant_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("grant_ids must be an ordered array of exact strings")
    grant_ids: list[str] = []
    for grant_id in value:
        grant_ids.append(_require_nonblank_string(grant_id, "grant_ids item"))
    if len(set(grant_ids)) != len(grant_ids):
        raise ValueError("grant_ids must be unique")
    return tuple(grant_ids)


def _registry_digest(entries: tuple[AuthorityGrant, ...]) -> str:
    payload = json.dumps(
        [grant.to_dict() for grant in entries],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_registry_entries(registry: object) -> tuple[AuthorityGrant, ...]:
    if type(registry) is not AuthorityGrantRegistry:
        raise ValueError("grant_registry must be an exact AuthorityGrantRegistry")
    entries = registry._entries
    seal = registry._seal
    if type(entries) is not tuple or type(seal) is not str:
        raise ValueError("authority registry integrity check failed")
    snapshots = _snapshot_grants(entries)
    if len({grant.grant_id for grant in snapshots}) != len(snapshots):
        raise ValueError("authority registry integrity check failed")
    if not hmac.compare_digest(seal, _registry_digest(snapshots)):
        raise ValueError("authority registry integrity check failed")
    return snapshots


def _resolve_registry_grants(
    registry: object,
    grant_ids: object,
) -> tuple[AuthorityGrant, ...]:
    if type(registry) is not AuthorityGrantRegistry:
        raise ValueError("grant_registry must be an exact AuthorityGrantRegistry")
    return cast(AuthorityGrantRegistry, registry).resolve(cast(Sequence[str], grant_ids))


def _snapshot_decision(decision: object) -> AuthorizationDecision:
    if type(decision) is not AuthorizationDecision:
        raise ValueError("decision must be an exact AuthorizationDecision")
    if type(decision.authorization_refs) is not tuple:
        raise ValueError("AuthorizationDecision authorization_refs must remain an exact tuple")
    return AuthorizationDecision(
        decision.action_id,
        decision.action_digest,
        decision.authorization_scope,
        decision.principal_id,
        decision.role,
        decision.capability,
        decision.authorized,
        decision.grant_id,
        decision.authorization_refs,
        decision.irreversible_approval,
        decision.reason,
        decision.observed_at,
        decision.effective_expires_at,
    )


def _matching_grants(
    action: ActionRecord,
    action_digest: str,
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
        and grant.action_digest == action_digest
        and grant.authorization_scope == action.authorization_scope
        and capability in grant.capabilities
    )


def _matching_irreversible_grants(
    action: ActionRecord,
    action_digest: str,
    approval: IrreversibleApprovalBinding | None,
    grants: tuple[AuthorityGrant, ...],
) -> tuple[AuthorityGrant, ...]:
    if approval is None:
        return ()
    if (
        approval.action_id != action.action_id
        or approval.action_digest != action_digest
        or approval.authorization_scope != action.authorization_scope
    ):
        return ()
    return tuple(
        grant
        for grant in grants
        if grant.role is RuntimeRole.HUMAN
        and grant.principal_id == approval.human_principal_id
        and grant.grant_id == approval.authorization_grant_id
        and grant.action_id == action.action_id
        and grant.action_digest == action_digest
        and grant.authorization_scope == action.authorization_scope
        and RuntimeCapability.AUTHORIZE_IRREVERSIBLE in grant.capabilities
        and grant.evidence_refs == approval.evidence_refs
    )


def _current_grants(
    grants: tuple[AuthorityGrant, ...],
    observed: datetime,
) -> tuple[AuthorityGrant, ...]:
    return tuple(
        grant
        for grant in grants
        if grant.expires_at is None or observed < _parse_rfc3339(grant.expires_at)
    )


def _verify_independence(
    principal_id: str,
    capability: RuntimeCapability,
    action_author_id: str | None,
    action_executor_id: str | None,
) -> AuthorizationReason | None:
    if capability is not RuntimeCapability.VERIFY:
        return None
    if action_author_id is None or action_executor_id is None:
        return AuthorizationReason.MISSING_VERIFIER_INDEPENDENCE_CONTEXT
    if principal_id == action_author_id:
        return AuthorizationReason.VERIFIER_AUTHORED_ACTION
    if principal_id == action_executor_id:
        return AuthorizationReason.VERIFIER_EXECUTED_ACTION
    return None


def _deny(
    action: ActionRecord,
    action_digest: str,
    principal_id: str,
    role: RuntimeRole,
    capability: RuntimeCapability,
    reason: AuthorizationReason,
    observed_at: str,
    effective_expires_at: str | None = None,
) -> AuthorizationDecision:
    return AuthorizationDecision(
        action_id=action.action_id,
        action_digest=action_digest,
        authorization_scope=action.authorization_scope,
        principal_id=principal_id,
        role=role,
        capability=capability,
        authorized=False,
        grant_id=None,
        authorization_refs=(),
        irreversible_approval=None,
        reason=reason,
        observed_at=observed_at,
        effective_expires_at=_normalize_optional_timestamp(effective_expires_at),
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


def _read_trusted_time(clock: object) -> datetime:
    now_method = getattr(clock, "now", None)
    if not callable(now_method):
        raise ValueError("clock must provide a callable now method")
    observed = now_method()
    if type(observed) is not datetime:
        raise ValueError("clock must return an exact datetime")
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("clock must return an aware datetime")
    return observed


def _format_datetime(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc)
    timespec = "microseconds" if utc_value.microsecond else "seconds"
    return utc_value.isoformat(timespec=timespec).replace("+00:00", "Z")


def _normalize_optional_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    return _format_datetime(_parse_rfc3339(value))


def _effective_expiry(grants: tuple[AuthorityGrant, ...]) -> str | None:
    expiry_values = [grant.expires_at for grant in grants if grant.expires_at is not None]
    if not expiry_values:
        return None
    return min(expiry_values, key=_parse_rfc3339)


def _minimum_expiry(first: str | None, second: str | None) -> str | None:
    if first is None:
        return second
    if second is None:
        return first
    return min((first, second), key=_parse_rfc3339)


def _snapshot_approval(value: object) -> IrreversibleApprovalBinding:
    if type(value) is not IrreversibleApprovalBinding:
        raise ValueError("irreversible approval must be an exact IrreversibleApprovalBinding")
    if type(value.evidence_refs) is not tuple:
        raise ValueError("IrreversibleApprovalBinding evidence_refs must remain an exact tuple")
    try:
        return IrreversibleApprovalBinding(
            value.human_principal_id,
            value.action_id,
            value.action_digest,
            value.authorization_scope,
            value.authorization_grant_id,
            value.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"irreversible approval contains invalid fields: {exc}") from exc


def _snapshot_optional_approval(
    value: object,
) -> IrreversibleApprovalBinding | None:
    if value is None:
        return None
    return _snapshot_approval(value)


def _restore_optional_approval(value: object) -> IrreversibleApprovalBinding | None:
    if value is None:
        return None
    return IrreversibleApprovalBinding.from_dict(value)


def _restore_reason(value: object) -> AuthorizationReason:
    if type(value) is not str:
        raise ValueError("AuthorizationDecision reason must be a built-in string")
    try:
        return AuthorizationReason(value)
    except ValueError as exc:
        raise ValueError(f"unknown AuthorizationDecision reason {value!r}") from exc


def _same_authority(
    stored: AuthorizationDecision,
    current: AuthorizationDecision,
) -> bool:
    return (
        stored.action_id == current.action_id
        and stored.action_digest == current.action_digest
        and stored.authorization_scope == current.authorization_scope
        and stored.principal_id == current.principal_id
        and stored.role is current.role
        and stored.capability is current.capability
        and stored.grant_id == current.grant_id
        and stored.authorization_refs == current.authorization_refs
        and stored.irreversible_approval == current.irreversible_approval
        and stored.effective_expires_at == current.effective_expires_at
    )


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
    "AuthorityGrantRegistry",
    "AuthorizationDecision",
    "AuthorizationReason",
    "IrreversibleApprovalBinding",
    "RuntimeCapability",
    "RuntimeRole",
    "TrustedClock",
    "action_record_digest",
    "authorize_action",
    "consume_authorization",
]
