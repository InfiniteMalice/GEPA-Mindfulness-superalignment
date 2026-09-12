"""Compatibility exports for runtime-governance authority contracts."""

from .verification.runtime_governance import (
    ActionAuthorityPolicy,
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

__all__ = [
    "ActionAuthorityPolicy",
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
