"""Compatibility exports for runtime-governance authority contracts."""

from .verification.runtime_governance import (
    AuthorityGrant,
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
    "AuthorityGrant",
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
