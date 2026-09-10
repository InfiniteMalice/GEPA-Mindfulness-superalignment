"""Immutable provenance records for optimizer-eligible reward components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evidence import EvidenceReference


class VerificationRoute(str, Enum):
    """The independently verifiable route that supports one reward component."""

    OBSERVABLE_EVIDENCE = "observable_evidence"
    TRUSTED_EVALUATOR = "trusted_evaluator"


def _required_string(value: object, field_name: str) -> str:
    """Return a non-blank identifier or method declaration."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class TrustedEvaluatorContract:
    """The stable identity and declared contract for a trusted evaluator."""

    evaluator_id: str
    evaluator_version: str
    contract_id: str

    def __post_init__(self) -> None:
        """Reject incomplete evaluator identity and contract records."""
        for field_name in ("evaluator_id", "evaluator_version", "contract_id"):
            _required_string(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class RewardProvenance:
    """One exclusive observable-evidence or trusted-evaluator verification route."""

    component_name: str
    verification_method: str
    route: VerificationRoute
    evidence_refs: tuple[EvidenceReference, ...] = ()
    evaluator: TrustedEvaluatorContract | None = None

    def __post_init__(self) -> None:
        """Bind each provenance record to exactly one complete verification route."""
        _required_string(self.component_name, "component_name")
        _required_string(self.verification_method, "verification_method")
        if not isinstance(self.route, VerificationRoute):
            raise ValueError("route must be a VerificationRoute.")
        if not isinstance(self.evidence_refs, (list, tuple)) or not all(
            isinstance(reference, EvidenceReference) for reference in self.evidence_refs
        ):
            raise ValueError("evidence_refs must contain EvidenceReference values.")
        evidence_refs = tuple(self.evidence_refs)
        object.__setattr__(self, "evidence_refs", evidence_refs)

        if self.route is VerificationRoute.OBSERVABLE_EVIDENCE:
            if self.evaluator is not None:
                raise ValueError("observable evidence route cannot include an evaluator.")
            if not evidence_refs:
                raise ValueError("observable evidence route requires observable evidence.")
            if any(not reference.is_observable for reference in evidence_refs):
                raise ValueError(
                    "observable evidence route requires observable evidence references."
                )
            return

        if evidence_refs:
            raise ValueError(
                "trusted evaluator route cannot include observable evidence references."
            )
        if not isinstance(self.evaluator, TrustedEvaluatorContract):
            raise ValueError("trusted evaluator route requires a trusted evaluator contract.")


__all__ = [
    "RewardProvenance",
    "TrustedEvaluatorContract",
    "VerificationRoute",
]
