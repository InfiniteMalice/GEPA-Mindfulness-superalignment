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
class PublicRationaleComparisonEvidence:
    """Observable records connecting a public rationale to a prediction, action, and outcome."""

    public_rationale: EvidenceReference
    committed_prediction: EvidenceReference
    selected_action: EvidenceReference
    observed_outcome: EvidenceReference

    def __post_init__(self) -> None:
        """Require four distinct, observable records for the public-rationale comparison."""
        references = self.references
        if not all(isinstance(reference, EvidenceReference) for reference in references):
            raise ValueError("public rationale comparison requires EvidenceReference values.")
        if any(not reference.is_observable for reference in references):
            raise ValueError("public rationale comparison requires observable evidence references.")
        if len({reference.reference_id for reference in references}) != len(references):
            raise ValueError("public rationale comparison requires distinct evidence references.")

    @property
    def references(self) -> tuple[EvidenceReference, ...]:
        """Return the complete, ordered comparison evidence boundary."""
        return (
            self.public_rationale,
            self.committed_prediction,
            self.selected_action,
            self.observed_outcome,
        )


@dataclass(frozen=True)
class RewardProvenance:
    """One exclusive observable-evidence or trusted-evaluator verification route."""

    component_name: str
    verification_method: str
    route: VerificationRoute
    evidence_refs: tuple[EvidenceReference, ...] = ()
    evaluator: TrustedEvaluatorContract | None = None
    public_rationale_comparison: PublicRationaleComparisonEvidence | None = None

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
        comparison = self.public_rationale_comparison
        if comparison is not None and not isinstance(comparison, PublicRationaleComparisonEvidence):
            raise ValueError(
                "public_rationale_comparison must be PublicRationaleComparisonEvidence or None."
            )

        if self.route is VerificationRoute.OBSERVABLE_EVIDENCE:
            if self.evaluator is not None:
                raise ValueError("observable evidence route cannot include an evaluator.")
            if not evidence_refs:
                raise ValueError("observable evidence route requires observable evidence.")
            if any(not reference.is_observable for reference in evidence_refs):
                raise ValueError(
                    "observable evidence route requires observable evidence references."
                )
            if self.component_name == "public_rationale_fidelity":
                if comparison is None:
                    raise ValueError(
                        "public rationale fidelity requires structured comparison evidence."
                    )
                if not set(comparison.references).issubset(evidence_refs):
                    raise ValueError(
                        "public rationale comparison evidence must be included in evidence_refs."
                    )
            elif comparison is not None:
                raise ValueError(
                    "public rationale comparison evidence is only valid for "
                    "public rationale fidelity."
                )
            return

        if evidence_refs:
            raise ValueError(
                "trusted evaluator route cannot include observable evidence references."
            )
        if comparison is not None:
            raise ValueError(
                "trusted evaluator route cannot include public rationale comparison evidence."
            )
        if not isinstance(self.evaluator, TrustedEvaluatorContract):
            raise ValueError("trusted evaluator route requires a trusted evaluator contract.")


__all__ = [
    "PublicRationaleComparisonEvidence",
    "RewardProvenance",
    "TrustedEvaluatorContract",
    "VerificationRoute",
]
