"""Check proposals and public verdict links; neither is a verifier credential or grant."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.evidence import EvidenceReference
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    _unit,
    choice,
    public_refs,
    restore_refs,
)
from .epistemic_state import _nonnegative


@dataclass(frozen=True, slots=True)
class CheckRequest(DiagnosticRecord):
    """Factors retained separately for future calibration; priority is heuristic."""

    check_id: str
    claim_id: str
    purpose: str
    procedure: str
    expected_information_gain: float
    decision_sensitivity: float
    decision_importance: float
    verification_cost: float
    evidence_refs: tuple[EvidenceReference, ...]
    action_id: str

    schema_version = "verification-check-request-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        for name in ("check_id", "claim_id", "procedure", "action_id"):
            _text(getattr(self, name), name)
        choice(self.purpose, "purpose", ("resolver", "falsifier", "coverage", "discrimination"))
        for name in ("expected_information_gain", "decision_sensitivity", "decision_importance"):
            _unit(getattr(self, name), name)
        if _nonnegative(self.verification_cost, "verification_cost") == 0:
            raise ValueError("verification_cost must be positive")
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))

    @property
    def priority(self) -> float:
        """Return information gain times sensitivity times importance per cost."""
        return (
            self.expected_information_gain
            * self.decision_sensitivity
            * self.decision_importance
            / self.verification_cost
        )


@dataclass(frozen=True, slots=True)
class CheckResult(DiagnosticRecord):
    """A claimed verdict awaiting host evidence and independence validation."""

    check_id: str
    claim_id: str
    action_id: str
    verdict: str
    evidence_refs: tuple[EvidenceReference, ...]
    verifier_id: str
    revision_claim_id: str | None

    schema_version = "verification-check-result-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        for name in ("check_id", "claim_id", "action_id", "verifier_id"):
            _text(getattr(self, name), name)
        choice(self.verdict, "verdict", ("supported", "contradicted", "unresolved"))
        object.__setattr__(
            self,
            "evidence_refs",
            public_refs(self.evidence_refs, required=self.verdict != "unresolved"),
        )
        if self.revision_claim_id is not None:
            _text(self.revision_claim_id, "revision_claim_id")
            if self.revision_claim_id == self.claim_id:
                raise ValueError("revision requires a distinct claim identity")
