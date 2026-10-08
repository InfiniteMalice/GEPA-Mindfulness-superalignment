"""Stakeholder interests and perspective provenance confer no authorization."""

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
    strings,
)

PERSPECTIVES = ("actor", "affected_party", "neutral_observer", "institutional", "role_reversal")


@dataclass(frozen=True, slots=True)
class Stakeholder(DiagnosticRecord):
    """Host-declared inclusion and preference uncertainty, separate from hard constraints."""

    stakeholder_id: str
    stakeholder_role: str
    impact: str
    inclusion: str
    evidence_refs: tuple[EvidenceReference, ...]
    interests: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    preferences: tuple[str, ...]
    preference_provenance: tuple[EvidenceReference, ...]
    preference_uncertainty: float | None

    schema_version = "stakeholder-v1"
    restorers = {"evidence_refs": restore_refs, "preference_provenance": restore_refs}

    def __post_init__(self) -> None:
        _text(self.stakeholder_id, "stakeholder_id")
        _text(self.stakeholder_role, "stakeholder_role")
        choice(self.impact, "impact", ("direct", "indirect"))
        choice(self.inclusion, "inclusion", ("explicit", "inferred"))
        for name in ("interests", "hard_constraints", "preferences"):
            object.__setattr__(self, name, strings(getattr(self, name), name))
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        object.__setattr__(
            self,
            "preference_provenance",
            public_refs(self.preference_provenance, required=bool(self.preferences)),
        )
        if self.inclusion == "inferred" and self.preference_uncertainty is None:
            raise ValueError("inferred preferences require explicit uncertainty")
        if self.preference_uncertainty is not None:
            _unit(self.preference_uncertainty, "preference_uncertainty")


@dataclass(frozen=True, slots=True)
class Perspective(DiagnosticRecord):
    """A transformation identity; semantic equivalence is a host assertion to audit."""

    variant_id: str
    semantic_core_id: str
    material_facts_id: str
    perspective: str
    stakeholder_ids: tuple[str, ...]
    material_facts_changed: bool
    source_variant_id: str

    schema_version = "perspective-v1"

    def __post_init__(self) -> None:
        for name in ("variant_id", "semantic_core_id", "material_facts_id", "source_variant_id"):
            _text(getattr(self, name), name)
        choice(self.perspective, "perspective", PERSPECTIVES)
        object.__setattr__(
            self, "stakeholder_ids", strings(self.stakeholder_ids, "stakeholder_ids")
        )
        if type(self.material_facts_changed) is not bool:
            raise ValueError("material_facts_changed must be boolean")
