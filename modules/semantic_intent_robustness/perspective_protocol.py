"""Public stakeholder capture records; declarations cannot establish authority."""

# Standard library
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, TypeVar

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.debate_records import _digest, _record
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    _text,
    _unit,
    choice,
    public_refs,
    restore_records,
    restore_refs,
    strings,
)
from gepa_mindfulness.verification.perspective_records import Perspective, Stakeholder
from gepa_mindfulness.verification.state import EvidenceClaim

T = TypeVar("T", bound=DiagnosticRecord)


def _items(value: object, cls: type[T]) -> tuple[T, ...]:
    if type(value) is not tuple:
        raise ValueError("records require a tuple")
    return tuple(_record(item, cls) for item in value)


def _unique(values: list[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


def _content_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def perspective_digest(record: DiagnosticRecord) -> str:
    """Bind a complete detached diagnostic record, without authenticating its claims."""
    return _content_digest(record.to_dict())


@dataclass(frozen=True)
class RoleAssignment(DiagnosticRecord):
    """Declared rights and obligations associated with one identified actor."""

    actor_id: str
    role_id: str
    rights: tuple[str, ...]
    duties: tuple[str, ...]
    authority: tuple[str, ...]
    schema_version = "perspective-role-v1"

    def __post_init__(self) -> None:
        _text(self.actor_id, "actor_id")
        _text(self.role_id, "role_id")
        for name in ("rights", "duties", "authority"):
            object.__setattr__(self, name, strings(getattr(self, name), name))


@dataclass(frozen=True)
class PerspectiveSource(DiagnosticRecord):
    """Public source assertions with retained usage restrictions, not verified facts."""

    variant_id: str
    semantic_core_id: str
    source_text: str
    facts: tuple[EvidenceClaim, ...]
    constraints: tuple[EvidenceClaim, ...]
    roles: tuple[RoleAssignment, ...]
    source_refs: tuple[EvidenceReference, ...]
    source_training_eligibility: TrainingEligibility
    schema_version = "perspective-source-v1"
    restorers = {
        "facts": lambda v: tuple(EvidenceClaim.from_dict(c) for c in v),
        "constraints": lambda v: tuple(EvidenceClaim.from_dict(c) for c in v),
        "roles": lambda v: restore_records(v, RoleAssignment),
        "source_refs": restore_refs,
        "source_training_eligibility": TrainingEligibility,
    }

    def __post_init__(self) -> None:
        for name in ("variant_id", "semantic_core_id", "source_text"):
            _text(getattr(self, name), name)
        for name in ("facts", "constraints"):
            values = getattr(self, name)
            if type(values) is not tuple or any(type(c) is not EvidenceClaim for c in values):
                raise ValueError("source claims require exact EvidenceClaim tuples")
            values = tuple(EvidenceClaim.from_dict(c.to_dict()) for c in values)
            for claim in values:
                public_refs(claim.evidence_refs, required=True)
                if claim.status != "unverified":
                    raise ValueError("source claims must remain unverified")
            object.__setattr__(self, name, values)
        _unique([c.claim_id for c in self.facts + self.constraints], "claim IDs")
        object.__setattr__(self, "roles", _items(self.roles, RoleAssignment))
        _unique([r.actor_id for r in self.roles], "actor IDs")
        object.__setattr__(self, "source_refs", public_refs(self.source_refs, required=True))
        if type(self.source_training_eligibility) is not TrainingEligibility or (
            self.source_training_eligibility is TrainingEligibility.TRAIN
        ):
            raise ValueError("perspective source must retain non-training admission")

    @property
    def facts_digest(self) -> str:
        """Bind the exact factual assertions used by perspective slots."""
        return _content_digest([c.to_dict() for c in self.facts])


@dataclass(frozen=True)
class PerspectiveSlot(DiagnosticRecord):
    """A planned candidate slot that remains present when no candidate arrives."""

    slot_id: str
    perspective: Perspective
    schema_version = "perspective-slot-v1"
    restorers = {"perspective": Perspective.from_dict}

    def __post_init__(self) -> None:
        _text(self.slot_id, "slot_id")
        object.__setattr__(self, "perspective", _record(self.perspective, Perspective))
        if self.perspective.variant_id != self.slot_id:
            raise ValueError("perspective variant must match slot ID")


@dataclass(frozen=True)
class PerspectivePlan(DiagnosticRecord):
    """Bound public sources and declared stakeholder/candidate rosters."""

    source: PerspectiveSource
    stakeholders: tuple[Stakeholder, ...]
    slots: tuple[PerspectiveSlot, ...]
    schema_version = "perspective-plan-v1"
    restorers = {
        "source": PerspectiveSource.from_dict,
        "stakeholders": lambda v: restore_records(v, Stakeholder),
        "slots": lambda v: restore_records(v, PerspectiveSlot),
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _record(self.source, PerspectiveSource))
        object.__setattr__(self, "stakeholders", _items(self.stakeholders, Stakeholder))
        object.__setattr__(self, "slots", _items(self.slots, PerspectiveSlot))
        if not 1 <= len(self.stakeholders) <= 16 or len(self.slots) > 64:
            raise ValueError("requires 1-16 stakeholders and at most 64 slots")
        ids = [s.stakeholder_id for s in self.stakeholders]
        _unique(ids, "stakeholder IDs")
        _unique([s.slot_id for s in self.slots], "slot IDs")
        if not {r.actor_id for r in self.source.roles} <= set(ids):
            raise ValueError("role references unknown stakeholder")
        for slot in self.slots:
            p = slot.perspective
            if (
                p.source_variant_id != self.source.variant_id
                or p.semantic_core_id != self.source.semantic_core_id
                or p.material_facts_id != self.source.facts_digest
                or p.material_facts_changed
                or not p.stakeholder_ids
                or not set(p.stakeholder_ids) <= set(ids)
            ):
                raise ValueError("perspective source/fact/stakeholder binding mismatch")


@dataclass(frozen=True)
class PerspectiveCandidate(DiagnosticRecord):
    """A simulated public response/preference with no field granting verification."""

    slot_id: str
    public_response: str
    preferences: tuple[str, ...]
    generator_id: str
    generator_version: str
    uncertainty: float | None
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version = "perspective-candidate-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        for name in ("slot_id", "public_response", "generator_id", "generator_version"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "preferences", strings(self.preferences, "preferences"))
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        if self.uncertainty is not None:
            _unit(self.uncertainty, "uncertainty")


@dataclass(frozen=True)
class PerspectiveCapture(DiagnosticRecord):
    """Attempt status and captured simulated candidates, bound to the original plan."""

    plan_digest: str
    candidates: tuple[PerspectiveCandidate, ...]
    status: str
    reason: str
    schema_version = "perspective-capture-v1"
    restorers = {"candidates": lambda v: restore_records(v, PerspectiveCandidate)}

    def __post_init__(self) -> None:
        _digest(self.plan_digest)
        _text(self.reason, "reason")
        choice(self.status, "status", ("observed", "censored", "callback_error"))
        object.__setattr__(self, "candidates", _items(self.candidates, PerspectiveCandidate))
        _unique([c.slot_id for c in self.candidates], "candidate slot IDs")
        if len(self.candidates) > 64 or (self.status != "observed" and self.candidates):
            raise ValueError("only observed captures may contain up to 64 candidates")
