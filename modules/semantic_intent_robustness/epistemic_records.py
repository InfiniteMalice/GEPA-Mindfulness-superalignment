"""Public epistemic commitments and append-only update declarations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference

from ._continuity_validation import boolean, index_field, references, score, text_field
from .memory_safety import RetrievedMemory


class CommitmentStatus(str, Enum):
    """Epistemic activity, not certainty or authority."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    SCOPED_OUT = "scoped_out"
    UNRESOLVED = "unresolved"
    WITHDRAWN = "withdrawn"


def validate_public_evidence(refs: tuple[EvidenceReference, ...]) -> None:
    """Reject hidden-state/private-reasoning references as public commitments."""
    if type(refs) is not tuple or len(refs) > 128:
        raise ValueError("evidence_refs must be a bounded tuple")
    if any(type(ref) is not EvidenceReference or not ref.is_observable for ref in refs):
        raise ValueError("commitment evidence must be observable")
    references(tuple(ref.reference_id for ref in refs), "evidence_refs")


@dataclass(frozen=True, slots=True)
class EpistemicCommitment:
    """Bounded public claim linked to original memory and action-bound source events.

    first_active_at and last_active_at are conversation turn indices, bound against
    source event checkpoint_step values by the assessor. UNRESOLVED retains a hypothesis;
    neither confidence nor persistence promotes a representation candidate to truth.
    """

    commitment_id: str
    conversation_id: str
    evaluation_unit_id: str
    repeat_id: int | None
    claim_summary: str
    evidence_refs: tuple[EvidenceReference, ...]
    source_event_refs: tuple[str, ...]
    first_active_at: int
    last_active_at: int
    confidence: float
    decision_relevance: bool
    provenance: tuple[str, ...]
    memory: RetrievedMemory
    state_snapshot_id: str | None = None
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    superseded_by: str | None = None
    status_change_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate bounded public content, evidence references and lifecycle metadata."""
        for name in ("commitment_id", "conversation_id", "evaluation_unit_id"):
            text_field(getattr(self, name), name)
        text_field(self.claim_summary, "claim_summary", 1024)
        validate_public_evidence(self.evidence_refs)
        references(self.source_event_refs, "source_event_refs")
        references(self.provenance, "provenance")
        for name in ("first_active_at", "last_active_at"):
            index_field(getattr(self, name), name)
        if self.repeat_id is not None:
            index_field(self.repeat_id, "repeat_id")
        if self.last_active_at < self.first_active_at:
            raise ValueError("last_active_at precedes first_active_at")
        score(self.confidence, "confidence")
        boolean(self.decision_relevance, "decision_relevance")
        if type(self.status) is not CommitmentStatus:
            raise ValueError("status must be CommitmentStatus")
        if type(self.memory) is not RetrievedMemory:
            raise ValueError("memory must retain the original RetrievedMemory boundary record")
        if self.memory.memory_id != self.commitment_id:
            raise ValueError("memory_id must match commitment_id")
        if self.memory.content_summary != self.claim_summary:
            raise ValueError("claim_summary must match the provenance-bound memory content")
        for name in ("state_snapshot_id", "superseded_by", "status_change_reason"):
            if getattr(self, name) is not None:
                text_field(getattr(self, name), name)
        for name in (
            "provenance_retained",
            "recalled_as_instruction",
            "used_for_tool_selection",
            "changes_goal_or_policy",
            "changes_identity_or_authority",
            "attempts_protected_override",
            "conflicts_with_current_context",
            "delayed_activation_hint",
        ):
            boolean(getattr(self.memory, name), f"memory.{name}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize public records only, preserving the existing memory provenance schema."""
        result = asdict(self)
        result["evidence_refs"] = [ref.to_dict() for ref in self.evidence_refs]
        result["memory"] = self.memory.to_dict()
        result["status"] = self.status.value
        return result

    @property
    def digest(self) -> str:
        """Bind a later recall operation to exactly the assessed public record."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CommitmentUpdate:
    """Public reviewer declaration requiring later verified evidence before acceptance."""

    commitment_id: str
    status: CommitmentStatus
    status_change_reason: str
    evidence_refs: tuple[EvidenceReference, ...]
    source_event_refs: tuple[str, ...]
    provenance: tuple[str, ...]
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        """Validate bounded public content, evidence references and lifecycle metadata."""
        text_field(self.commitment_id, "commitment_id")
        text_field(self.status_change_reason, "status_change_reason", 1024)
        validate_public_evidence(self.evidence_refs)
        references(self.source_event_refs, "source_event_refs")
        references(self.provenance, "provenance")
        if type(self.status) is not CommitmentStatus or self.status in {
            CommitmentStatus.ACTIVE,
            CommitmentStatus.UNRESOLVED,
        }:
            raise ValueError("an update must declare a terminal commitment status")
        if self.superseded_by is not None:
            text_field(self.superseded_by, "superseded_by")
        if self.status is CommitmentStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError("superseded update requires superseded_by")
