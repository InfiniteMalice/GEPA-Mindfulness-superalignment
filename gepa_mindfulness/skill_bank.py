"""Immutable skill metadata and change proposals beside the durable skill lifecycle.

Host-owned cards describe WHEN and HOW separately. Binding a card to an artifact does
not certify its text or authorize deployment. Only the existing lifecycle can persist skills.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
from typing import Any

from .skill_lifecycle import SkillLifecycleHistory, skill_artifact_digest
from .verification.failure_layers import FailureLayer


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


class SkillChangeKind(str, Enum):
    ROUTING_DESCRIPTION = "routing_description"
    OPERATIONAL_GUIDANCE = "operational_guidance"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class SkillCard:
    """Versioned, host-curated metadata; foundational is a governance classification."""

    skill_id: str
    version: str
    artifact_id: str
    artifact_digest: str
    routing_description: str
    operational_guidance: str
    foundational: bool = False

    def __post_init__(self) -> None:
        for name in (
            "skill_id",
            "version",
            "artifact_id",
            "routing_description",
            "operational_guidance",
            "artifact_digest",
        ):
            _text(getattr(self, name), name)
        if (
            len(self.artifact_digest) != 71
            or not self.artifact_digest.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in self.artifact_digest[7:])
        ):
            raise ValueError("artifact_digest must be sha256:<64 lowercase hex>")
        if type(self.foundational) is not bool:
            raise ValueError("foundational must be boolean")

    @classmethod
    def from_history(
        cls,
        history: SkillLifecycleHistory,
        routing_description: str,
        operational_guidance: str,
        *,
        foundational: bool = False,
    ) -> SkillCard:
        """Bind metadata to the current canonical artifact, including draft states."""
        if type(history) is not SkillLifecycleHistory:
            raise ValueError("history must be a canonical SkillLifecycleHistory handle")
        artifact = history.current()
        return cls(
            artifact.skill_id,
            artifact.version,
            artifact.artifact_id,
            skill_artifact_digest(artifact),
            routing_description,
            operational_guidance,
            foundational,
        )

    def matches_history(self, history: SkillLifecycleHistory) -> bool:
        """Check identity freshness only; this is not a text validation or authority check."""
        self.__post_init__()
        if type(history) is not SkillLifecycleHistory:
            raise ValueError("history must be a canonical SkillLifecycleHistory handle")
        artifact = history.current()
        return (self.skill_id, self.version, self.artifact_id, self.artifact_digest) == (
            artifact.skill_id,
            artifact.version,
            artifact.artifact_id,
            skill_artifact_digest(artifact),
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self) | {"schema_version": "skill-card-v1", "confers_authority": False}


@dataclass(frozen=True, slots=True)
class SkillChangeProposal:
    """Review request only; no API applies it or issues a lifecycle receipt."""

    skill_id: str
    card_digest: str
    kind: SkillChangeKind
    review_ref: str
    replacement_text: str | None
    blocked: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "schema_version": "skill-change-proposal-v1",
            "confers_authority": False,
            "requires_held_out_validation": True,
        }


@dataclass(frozen=True, slots=True)
class SkillBank:
    """Immutable host-owned catalog; agent output must not replace this configuration."""

    cards: tuple[SkillCard, ...]

    def __post_init__(self) -> None:
        if type(self.cards) is not tuple or len(self.cards) > 10000:
            raise ValueError("cards must be a tuple of at most 10000 SkillCard records")
        if any(type(card) is not SkillCard for card in self.cards):
            raise ValueError("cards require SkillCard records")
        cards = tuple(replace(card) for card in self.cards)
        if len({card.skill_id for card in cards}) != len(cards):
            raise ValueError("skill IDs must be unique in a bank")
        object.__setattr__(self, "cards", cards)

    def propose(
        self,
        skill_id: str,
        kind: SkillChangeKind,
        review_ref: str,
        replacement_text: str | None = None,
    ) -> SkillChangeProposal:
        """Produce a non-executable proposal; performance cannot override foundational status."""
        self.__post_init__()
        _text(skill_id, "skill_id")
        _text(review_ref, "review_ref")
        if type(kind) is not SkillChangeKind:
            raise ValueError("kind must be SkillChangeKind")
        card = next((card for card in self.cards if card.skill_id == skill_id), None)
        if card is None:
            raise ValueError("skill_id must occur in the bank")
        if kind != SkillChangeKind.RETIRE:
            _text(replacement_text, "replacement_text")
        elif replacement_text is not None:
            _text(replacement_text, "replacement_text")
        reason = (
            "foundational_norm_requires_human_governance"
            if card.foundational
            else "review_and_lifecycle_validation_required"
        )
        digest = sha256(json.dumps(card.to_dict(), sort_keys=True).encode()).hexdigest()
        return SkillChangeProposal(
            skill_id, digest, kind, review_ref, replacement_text, card.foundational, reason
        )

    def propose_for_layer(
        self,
        skill_id: str,
        layer: FailureLayer,
        review_ref: str,
        replacement_text: str,
    ) -> SkillChangeProposal:
        """Only routing and knowledge hypotheses select a skill-text review surface."""
        if type(layer) is not FailureLayer:
            raise ValueError("layer must be FailureLayer")
        kinds = {
            FailureLayer.ROUTING: SkillChangeKind.ROUTING_DESCRIPTION,
            FailureLayer.KNOWLEDGE_SKILL: SkillChangeKind.OPERATIONAL_GUIDANCE,
        }
        if layer not in kinds:
            raise ValueError("this failure layer requires review outside skill text")
        return self.propose(skill_id, kinds[layer], review_ref, replacement_text)
