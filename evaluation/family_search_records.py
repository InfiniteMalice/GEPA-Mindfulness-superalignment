"""Bounded family declarations over existing improvement protocols; never authority."""

# Standard library
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TypeVar, cast

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.training.causal_reward_records import _text, validate_input_envelope

from .improvement_records import ImprovementProtocol, ImprovementRecord, _encode

R = TypeVar("R", bound="_SearchRecord")


def identity_key(value: str) -> str:
    """Detect simple spelling aliases without claiming authenticated identity."""
    return unicodedata.normalize("NFC", value).casefold()


def unique_ids(values: tuple[str, ...]) -> None:
    """Reject blank, oversized, duplicate or normalized-alias declared IDs."""
    for value in values:
        _text(value, "identity", 256)
        if value != value.strip():
            raise ValueError("identity cannot have surrounding whitespace")
    if len({identity_key(v) for v in values}) != len(values):
        raise ValueError("duplicate or alias identity")


def _refs(values: tuple[EvidenceReference, ...]) -> None:
    """Bound observable source labels; labels do not verify evidence truth."""
    if not 1 <= len(values) <= 16:
        raise ValueError("bounded public references required")
    unique_ids(tuple(v.reference_id for v in values))


@dataclass(frozen=True, kw_only=True)
class _SearchRecord(ImprovementRecord):
    training_eligibility: str = "DEVELOPMENT"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        """Preserve exact existing primitive/record types and non-authority markers."""
        try:
            super().__post_init__()
        except (TypeError, OverflowError) as error:
            raise ValueError("invalid family-search record") from error
        if (
            self.training_eligibility != "DEVELOPMENT"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("family search cannot confer optimizer or runtime authority")
        validate_input_envelope(_encode(self))

    def _schema(self, expected: str) -> None:
        """Reject future or substituted wire versions."""
        if self.schema_version != expected:
            raise ValueError("unsupported family-search schema")

    @classmethod
    def from_dict(cls: type[R], value: object) -> R:
        """Check the JSON envelope before restoring typed nested arrays."""
        validate_input_envelope(value)
        return cast(R, super().from_dict(value))


@dataclass(frozen=True)
class FamilySearchConfig(_SearchRecord):
    """Operator opt-in only; no runtime is configured or executed by this flag."""

    enabled: bool = False
    schema_version: str = "family-search-config-v1"

    def _validate(self) -> None:
        """Keep configuration versioning exact."""
        self._schema("family-search-config-v1")


@dataclass(frozen=True)
class SearchFamily(_SearchRecord):
    """An experimental structural hypothesis, not a canonical task family or eligibility."""

    family_id: str
    surface: str
    hypothesis: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "search-family-v1"

    def _validate(self) -> None:
        """Bound inert hypotheses and reject normative-authority surface labels."""
        self._schema("search-family-v1")
        unique_ids((self.family_id,))
        if self.surface not in ("model", "data", "memory", "harness"):
            raise ValueError("unsupported experimental surface")
        _text(self.hypothesis, "hypothesis", 2048)
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class FamilyMembership(_SearchRecord):
    """One candidate's declared family and synchronous proposal round."""

    candidate_id: str
    family_id: str
    round_index: int
    round_id: str
    schema_version: str = "family-membership-v1"

    def _validate(self) -> None:
        """Reject malformed or unsupported round coordinates."""
        self._schema("family-membership-v1")
        for value in (self.candidate_id, self.family_id, self.round_id):
            unique_ids((value,))
        if not 0 <= self.round_index <= 15:
            raise ValueError("round index outside supported range")


@dataclass(frozen=True)
class FamilySearchPlan(_SearchRecord):
    """Fixed experimental commitments; semantic validation occurs during preparation."""

    protocol: ImprovementProtocol
    task_spec_digest: str
    protected_manifest_digest: str
    parent_checkpoint_digest: str
    inference_budget_digest: str
    training_budget_digest: str
    families: tuple[SearchFamily, ...]
    members: tuple[FamilyMembership, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    independent_rounds: int = 1
    sharing_interval: int = 1
    max_candidates_per_family: int = 16
    max_evaluations_per_family: int = 512
    schema_version: str = "family-search-plan-v1"

    def _validate(self) -> None:
        """Bound declarations before protocol/lineage and task-roster checks."""
        self._schema("family-search-plan-v1")
        if not 2 <= len(self.families) <= 8 or not 1 <= len(self.members) <= 64:
            raise ValueError("requires 2..8 families and 1..64 candidate memberships")
        unique_ids(tuple(f.family_id for f in self.families))
        unique_ids(tuple(m.candidate_id for m in self.members))
        for value in (self.independent_rounds, self.sharing_interval):
            if not 1 <= value <= 16:
                raise ValueError("unsupported separation/sharing schedule")
        if not 1 <= self.max_candidates_per_family <= 64:
            raise ValueError("unsupported candidate cap")
        if not 1 <= self.max_evaluations_per_family <= 2048:
            raise ValueError("unsupported evaluation cap")
        if (
            not 1 <= len(self.protocol.candidates) <= 64
            or not 1 <= len(self.protocol.slots) <= 2048
        ):
            raise ValueError("bounded candidate and slot roster required")
        if not 1 <= len(self.protocol.metrics) <= 16:
            raise ValueError("bounded component metric roster required")
        _refs(self.evidence_refs)
