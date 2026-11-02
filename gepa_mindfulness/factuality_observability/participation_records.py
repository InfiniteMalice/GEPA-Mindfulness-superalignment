"""Strict, non-authoritative records for optional specialist-participation research."""

# Standard library
from __future__ import annotations

import re
from dataclasses import dataclass, fields

# Third-party
# Local
from evaluation.improvement_records import ImprovementRecord, SystemConfig, _encode
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import _text, validate_input_envelope

from .decision_routing import PublicDecisionContext

MAX_CANDIDATES = 16
MAX_RECORD_BYTES = 65536
MAX_OUTPUT_BYTES = 4 * 1024 * 1024


def identity(value: object) -> None:
    """Restrict inert identifiers; this does not authenticate a principal."""
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@/\-]{0,127}", value) is None
    ):
        raise ValueError("bounded ASCII identity required")


def _refs(values: tuple[EvidenceReference, ...]) -> None:
    if not 1 <= len(values) <= 8 or len({r.reference_id for r in values}) != len(values):
        raise ValueError("bounded unique observable evidence required")
    for value in values:
        _text(value.reference_id, "reference_id")


def _system(value: SystemConfig) -> None:
    for item in fields(value):
        _text(getattr(value, item.name), item.name, 128)


@dataclass(frozen=True, kw_only=True)
class _ParticipationRecord(ImprovementRecord):
    training_eligibility: str = "DEVELOPMENT"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except (TypeError, OverflowError) as error:
            raise ValueError("invalid participation record") from error
        if (
            self.training_eligibility != "DEVELOPMENT"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("participation records cannot grant authority or optimizer credit")
        validate_input_envelope(_encode(self))

    def _schema(self, expected: str) -> None:
        if self.schema_version != expected:
            raise ValueError("unsupported participation schema")


@dataclass(frozen=True)
class ParticipationConfig(_ParticipationRecord):
    enabled: bool = False
    seed: int = 0
    max_output_bytes: int = MAX_OUTPUT_BYTES
    schema_version: str = "specialist-participation-config-v1"

    def _validate(self) -> None:
        self._schema("specialist-participation-config-v1")
        if not 0 <= self.seed < 2**32 or not 4096 <= self.max_output_bytes <= MAX_OUTPUT_BYTES:
            raise ValueError("unsupported seed or output byte budget")


@dataclass(frozen=True)
class SpecialistCandidate(_ParticipationRecord):
    specialist_id: str
    principal_id: str
    system: SystemConfig
    checkpoint_digest: str
    description: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "specialist-candidate-v1"

    def _validate(self) -> None:
        self._schema("specialist-candidate-v1")
        identity(self.specialist_id)
        identity(self.principal_id)
        _system(self.system)
        _text(self.description, "description", 512)
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class ParticipationPacket(_ParticipationRecord):
    request_id: str
    context: PublicDecisionContext
    scope: str
    candidates: tuple[SpecialistCandidate, ...]
    schema_version: str = "specialist-participation-packet-v1"

    def _validate(self) -> None:
        self._schema("specialist-participation-packet-v1")
        identity(self.request_id)
        _text(self.scope, "scope")
        if not 1 <= len(self.candidates) <= MAX_CANDIDATES:
            raise ValueError("1..16 specialist candidates required")
        for name in ("specialist_id", "principal_id"):
            values = [getattr(c, name).casefold() for c in self.candidates]
            if len(set(values)) != len(values):
                raise ValueError("duplicate or alias specialist/principal identity")


@dataclass(frozen=True)
class ParticipationReview(_ParticipationRecord):
    """An independently supplied assessment; declarations alone are not authentication."""

    packet_digest: str
    candidate_digest: str
    evaluator: TrustedEvaluatorContract
    reviewer_principal_id: str
    review_id: str
    status: str
    reason: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "specialist-participation-review-v1"

    def _validate(self) -> None:
        self._schema("specialist-participation-review-v1")
        identity(self.reviewer_principal_id)
        identity(self.review_id)
        for item in fields(self.evaluator):
            _text(getattr(self.evaluator, item.name), item.name, 128)
        if self.status not in ("eligible", "rejected", "uncertain"):
            raise ValueError("invalid participation review status")
        _text(self.reason, "reason", 512)
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class SpecialistBid(_ParticipationRecord):
    """Declared willingness to participate, never competence, truth or permission."""

    packet_digest: str
    candidate_digest: str
    specialist_id: str
    system: SystemConfig
    value: float
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "specialist-participation-bid-v1"

    def _validate(self) -> None:
        self._schema("specialist-participation-bid-v1")
        identity(self.specialist_id)
        _system(self.system)
        if not 0 <= self.value <= 1:
            raise ValueError("finite bid in [0,1] required")
        _refs(self.evidence_refs)
