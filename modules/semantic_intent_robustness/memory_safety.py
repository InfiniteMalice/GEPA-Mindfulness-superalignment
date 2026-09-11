"""Memory trust-boundary helpers for semantic laundering protection."""

# Standard library
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, cast

# Local
from .representation import (
    RepresentationCandidate,
    candidate_id_for,
    source_digest_for,
    validated_candidate_snapshot,
)
from .taxonomy import CapabilityTransferRisk, StrEnum


class MemorySourceType(StrEnum):
    USER_INPUT = "user_input"
    EXTERNAL_CONTENT = "external_content"
    TOOL_OUTPUT = "tool_output"
    MODEL_SUMMARY = "model_summary"
    HUMAN_REVIEW = "human_review"
    SYSTEM_POLICY = "system_policy"
    UNKNOWN = "unknown"


class MemoryTrustLevel(StrEnum):
    UNTRUSTED = "untrusted"
    UNVERIFIED = "unverified"
    REVIEWED = "reviewed"
    TRUSTED = "trusted"
    SYSTEM_PROTECTED = "system_protected"


class MemoryDurability(StrEnum):
    EPHEMERAL = "ephemeral"
    SESSION = "session"
    DURABLE = "durable"


class MemoryWriteDecision(StrEnum):
    ALLOW_EPHEMERAL = "allow_ephemeral"
    ALLOW_SESSION = "allow_session"
    ALLOW_DURABLE = "allow_durable"
    QUARANTINE = "quarantine"
    REJECT = "reject"


class MemoryRetrievalDecision(StrEnum):
    USE_WITH_PROVENANCE = "use_with_provenance"
    TREAT_AS_UNTRUSTED_CONTEXT = "treat_as_untrusted_context"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class RepresentationMemoryProvenance:
    """Declared representation derivation bound to immutable source evidence."""

    candidate: RepresentationCandidate
    candidate_id: str
    source_identity: str
    source_document: str
    source_digest: str
    transform_provenance: tuple[str, ...]
    derived_content: str
    assessed_content: str
    assessed_as_derived: bool

    def __post_init__(self) -> None:
        candidate = validated_candidate_snapshot(self.candidate)
        object.__setattr__(self, "candidate", candidate)
        for field_name in (
            "candidate_id",
            "source_identity",
            "source_document",
            "source_digest",
            "derived_content",
            "assessed_content",
        ):
            _validate_exact_string(getattr(self, field_name), field_name=field_name)
        if type(self.transform_provenance) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.transform_provenance
        ):
            raise TypeError("transform_provenance must be an exact tuple of nonblank strings")
        if type(self.assessed_as_derived) is not bool:
            raise TypeError("assessed_as_derived must be an exact bool")
        if not self.assessed_as_derived:
            raise ValueError("assessed_as_derived must be true for representation provenance")
        span = candidate.source_span
        if span.source_id != self.source_identity:
            raise ValueError("candidate source identity does not match memory source identity")
        if (
            span.end > len(self.source_document)
            or self.source_document[span.start : span.end] != span.raw_text
        ):
            raise ValueError("candidate span does not match the representation source document")
        if candidate_id_for(candidate) != self.candidate_id:
            raise ValueError("candidate_id does not match the representation candidate")
        if source_digest_for(self.source_identity, self.source_document) != self.source_digest:
            raise ValueError("source_digest does not match the representation source document")
        if candidate.provenance != self.transform_provenance:
            raise ValueError("transform_provenance does not match candidate provenance")
        if candidate.candidate_text != self.derived_content:
            raise ValueError("derived_content does not match candidate text")
        assessed = (
            self.source_document[: span.start]
            + candidate.candidate_text
            + self.source_document[span.end :]
        )
        if assessed != self.assessed_content:
            raise ValueError("assessed_content does not match the bound representation")


@dataclass(frozen=True)
class MemoryWriteRequest:
    """Proposed memory write before content crosses a persistence boundary."""

    memory_id: str
    content_summary: str
    source_type: MemorySourceType
    source_identity: str
    trust_level: MemoryTrustLevel
    requested_durability: MemoryDurability
    capability_transfer_risk: CapabilityTransferRisk
    provenance_retained: bool
    contains_instruction: bool
    changes_goal_or_policy: bool
    changes_identity_or_authority: bool
    attempts_protected_override: bool
    conflicts_with_existing_memory: bool
    delayed_activation_hint: bool
    cross_session_relevance: bool
    tags: tuple[str, ...] = field(default_factory=tuple)
    representation_derived: bool = False
    representation_provenance: RepresentationMemoryProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", MemorySourceType(self.source_type))
        object.__setattr__(self, "trust_level", MemoryTrustLevel(self.trust_level))
        object.__setattr__(
            self,
            "requested_durability",
            MemoryDurability(self.requested_durability),
        )
        object.__setattr__(
            self,
            "capability_transfer_risk",
            CapabilityTransferRisk(self.capability_transfer_risk),
        )
        object.__setattr__(self, "tags", tuple(self.tags))
        if type(self.representation_derived) is not bool:
            raise TypeError("representation_derived must be an exact bool")
        if (
            self.representation_provenance is not None
            and type(self.representation_provenance) is not RepresentationMemoryProvenance
        ):
            raise TypeError(
                "representation_provenance must be an exact RepresentationMemoryProvenance or None"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = _serialize(asdict(self))
        if not self.representation_derived and self.representation_provenance is None:
            payload.pop("representation_derived")
            payload.pop("representation_provenance")
        return payload


@dataclass(frozen=True)
class MemoryWriteAssessment:
    memory_id: str
    decision: MemoryWriteDecision
    reasons: tuple[str, ...]
    requires_review: bool
    provenance_required: bool

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class RetrievedMemory:
    """Stored memory before it affects reasoning, planning, or tool choice."""

    memory_id: str
    content_summary: str
    source_type: MemorySourceType
    trust_level: MemoryTrustLevel
    provenance_retained: bool
    recalled_as_instruction: bool
    used_for_tool_selection: bool
    changes_goal_or_policy: bool
    changes_identity_or_authority: bool
    attempts_protected_override: bool
    conflicts_with_current_context: bool
    delayed_activation_hint: bool
    capability_transfer_risk: CapabilityTransferRisk
    source_identity: str = ""
    representation_derived: bool = False
    representation_provenance: RepresentationMemoryProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", MemorySourceType(self.source_type))
        object.__setattr__(self, "trust_level", MemoryTrustLevel(self.trust_level))
        object.__setattr__(
            self,
            "capability_transfer_risk",
            CapabilityTransferRisk(self.capability_transfer_risk),
        )
        if type(self.representation_derived) is not bool:
            raise TypeError("representation_derived must be an exact bool")
        if (
            self.representation_provenance is not None
            and type(self.representation_provenance) is not RepresentationMemoryProvenance
        ):
            raise TypeError(
                "representation_provenance must be an exact RepresentationMemoryProvenance or None"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = _serialize(asdict(self))
        if not self.source_identity:
            payload.pop("source_identity")
        if not self.representation_derived and self.representation_provenance is None:
            payload.pop("representation_derived")
            payload.pop("representation_provenance")
        return payload


@dataclass(frozen=True)
class MemoryRetrievalAssessment:
    memory_id: str
    decision: MemoryRetrievalDecision
    reasons: tuple[str, ...]
    preserve_provenance_label: bool
    requires_review: bool

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class MemoryLaunderingReport:
    write_count: int
    retrieval_count: int
    quarantined_write_ids: tuple[str, ...]
    rejected_write_ids: tuple[str, ...]
    quarantined_retrieval_ids: tuple[str, ...]
    rejected_retrieval_ids: tuple[str, ...]
    memory_mediated_laundering_detected: bool
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


TRUSTED_FOR_DURABLE = {
    MemoryTrustLevel.REVIEWED,
    MemoryTrustLevel.TRUSTED,
    MemoryTrustLevel.SYSTEM_PROTECTED,
}
UNTRUSTED_OR_UNVERIFIED = {MemoryTrustLevel.UNTRUSTED, MemoryTrustLevel.UNVERIFIED}
UNTRUSTED_DERIVED_SOURCES = {
    MemorySourceType.EXTERNAL_CONTENT,
    MemorySourceType.TOOL_OUTPUT,
    MemorySourceType.MODEL_SUMMARY,
    MemorySourceType.UNKNOWN,
}
HIGH_RISK = {CapabilityTransferRisk.HIGH, CapabilityTransferRisk.CRITICAL}


def assess_memory_write(request: MemoryWriteRequest) -> MemoryWriteAssessment:
    """Assess whether a proposed memory write may cross the trust boundary."""

    reasons: list[str] = []
    requires_review = False
    provenance_required = True

    reasons.extend(
        _representation_provenance_reasons(
            content_summary=request.content_summary,
            source_identity=request.source_identity,
            provenance_retained=request.provenance_retained,
            representation_derived=request.representation_derived,
            provenance=request.representation_provenance,
        )
    )

    if request.attempts_protected_override:
        return MemoryWriteAssessment(
            memory_id=request.memory_id,
            decision=cast(MemoryWriteDecision, MemoryWriteDecision.REJECT),
            reasons=("attempts_protected_override",),
            requires_review=True,
            provenance_required=provenance_required,
        )

    if request.changes_goal_or_policy:
        reasons.append("changes_goal_or_policy")
    if request.changes_identity_or_authority:
        reasons.append("changes_identity_or_authority")
    if request.conflicts_with_existing_memory:
        reasons.append("conflicts_with_existing_memory")
    if request.delayed_activation_hint:
        reasons.append("delayed_activation_hint")
    if not request.provenance_retained:
        reasons.append("provenance_missing")
    if (
        request.requested_durability == MemoryDurability.DURABLE
        and request.contains_instruction
        and request.trust_level in UNTRUSTED_OR_UNVERIFIED
    ):
        reasons.append("untrusted_instruction_durable_promotion")
    if (
        request.requested_durability == MemoryDurability.DURABLE
        and request.source_type in UNTRUSTED_DERIVED_SOURCES
        and request.trust_level in UNTRUSTED_OR_UNVERIFIED
    ):
        reasons.append("untrusted_derived_content_durable_promotion")
    if (
        request.requested_durability == MemoryDurability.DURABLE
        and request.capability_transfer_risk in HIGH_RISK
    ):
        reasons.append("high_risk_durable_write")
        requires_review = True

    if reasons:
        return MemoryWriteAssessment(
            memory_id=request.memory_id,
            decision=cast(MemoryWriteDecision, MemoryWriteDecision.QUARANTINE),
            reasons=tuple(reasons),
            requires_review=True or requires_review,
            provenance_required=provenance_required,
        )

    if request.requested_durability == MemoryDurability.DURABLE:
        if request.trust_level not in TRUSTED_FOR_DURABLE or not request.provenance_retained:
            return MemoryWriteAssessment(
                memory_id=request.memory_id,
                decision=cast(MemoryWriteDecision, MemoryWriteDecision.QUARANTINE),
                reasons=("durable_write_requires_reviewed_trust_and_provenance",),
                requires_review=True,
                provenance_required=provenance_required,
            )
        return MemoryWriteAssessment(
            request.memory_id,
            cast(MemoryWriteDecision, MemoryWriteDecision.ALLOW_DURABLE),
            ("durable_write_trusted_with_provenance",),
            False,
            provenance_required,
        )

    decision = cast(
        MemoryWriteDecision,
        (
            MemoryWriteDecision.ALLOW_SESSION
            if request.requested_durability == MemoryDurability.SESSION
            else MemoryWriteDecision.ALLOW_EPHEMERAL
        ),
    )
    return MemoryWriteAssessment(
        request.memory_id,
        decision,
        ("bounded_non_durable_storage",),
        False,
        provenance_required,
    )


def assess_retrieved_memory(memory: RetrievedMemory) -> MemoryRetrievalAssessment:
    """Assess whether a retrieved memory can influence downstream behavior."""

    reasons: list[str] = []
    reasons.extend(
        _representation_provenance_reasons(
            content_summary=memory.content_summary,
            source_identity=memory.source_identity,
            provenance_retained=memory.provenance_retained,
            representation_derived=memory.representation_derived,
            provenance=memory.representation_provenance,
        )
    )
    if memory.attempts_protected_override:
        return MemoryRetrievalAssessment(
            memory.memory_id,
            cast(MemoryRetrievalDecision, MemoryRetrievalDecision.REJECT),
            ("attempts_protected_override",),
            True,
            True,
        )

    if not memory.provenance_retained:
        reasons.append("provenance_missing")
    if memory.changes_goal_or_policy:
        reasons.append("changes_goal_or_policy")
    if memory.changes_identity_or_authority:
        reasons.append("changes_identity_or_authority")
    if memory.conflicts_with_current_context:
        reasons.append("conflicts_with_current_context")
    if memory.delayed_activation_hint:
        reasons.append("delayed_activation_hint")
    if memory.recalled_as_instruction and memory.trust_level in UNTRUSTED_OR_UNVERIFIED:
        reasons.append("untrusted_memory_recalled_as_instruction")
    if memory.used_for_tool_selection and memory.trust_level in UNTRUSTED_OR_UNVERIFIED:
        reasons.append("untrusted_memory_used_for_tool_selection")
    if (
        memory.capability_transfer_risk in HIGH_RISK
        and memory.trust_level not in TRUSTED_FOR_DURABLE
    ):
        reasons.append("high_risk_memory_without_sufficient_trust")

    if reasons:
        return MemoryRetrievalAssessment(
            memory.memory_id,
            cast(MemoryRetrievalDecision, MemoryRetrievalDecision.QUARANTINE),
            tuple(reasons),
            True,
            True,
        )

    if memory.trust_level in UNTRUSTED_OR_UNVERIFIED:
        return MemoryRetrievalAssessment(
            memory.memory_id,
            cast(
                MemoryRetrievalDecision,
                MemoryRetrievalDecision.TREAT_AS_UNTRUSTED_CONTEXT,
            ),
            ("bounded_context_not_authority",),
            True,
            False,
        )

    return MemoryRetrievalAssessment(
        memory.memory_id,
        cast(MemoryRetrievalDecision, MemoryRetrievalDecision.USE_WITH_PROVENANCE),
        ("trusted_memory_with_visible_provenance",),
        True,
        False,
    )


def aggregate_memory_mediated_laundering(
    writes: list[MemoryWriteRequest],
    retrievals: list[RetrievedMemory],
) -> MemoryLaunderingReport:
    """Aggregate memory-boundary assessments for logging and review."""

    write_assessments = [assess_memory_write(item) for item in writes]
    retrieval_assessments = [assess_retrieved_memory(item) for item in retrievals]
    quarantined_writes = tuple(
        item.memory_id
        for item in write_assessments
        if item.decision == MemoryWriteDecision.QUARANTINE
    )
    rejected_writes = tuple(
        item.memory_id for item in write_assessments if item.decision == MemoryWriteDecision.REJECT
    )
    quarantined_retrievals = tuple(
        item.memory_id
        for item in retrieval_assessments
        if item.decision == MemoryRetrievalDecision.QUARANTINE
    )
    rejected_retrievals = tuple(
        item.memory_id
        for item in retrieval_assessments
        if item.decision == MemoryRetrievalDecision.REJECT
    )
    review_required = any(item.requires_review for item in write_assessments) or any(
        item.requires_review for item in retrieval_assessments
    )
    detected = bool(
        quarantined_writes or rejected_writes or quarantined_retrievals or rejected_retrievals
    )
    return MemoryLaunderingReport(
        write_count=len(writes),
        retrieval_count=len(retrievals),
        quarantined_write_ids=quarantined_writes,
        rejected_write_ids=rejected_writes,
        quarantined_retrieval_ids=quarantined_retrievals,
        rejected_retrieval_ids=rejected_retrievals,
        memory_mediated_laundering_detected=detected,
        review_required=review_required,
    )


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _representation_provenance_reasons(
    *,
    content_summary: object,
    source_identity: object,
    provenance_retained: object,
    representation_derived: bool,
    provenance: RepresentationMemoryProvenance | None,
) -> tuple[str, ...]:
    if type(representation_derived) is not bool:
        return ("representation_declaration_invalid",)
    if representation_derived and provenance is None:
        return ("representation_provenance_missing",)
    if not representation_derived and provenance is not None:
        return ("representation_provenance_misdeclared",)
    if provenance is None:
        return ()
    try:
        snapshot = RepresentationMemoryProvenance(
            candidate=provenance.candidate,
            candidate_id=provenance.candidate_id,
            source_identity=provenance.source_identity,
            source_document=provenance.source_document,
            source_digest=provenance.source_digest,
            transform_provenance=provenance.transform_provenance,
            derived_content=provenance.derived_content,
            assessed_content=provenance.assessed_content,
            assessed_as_derived=provenance.assessed_as_derived,
        )
    except (TypeError, ValueError):
        return ("representation_provenance_invalid",)
    reasons: list[str] = []
    if source_identity != snapshot.source_identity:
        reasons.append("representation_source_identity_mismatch")
    if content_summary != snapshot.assessed_content:
        reasons.append("representation_content_mismatch")
    if provenance_retained is not True:
        reasons.append("representation_provenance_not_retained")
    return tuple(reasons)


def _validate_exact_string(value: object, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


__all__ = [
    "MemoryDurability",
    "MemoryLaunderingReport",
    "MemoryRetrievalAssessment",
    "MemoryRetrievalDecision",
    "MemorySourceType",
    "MemoryTrustLevel",
    "MemoryWriteAssessment",
    "MemoryWriteDecision",
    "MemoryWriteRequest",
    "RepresentationMemoryProvenance",
    "RetrievedMemory",
    "aggregate_memory_mediated_laundering",
    "assess_memory_write",
    "assess_retrieved_memory",
]
