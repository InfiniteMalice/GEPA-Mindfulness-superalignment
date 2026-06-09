"""Memory trust-boundary helpers for semantic laundering protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", MemorySourceType(self.source_type))
        object.__setattr__(self, "trust_level", MemoryTrustLevel(self.trust_level))
        object.__setattr__(
            self,
            "capability_transfer_risk",
            CapabilityTransferRisk(self.capability_transfer_risk),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


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

    if request.attempts_protected_override:
        return MemoryWriteAssessment(
            memory_id=request.memory_id,
            decision=MemoryWriteDecision.REJECT,
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
            decision=MemoryWriteDecision.QUARANTINE,
            reasons=tuple(reasons),
            requires_review=True or requires_review,
            provenance_required=provenance_required,
        )

    if request.requested_durability == MemoryDurability.DURABLE:
        if request.trust_level not in TRUSTED_FOR_DURABLE or not request.provenance_retained:
            return MemoryWriteAssessment(
                memory_id=request.memory_id,
                decision=MemoryWriteDecision.QUARANTINE,
                reasons=("durable_write_requires_reviewed_trust_and_provenance",),
                requires_review=True,
                provenance_required=provenance_required,
            )
        return MemoryWriteAssessment(
            request.memory_id,
            MemoryWriteDecision.ALLOW_DURABLE,
            ("durable_write_trusted_with_provenance",),
            False,
            provenance_required,
        )

    decision = (
        MemoryWriteDecision.ALLOW_SESSION
        if request.requested_durability == MemoryDurability.SESSION
        else MemoryWriteDecision.ALLOW_EPHEMERAL
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
    if memory.attempts_protected_override:
        return MemoryRetrievalAssessment(
            memory.memory_id,
            MemoryRetrievalDecision.REJECT,
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
            MemoryRetrievalDecision.QUARANTINE,
            tuple(reasons),
            True,
            True,
        )

    if memory.trust_level in UNTRUSTED_OR_UNVERIFIED:
        return MemoryRetrievalAssessment(
            memory.memory_id,
            MemoryRetrievalDecision.TREAT_AS_UNTRUSTED_CONTEXT,
            ("bounded_context_not_authority",),
            True,
            False,
        )

    return MemoryRetrievalAssessment(
        memory.memory_id,
        MemoryRetrievalDecision.USE_WITH_PROVENANCE,
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
    "RetrievedMemory",
    "aggregate_memory_mediated_laundering",
    "assess_memory_write",
    "assess_retrieved_memory",
]
