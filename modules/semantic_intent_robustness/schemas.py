"""Typed schemas for semantic intent robustness records."""

# Standard library
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Local
from .taxonomy import (
    CapabilityTransferRisk,
    ExecutionalityLevel,
    HarmDomain,
    HarmSeverity,
    IntentPrimary,
    IntentSecondary,
    OperationalSpecificity,
    PolicyAction,
    RequestedCapability,
    Reversibility,
    ReviewStatus,
    SafeAlternativeMode,
    ScaleOfHarm,
    SourceType,
    TargetType,
    UncertaintyLevel,
    VariantType,
)


@dataclass(frozen=True)
class SemanticSafetyRecord:
    """Structured semantic safety annotation for one prompt or turn."""

    prompt_id: str
    prompt_text: str
    semantic_cluster_id: str
    parent_example_id: str | None
    variant_type: VariantType
    language: str
    turn_index: int = 0
    conversation_id: str | None = None
    intent_primary: IntentPrimary = IntentPrimary.UNCERTAIN
    intent_secondary: IntentSecondary = IntentSecondary.UNKNOWN
    benign_plausibility: float = 0.0
    dual_use_probability: float = 0.0
    evasion_likelihood: float = 0.0
    user_goal_summary: str = ""
    requested_capability: RequestedCapability = RequestedCapability.ANALYSIS
    capability_transfer_risk: CapabilityTransferRisk = CapabilityTransferRisk.LOW
    executionality_level: ExecutionalityLevel = ExecutionalityLevel.ABSTRACT
    operational_specificity: OperationalSpecificity = OperationalSpecificity.LOW
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MEDIUM
    harm_domain: HarmDomain = HarmDomain.UNKNOWN
    harm_severity: HarmSeverity = HarmSeverity.MINIMAL
    reversibility: Reversibility = Reversibility.UNKNOWN
    scale_of_harm: ScaleOfHarm = ScaleOfHarm.INDIVIDUAL
    target_type: TargetType = TargetType.UNKNOWN
    deception_component: bool = False
    concealment_component: bool = False
    policy_action: PolicyAction = PolicyAction.ABSTAIN
    safe_alternative_mode: SafeAlternativeMode = SafeAlternativeMode.CLARIFY
    abstain_recommended: bool = True
    explanation_brief: str = ""
    allowed_high_level_help: list[str] = field(default_factory=list)
    disallowed_operational_help: list[str] = field(default_factory=list)
    rationale_summary: str = ""
    consistency_target_group: str = ""
    source_type: SourceType = SourceType.SYNTHETIC
    generated_by: str = "semantic_intent_robustness"
    review_status: ReviewStatus = ReviewStatus.DRAFT

    def __post_init__(self) -> None:
        for field_name in (
            "benign_plausibility",
            "dual_use_probability",
            "evasion_likelihood",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if hasattr(value, "value"):
                payload[key] = value.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticSafetyRecord":
        """Hydrate a record from serialized data."""

        data = dict(payload)
        data["variant_type"] = VariantType(data["variant_type"])
        data["intent_primary"] = IntentPrimary(data["intent_primary"])
        data["intent_secondary"] = IntentSecondary(data["intent_secondary"])
        data["requested_capability"] = RequestedCapability(data["requested_capability"])
        data["capability_transfer_risk"] = CapabilityTransferRisk(data["capability_transfer_risk"])
        data["executionality_level"] = ExecutionalityLevel(data["executionality_level"])
        data["operational_specificity"] = OperationalSpecificity(data["operational_specificity"])
        data["uncertainty_level"] = UncertaintyLevel(data["uncertainty_level"])
        data["harm_domain"] = HarmDomain(data["harm_domain"])
        data["harm_severity"] = HarmSeverity(data["harm_severity"])
        data["reversibility"] = Reversibility(data["reversibility"])
        data["scale_of_harm"] = ScaleOfHarm(data["scale_of_harm"])
        data["target_type"] = TargetType(data["target_type"])
        data["policy_action"] = PolicyAction(data["policy_action"])
        data["safe_alternative_mode"] = SafeAlternativeMode(data["safe_alternative_mode"])
        data["source_type"] = SourceType(data["source_type"])
        data["review_status"] = ReviewStatus(data["review_status"])
        return cls(**data)


@dataclass(frozen=True)
class SemanticCluster:
    """A cluster of semantically related records and negative controls."""

    cluster_id: str
    records: tuple[SemanticSafetyRecord, ...]
    negative_controls: tuple[SemanticSafetyRecord, ...] = field(default_factory=tuple)
    cluster_summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "negative_controls", tuple(self.negative_controls))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_summary": self.cluster_summary,
            "records": [record.to_dict() for record in self.records],
            "negative_controls": [record.to_dict() for record in self.negative_controls],
        }


@dataclass(frozen=True)
class MultiTurnConversation:
    """Conversation wrapper used by aggregation and evaluation utilities."""

    conversation_id: str
    turns: tuple[SemanticSafetyRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))


__all__ = [
    "MultiTurnConversation",
    "SemanticCluster",
    "SemanticSafetyRecord",
]
