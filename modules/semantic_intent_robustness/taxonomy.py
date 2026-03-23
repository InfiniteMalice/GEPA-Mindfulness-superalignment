"""Taxonomy enums for semantic intent robustness analysis."""

# Standard library
from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum helper with stable serialization."""

    def __str__(self) -> str:
        return self.value


class VariantType(StrEnum):
    """Meaning relationship between a prompt and its seed example."""

    ORIGINAL = "original"
    PARAPHRASE = "paraphrase"
    EUPHEMISTIC_REWRITE = "euphemistic_rewrite"
    INDIRECT_PHRASING = "indirect_phrasing"
    ROLEPLAY_WRAPPER = "roleplay_wrapper"
    FICTIONAL_WRAPPER = "fictional_wrapper"
    ACADEMIC_WRAPPER = "academic_wrapper"
    MULTILINGUAL_TRANSLATION = "multilingual_translation"
    BACK_TRANSLATION_DRIFT = "back_translation_drift"
    CODE_SWITCH = "code_switch"
    TYPO_NOISE = "typo_noise"
    MULTI_TURN_DECOMPOSITION = "multi_turn_decomposition"
    BENIGN_WRAPPER_HARMFUL_CORE = "benign_wrapper_harmful_core"
    TOPIC_PRESERVING_INTENT_SHIFT = "topic_preserving_intent_shift"


class IntentPrimary(StrEnum):
    """Primary intent labels aligned to value decomposition."""

    BENIGN_INFORMATION = "benign_information"
    BENIGN_CREATIVE = "benign_creative"
    DUAL_USE_EXPLORATION = "dual_use_exploration"
    HARMFUL_ENABLEMENT = "harmful_enablement"
    DECEPTIVE_COORDINATION = "deceptive_coordination"
    UNCERTAIN = "uncertain"


class IntentSecondary(StrEnum):
    """Secondary nuance labels for downstream policy selection."""

    EDUCATIONAL = "educational"
    COMPLIANCE = "compliance"
    REPUTATIONAL = "reputational"
    OPERATIONALIZATION = "operationalization"
    EVASION = "evasion"
    RESEARCH = "research"
    FICTION = "fiction"
    SAFETY_EVALUATION = "safety_evaluation"
    UNKNOWN = "unknown"


class RequestedCapability(StrEnum):
    """Capability class requested by the user."""

    EXPLANATION = "explanation"
    CLASSIFICATION = "classification"
    PLANNING = "planning"
    PROCEDURAL_GUIDANCE = "procedural_guidance"
    TOOLING = "tooling"
    PERSUASION = "persuasion"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"


class CapabilityTransferRisk(StrEnum):
    """Risk that the response materially transfers harmful capability."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionalityLevel(StrEnum):
    """How directly the prompt seeks executable action."""

    ABSTRACT = "abstract"
    STRATEGIC = "strategic"
    PROCEDURAL = "procedural"
    ACTIONABLE = "actionable"


class OperationalSpecificity(StrEnum):
    """How concretely the request specifies steps, tools, or targets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UncertaintyLevel(StrEnum):
    """Confidence/ambiguity estimate for semantic interpretation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HarmDomain(StrEnum):
    """Broad harm domain for value decomposition."""

    NONE = "none"
    CYBER = "cyber"
    FRAUD = "fraud"
    PRIVACY = "privacy"
    SELF_HARM = "self_harm"
    PHYSICAL = "physical"
    REPUTATIONAL = "reputational"
    SOCIAL_ENGINEERING = "social_engineering"
    MISINFORMATION = "misinformation"
    UNKNOWN = "unknown"


class HarmSeverity(StrEnum):
    """Severity estimate for potential harm."""

    MINIMAL = "minimal"
    MODERATE = "moderate"
    SERIOUS = "serious"
    EXTREME = "extreme"


class Reversibility(StrEnum):
    """Whether likely harm is reversible."""

    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class ScaleOfHarm(StrEnum):
    """Expected scope of impact."""

    INDIVIDUAL = "individual"
    LOCALIZED = "localized"
    BROAD = "broad"
    SYSTEMIC = "systemic"


class TargetType(StrEnum):
    """Who or what is affected by the requested capability."""

    NONE = "none"
    SELF = "self"
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    PUBLIC = "public"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class PolicyAction(StrEnum):
    """Policy action to take after structured decomposition."""

    ALLOW = "allow"
    ALLOW_WITH_BOUNDARIES = "allow_with_boundaries"
    REDIRECT = "redirect"
    ABSTAIN = "abstain"
    REFUSE = "refuse"


class SafeAlternativeMode(StrEnum):
    """Preferred safe response shape."""

    NONE = "none"
    HIGH_LEVEL_SAFETY = "high_level_safety"
    BENIGN_SUBSTITUTE = "benign_substitute"
    CLARIFY = "clarify"
    SUPPORT_RESOURCES = "support_resources"
    POLICY_EXPLANATION = "policy_explanation"


class SourceType(StrEnum):
    """Record provenance."""

    HAND_AUTHORED = "hand_authored"
    SYNTHETIC = "synthetic"
    TRANSLATED = "translated"
    DERIVED = "derived"


class ReviewStatus(StrEnum):
    """Human review status for examples and labels."""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"


ALL_TAXONOMIES = {
    "variant_type": VariantType,
    "intent_primary": IntentPrimary,
    "intent_secondary": IntentSecondary,
    "requested_capability": RequestedCapability,
    "capability_transfer_risk": CapabilityTransferRisk,
    "executionality_level": ExecutionalityLevel,
    "operational_specificity": OperationalSpecificity,
    "uncertainty_level": UncertaintyLevel,
    "harm_domain": HarmDomain,
    "harm_severity": HarmSeverity,
    "reversibility": Reversibility,
    "scale_of_harm": ScaleOfHarm,
    "target_type": TargetType,
    "policy_action": PolicyAction,
    "safe_alternative_mode": SafeAlternativeMode,
    "source_type": SourceType,
    "review_status": ReviewStatus,
}


__all__ = [
    "ALL_TAXONOMIES",
    "CapabilityTransferRisk",
    "ExecutionalityLevel",
    "HarmDomain",
    "HarmSeverity",
    "IntentPrimary",
    "IntentSecondary",
    "OperationalSpecificity",
    "PolicyAction",
    "RequestedCapability",
    "ReviewStatus",
    "Reversibility",
    "SafeAlternativeMode",
    "ScaleOfHarm",
    "SourceType",
    "StrEnum",
    "TargetType",
    "UncertaintyLevel",
    "VariantType",
]
