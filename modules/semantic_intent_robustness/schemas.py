"""Typed schemas for semantic intent robustness records."""

# Standard library
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

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
    allowed_high_level_help: tuple[str, ...] = field(default_factory=tuple)
    disallowed_operational_help: tuple[str, ...] = field(default_factory=tuple)
    rationale_summary: str = ""
    consistency_target_group: str = ""
    source_type: SourceType = SourceType.SYNTHETIC
    generated_by: str = "semantic_intent_robustness"
    review_status: ReviewStatus = ReviewStatus.DRAFT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_high_level_help",
            tuple(self.allowed_high_level_help or []),
        )
        object.__setattr__(
            self,
            "disallowed_operational_help",
            tuple(self.disallowed_operational_help or []),
        )
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
            if isinstance(value, Enum):
                payload[key] = value.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticSafetyRecord":
        """Hydrate a record from serialized data while preserving field defaults."""

        data = dict(payload)
        default_map: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            if dataclass_field.default is not MISSING:
                default_map[dataclass_field.name] = dataclass_field.default
            elif dataclass_field.default_factory is not MISSING:
                default_map[dataclass_field.name] = dataclass_field.default_factory()
        for field_name, enum_cls in ENUM_FIELDS.items():
            if field_name in data:
                data[field_name] = enum_cls(data[field_name])
            elif field_name in default_map:
                data[field_name] = default_map[field_name]
        for field_name in TUPLE_FIELDS:
            if field_name in data:
                data[field_name] = tuple(data[field_name] or [])
            elif field_name in default_map:
                data[field_name] = tuple(default_map[field_name] or [])
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
    ground_truth_blocked: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))


ENUM_TYPES = (
    VariantType,
    IntentPrimary,
    IntentSecondary,
    RequestedCapability,
    CapabilityTransferRisk,
    ExecutionalityLevel,
    OperationalSpecificity,
    UncertaintyLevel,
    HarmDomain,
    HarmSeverity,
    Reversibility,
    ScaleOfHarm,
    TargetType,
    PolicyAction,
    SafeAlternativeMode,
    SourceType,
    ReviewStatus,
)


def _is_tuple_string_field(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not isinstance(args, tuple):
            return False
        return len(args) == 2 and args[0] is str and args[1] is Ellipsis
    if isinstance(annotation, UnionType):
        return any(_is_tuple_string_field(arg) for arg in annotation.__args__)
    return False


def _enum_from_annotation(annotation: object) -> type[Enum] | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    if isinstance(annotation, UnionType):
        for option in annotation.__args__:
            enum_option = _enum_from_annotation(option)
            if enum_option is not None:
                return enum_option
    return None


def _build_field_maps(
    schema_cls: type[SemanticSafetyRecord],
) -> tuple[dict[str, type[Enum]], set[str]]:
    hints = get_type_hints(schema_cls)
    enum_fields: dict[str, type[Enum]] = {}
    tuple_fields: set[str] = set()
    for field_name, annotation in hints.items():
        enum_cls = _enum_from_annotation(annotation)
        if enum_cls in ENUM_TYPES:
            enum_fields[field_name] = enum_cls
        if _is_tuple_string_field(annotation):
            tuple_fields.add(field_name)
    return enum_fields, tuple_fields


ENUM_FIELDS, TUPLE_FIELDS = _build_field_maps(SemanticSafetyRecord)


__all__ = [
    "MultiTurnConversation",
    "SemanticCluster",
    "SemanticSafetyRecord",
]
