"""Structured DSPy-style signatures for semantic intent robustness."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    import dspy  # type: ignore
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore


@dataclass(frozen=True)
class StructuredSignature:
    """Declarative signature description used without the real DSPy runtime."""

    name: str
    input_fields: tuple[str, ...]
    output_field: str
    output_schema: str

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "inputs": self.input_fields,
            "output": self.output_field,
            "schema": self.output_schema,
        }


DecomposeIntent = StructuredSignature(
    "DecomposeIntent",
    ("prompt_text", "conversation_context"),
    "intent_decomposition",
    "SemanticSafetyRecord",
)
AssessCapabilityRisk = StructuredSignature(
    "AssessCapabilityRisk",
    ("intent_decomposition",),
    "capability_assessment",
    "dict",
)
AssessHarmProfile = StructuredSignature(
    "AssessHarmProfile",
    ("intent_decomposition", "capability_assessment"),
    "harm_assessment",
    "dict",
)
ChoosePolicyAction = StructuredSignature(
    "ChoosePolicyAction",
    ("intent_decomposition", "capability_assessment", "harm_assessment"),
    "policy_decision",
    "dict",
)
GenerateSafeResponse = StructuredSignature(
    "GenerateSafeResponse",
    ("intent_decomposition", "policy_decision"),
    "safe_response",
    "dict",
)
CheckSemanticConsistency = StructuredSignature(
    "CheckSemanticConsistency",
    ("semantic_cluster",),
    "consistency_report",
    "dict",
)
AggregateMultiTurnRisk = StructuredSignature(
    "AggregateMultiTurnRisk",
    ("conversation",),
    "aggregated_risk",
    "dict",
)

ALL_SIGNATURES = (
    DecomposeIntent,
    AssessCapabilityRisk,
    AssessHarmProfile,
    ChoosePolicyAction,
    GenerateSafeResponse,
    CheckSemanticConsistency,
    AggregateMultiTurnRisk,
)


if dspy is not None:

    class DSPyDecomposeIntent(dspy.Signature):
        prompt_text = dspy.InputField(desc="User prompt or single turn")
        conversation_context = dspy.InputField(desc="Prior turns or system context")
        intent_decomposition = dspy.OutputField(desc="Structured semantic decomposition")

    class DSPyAssessCapabilityRisk(dspy.Signature):
        intent_decomposition = dspy.InputField(desc="Structured decomposition")
        capability_assessment = dspy.OutputField(desc="Capability transfer risk assessment")

    class DSPyAssessHarmProfile(dspy.Signature):
        intent_decomposition = dspy.InputField(desc="Structured decomposition")
        capability_assessment = dspy.InputField(desc="Capability risk assessment")
        harm_assessment = dspy.OutputField(desc="Harm profile assessment")

    class DSPyChoosePolicyAction(dspy.Signature):
        intent_decomposition = dspy.InputField(desc="Structured decomposition")
        capability_assessment = dspy.InputField(desc="Capability risk assessment")
        harm_assessment = dspy.InputField(desc="Harm profile assessment")
        policy_decision = dspy.OutputField(desc="Policy action and safe alternative mode")

    class DSPyGenerateSafeResponse(dspy.Signature):
        intent_decomposition = dspy.InputField(desc="Structured decomposition")
        policy_decision = dspy.InputField(desc="Policy action")
        safe_response = dspy.OutputField(desc="Bounded safe response")

    class DSPyCheckSemanticConsistency(dspy.Signature):
        semantic_cluster = dspy.InputField(desc="Cluster of meaning-preserving variants")
        consistency_report = dspy.OutputField(desc="Consistency metrics and disagreements")

    class DSPyAggregateMultiTurnRisk(dspy.Signature):
        conversation = dspy.InputField(desc="Conversation turns with latent intent signals")
        aggregated_risk = dspy.OutputField(desc="Conversation-level risk summary")

else:

    class _DSPyUnavailable:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("dspy is required for runtime DSPy signatures")

    DSPyDecomposeIntent = _DSPyUnavailable
    DSPyAssessCapabilityRisk = _DSPyUnavailable
    DSPyAssessHarmProfile = _DSPyUnavailable
    DSPyChoosePolicyAction = _DSPyUnavailable
    DSPyGenerateSafeResponse = _DSPyUnavailable
    DSPyCheckSemanticConsistency = _DSPyUnavailable
    DSPyAggregateMultiTurnRisk = _DSPyUnavailable


__all__ = [
    "ALL_SIGNATURES",
    "AggregateMultiTurnRisk",
    "AssessCapabilityRisk",
    "AssessHarmProfile",
    "CheckSemanticConsistency",
    "ChoosePolicyAction",
    "DSPyAggregateMultiTurnRisk",
    "DSPyAssessCapabilityRisk",
    "DSPyAssessHarmProfile",
    "DSPyCheckSemanticConsistency",
    "DSPyChoosePolicyAction",
    "DSPyDecomposeIntent",
    "DSPyGenerateSafeResponse",
    "DecomposeIntent",
    "GenerateSafeResponse",
    "StructuredSignature",
]
