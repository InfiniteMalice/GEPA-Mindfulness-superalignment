"""Structured modules for semantic intent robustness pipelines."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Local
from .consistency import aggregate_multi_turn_risk, semantic_cluster_agreement
from .schemas import MultiTurnConversation, SemanticCluster, SemanticSafetyRecord
from .taxonomy import PolicyAction, SafeAlternativeMode


@dataclass(frozen=True)
class SemanticPipelineResult:
    """End-to-end structured pipeline output."""

    decomposition: SemanticSafetyRecord
    capability_assessment: dict[str, Any]
    harm_assessment: dict[str, Any]
    policy_decision: dict[str, Any]
    safe_response: dict[str, Any]
    consistency_report: dict[str, Any] | None = None


class DecomposeIntentModule:
    """Return the provided structured decomposition or validate a seed record."""

    def __call__(
        self, record: SemanticSafetyRecord, conversation_context: str = ""
    ) -> SemanticSafetyRecord:
        del conversation_context
        return record


class AssessCapabilityRiskModule:
    """Summarize requested capability transfer and executionality."""

    def __call__(self, record: SemanticSafetyRecord) -> dict[str, Any]:
        return {
            "requested_capability": record.requested_capability.value,
            "capability_transfer_risk": record.capability_transfer_risk.value,
            "executionality_level": record.executionality_level.value,
            "operational_specificity": record.operational_specificity.value,
            "dual_use_probability": record.dual_use_probability,
        }


class AssessHarmProfileModule:
    """Summarize harm-related dimensions for policy selection."""

    def __call__(self, record: SemanticSafetyRecord, capability: dict[str, Any]) -> dict[str, Any]:
        del capability
        return {
            "harm_domain": record.harm_domain.value,
            "harm_severity": record.harm_severity.value,
            "reversibility": record.reversibility.value,
            "scale_of_harm": record.scale_of_harm.value,
            "target_type": record.target_type.value,
            "deception_component": record.deception_component,
            "concealment_component": record.concealment_component,
        }


class ChoosePolicyActionModule:
    """Choose a policy and safe alternative mode from the decomposition."""

    def __call__(
        self,
        record: SemanticSafetyRecord,
        capability: dict[str, Any],
        harm: dict[str, Any],
    ) -> dict[str, Any]:
        del capability, harm
        return {
            "policy_action": record.policy_action.value,
            "safe_alternative_mode": record.safe_alternative_mode.value,
            "abstain_recommended": record.abstain_recommended,
            "explanation_brief": record.explanation_brief,
        }


class GenerateSafeResponseModule:
    """Generate a bounded response specification rather than raw free-form text."""

    def __call__(self, record: SemanticSafetyRecord, policy: dict[str, Any]) -> dict[str, Any]:
        alternative = record.safe_alternative_mode.value
        response_text = self._response_text(policy)
        return {
            "policy_action": policy["policy_action"],
            "response_text": response_text,
            "allowed_high_level_help": record.allowed_high_level_help,
            "disallowed_operational_help": record.disallowed_operational_help,
            "safe_alternative_mode": alternative,
        }

    def _response_text(self, policy: dict[str, Any]) -> str:
        action = PolicyAction(policy["policy_action"])
        if action == PolicyAction.ALLOW:
            return "Provide the requested benign help while preserving transparency."
        if action == PolicyAction.ALLOW_WITH_BOUNDARIES:
            return "Offer bounded, high-level guidance and avoid operational detail."
        if action == PolicyAction.REDIRECT:
            return "Redirect toward safety-focused, preventive, or educational framing."
        if action == PolicyAction.ABSTAIN:
            return "Acknowledge uncertainty and ask clarifying questions or abstain."
        return "Refuse operational assistance and provide a safer alternative."


class CheckSemanticConsistencyModule:
    """Summarize semantic cluster consistency."""

    def __call__(self, cluster: SemanticCluster) -> dict[str, Any]:
        return {
            "cluster_id": cluster.cluster_id,
            "semantic_cluster_agreement": semantic_cluster_agreement(cluster),
            "record_count": len(cluster.records),
            "negative_control_count": len(cluster.negative_controls),
        }


class AggregateMultiTurnRiskModule:
    """Aggregate latent risk across a conversation."""

    def __call__(self, conversation: MultiTurnConversation) -> dict[str, Any]:
        return aggregate_multi_turn_risk(conversation)


class SemanticIntentPipeline:
    """Structured semantic intent pipeline mirroring the requested DSPy flow."""

    def __init__(self) -> None:
        self.decompose = DecomposeIntentModule()
        self.capability = AssessCapabilityRiskModule()
        self.harm = AssessHarmProfileModule()
        self.policy = ChoosePolicyActionModule()
        self.response = GenerateSafeResponseModule()
        self.consistency = CheckSemanticConsistencyModule()
        self.multi_turn = AggregateMultiTurnRiskModule()

    def run(
        self,
        record: SemanticSafetyRecord,
        *,
        cluster: SemanticCluster | None = None,
    ) -> SemanticPipelineResult:
        decomposition = self.decompose(record)
        capability = self.capability(decomposition)
        harm = self.harm(decomposition, capability)
        policy = self.policy(decomposition, capability, harm)
        response = self.response(decomposition, policy)
        consistency = self.consistency(cluster) if cluster is not None else None
        return SemanticPipelineResult(
            decomposition=decomposition,
            capability_assessment=capability,
            harm_assessment=harm,
            policy_decision=policy,
            safe_response=response,
            consistency_report=consistency,
        )

    def run_conversation(self, conversation: MultiTurnConversation) -> dict[str, Any]:
        summary = self.multi_turn(conversation)
        if any(turn.policy_action == PolicyAction.REFUSE for turn in conversation.turns):
            summary["policy_action"] = PolicyAction.REFUSE.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        elif any(turn.policy_action == PolicyAction.ABSTAIN for turn in conversation.turns):
            summary["policy_action"] = PolicyAction.ABSTAIN.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.CLARIFY.value
        elif summary["abstain_recommended"]:
            summary["policy_action"] = PolicyAction.ABSTAIN.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.CLARIFY.value
        elif any(
            turn.policy_action in {PolicyAction.ALLOW_WITH_BOUNDARIES, PolicyAction.REDIRECT}
            for turn in conversation.turns
        ):
            summary["policy_action"] = PolicyAction.ALLOW_WITH_BOUNDARIES.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        else:
            summary["policy_action"] = PolicyAction.ALLOW.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.NONE.value
        return summary


SEMANTIC_PIPELINE_REGISTRY = {
    "semantic_intent_pipeline": SemanticIntentPipeline,
}


__all__ = [
    "AggregateMultiTurnRiskModule",
    "AssessCapabilityRiskModule",
    "AssessHarmProfileModule",
    "CheckSemanticConsistencyModule",
    "ChoosePolicyActionModule",
    "DecomposeIntentModule",
    "GenerateSafeResponseModule",
    "SEMANTIC_PIPELINE_REGISTRY",
    "SemanticIntentPipeline",
    "SemanticPipelineResult",
]
