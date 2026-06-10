"""Structured modules for semantic intent robustness pipelines."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Local
from .consistency import (
    aggregate_multi_turn_risk,
    cooperative_alternative_quality,
    pressure_resistance_score,
    principle_consistency_score,
    semantic_cluster_agreement,
)
from .kv_context_safety import (
    KVCacheFeatureAdapter,
    KVCacheFeatureSnapshot,
    KVContextRiskScorer,
    KVContextSafetyConfig,
    PromptRiskAssessment,
    TranscriptFallbackAdapter,
)
from .memory_safety import (
    MemoryLaunderingReport,
    MemoryWriteRequest,
    RetrievedMemory,
    aggregate_memory_mediated_laundering,
)
from .schemas import (
    MultiTurnConversation,
    PrincipleRobustnessRecord,
    SemanticCluster,
    SemanticSafetyRecord,
)
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
    principle_decomposition: PrincipleRobustnessRecord | None = None
    principle_defense: dict[str, Any] | None = None
    principle_consistency_report: dict[str, Any] | None = None


class DecomposeIntentModule:
    """Return the provided structured decomposition or validate a seed record."""

    def __call__(
        self, record: SemanticSafetyRecord, _conversation_context: str = ""
    ) -> SemanticSafetyRecord:
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

    def __call__(self, record: SemanticSafetyRecord, _capability: dict[str, Any]) -> dict[str, Any]:
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
        _capability: dict[str, Any],
        _harm: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "policy_action": record.policy_action.value,
            "safe_alternative_mode": record.safe_alternative_mode.value,
            "abstain_recommended": record.abstain_recommended,
            "explanation_brief": record.explanation_brief,
        }


class GenerateSafeResponseModule:
    """Generate a bounded response specification rather than raw free-form text."""

    def __call__(self, record: SemanticSafetyRecord, policy: dict[str, Any]) -> dict[str, Any]:
        alternative = policy.get("safe_alternative_mode")
        if alternative is None:
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
        if action == PolicyAction.REFUSE:
            return "Refuse operational assistance and provide a safer alternative."
        raise ValueError(f"Unsupported policy_action: {action.value}")


class CheckSemanticConsistencyModule:
    """Summarize semantic cluster consistency."""

    def __call__(self, cluster: SemanticCluster) -> dict[str, Any]:
        return {
            "cluster_id": cluster.cluster_id,
            "semantic_cluster_agreement": semantic_cluster_agreement(cluster),
            "record_count": len(cluster.records),
            "negative_control_count": len(cluster.negative_controls),
        }


class DecomposePrinciplesModule:
    """Return a provided principle decomposition for pressure-aware flows."""

    def __call__(
        self,
        record: PrincipleRobustnessRecord,
        _semantic_decomposition: SemanticSafetyRecord | None = None,
    ) -> PrincipleRobustnessRecord:
        return record


class AssessAdversarialPressureModule:
    """Summarize pressure labels and tempting failure modes."""

    def __call__(self, record: PrincipleRobustnessRecord) -> dict[str, Any]:
        return {
            "pressure_types": [item.value for item in record.pressure_types],
            "adversarial_move": record.adversarial_move,
            "tempting_failure_mode": record.tempting_failure_mode.value,
            "failure_if_model_yields": record.failure_if_model_yields,
        }


class DefendPrincipledCooperationModule:
    """Produce public, principle-grounded response guidance without hidden reasoning."""

    def __call__(
        self,
        record: PrincipleRobustnessRecord,
        pressure: dict[str, Any],
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "target_principles": [item.value for item in record.target_principles],
            "critique_of_adversarial_frame": record.critique_of_adversarial_frame,
            "preferred_reasoning_summary": record.preferred_reasoning,
            "safe_response_target": record.safe_response_target,
            "uncertainty_handling": record.uncertainty_handling,
            "abstention_or_redirect_target": record.abstention_or_redirect_target,
            "pressure_types": pressure.get("pressure_types", []),
            "policy_action": policy.get("policy_action"),
        }


class CheckPrincipleConsistencyModule:
    """Summarize value-level stability across pressure variants."""

    def __call__(self, records: list[PrincipleRobustnessRecord]) -> dict[str, Any]:
        return {
            "record_count": len(records),
            "principle_consistency": principle_consistency_score(records),
            "pressure_resistance": pressure_resistance_score(records),
            "cooperative_alternative_quality": cooperative_alternative_quality(records),
        }


class AggregateMultiTurnRiskModule:
    """Aggregate latent risk across a conversation."""

    def __call__(self, conversation: MultiTurnConversation) -> dict[str, Any]:
        return aggregate_multi_turn_risk(conversation)


class MemoryBoundaryModule:
    """Aggregate memory write and retrieval boundary decisions."""

    def __call__(
        self,
        writes: list[MemoryWriteRequest],
        retrievals: list[RetrievedMemory],
    ) -> MemoryLaunderingReport:
        return aggregate_memory_mediated_laundering(writes, retrievals)


class KVContextSafetyModule:
    """Run the opt-in KV-context safety scorer."""

    def __init__(self) -> None:
        self.scorer = KVContextRiskScorer()

    def __call__(
        self,
        *,
        latest_record: SemanticSafetyRecord,
        conversation: MultiTurnConversation,
        snapshot: KVCacheFeatureSnapshot,
        config: KVContextSafetyConfig,
        candidate_response: str | None = None,
    ) -> PromptRiskAssessment:
        return self.scorer.assess(
            latest_record=latest_record,
            conversation=conversation,
            snapshot=snapshot,
            config=config,
            candidate_response=candidate_response,
        )


class SemanticIntentPipeline:
    """Structured semantic intent pipeline mirroring the requested DSPy flow."""

    def __init__(self) -> None:
        self.decompose = DecomposeIntentModule()
        self.capability = AssessCapabilityRiskModule()
        self.harm = AssessHarmProfileModule()
        self.policy = ChoosePolicyActionModule()
        self.response = GenerateSafeResponseModule()
        self.consistency = CheckSemanticConsistencyModule()
        self.decompose_principles = DecomposePrinciplesModule()
        self.pressure = AssessAdversarialPressureModule()
        self.principle_defense = DefendPrincipledCooperationModule()
        self.principle_consistency = CheckPrincipleConsistencyModule()
        self.multi_turn = AggregateMultiTurnRiskModule()
        self.memory_boundary = MemoryBoundaryModule()
        self.kv_context_safety = KVContextSafetyModule()

    def run(
        self,
        record: SemanticSafetyRecord,
        *,
        cluster: SemanticCluster | None = None,
        principle_record: PrincipleRobustnessRecord | None = None,
        principle_records: list[PrincipleRobustnessRecord] | None = None,
    ) -> SemanticPipelineResult:
        decomposition = self.decompose(record)
        capability = self.capability(decomposition)
        harm = self.harm(decomposition, capability)
        policy = self.policy(decomposition, capability, harm)
        response = self.response(decomposition, policy)
        consistency = self.consistency(cluster) if cluster is not None else None
        principle_decomposition = None
        principle_defense = None
        principle_report = None
        if principle_record is not None:
            principle_decomposition = self.decompose_principles(principle_record, decomposition)
            pressure = self.pressure(principle_decomposition)
            principle_defense = self.principle_defense(
                principle_decomposition,
                pressure,
                policy,
            )
        if principle_records is not None:
            principle_report = self.principle_consistency(principle_records)
        return SemanticPipelineResult(
            decomposition=decomposition,
            capability_assessment=capability,
            harm_assessment=harm,
            policy_decision=policy,
            safe_response=response,
            consistency_report=consistency,
            principle_decomposition=principle_decomposition,
            principle_defense=principle_defense,
            principle_consistency_report=principle_report,
        )

    def run_conversation(self, conversation: MultiTurnConversation) -> dict[str, Any]:
        summary = self.multi_turn(conversation)
        if not conversation.turns:
            summary["policy_action"] = PolicyAction.REFUSE.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
            summary["conversation_blocked"] = True
            return summary
        max_risk = summary.get("max_capability_transfer_risk", "low")
        contains_concealment = summary.get("contains_concealment", False)
        contains_deception = summary.get("contains_deception", False)
        abstain_recommended = summary.get("abstain_recommended", False)
        observed_actions = {turn.policy_action for turn in conversation.turns}
        if PolicyAction.REFUSE in observed_actions:
            summary["policy_action"] = PolicyAction.REFUSE.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        elif PolicyAction.ABSTAIN in observed_actions:
            summary["policy_action"] = PolicyAction.ABSTAIN.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.CLARIFY.value
        elif (
            max_risk in {"high", "critical"}
            or contains_concealment
            or contains_deception
            or summary.get("conversation_blocked", False)
        ):
            summary["policy_action"] = PolicyAction.REFUSE.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        elif PolicyAction.REDIRECT in observed_actions:
            summary["policy_action"] = PolicyAction.REDIRECT.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        elif PolicyAction.ALLOW_WITH_BOUNDARIES in observed_actions:
            summary["policy_action"] = PolicyAction.ALLOW_WITH_BOUNDARIES.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.HIGH_LEVEL_SAFETY.value
        elif abstain_recommended:
            summary["policy_action"] = PolicyAction.ABSTAIN.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.CLARIFY.value
        else:
            summary["policy_action"] = PolicyAction.ALLOW.value
            summary["safe_alternative_mode"] = SafeAlternativeMode.NONE.value
        summary["conversation_blocked"] = summary["policy_action"] in {
            PolicyAction.REFUSE.value,
            PolicyAction.ABSTAIN.value,
        }
        return summary

    def run_memory_boundary(
        self,
        writes: list[MemoryWriteRequest],
        retrievals: list[RetrievedMemory],
    ) -> MemoryLaunderingReport:
        """Run the separate memory-mediated laundering boundary layer."""

        return self.memory_boundary(writes, retrievals)

    def run_kv_context_assessment(
        self,
        *,
        latest_record: SemanticSafetyRecord,
        conversation: MultiTurnConversation,
        conversation_history: tuple[str, ...],
        config: KVContextSafetyConfig,
        kv_cache: object | None = None,
        candidate_response: str | None = None,
        adapter: KVCacheFeatureAdapter | None = None,
    ) -> PromptRiskAssessment:
        """Run the disabled-by-default contextual safety overlay."""

        if not config.enabled:
            return PromptRiskAssessment(
                conversation_id=conversation.conversation_id,
                turn_index=latest_record.turn_index,
                prompt_text=latest_record.prompt_text,
                single_prompt_risk=0.0,
                contextual_risk=0.0,
                contextual_uplift=0.0,
                contextual_ratio=0.0,
                trajectory_flag=False,
                trajectory_reasons=(),
                recommended_action="allow",
                requires_review=False,
                cache_mode=config.mode,
            )
        selected_adapter = adapter or TranscriptFallbackAdapter()
        snapshot = selected_adapter.extract_snapshot(
            conversation_id=conversation.conversation_id,
            turn_index=latest_record.turn_index,
            prompt_text=latest_record.prompt_text,
            conversation_history=conversation_history,
            kv_cache=kv_cache,
            candidate_response=candidate_response,
        )
        return self.kv_context_safety(
            latest_record=latest_record,
            conversation=conversation,
            snapshot=snapshot,
            config=config,
            candidate_response=candidate_response,
        )


SEMANTIC_PIPELINE_REGISTRY = {
    "semantic_intent_pipeline": SemanticIntentPipeline,
}


__all__ = [
    "AggregateMultiTurnRiskModule",
    "AssessAdversarialPressureModule",
    "AssessCapabilityRiskModule",
    "AssessHarmProfileModule",
    "CheckPrincipleConsistencyModule",
    "CheckSemanticConsistencyModule",
    "ChoosePolicyActionModule",
    "DecomposeIntentModule",
    "DecomposePrinciplesModule",
    "DefendPrincipledCooperationModule",
    "GenerateSafeResponseModule",
    "KVContextSafetyModule",
    "MemoryBoundaryModule",
    "SEMANTIC_PIPELINE_REGISTRY",
    "SemanticIntentPipeline",
    "SemanticPipelineResult",
]
