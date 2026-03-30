"""Semantic intent robustness package for GEPA Mindfulness Superalignment."""

# Standard library
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE: dict[str, str] = {
    "ALL_SIGNATURES": ".signatures",
    "DEFAULT_CONFIG": ".config",
    "EvaluationSummary": ".evaluators",
    "LossBreakdown": ".losses",
    "MultiTurnConversation": ".schemas",
    "SEMANTIC_PIPELINE_REGISTRY": ".modules",
    "SemanticBatch": ".losses",
    "SemanticCluster": ".schemas",
    "SemanticIntentConfig": ".config",
    "SemanticIntentPipeline": ".modules",
    "SemanticPipelineResult": ".modules",
    "SemanticRobustnessEvaluator": ".evaluators",
    "SemanticSafetyRecord": ".schemas",
    "aggregate_multi_turn_risk": ".consistency",
    "batch_format_expectations": ".losses",
    "build_example_dataset": ".dataset_builder",
    "compute_loss_breakdown": ".losses",
    "decomposition_consistency_score": ".consistency",
    "export_example_clusters": ".dataset_builder",
    "export_example_jsonl": ".dataset_builder",
    "policy_consistency_score": ".consistency",
    "semantic_cluster_agreement": ".consistency",
    "topic_vs_intent_discrimination": ".consistency",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports to avoid eager optional imports."""

    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
