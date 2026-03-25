"""Semantic intent robustness package for GEPA Mindfulness Superalignment."""

# Local
from .config import DEFAULT_CONFIG, SemanticIntentConfig
from .consistency import (
    aggregate_multi_turn_risk,
    decomposition_consistency_score,
    policy_consistency_score,
    semantic_cluster_agreement,
    topic_vs_intent_discrimination,
)
from .dataset_builder import build_example_dataset, export_example_clusters, export_example_jsonl
from .evaluators import EvaluationSummary, SemanticRobustnessEvaluator
from .losses import LossBreakdown, SemanticBatch, batch_format_expectations, compute_loss_breakdown
from .modules import SEMANTIC_PIPELINE_REGISTRY, SemanticIntentPipeline, SemanticPipelineResult
from .schemas import MultiTurnConversation, SemanticCluster, SemanticSafetyRecord
from .signatures import ALL_SIGNATURES

__all__ = [
    "ALL_SIGNATURES",
    "DEFAULT_CONFIG",
    "EvaluationSummary",
    "LossBreakdown",
    "MultiTurnConversation",
    "SEMANTIC_PIPELINE_REGISTRY",
    "SemanticBatch",
    "SemanticCluster",
    "SemanticIntentConfig",
    "SemanticIntentPipeline",
    "SemanticPipelineResult",
    "SemanticRobustnessEvaluator",
    "SemanticSafetyRecord",
    "aggregate_multi_turn_risk",
    "batch_format_expectations",
    "build_example_dataset",
    "compute_loss_breakdown",
    "decomposition_consistency_score",
    "export_example_clusters",
    "export_example_jsonl",
    "policy_consistency_score",
    "semantic_cluster_agreement",
    "topic_vs_intent_discrimination",
]
