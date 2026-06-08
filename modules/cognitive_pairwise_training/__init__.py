"""CPT-inspired metacognitive pairwise training helpers."""

from .pipeline import (
    CognitivePairwiseTrainingConfig,
    build_pairwise_examples,
    compute_cpt_evaluation,
    compute_pairwise_label_loss,
    export_pairwise_jsonl,
)
from .schemas import (
    CPTBatchMetrics,
    PairType,
    PairwiseLabel,
    PairwiseReasoningExample,
    ReasoningTraceCandidate,
)

__all__ = [
    "CPTBatchMetrics",
    "CognitivePairwiseTrainingConfig",
    "PairType",
    "PairwiseLabel",
    "PairwiseReasoningExample",
    "ReasoningTraceCandidate",
    "build_pairwise_examples",
    "compute_cpt_evaluation",
    "compute_pairwise_label_loss",
    "export_pairwise_jsonl",
]
