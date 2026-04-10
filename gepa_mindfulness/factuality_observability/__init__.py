"""Atomic factuality, observability, diagnostics, and trace logging toolkit."""

from .calibration import CalibrationOutput, ConfidenceSignals, fuse_confidence
from .config import FactualityObservabilityConfig
from .consistency import QueryRecord, compute_related_query_consistency
from .decomposition import AtomicDecompositionResult, decompose_verify_and_repair
from .logging import (
    SampleLogBundle,
    TracePackage,
    build_sample_log_bundle,
    build_trace_package,
    mark_correct_but_wrong_reason,
)
from .pipeline import PipelineInputs, PipelineOutputs, run_v2_pipeline
from .routing import RoutingContext, RoutingDecision, choose_routing_action
from .scoring import ScoringInputs, compute_scores
from .schemas import CaseOverlayV2, EvaluationScoresV2, ObservabilityTier

__all__ = [
    "AtomicDecompositionResult",
    "CalibrationOutput",
    "CaseOverlayV2",
    "ConfidenceSignals",
    "EvaluationScoresV2",
    "FactualityObservabilityConfig",
    "ObservabilityTier",
    "PipelineInputs",
    "PipelineOutputs",
    "QueryRecord",
    "RoutingContext",
    "RoutingDecision",
    "SampleLogBundle",
    "ScoringInputs",
    "TracePackage",
    "build_sample_log_bundle",
    "build_trace_package",
    "choose_routing_action",
    "compute_related_query_consistency",
    "compute_scores",
    "decompose_verify_and_repair",
    "fuse_confidence",
    "mark_correct_but_wrong_reason",
    "run_v2_pipeline",
]
