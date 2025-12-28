"""Core GEPA logic exports."""

from .abstention import ABSTAIN_OUTPUT, ConfidenceDecision, enforce_abstention
from .abstention_rewards import (
    AbstentionReward,
    AbstentionRewardWeights,
    compute_abstention_reward,
    is_abstention_response,
)
from .contemplative_principles import ContemplativePrinciple, GEPAPrinciples, GEPAPrincipleScore
from .dual_path import DualPathProbeScenario, iterate_dual_path_pool, sample_dual_path_batch
from .imperatives import AlignmentImperative, ImperativeEvaluator, ImperativeSignal
from .paraconsistent import ParaconsistentTruthValue, dialetheic_and
from .rewards import RewardSignal, RewardWeights
from .thought_alignment import (
    classify_thought_alignment,
    compute_epistemic_score,
    compute_match_score,
)
from .tracing import CircuitTracerLogger, ThoughtTrace, TraceEvent

AdversarialScenario = DualPathProbeScenario
iterate_adversarial_pool = iterate_dual_path_pool
sample_adversarial_batch = sample_dual_path_batch

__all__ = [
    "ABSTAIN_OUTPUT",
    "AbstentionReward",
    "AbstentionRewardWeights",
    "AlignmentImperative",
    "AdversarialScenario",
    "CircuitTracerLogger",
    "ConfidenceDecision",
    "ContemplativePrinciple",
    "DualPathProbeScenario",
    "GEPAPrincipleScore",
    "GEPAPrinciples",
    "ImperativeEvaluator",
    "ImperativeSignal",
    "ParaconsistentTruthValue",
    "RewardSignal",
    "RewardWeights",
    "ThoughtTrace",
    "TraceEvent",
    "classify_thought_alignment",
    "compute_abstention_reward",
    "compute_epistemic_score",
    "compute_match_score",
    "iterate_dual_path_pool",
    "iterate_adversarial_pool",
    "iterate_dual_path_pool",
    "sample_adversarial_batch",
    "sample_dual_path_batch",
]
