"""Core GEPA logic exports."""

import random
import warnings
from collections.abc import Iterator
from typing import Any

from .abstention import ABSTAIN_OUTPUT, ConfidenceDecision, enforce_abstention
from .abstention_rewards import (
    AbstentionReward,
    AbstentionRewardWeights,
    compute_abstention_reward,
    is_abstention_response,
)
from .clarifying_abstention import (
    APPENDED_AMBIGUITY_CASES,
    FRAMEWORK_CASE_IDS,
    ORIGINAL_CASE_IDS,
    AbstentionType,
    AmbiguityHandlingMode,
    FrameworkCaseDefinition,
    StakesDimension,
    score_ambiguity_handling,
)
from .contemplative_principles import (
    ContemplativePrinciple,
    GEPAPrinciples,
    GEPAPrincipleScore,
)
from .dual_path import (
    DualPathProbeScenario,
    iterate_dual_path_pool,
    sample_dual_path_batch,
)
from .imperatives import AlignmentImperative, ImperativeEvaluator, ImperativeSignal
from .paraconsistent import ParaconsistentTruthValue, dialetheic_and
from .rewards import RewardSignal, RewardWeights
from .thought_alignment import (
    classify_thought_alignment,
    compute_epistemic_score,
    compute_match_score,
)
from .tracing import CircuitTracerLogger, ThoughtTrace, TraceEvent


def iterate_adversarial_pool() -> Iterator[DualPathProbeScenario]:
    """Deprecated alias for iterate_dual_path_pool."""
    warnings.warn(
        "iterate_adversarial_pool is deprecated; use iterate_dual_path_pool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return iterate_dual_path_pool()


def sample_adversarial_batch(
    batch_size: int,
    rng: random.Random | None = None,
) -> list[DualPathProbeScenario]:
    """Deprecated alias for sample_dual_path_batch."""
    warnings.warn(
        "sample_adversarial_batch is deprecated; use sample_dual_path_batch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sample_dual_path_batch(batch_size, rng=rng)


def __getattr__(name: str) -> Any:
    if name == "AdversarialScenario":
        warnings.warn(
            "AdversarialScenario is deprecated; use DualPathProbeScenario instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DualPathProbeScenario
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ABSTAIN_OUTPUT",
    "APPENDED_AMBIGUITY_CASES",
    "AbstentionReward",
    "AbstentionRewardWeights",
    "AbstentionType",
    "AlignmentImperative",
    "AdversarialScenario",
    "AmbiguityHandlingMode",
    "CircuitTracerLogger",
    "ConfidenceDecision",
    "ContemplativePrinciple",
    "DualPathProbeScenario",
    "FRAMEWORK_CASE_IDS",
    "FrameworkCaseDefinition",
    "GEPAPrincipleScore",
    "GEPAPrinciples",
    "ImperativeEvaluator",
    "ImperativeSignal",
    "ORIGINAL_CASE_IDS",
    "ParaconsistentTruthValue",
    "RewardSignal",
    "RewardWeights",
    "StakesDimension",
    "ThoughtTrace",
    "TraceEvent",
    "classify_thought_alignment",
    "compute_abstention_reward",
    "compute_epistemic_score",
    "compute_match_score",
    "dialetheic_and",
    "enforce_abstention",
    "is_abstention_response",
    "iterate_adversarial_pool",
    "iterate_dual_path_pool",
    "sample_adversarial_batch",
    "sample_dual_path_batch",
    "score_ambiguity_handling",
]
