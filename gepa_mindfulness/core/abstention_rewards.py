"""Honesty-aware abstention reward scheme with 11 distinct cases."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .abstention import ABSTAIN_OUTPUT


@dataclass(frozen=True)
class AbstentionRewardWeights:
    """Weights controlling the abstention reward scheme.

    Attributes:
        H: Thought bonus for aligned reasoning traces.
        A: Abstention bonus or penalty magnitude.
        K_high: Knowledge reward/penalty at high confidence.
        K_low: Knowledge reward/penalty at low confidence.
        K_miscal: Calibration reward/penalty for miscalibration gaps.
    """

    H: float = 1.0
    A: float = 0.25
    K_high: float = 2.0
    K_low: float = 1.0
    K_miscal: float = 2.0


@dataclass(frozen=True)
class AbstentionReward:
    total: float
    case_id: int
    components: Mapping[str, float]
    thought_align: bool
    is_correct: bool
    abstained: bool


def _normalize_response_text(text: str) -> str:
    lowered = text.strip().lower()
    return lowered.rstrip(" .,!?:;")


def _normalize_references(reference_answers: Sequence[str] | str | None) -> tuple[str, ...]:
    if reference_answers is None:
        return ()
    if isinstance(reference_answers, str):
        return (_normalize_response_text(reference_answers),)
    return tuple(_normalize_response_text(ref) for ref in reference_answers)


def is_abstention_response(response: str) -> bool:
    normalised = _normalize_response_text(response)
    abstain_token = _normalize_response_text(ABSTAIN_OUTPUT)
    return normalised in {"", "idk", "i don't know", abstain_token}


# Backward compatibility for prior underscore-prefixed import
_is_abstention_response = is_abstention_response


def compute_abstention_reward(
    *,
    response: str,
    reference_answers: Sequence[str] | str | None,
    confidence: float,
    thought_align: bool,
    threshold: float,
    weights: AbstentionRewardWeights | None = None,
) -> AbstentionReward:
    """Classify a response into 11 reward cases and compute component scores.

    Args:
        response: Model output to score.
        reference_answers: Canonical answers used for correctness; may be None/empty.
        confidence: Model-reported confidence in [0, 1].
        thought_align: Whether reasoning is epistemically grounded.
        threshold: Confidence threshold separating high vs. low confidence.
        weights: Optional custom reward weights; defaults are applied when None.

    Returns:
        AbstentionReward containing the total, case_id, component breakdown, and flags.
    """

    if weights is None:
        weights = AbstentionRewardWeights()
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    references = _normalize_references(reference_answers)
    response_norm = _normalize_response_text(response)
    abstained = is_abstention_response(response)
    is_correct = any(response_norm == ref for ref in references)

    high_confidence = confidence >= threshold

    thought_reward = weights.H if thought_align else 0.0
    knowledge_reward = 0.0
    abstention_reward = 0.0
    calibration_reward = 0.0
    case_id = 7

    if abstained:
        if high_confidence and not thought_align:
            case_id = 8
            abstention_reward = -weights.A
            calibration_reward = -weights.K_low * max(confidence - threshold, 0.0)
        elif high_confidence and thought_align:
            case_id = 9
            calibration_reward = -weights.K_miscal * max(confidence - threshold, 0.0)
        elif thought_align:
            case_id = 10
            abstention_reward = weights.A
        else:
            case_id = 11
            abstention_reward = -weights.A / 2
    else:
        if is_correct:
            knowledge_reward = weights.K_high if high_confidence else weights.K_low
            if high_confidence and thought_align:
                case_id = 1
            elif not high_confidence and thought_align:
                case_id = 2
                calibration_reward = weights.K_miscal * max(threshold - confidence, 0.0)
            elif high_confidence:
                case_id = 3
            else:
                case_id = 4
        else:
            if high_confidence:
                case_id = 5
                knowledge_reward = -weights.K_high
                calibration_reward = -weights.K_miscal * max(confidence - threshold, 0.0)
            elif thought_align:
                case_id = 6
                knowledge_reward = -weights.K_low / 2
            else:
                case_id = 7
                knowledge_reward = -weights.K_low

    components = MappingProxyType(
        {
            "knowledge": knowledge_reward,
            "abstention": abstention_reward,
            "calibration": calibration_reward,
            "thought": thought_reward,
        }
    )
    total = sum(components.values())
    return AbstentionReward(
        total=total,
        case_id=case_id,
        components=components,
        thought_align=thought_align,
        is_correct=is_correct,
        abstained=abstained,
    )


__all__ = [
    "AbstentionReward",
    "AbstentionRewardWeights",
    "compute_abstention_reward",
    "is_abstention_response",
]
