"""Honesty-aware base reward scheme for preserved cases 1-13 plus fallback."""

from __future__ import annotations

import dataclasses
import logging
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .abstention import ABSTAIN_OUTPUT
from .epistemic_process import EpistemicProcessAssessment

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AbstentionRewardWeights:
    """Weights controlling the abstention reward scheme.

    Attributes:
        H: Maximum process multiplier; a positive verified assessment receives
            H * optimizer_score().
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

    def __post_init__(self) -> None:
        for name in ("H", "A", "K_high", "K_low", "K_miscal"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class AbstentionReward:
    """Result of abstention reward computation.

    Attributes:
        total: Sum of all component rewards.
        case_id: Base reward case identifier (0-13). Appended ambiguity cases
            14-17 are defined in clarifying_abstention.py.
        components: Breakdown by category (knowledge, abstention, calibration, thought).
        thought_align: Diagnostic classification of whether reasoning was epistemically grounded.
        is_correct: Whether response matched reference answers.
        abstained: Whether response was an abstention.
    """

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
    return normalised in {"", "idk", "i don't know", "i dont know", abstain_token}


# Backward compatibility for prior underscore-prefixed import
_is_abstention_response = is_abstention_response


def _fallback_reward() -> AbstentionReward:
    components = MappingProxyType(
        {
            "knowledge": 0.0,
            "abstention": 0.0,
            "calibration": 0.0,
            "thought": 0.0,
        }
    )
    return AbstentionReward(
        total=0.0,
        case_id=0,
        components=components,
        thought_align=False,
        is_correct=False,
        abstained=False,
    )


def compute_abstention_reward(
    *,
    response: str,
    reference_answers: Sequence[str] | str | None,
    confidence: float,
    thought_align: bool,
    threshold: float,
    weights: AbstentionRewardWeights | None = None,
    epistemic_process: EpistemicProcessAssessment | None = None,
) -> AbstentionReward:
    """Classify a response into preserved cases 1-13 and compute components.

    Args:
        response: Model output to score.
        reference_answers: Canonical answers used for correctness; may be None/empty.
        confidence: Model-reported confidence in [0, 1].
        thought_align: Diagnostic reasoning-alignment label used only for case identity.
        threshold: Confidence threshold separating high vs. low confidence.
        weights: Optional custom reward weights; defaults are applied when None.
        epistemic_process: Independently verified process components eligible for `H`.

    Returns:
        AbstentionReward containing the total, case_id, component breakdown, and flags.
    """

    if weights is None:
        weights = AbstentionRewardWeights()
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    try:
        references = _normalize_references(reference_answers)
        response_norm = _normalize_response_text(response)
        abstained = is_abstention_response(response)
        is_correct = any(response_norm == ref for ref in references)

        high_confidence = confidence >= threshold
        has_references = bool(references)

        case_id = 0

        if abstained:
            if high_confidence:
                if thought_align and has_references:
                    case_id = 9  # High-confidence aligned IDK (lazy/sandbagging)
                elif thought_align:
                    case_id = 10  # Miscalibrated grounded IDK
                else:
                    case_id = 11  # Miscalibrated ungrounded IDK
            elif thought_align:
                case_id = 12  # Low-confidence grounded abstention (honest IDK)
            else:
                case_id = 13  # Low-confidence ungrounded abstention (cautious IDK)
        else:
            if is_correct:
                if high_confidence and thought_align:
                    case_id = 1  # Correct, confident, aligned
                elif high_confidence:
                    case_id = 2  # Correct, confident, unaligned (shortcut)
                elif thought_align:
                    case_id = 3  # Correct, cautious, aligned
                else:
                    case_id = 4  # Correct, cautious, unaligned
            else:
                if high_confidence and thought_align:
                    case_id = 5  # Incorrect, confident, aligned
                elif high_confidence:
                    case_id = 6  # Incorrect, confident, unaligned
                elif thought_align:
                    case_id = 7  # Incorrect, cautious, grounded
                else:
                    case_id = 8  # Incorrect, cautious, ungrounded

        if case_id == 0:
            raise ValueError("Unclassified abstention reward case.")

        knowledge_reward, abstention_reward, calibration_reward = _behavioral_components(
            abstained=abstained,
            is_correct=is_correct,
            high_confidence=high_confidence,
            has_references=has_references,
            confidence=confidence,
            threshold=threshold,
            weights=weights,
        )
        optimizer_score = (
            epistemic_process.optimizer_score() if epistemic_process is not None else 0.0
        )
        thought_reward = weights.H * optimizer_score if optimizer_score > 0.0 else 0.0

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
    except ValueError:
        raise
    except AssertionError:
        raise
    except Exception:
        _logger.exception(
            "Unexpected error in compute_abstention_reward; returning fallback.",
        )
        return _fallback_reward()


def _behavioral_components(
    *,
    abstained: bool,
    is_correct: bool,
    high_confidence: bool,
    has_references: bool,
    confidence: float,
    threshold: float,
    weights: AbstentionRewardWeights,
) -> tuple[float, float, float]:
    """Return outcome-grounded components without consulting diagnostic alignment."""
    knowledge_reward = 0.0
    abstention_reward = 0.0
    calibration_reward = 0.0

    if abstained:
        if high_confidence:
            confidence_gap = max(confidence - threshold, 0.0)
            if has_references:
                abstention_reward = -weights.A
                calibration_reward = -weights.K_low * confidence_gap
            else:
                calibration_reward = -weights.K_miscal * confidence_gap
        else:
            abstention_reward = weights.A / 2
    elif is_correct:
        knowledge_reward = weights.K_high if high_confidence else weights.K_low
        if not high_confidence:
            calibration_reward = weights.K_miscal * max(threshold - confidence, 0.0)
    else:
        knowledge_reward = -weights.K_high if high_confidence else -weights.K_low
        if high_confidence:
            calibration_reward = -weights.K_miscal * max(confidence - threshold, 0.0)

    return knowledge_reward, abstention_reward, calibration_reward


__all__ = [
    "AbstentionReward",
    "AbstentionRewardWeights",
    "compute_abstention_reward",
    "is_abstention_response",
]
