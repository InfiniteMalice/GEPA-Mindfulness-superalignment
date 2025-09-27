"""Confidence-aware abstention utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


ABSTAIN_OUTPUT = "I don't know"


@dataclass
class ConfidenceDecision:
    confidence: float
    output: str
    metadata: Dict[str, float]


def enforce_abstention(raw_output: str, confidence: float, threshold: float = 0.75) -> ConfidenceDecision:
    """Return abstain output whenever confidence falls below the threshold."""

    decision = ConfidenceDecision(
        confidence=confidence,
        output=raw_output,
        metadata={"threshold": threshold},
    )
    if confidence < threshold:
        decision.output = ABSTAIN_OUTPUT
        decision.metadata["abstained"] = 1.0
    else:
        decision.metadata["abstained"] = 0.0
    return decision


def honesty_reward_from_trace(confidence: float, mindfulness: float, emptiness: float) -> float:
    """Compute honesty reward emphasising Mindfulness and Emptiness."""

    honesty = (mindfulness + emptiness) / 2.0
    calibrated = honesty * confidence
    return calibrated


def abstention_signal(confidence: float, threshold: float) -> Tuple[float, float]:
    """Return tuple of (task_reward_multiplier, honesty_penalty)."""

    if confidence < threshold:
        return 0.0, 0.0
    return 1.0, -max(0.0, threshold - confidence)
