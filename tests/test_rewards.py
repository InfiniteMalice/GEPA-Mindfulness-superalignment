from __future__ import annotations

import pytest

from gepa_mindfulness.core.abstention import AbstentionAssessment, AbstentionQuality
from gepa_mindfulness.core.rewards import (
    GEPARewardCalculator,
    HallucinationConfig,
    RewardWeights,
)


@pytest.fixture()
def calculator() -> GEPARewardCalculator:
    weights = RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
    hallucination = HallucinationConfig(
        confidence_threshold=0.75,
        confident_wrong_penalty=-2.0,
        uncertain_wrong_penalty=-0.5,
        appropriate_abstention_reward=0.5,
        lazy_abstention_penalty=-0.2,
    )
    return GEPARewardCalculator(weights=weights, hallucination=hallucination)


def test_reward_weights_validation() -> None:
    weights = RewardWeights(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25)
    weights.validate()
    with pytest.raises(ValueError):
        RewardWeights(alpha=0.5, beta=0.5, gamma=0.5, delta=-0.5).validate()


def test_correct_answer_reward_positive(calculator: GEPARewardCalculator) -> None:
    breakdown = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores={"mindfulness": 0.9},
        imperatives={"reduce_suffering": {"support": 0.8, "opposition": 0.1}},
        confidence=0.9,
        trace_summary={},
    )
    assert breakdown.total > 0


def test_confident_hallucination_penalised(calculator: GEPARewardCalculator) -> None:
    breakdown = calculator.compute_reward(
        response="wrong",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.95,
        trace_summary={},
    )
    assert pytest.approx(breakdown.hallucination, rel=1e-6) == -2.0


def test_appropriate_abstention_reward(calculator: GEPARewardCalculator) -> None:
    assessment = AbstentionAssessment(
        quality=AbstentionQuality.GENUINE,
        evidence_markers={"evidence": 0.9, "lazy": 0.0},
    )
    breakdown = calculator.compute_reward(
        response="I don't know",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.2,
        trace_summary={"evidence": "consulted sources"},
        abstention=assessment,
    )
    assert pytest.approx(breakdown.hallucination, rel=1e-6) == 0.5
    assert breakdown.abstention_quality is AbstentionQuality.GENUINE


def test_lazy_abstention_penalty(calculator: GEPARewardCalculator) -> None:
    assessment = AbstentionAssessment(
        quality=AbstentionQuality.LAZY,
        evidence_markers={"evidence": 0.0, "lazy": 1.0},
    )
    breakdown = calculator.compute_reward(
        response="Not sure",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.3,
        trace_summary={},
        abstention=assessment,
    )
    assert pytest.approx(breakdown.hallucination, rel=1e-6) == -0.2


def test_trace_summary_supports_genuine_uncertainty(calculator: GEPARewardCalculator) -> None:
    breakdown = calculator.compute_reward(
        response="I am uncertain after reviewing conflicting evidence",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.4,
        trace_summary={"evidence": "consulted", "tensions": "noted"},
    )
    assert breakdown.honesty > 0
    assert breakdown.abstention_quality is not None
