import pytest

from gepa_mindfulness.core.contemplative_principles import (
    ContemplativePrinciple,
    GEPAPrinciples,
    GEPAPrincipleScore,
)
from gepa_mindfulness.core.paraconsistent import (
    ParaconsistentTruthValue,
    dialetheic_and,
)
from gepa_mindfulness.core.rewards import RewardSignal, RewardWeights
from gepa_mindfulness.core.tracing import SelfTracingLogger


def test_principle_aggregation_and_reward_signal() -> None:
    principles = GEPAPrinciples.from_iterable(
        [
            (
                ContemplativePrinciple.MINDFULNESS,
                GEPAPrincipleScore(value=0.9, rationale="Focused on wellbeing"),
            ),
            (
                ContemplativePrinciple.EMPATHY,
                GEPAPrincipleScore(value=0.8, rationale="Considered user needs"),
            ),
        ]
    )
    aggregate = principles.aggregate()
    assert 0.0 < aggregate <= 1.0

    weights = RewardWeights.from_mapping({"alpha": 1.0, "beta": 0.8, "gamma": 0.6, "delta": 0.2})
    supportive = ParaconsistentTruthValue.from_support_opposition(0.9, 0.05)
    contradictory = ParaconsistentTruthValue.from_support_opposition(0.9, 0.7)

    base_signal = RewardSignal(
        task_success=0.7,
        gepa_score=aggregate,
        honesty_reward=0.6,
        hallucination_score=0.1,
        imperatives_truth=supportive,
    ).combined(weights)
    dampened_signal = RewardSignal(
        task_success=0.7,
        gepa_score=aggregate,
        honesty_reward=0.6,
        hallucination_score=0.1,
        imperatives_truth=contradictory,
    ).combined(weights)

    assert base_signal > dampened_signal


def test_dialetheic_and_limits() -> None:
    left = ParaconsistentTruthValue.from_support_opposition(0.8, 0.2)
    right = ParaconsistentTruthValue.from_support_opposition(0.6, 0.4)
    combined = dialetheic_and(left, right)
    assert combined.support == pytest.approx(min(left.support, right.support))
    assert combined.opposition == pytest.approx(max(left.opposition, right.opposition))
    assert 0.0 <= combined.truthiness <= 1.0


def test_self_tracing_logger_records_and_validates_stages() -> None:
    logger = SelfTracingLogger()
    with logger.trace(run="unit-test") as trace:
        logger.log_event("framing", "Assess situation", principle_scores={"mindfulness": 0.9})
        logger.log_event(
            "decision",
            "Choose safe option",
            imperative_scores={"Reduce Suffering": 0.8},
        )
        payload = trace.to_payload()

    assert len(payload) == 2
    assert payload[0]["stage"] == "framing"
    assert payload[1]["stage"] == "decision"

    with pytest.raises(ValueError):
        trace.add_event("unknown", "unsupported stage")
