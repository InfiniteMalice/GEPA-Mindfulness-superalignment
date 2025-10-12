import pytest

from mindful_trace_gepa.scoring.aggregate import DEFAULT_CONFIG, aggregate_tiers, build_config
from mindful_trace_gepa.scoring.schema import DIMENSIONS, TierScores


def make_tier(name: str, base: int, confidence: float) -> TierScores:
    scores = {dim: base for dim in DIMENSIONS}
    conf = {dim: confidence for dim in DIMENSIONS}
    return TierScores(tier=name, scores=scores, confidence=conf, meta={})


def test_aggregation_escalates_on_disagreement():
    heuristic = make_tier("heuristic", 2, 0.7)
    judge = make_tier("judge", 4, 0.9)
    classifier = make_tier("classifier", 0, 0.8)
    result = aggregate_tiers(
        [heuristic, judge, classifier],
        {
            "weights": {"heuristic": 0.2, "judge": 0.5, "classifier": 0.3},
            "abstention_thresholds": {dim: 0.75 for dim in DIMENSIONS},
            "disagreement_penalty": 0.2,
            "escalate_if_any_below": 0.6,
        },
    )
    assert result.escalate is True
    assert any("disagreement" in reason for reason in result.reasons)


def test_aggregation_escalates_on_low_confidence():
    heuristic = make_tier("heuristic", 2, 0.2)
    result = aggregate_tiers(
        [heuristic],
        {
            "weights": {"heuristic": 1.0},
            "abstention_thresholds": {dim: 0.5 for dim in DIMENSIONS},
            "disagreement_penalty": 0.0,
            "escalate_if_any_below": 0.4,
        },
    )
    assert result.escalate is True
    assert any("below" in reason for reason in result.reasons)


def test_partial_weight_override_preserves_defaults():
    heuristic = make_tier("heuristic", 0, 1.0)
    judge = make_tier("judge", 4, 1.0)
    classifier = make_tier("classifier", 2, 1.0)
    result = aggregate_tiers(
        [heuristic, judge, classifier],
        {
            "weights": {"judge": 0.7},
        },
    )
    assert result.final["mindfulness"] == 3


def test_build_config_sanitizes_numeric_values():
    config = build_config(
        {
            "weights": {"judge": "0.4", "classifier": None},
            "abstention_thresholds": {"mindfulness": "0.8", "integrity": None},
            "disagreement_penalty": "0.5",
            "escalate_if_any_below": None,
        }
    )

    assert config["weights"]["judge"] == 0.4
    # Classifier weight should retain default 0.3 because override was None
    assert config["weights"]["classifier"] == DEFAULT_CONFIG["weights"]["classifier"]
    assert config["abstention_thresholds"]["mindfulness"] == 0.8
    # Purpose threshold fallback should remain default value
    expected_integrity = DEFAULT_CONFIG["abstention_thresholds"]["integrity"]
    assert config["abstention_thresholds"]["integrity"] == expected_integrity
    # Numeric fields should coerce to float and ignore None override
    assert config["disagreement_penalty"] == 0.5
    assert config["escalate_if_any_below"] == DEFAULT_CONFIG["escalate_if_any_below"]


def test_small_disagreement_does_not_trigger_penalty_or_escalation():
    judge = make_tier("judge", 3, 0.9)
    classifier = make_tier("classifier", 2, 0.8)

    result = aggregate_tiers(
        [judge, classifier],
        {
            "weights": {"judge": 0.5, "classifier": 0.3},
            "abstention_thresholds": {dim: 0.75 for dim in DIMENSIONS},
            "disagreement_penalty": 0.25,
            "escalate_if_any_below": 0.6,
        },
    )

    assert result.escalate is False
    for dim in DIMENSIONS:
        assert result.confidence[dim] == pytest.approx(0.85)
