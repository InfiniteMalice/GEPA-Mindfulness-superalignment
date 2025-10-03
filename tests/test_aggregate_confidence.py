from mindful_trace_gepa.scoring.aggregate import aggregate_tiers
from mindful_trace_gepa.scoring.schema import DIMENSIONS, TierScores


def make_tier(name: str, base: int, confidence: float) -> TierScores:
    scores = {dim: base for dim in DIMENSIONS}
    conf = {dim: confidence for dim in DIMENSIONS}
    return TierScores(tier=name, scores=scores, confidence=conf, meta={})


def test_aggregation_escalates_on_disagreement():
    heuristic = make_tier("heuristic", 2, 0.7)
    judge = make_tier("judge", 4, 0.9)
    classifier = make_tier("classifier", 0, 0.8)
    result = aggregate_tiers([heuristic, judge, classifier], {
        "weights": {"heuristic": 0.2, "judge": 0.5, "classifier": 0.3},
        "abstention_thresholds": {dim: 0.75 for dim in DIMENSIONS},
        "disagreement_penalty": 0.2,
        "escalate_if_any_below": 0.6,
    })
    assert result.escalate is True
    assert any("disagreement" in reason for reason in result.reasons)


def test_aggregation_escalates_on_low_confidence():
    heuristic = make_tier("heuristic", 2, 0.2)
    result = aggregate_tiers([heuristic], {
        "weights": {"heuristic": 1.0},
        "abstention_thresholds": {dim: 0.5 for dim in DIMENSIONS},
        "disagreement_penalty": 0.0,
        "escalate_if_any_below": 0.4,
    })
    assert result.escalate is True
    assert any("below" in reason for reason in result.reasons)


def test_partial_weight_override_preserves_defaults():
    heuristic = make_tier("heuristic", 0, 1.0)
    judge = make_tier("judge", 4, 1.0)
    classifier = make_tier("classifier", 2, 1.0)
    result = aggregate_tiers([heuristic, judge, classifier], {
        "weights": {"judge": 0.7},
    })
    assert result.final["mindfulness"] == 3