import math
def test_aggregate_gepa_score_basic_weighting():
    sessions = [
        PracticeSession(duration_minutes=30, grounding=0.6, equanimity=0.5, purpose=0.8, awareness=0.7),
        PracticeSession(duration_minutes=15, grounding=0.9, equanimity=0.8, purpose=0.9, awareness=0.85),
    ]

    # expected value computed manually
    expected = (
        ((0.6 + 0.5 + 0.8 + 0.7) / 4.0) * 30
        + ((0.9 + 0.8 + 0.9 + 0.85) / 4.0) * 15
    ) / 45

    assert math.isclose(aggregate_gepa_score(sessions), expected)

def test_zero_duration_sessions_are_ignored():
    sessions = [
        PracticeSession(duration_minutes=0, grounding=0.4, equanimity=0.5, purpose=0.6, awareness=0.7),
        PracticeSession(duration_minutes=0, grounding=0.9, equanimity=0.9, purpose=0.9, awareness=0.9),
    ]

    assert aggregate_gepa_score(sessions) == 0.0


def test_validation_rejects_out_of_range_scores():
    session = PracticeSession(duration_minutes=10, grounding=1.5, equanimity=0.5, purpose=0.5, awareness=0.5)

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])


def test_validation_rejects_negative_duration():
    session = PracticeSession(duration_minutes=-1, grounding=0.5, equanimity=0.5, purpose=0.5, awareness=0.5)

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])

