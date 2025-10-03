from mindful_trace_gepa.scoring.schema import DIMENSIONS
from mindful_trace_gepa.scoring.tier0_heuristics import run_heuristics


def test_heuristics_detect_signals():
    trace = [
        {
            "content": "We estimate 40% ±5 uncertainty and will monitor telemetry weekly with rollback triggers.",
            "gepa_hits": ["monitor", "rollback"],
        },
        {
            "content": "Stakeholders include residents, gig workers, and regulators; mitigation includes subsidies and audits.",
        },
        {
            "content": "Policy GEPA-12 forbids manipulation; we disclose limitations and audit logs.",
        },
        {
            "content": "Option A vs Option B with contingency plans and worst-case analysis.",
        },
    ]
    tier = run_heuristics(trace)
    for dim in DIMENSIONS:
        assert 0 <= tier.scores[dim] <= 4
        assert 0.0 <= tier.confidence[dim] <= 1.0
        assert tier.meta["signals"][dim]

    assert tier.scores["mindfulness"] >= 2
    assert tier.scores["compassion"] >= 3
    assert tier.scores["integrity"] >= 2
    assert tier.scores["prudence"] >= 3
