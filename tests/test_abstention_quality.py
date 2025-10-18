from gepa_mindfulness.core.abstention import (
    AbstentionQuality,
    assess_abstention_quality,
)


def test_assess_abstention_genuine_when_evidence_present():
    summary = {
        "tensions": "Conflict between sources",
        "evidence": "Reviewed multiple reports",
        "reflection": "Acknowledged limits",
    }
    response_segments = [
        "I examined several sources",
        "There is tension between the data",
        "I acknowledge a limitation",
    ]
    assessment = assess_abstention_quality(summary, response_segments)
    assert assessment.quality is AbstentionQuality.GENUINE
    assert assessment.evidence_markers["evidence"] > assessment.evidence_markers["lazy"]


def test_assess_abstention_lazy_when_short_and_vague():
    summary = {"path_1_reasoning": "Not sure"}
    response_segments = ["Maybe", "I am unsure"]
    assessment = assess_abstention_quality(summary, response_segments)
    assert assessment.quality in {AbstentionQuality.LAZY, AbstentionQuality.UNKNOWN}
