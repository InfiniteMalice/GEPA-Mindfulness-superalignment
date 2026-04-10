from gepa_mindfulness.factuality_observability.logging import mark_correct_but_wrong_reason


def test_correct_answer_wrong_provenance_flagged() -> None:
    assert mark_correct_but_wrong_reason(
        answer_correctness_score=1.0,
        provenance_status="textual_only",
        support_coverage_score=0.3,
        uncertainty_disagreement=0.7,
    )
