from mindful_trace_gepa.scoring.llm_judge import JudgeConfig, LLMJudge
from mindful_trace_gepa.scoring.schema import DIMENSIONS


def test_mock_judge_matches_schema(monkeypatch):
    monkeypatch.setenv("GEPA_JUDGE_MOCK", "1")
    judge = LLMJudge(JudgeConfig(mock=True))
    tier = judge.score_trace([{"content": "Simple trace"}])
    assert tier.tier == "judge"
    for dim in DIMENSIONS:
        assert dim in tier.scores
        assert 0 <= tier.scores[dim] <= 4
        assert 0.0 < tier.confidence[dim] <= 1.0
        spans = tier.meta["spans"][dim]
        assert isinstance(spans, list)
        assert spans[0]["start"] == 0
