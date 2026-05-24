from evaluation.schema import (
    EvalResult,
    read_results_jsonl,
    result_from_jsonl,
    result_to_jsonl,
    write_results_jsonl,
)


def test_eval_result_jsonl_round_trip(tmp_path) -> None:
    result = EvalResult(
        eval_id="case-1",
        suite="simpleqa",
        category="factuality",
        prompt="What is 2+2?",
        model_answer="4",
        gold_answer="4",
        outcome="correct",
        confidence=0.8,
        gepa_score=4,
        trace_flags={"unsupported_claim": False},
        metadata={"fixture": True},
    )

    parsed = result_from_jsonl(result_to_jsonl(result))

    assert parsed == result
    assert set(parsed.trace_flags) >= {"unsupported_claim", "strategic_deception"}

    path = tmp_path / "results.jsonl"
    write_results_jsonl([result], path)
    assert read_results_jsonl(path) == [result]
