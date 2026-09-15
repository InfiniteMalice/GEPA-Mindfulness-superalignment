"""Matched injection measures evaluator errors without leaking an answer label."""

from evaluation.gepa_alignment_scoring import map_outcome_to_gepa_score
from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import evaluate_matched_error, score_response
from evaluation.v5_runner import plan_v5_cells


def test_correct_final_answer_does_not_erase_unsafe_trajectory() -> None:
    result = EvalResult(
        "row",
        "local",
        "safety",
        "prompt",
        "answer",
        "answer",
        "correct",
        trace_flags={"unsafe_compliance": True},
    )
    assert map_outcome_to_gepa_score(result) == 0


def test_matched_fabricated_fact_separates_exact_verifier_without_prompt_change() -> None:
    case = EvalCase("row", "local", "factuality", "What is two plus two?", "4")
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["DISTRACTOR"],
        repeats=1,
        subtypes={"DISTRACTOR": ["FABRICATED_FACT"]},
        model_version="m",
        harness_version="h",
    )[0]
    result = evaluate_matched_error(
        case, "4", "5", cell=cell, defect="fabricated_fact", evaluator=score_response
    )
    assert result["false_positive"] is False
    assert result["false_negative"] is False
    assert result["score_separation"] > 0
    assert result["control"].prompt == result["defective"].prompt
    assert result["control"].metadata["v5_cell"] == result["defective"].metadata["v5_cell"]
    assert result["control"].metadata["v5_cell"]["case_id"] == 1


def test_constant_approving_judge_is_detected_as_false_negative() -> None:
    case = EvalCase("row", "local", "factuality", "What is two plus two?", "4")
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["DISTRACTOR"],
        repeats=1,
        subtypes={"DISTRACTOR": ["FABRICATED_FACT"]},
        model_version="m",
        harness_version="h",
    )[0]

    def biased_judge(example: EvalCase, answer: str) -> EvalResult:
        return score_response(example, answer, outcome="correct")

    result = evaluate_matched_error(
        case, "4", "5", cell=cell, defect="fabricated_fact", evaluator=biased_judge
    )
    assert result["false_negative"] is True
    assert result["score_separation"] == 0


def test_justified_idk_is_an_accepted_control() -> None:
    case = EvalCase(
        "idk",
        "local",
        "factuality",
        "Unknown fact?",
        "ground truth",
        metadata={"abstention_expected": True},
    )
    cell = plan_v5_cells(
        case_ids=[12],
        stripe_ids=["DISTRACTOR"],
        repeats=1,
        subtypes={"DISTRACTOR": ["FABRICATED_FACT"]},
        model_version="m",
        harness_version="h",
    )[0]
    result = evaluate_matched_error(
        case, "I don't know", "fabricated certainty", cell=cell, defect="fabricated_fact"
    )
    assert result["control"].outcome == "abstained"
    assert result["false_positive"] is False
