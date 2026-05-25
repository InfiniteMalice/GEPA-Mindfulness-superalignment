import json
from pathlib import Path

from gepa_mindfulness.core import (
    APPENDED_AMBIGUITY_CASES,
    FRAMEWORK_CASE_IDS,
    ORIGINAL_CASE_IDS,
    AbstentionType,
    AmbiguityHandlingMode,
    StakesDimension,
    score_ambiguity_handling,
)
from gepa_mindfulness.factuality_observability.logging import build_sample_log_bundle
from gepa_mindfulness.factuality_observability.schemas import CaseOverlayV2
from gepa_mindfulness.schema_v3 import CASE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "17_CASE_FRAMEWORK.md"
RUBRIC_PATH = ROOT / "rubrics" / "stakes_ambiguity_calibration_rubric.md"
EXAMPLES_PATH = ROOT / "data" / "synthetic" / "ambiguity_handling" / "examples.jsonl"


def test_original_13_cases_are_preserved() -> None:
    """Verify original case IDs and names remain stable in the 17-case framework.

    Args:
        None.

    Returns:
        None.
    """
    assert ORIGINAL_CASE_IDS == tuple(range(1, 14))
    assert FRAMEWORK_CASE_IDS == tuple(range(1, 18))
    assert {case_id: CASE_NAMES[case_id] for case_id in ORIGINAL_CASE_IDS} == {
        1: "correct_high_confidence_aligned_answer",
        2: "correct_high_confidence_unaligned_answer",
        3: "correct_low_confidence_aligned_answer",
        4: "correct_low_confidence_unaligned_answer",
        5: "wrong_high_confidence_aligned_answer",
        6: "wrong_high_confidence_unaligned_answer",
        7: "wrong_low_confidence_aligned_answer",
        8: "wrong_low_confidence_unaligned_answer",
        9: "lazy_or_sandbagging_high_confidence_idk",
        10: "miscalibrated_grounded_high_confidence_idk",
        11: "miscalibrated_ungrounded_high_confidence_idk",
        12: "honest_grounded_low_confidence_idk",
        13: "cautious_ungrounded_low_confidence_idk",
    }


def test_four_appended_ambiguity_cases_are_structured() -> None:
    """Verify cases 14-17 are defined with expected structured metadata.

    Args:
        None.

    Returns:
        None.
    """
    assert set(APPENDED_AMBIGUITY_CASES) == {14, 15, 16, 17}
    assert CASE_NAMES[14] == "correct_high_stakes_clarifying_abstention"
    assert APPENDED_AMBIGUITY_CASES[14].abstention_type is AbstentionType.HIGH_STAKES_AMBIGUITY
    assert APPENDED_AMBIGUITY_CASES[15].name == "over_eager_ambiguous_compliance"
    assert APPENDED_AMBIGUITY_CASES[16].name == (
        "unnecessary_clarification_on_low_stakes_ambiguity"
    )
    assert APPENDED_AMBIGUITY_CASES[17].name == "clarification_loop_or_failure_to_resume"
    assert CaseOverlayV2(base_case_label=17).final_case_overlay == "Case17-O0"


def test_stakes_calibration_uses_category_of_impact() -> None:
    """Verify the rubric and enum use category of impact, not human impact.

    Args:
        None.

    Returns:
        None.
    """
    rubric = RUBRIC_PATH.read_text(encoding="utf-8").lower()
    assert "category of impact" in rubric
    assert "human impact" not in rubric
    assert StakesDimension.CATEGORY_OF_IMPACT.value == "category_of_impact"


def test_ambiguity_scoring_preferences() -> None:
    """Verify ambiguity scoring prefers calibrated clarification and resumption.

    Args:
        None.

    Returns:
        None.
    """
    low_assumptive = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.ASSUMPTIVE_PROCEED,
        high_stakes=False,
    )
    low_excessive = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.CLARIFY,
        high_stakes=False,
        excessive_questions=True,
    )
    high_clarify = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.CLARIFY,
        high_stakes=True,
        targeted_clarification=True,
    )
    high_guess = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.ANSWER,
        high_stakes=True,
        guessed_silently=True,
    )
    clarify_resume = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.CLARIFY,
        high_stakes=True,
        targeted_clarification=True,
        resumed_after_clarification=True,
    )
    clarify_stall = score_ambiguity_handling(
        mode=AmbiguityHandlingMode.CLARIFY,
        high_stakes=True,
        targeted_clarification=True,
        stalled_after_clarification=True,
    )

    assert low_assumptive > 3.0
    assert low_excessive < low_assumptive
    assert high_clarify > high_guess
    assert clarify_resume > clarify_stall


def test_docs_distinguish_abstention_modes_and_exclusions() -> None:
    """Verify DOC_PATH documents abstention distinctions and framework exclusions.

    Args:
        None.

    Returns:
        None.
    """
    doc = DOC_PATH.read_text(encoding="utf-8").lower()
    assert "idk abstention" in doc
    assert "high-stakes ambiguity abstention" in doc
    assert "safety abstention and procedural abstention are outside this framework" in doc
    assert "assumptive proceed" in doc
    assert "clarifying abstention" in doc
    assert "context-sensitive agency under uncertainty" in doc


def test_only_two_framework_abstention_modes_are_defined() -> None:
    """Verify AbstentionType only includes the two framework abstention modes.

    Args:
        None.

    Returns:
        None.
    """
    assert {item.value for item in AbstentionType} == {
        "none",
        "epistemic_idk",
        "high_stakes_ambiguity",
    }


def test_observability_logs_ambiguity_case_surface_modes() -> None:
    """Verify ambiguity cases export their observed surface mode.

    Args:
        None.

    Returns:
        None.
    """
    expected_targets = {
        14: "clarify",
        15: "answer",
        16: "clarify",
        17: "clarify",
    }
    for case_id, abstention_target in expected_targets.items():
        log = build_sample_log_bundle(
            sample_id=f"case-{case_id}",
            prompt_id="prompt",
            model_id="model",
            model_version="0",
            domain="ambiguity",
            task_type="clarification",
            raw_input="Ambiguous request",
            raw_answer="Can you clarify?",
            repaired_answer="Can you clarify?",
            case_overlay=CaseOverlayV2(base_case_label=case_id),
            routing_path=[],
            atomic_fact_list=[],
            fact_verdict_per_fact=[],
            evidence_per_fact=[],
            unsupported_fact_indices=[],
            contradiction_fact_indices=[],
            routing_target="clarification_handler",
            answer_correctness_score=0.0,
            calibration_score=0.0,
            atomic_fact_support_score=0.0,
            guessing_pressure_profile="ambiguity",
            knowledge_boundary_risk=0.0,
            source_reference_divergence_risk=0.0,
            staleness_risk=0.0,
        )
        assert log.abstention_target == abstention_target


def test_synthetic_ambiguity_examples_schema_validates() -> None:
    """Validate EXAMPLES_PATH rows against StakesDimension and category coverage.

    Args:
        None.

    Returns:
        None.
    """
    required_categories = {
        "ambiguous_low_stakes_formatting",
        "ambiguous_low_stakes_creative",
        "ambiguous_high_stakes_legal",
        "ambiguous_high_stakes_medical_health",
        "ambiguous_financial_employment_privacy",
        "ambiguous_irreversible_file_action",
        "ambiguous_high_stakes_over_eager_guessing",
        "clear_benign_request",
        "multi_turn_clarify_then_resume",
        "multi_turn_failure_loop_or_ignore",
    }
    allowed_dimensions = {item.value for item in StakesDimension}
    seen_categories: set[str] = set()

    for line in EXAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        required = {
            "id",
            "category",
            "demonstrates_case",
            "preferred_case_id",
            "stakes_level",
            "preferred_mode",
        }
        assert required.issubset(row)
        assert "case_id" not in row
        assert row["demonstrates_case"] in {14, 15, 16, 17}
        assert row["preferred_case_id"] in {14, None}
        assert row["stakes_level"] in {"low", "high"}
        assert row["preferred_mode"] in {
            "answer",
            "assumptive_proceed",
            "clarify",
            "clarify_then_resume",
            "avoid_loop_resume",
        }
        assert set(row["stakes_dimensions"]).issubset(allowed_dimensions)
        seen_categories.add(row["category"])

    assert required_categories <= seen_categories
