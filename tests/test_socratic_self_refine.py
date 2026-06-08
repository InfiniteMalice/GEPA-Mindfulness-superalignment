from __future__ import annotations

from gepa_mindfulness.training.configs import TrainingConfig
from socratic_self_refine import (
    SocraticSelfRefineConfig,
    assess_reasoning_units,
    decompose_reasoning_trace,
    run_socratic_self_refine,
)


def _trace() -> list[dict[str, object]]:
    return [
        {
            "unit_id": "u1",
            "sub_question": "What is given?",
            "sub_answer": "A safe abstract premise.",
            "confidence": 0.9,
            "verifier_status": "verified",
        },
        {
            "unit_id": "u2",
            "sub_question": "What follows?",
            "sub_answer": "Unsupported conclusion.",
            "confidence": 0.3,
            "verifier_status": "failed",
            "dependencies": ["u1"],
        },
    ]


def test_structured_response_decomposes_into_units_and_dependencies() -> None:
    units = decompose_reasoning_trace(_trace())
    assert units[0].sub_question == "What is given?"
    assert units[1].dependencies == ("u1",)


def test_controlled_resolving_creates_attempts_and_scores() -> None:
    assessments = assess_reasoning_units(decompose_reasoning_trace(_trace()))
    weak = next(item for item in assessments if item.unit_id == "u2")
    assert weak.resolve_attempts
    assert weak.self_consistency_score == 0.0


def test_weak_units_are_selected_and_strong_units_are_not() -> None:
    assessments = assess_reasoning_units(decompose_reasoning_trace(_trace()))
    by_id = {item.unit_id: item for item in assessments}
    assert by_id["u2"].repair_recommended is True
    assert by_id["u1"].repair_recommended is False


def test_repair_is_bounded_and_original_trace_preserved() -> None:
    report = run_socratic_self_refine(
        _trace(),
        config=SocraticSelfRefineConfig(enabled=True, mode="evaluation", max_iterations=1),
        initial_answer_reference="answer-v1",
    )
    assert report.units_repaired == 1
    assert report.max_iterations == 1
    assert report.initial_answer_reference == "answer-v1"
    assert report.repair_events[0].dependency_updates == ()


def test_dependency_updates_propagate_to_downstream_units() -> None:
    trace = _trace() + [
        {
            "unit_id": "u3",
            "sub_question": "What depends on u2?",
            "sub_answer": "Downstream answer.",
            "confidence": 0.8,
            "verifier_status": "verified",
            "dependencies": ["u2"],
        }
    ]
    report = run_socratic_self_refine(
        trace,
        config=SocraticSelfRefineConfig(enabled=True, mode="evaluation"),
    )
    assert report.repair_events[0].dependency_updates == ("u3",)


def test_policy_checks_run_after_repair_and_block_bypass() -> None:
    def policy_check(_item) -> str:
        return "bounded"

    report = run_socratic_self_refine(
        _trace(),
        config=SocraticSelfRefineConfig(enabled=True, mode="inference"),
        policy_check=policy_check,
    )
    assert report.units_repaired == 0
    assert report.review_required is True


def test_repair_policy_meaning_change_escalates_review() -> None:
    def policy_check(item) -> str:
        return "policy_changed" if item.__class__.__name__ == "RepairEvent" else "pass"

    report = run_socratic_self_refine(
        _trace(),
        config=SocraticSelfRefineConfig(enabled=True, mode="review-only"),
        policy_check=policy_check,
    )
    assert report.review_required is True


def test_ssr_disabled_by_default_and_existing_config_unchanged() -> None:
    config = TrainingConfig.from_mapping({})
    assert config.socratic_self_refine.enabled is False
    report = run_socratic_self_refine(_trace())
    assert report.stopped_reason == "disabled"
