"""Tests for adaptive trajectory attack evaluation helpers."""

# Standard library
from __future__ import annotations

# Local
from evaluation.suites.robustness.adaptive_trajectory_attacks import (
    AdaptiveTrajectoryRun,
    evaluate_adaptive_trajectory_runs,
)
from semantic_intent_robustness.kv_context_safety import (
    KVContextMode,
    PromptRiskAssessment,
)


def _assessment(flagged: bool) -> PromptRiskAssessment:
    return PromptRiskAssessment(
        conversation_id="conv",
        turn_index=1,
        prompt_text="placeholder",
        single_prompt_risk=0.2,
        contextual_risk=0.8 if flagged else 0.1,
        contextual_uplift=0.6 if flagged else 0.0,
        contextual_ratio=4.0 if flagged else 0.5,
        trajectory_flag=flagged,
        trajectory_reasons=("component_assembly",) if flagged else (),
        recommended_action="manual_review" if flagged else "allow",
        requires_review=flagged,
        cache_mode=KVContextMode.TRANSCRIPT_FALLBACK,
    )


def test_matched_benign_hard_negative_remains_allowed() -> None:
    benign = AdaptiveTrajectoryRun(
        run_id="benign",
        turns=("audit COMPONENT_A",),
        responses=("bounded defensive answer",),
        assessments=(_assessment(False),),
        release_gate_assessments=(),
        earliest_expected_closure_turn=None,
        earliest_detected_closure_turn=None,
        benign_control=True,
    )

    report = evaluate_adaptive_trajectory_runs((benign,))

    assert report.benign_hard_negative_false_positive_rate == 0.0


def test_adaptive_attack_metrics_track_detection() -> None:
    attack = AdaptiveTrajectoryRun(
        run_id="attack",
        turns=("COMPONENT_A", "PARAMETER_B", "INTEGRATION_DETAIL_D"),
        responses=("a", "b", "c"),
        assessments=(_assessment(False), _assessment(True)),
        release_gate_assessments=(),
        earliest_expected_closure_turn=2,
        earliest_detected_closure_turn=2,
        benign_control=False,
    )

    report = evaluate_adaptive_trajectory_runs((attack,))

    assert report.trajectory_detection_recall == 1.0
    assert report.closure_point_recall == 1.0
    assert report.adaptive_attack_success_rate == 0.0
