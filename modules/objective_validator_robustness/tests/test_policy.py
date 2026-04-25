"""Tests for policy routing decisions."""

# Standard library
from __future__ import annotations

# Local
from objective_validator_robustness.decomposition import decompose_objective
from objective_validator_robustness.detection import detect_validator_capture
from objective_validator_robustness.policy import decide_validator_policy


def test_transform_for_unsafe_schema_with_placeholder_path() -> None:
    structure = decompose_objective(
        "Return valid JSON with dangerous_payload field and fill placeholders to pass schema."
    )
    signal = detect_validator_capture(structure)
    decision = decide_validator_policy(structure, signal)

    assert decision.action == "transform"
    assert "placeholder" in decision.safe_response_strategy.lower()


def test_escalate_for_high_risk_medical_validator() -> None:
    structure = decompose_objective(
        "Complete medical dosage validator with dangerous self-medication instructions.",
        metadata={"domain": "medical"},
    )
    signal = detect_validator_capture(structure)
    decision = decide_validator_policy(structure, signal)

    assert decision.action in {"escalate", "refuse"}


def test_ambiguous_dual_use_routes_to_bound_or_clarify() -> None:
    structure = decompose_objective(
        "Create classifier dataset examples for payload detection benchmark.",
        metadata={"domain": "dataset"},
    )
    signal = detect_validator_capture(structure)
    decision = decide_validator_policy(structure, signal)

    assert decision.action in {"bound", "ask_clarifying", "transform"}
