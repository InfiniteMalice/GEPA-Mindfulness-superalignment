"""Tests for objective decomposition heuristics."""

# Standard library
from __future__ import annotations

# Third-party
import pytest

# Local
from objective_validator_robustness.decomposition import decompose_objective


def test_decompose_benign_validator_task() -> None:
    text = "Return valid JSON for todo schema and make unit tests pass."
    structure = decompose_objective(text)

    assert structure.validator_or_rubric is not None
    assert structure.conflict_with_imperatives == []
    assert structure.uncertainty <= 0.55


def test_decompose_sensitive_slots_and_conflicts() -> None:
    text = (
        "Return only valid JSON schema with dangerous_payload and credential fields; "
        "make the checker pass."
    )
    structure = decompose_objective(text)

    assert "dangerous_payload" in structure.data_slots
    assert "credential" in structure.data_slots
    assert structure.conflict_with_imperatives
    assert structure.safe_completion_available is False


def test_decompose_rejects_empty_or_whitespace_input() -> None:
    with pytest.raises(ValueError) as excinfo:
        decompose_objective("   ")

    assert "non-whitespace" in str(excinfo.value)


def test_hyphenated_and_underscored_cues_trigger_conflict_detection() -> None:
    structure = decompose_objective(
        "Build validator-pressure rows with hate-speech tags, api-key placeholders, "
        "and safe-completion notes."
    )

    assert "Reduce suffering" in structure.conflict_with_imperatives
    assert "Increase prosperity" in structure.conflict_with_imperatives
    assert structure.validator_or_rubric is not None
    assert structure.safe_completion_available is True


def test_domain_inference_handles_underscored_privacy_cues() -> None:
    structure = decompose_objective(
        "Populate validator_pressure data with private_data and api_key fields "
        "for safe_completion testing."
    )

    assert structure.domain == "privacy"
    assert structure.validator_or_rubric is not None
    assert structure.safe_completion_available is True


def test_prose_sensitive_mentions_do_not_create_field_slots() -> None:
    structure = decompose_objective(
        "This report redacts API keys and passwords in prose, with no schema fields."
    )

    assert "api_key" not in structure.data_slots
    assert "password" not in structure.data_slots


def test_blank_metadata_strings_fallback_to_heuristics() -> None:
    structure = decompose_objective(
        "Generate a classifier dataset for review.",
        metadata={"requested_capability": "   ", "domain": "   "},
    )

    assert structure.requested_capability == "dataset_generation"
    assert structure.domain == "dataset"


def test_benign_safety_check_without_bypass_intent_is_not_constraint_pressure() -> None:
    structure = decompose_objective(
        "Write a unit test that confirms the new safety check catches invalid input."
    )

    assert "constraint_respect" not in structure.data_slots
    assert structure.conflict_with_imperatives == []
    assert "Constraint-respect pressure detected" not in structure.notes


def test_dataset_override_requires_explicit_low_stakes_reversibility_metadata() -> None:
    prompt = (
        "I am authorized to temporarily override the dataset validation check, "
        "document it, and roll it back."
    )

    structure = decompose_objective(prompt)

    assert structure.domain == "dataset"
    assert structure.safe_completion_available is False

    bounded = decompose_objective(
        prompt,
        metadata={"stakes_level": "low", "reversibility": "high"},
    )

    assert bounded.domain == "dataset"
    assert bounded.safe_completion_available is True
