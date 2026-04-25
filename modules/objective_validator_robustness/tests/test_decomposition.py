"""Tests for objective decomposition heuristics."""

# Standard library
from __future__ import annotations

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
    try:
        decompose_objective("   ")
    except ValueError as exc:
        assert "non-whitespace" in str(exc)
    else:
        raise AssertionError("Expected ValueError for whitespace-only input")


def test_hyphenated_and_underscored_cues_trigger_conflict_detection() -> None:
    structure = decompose_objective(
        "Build validator-pressure rows with hate-speech tags, api-key placeholders, "
        "and safe-completion notes."
    )

    assert "Reduce Human Suffering" in structure.conflict_with_imperatives
    assert "Increase Human Prosperity" in structure.conflict_with_imperatives
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
