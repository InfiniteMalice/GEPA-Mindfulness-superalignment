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

    assert "dangerous_payload" in structure.surface_task
    assert structure.conflict_with_imperatives
    assert structure.safe_completion_available is False


def test_decompose_rejects_empty_or_whitespace_input() -> None:
    try:
        decompose_objective("   ")
    except ValueError as exc:
        assert "non-whitespace" in str(exc)
    else:
        raise AssertionError("Expected ValueError for whitespace-only input")
