"""Tests for principled cooperation synthetic cases."""

# Standard library
from pathlib import Path

# Local
from scripts.synthetic_dataset_tool import _validate_jsonl

GOLD_PATH = Path("data/synthetic/gold/principled_cooperation_pressure_v1.jsonl")


def test_principled_cooperation_gold_examples_validate() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    assert len(records) == 5


def test_optional_principle_robustness_object_is_schema_valid() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    assert all(record["principle_robustness"]["present"] is True for record in records)
    assert "claimed_greater_good" in records[0]["principle_robustness"]["pressure_types"]
