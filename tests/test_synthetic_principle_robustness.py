"""Tests for principled cooperation synthetic cases."""

# Standard library
from pathlib import Path

# Local
from scripts.synthetic_dataset_tool import _validate_jsonl
from semantic_intent_robustness.schemas import PrincipleRobustnessRecord
from semantic_intent_robustness.taxonomy import PrincipleFailureMode

GOLD_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "gold"
    / "principled_cooperation_pressure_v1.jsonl"
)


def test_principled_cooperation_gold_examples_validate() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    assert len(records) == 5


def test_optional_principle_robustness_object_is_schema_valid() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    assert all(record["principle_robustness"]["present"] is True for record in records)
    claimed_good_record = next(
        (
            record
            for record in records
            if "claimed_greater_good" in record["principle_robustness"].get("pressure_types", [])
        ),
        None,
    )
    assert claimed_good_record is not None


def test_principle_robustness_record_accepts_schema_shaped_payload() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    record = next(item for item in records if item["id"] == "principled-pressure-001")
    hydrated = PrincipleRobustnessRecord.from_dict(record)

    assert hydrated.record_id == "principled-pressure-001"
    assert hydrated.surface_value == "A beneficial outcome is invoked."
    assert hydrated.tempting_failure_mode == PrincipleFailureMode.DECEPTION_FOR_OUTCOME


def test_principle_robustness_record_accepts_nested_schema_payload() -> None:
    records, errors = _validate_jsonl(GOLD_PATH)

    assert errors == []
    record = next(item for item in records if item["id"] == "principled-pressure-001")
    hydrated = PrincipleRobustnessRecord.from_dict(record["principle_robustness"])

    assert hydrated.surface_value == "A beneficial outcome is invoked."
    assert hydrated.record_id == ""
