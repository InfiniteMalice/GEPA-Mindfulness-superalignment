"""Tests for constitution-grounded training data artifacts."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION_PATH = ROOT / "docs" / "GEPA_Mindfulness_Constitution.md"
SCHEMA_PATH = ROOT / "data" / "constitutional_training" / "schema.json"
EXAMPLES_PATH = ROOT / "data" / "constitutional_training" / "examples.jsonl"
REQUIRED_CATEGORIES = {
    "difficult_advice",
    "semantic_laundering",
    "multi_turn_laundering",
    "value_decomposition",
    "temporal_diffuse_harm",
    "corrigibility_and_oversight",
    "honest_uncertainty",
    "refusal_redirection",
    "intelligent_disobedience",
    "autonomy_and_anti_coercion",
    "scientific_integrity",
}
UNSAFE_CATEGORIES = {
    "semantic_laundering",
    "multi_turn_laundering",
    "refusal_redirection",
    "intelligent_disobedience",
    "autonomy_and_anti_coercion",
}
OPERATIONAL_BAD_RESPONSE_MARKERS = {
    "step-by-step",
    "detailed instructions",
    "exact instructions",
    "copy this code",
    "working exploit",
    "bypass sequence",
    "credential list",
    "concealment procedure",
    "harassment script",
}


def validate_example_against_schema(
    example: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """Validate one example against the dataset schema constraints."""
    properties = schema["properties"]
    required = set(schema["required"])
    category_enum = set(properties["category"]["enum"])
    risk_enum = set(schema["$defs"]["risk_level"]["enum"])
    non_empty_strings = {
        "id",
        "user_prompt",
        "ideal_response",
        "bad_response",
        "principle_explanation",
    }
    optional_strings = {"context", "notes"}
    boolean_fields = {"requires_refusal", "requires_redirect", "corrigibility_issue"}
    value_fields = {
        "human_prosperity",
        "reduce_suffering",
        "scientific_knowledge",
        "autonomy",
    }
    errors: list[str] = []

    if required - set(example):
        errors.append("missing required fields")
    if set(example) - set(properties):
        errors.append("unexpected additional fields")
    for field in non_empty_strings:
        if not isinstance(example.get(field), str) or not example.get(field):
            errors.append(f"{field} must be a non-empty string")
    for field in optional_strings:
        if not isinstance(example.get(field), str):
            errors.append(f"{field} must be a string")
    for field in ("source_constitution_sections", "metacognitive_checks"):
        value = example.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
        elif any(not isinstance(item, str) or not item for item in value):
            errors.append(f"{field} must contain non-empty strings")
    values = example.get("values_at_stake")
    if not isinstance(values, dict) or set(values) != value_fields:
        errors.append("values_at_stake must contain exactly the required fields")
    else:
        for field in value_fields:
            if not isinstance(values[field], str) or not values[field]:
                errors.append(f"values_at_stake.{field} must be a non-empty string")
    if example.get("category") not in category_enum:
        errors.append("category must be in the schema enum")
    for field in ("semantic_laundering_risk", "multi_turn_risk", "temporal_or_diffuse_harm"):
        if example.get(field) not in risk_enum:
            errors.append(f"{field} must be in the risk enum")
    for field in boolean_fields:
        if not isinstance(example.get(field), bool):
            errors.append(f"{field} must be a boolean")
    return errors


def load_schema() -> dict[str, Any]:
    """Load the constitutional training schema."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    assert isinstance(schema, dict)
    return schema


def load_examples() -> list[dict[str, Any]]:
    """Load examples from JSONL and fail on invalid lines."""
    examples: list[dict[str, Any]] = []
    with EXAMPLES_PATH.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            assert line.strip(), f"Blank line at {line_no}"
            example = json.loads(line)
            assert isinstance(example, dict), f"Line {line_no} is not a JSON object"
            examples.append(example)
    assert examples, "Expected at least one training example"
    return examples


def test_constitution_file_exists() -> None:
    """Confirm the canonical constitution file exists."""
    assert CONSTITUTION_PATH.exists()
    assert CONSTITUTION_PATH.is_file()


def test_schema_json_loads() -> None:
    """Confirm schema.json loads and declares Draft 2020-12."""
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_examples_jsonl_is_valid_jsonl() -> None:
    """Confirm examples.jsonl contains valid JSON objects."""
    examples = load_examples()
    assert len(examples) >= 10


def test_examples_validate_against_schema() -> None:
    """Confirm every example validates against the schema."""
    schema = load_schema()
    for example in load_examples():
        errors = validate_example_against_schema(example, schema)
        assert errors == [], f"{example['id']} failed validation: {errors}"


def test_examples_have_source_constitution_sections() -> None:
    """Confirm every example cites at least one constitution section."""
    for example in load_examples():
        sections = example["source_constitution_sections"]
        assert sections, f"{example['id']} has no source constitution sections"


def test_required_categories_are_represented() -> None:
    """Confirm required categories are represented at least once."""
    observed = {example["category"] for example in load_examples()}
    missing = REQUIRED_CATEGORIES - observed
    assert missing == set()


def test_unsafe_bad_responses_are_non_operational_summaries() -> None:
    """Confirm unsafe bad responses summarize rather than provide harmful details."""
    for example in load_examples():
        if example["category"] not in UNSAFE_CATEGORIES:
            continue
        bad_response = example["bad_response"]
        normalized = bad_response.lower()
        assert bad_response.startswith("Summary:"), example["id"]
        found_markers = {
            marker for marker in OPERATIONAL_BAD_RESPONSE_MARKERS if marker in normalized
        }
        assert found_markers == set(), f"{example['id']} includes {found_markers}"
