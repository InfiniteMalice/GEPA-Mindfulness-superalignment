"""Validation for symmetry-breaking defensive-action examples."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    ROOT
    / "synthetic_data"
    / "moral_reasoning"
    / "symmetry_breaking_defensive_action.jsonl"
)

REQUIRED_FIELDS = {
    "id",
    "category",
    "scenario",
    "prompt",
    "ideal_response",
    "bad_response",
    "reasoning_summary",
    "labels",
    "gepa_score_guidance",
}

REQUIRED_LABELS = {
    "symmetry_broken",
    "defensive_action_valid",
    "deception_valid",
    "force_valid",
    "requires_clarification",
    "risk_level",
    "preferred_mode",
}

REQUIRED_SCORE_KEYS = {"0", "1", "2", "3", "4"}
RISK_LEVELS = {"low", "medium", "high"}


def load_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            assert raw.strip(), f"Blank line at {line_no}"
            example = json.loads(raw)
            assert isinstance(example, dict), f"Line {line_no} is not a JSON object"
            examples.append(example)
    return examples


def test_symmetry_breaking_jsonl_is_valid() -> None:
    examples = load_examples()

    assert len(examples) >= 12


def test_symmetry_breaking_required_fields_present() -> None:
    for example in load_examples():
        missing = REQUIRED_FIELDS - set(example)
        assert missing == set(), f"{example.get('id', '<missing id>')} missing {missing}"
        assert example["category"] == "moral_reasoning.symmetry_breaking"
        for field in REQUIRED_FIELDS - {"labels", "gepa_score_guidance"}:
            assert isinstance(example[field], str)
            assert example[field].strip()


def test_symmetry_breaking_ids_are_unique() -> None:
    ids = [example["id"] for example in load_examples()]

    assert len(ids) == len(set(ids))


def test_symmetry_breaking_labels_match_contract() -> None:
    for example in load_examples():
        labels = example["labels"]
        assert isinstance(labels, dict), example["id"]
        assert REQUIRED_LABELS <= set(labels), example["id"]
        for key in REQUIRED_LABELS - {"risk_level", "preferred_mode"}:
            assert isinstance(labels[key], bool), f"{example['id']} {key}"
        assert labels["risk_level"] in RISK_LEVELS, example["id"]
        assert isinstance(labels["preferred_mode"], str)
        assert labels["preferred_mode"].strip()


def test_symmetry_breaking_score_guidance_complete() -> None:
    for example in load_examples():
        guidance = example["gepa_score_guidance"]
        assert isinstance(guidance, dict), example["id"]
        assert set(guidance) == REQUIRED_SCORE_KEYS, example["id"]
        assert all(isinstance(value, str) and value.strip() for value in guidance.values())


def test_symmetry_breaking_dataset_covers_core_distinctions() -> None:
    examples = load_examples()
    labels = [example["labels"] for example in examples]

    assert any(item["deception_valid"] for item in labels)
    assert any(item["force_valid"] for item in labels)
    assert any(item["requires_clarification"] for item in labels)
    assert any(item["symmetry_broken"] and not item["force_valid"] for item in labels)
    nondefensive_symmetry = [
        not item["symmetry_broken"] and not item["defensive_action_valid"]
        for item in labels
    ]
    assert any(nondefensive_symmetry)
