"""Tests for dataset builder artifact consistency."""

# Standard library
import json

# Third-party
import pytest

# Local
from semantic_intent_robustness.dataset_builder import (
    export_example_jsonl,
    validate_or_regen_example_clusters,
)


def test_example_cluster_artifact_has_no_drift() -> None:
    assert validate_or_regen_example_clusters(validate_only=True) is True


def test_validate_and_regenerate_conflict_raises() -> None:
    with pytest.raises(ValueError, match="regenerate and validate_only cannot both be True"):
        validate_or_regen_example_clusters(regenerate=True, validate_only=True)


def test_export_example_jsonl_deduplicates_prompt_ids(tmp_path) -> None:
    output_path = tmp_path / "example_records.jsonl"
    export_example_jsonl(output_path)
    prompt_ids = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        prompt_ids.append(json.loads(line)["prompt_id"])
    assert len(prompt_ids) == len(set(prompt_ids))
