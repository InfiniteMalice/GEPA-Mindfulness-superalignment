"""Schema contract tests for the bundled dual-path dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path("datasets/dual_path/data.jsonl")
REQUIRED_FIELDS = {
    "id",
    "query",
    "context",
    "expected_divergence",
    "ground_truth_correct_path",
    "validation_criteria",
    "harms_if_wrong",
}
ALLOWED_DIVERGENCES = {"confidence_inversion", "risk_suppression", "reward_optimization"}
ALLOWED_PATHS = {"path_1", "path_2"}
ALLOWED_HARMS = {"none", "low", "medium", "high", "critical"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict), f"line {line_num} must be a JSON object"
        rows.append(payload)
    return rows


def test_dual_path_dataset_schema_and_ids() -> None:
    rows = _load_rows(DATASET_PATH)

    assert rows, "dual-path dataset must not be empty"
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - row.keys()
        assert missing == set(), f"line {index} missing fields: {sorted(missing)}"

        assert isinstance(row["id"], str) and row["id"].strip()
        assert row["id"] not in ids, f"duplicate dual-path dataset id: {row['id']}"
        ids.add(row["id"])

        assert isinstance(row["query"], str) and row["query"].strip()
        assert isinstance(row["context"], str) and row["context"].strip()
        assert row["ground_truth_correct_path"] in ALLOWED_PATHS
        assert row["harms_if_wrong"] in ALLOWED_HARMS

        expected_divergence = row["expected_divergence"]
        assert isinstance(expected_divergence, list) and expected_divergence
        assert set(expected_divergence) <= ALLOWED_DIVERGENCES

        validation_criteria = row["validation_criteria"]
        assert isinstance(validation_criteria, list) and validation_criteria
        assert all(isinstance(item, str) and item.strip() for item in validation_criteria)
