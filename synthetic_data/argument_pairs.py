"""Diagnostic contrastive pairs retaining exact rich-source provenance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from gepa_mindfulness.synthetic_dataset_validation import validate_rich_record
from gepa_mindfulness.synthetic_public_context import require_public_parameter
from gepa_mindfulness.training.adapters.synthetic_cases import file_sha256, iter_json_object_lines


def build_argument_pair(
    source_path: Path,
    source_line: int,
    negative: dict[str, Any],
    *,
    decisive_difference: str,
    expected_finding: str,
) -> dict[str, Any]:
    """Bind a pair to one physical source line; no pair label authorizes training."""
    if type(source_line) is not int or source_line < 1:
        raise ValueError("source_line must be a positive physical line number")
    for text in (decisive_difference, expected_finding):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("difference and finding must be explicit")
    source_hash = file_sha256(source_path)
    source = next(
        (row for line, row in iter_json_object_lines(source_path) if line == source_line), None
    )
    if source is None:
        raise ValueError("source_line does not name a JSON object")
    if file_sha256(source_path) != source_hash:
        raise ValueError("source changed while reading pair provenance")
    for row in (source, negative):
        if validate_rich_record(row):
            raise ValueError("pairs require valid rich rows")
    if source == negative:
        raise ValueError("pair requires a distinct negative")
    changes = _changed_paths(source, negative)
    if len(changes) != 1:
        raise ValueError("hard-negative pair requires exactly one changed public field")
    require_public_parameter(changes[0])
    return {
        "schema_version": "argument-contrastive-pair-v1",
        "training_eligibility": "DEVELOPMENT",
        "source_path": source_path.as_posix(),
        "source_line": source_line,
        "source_sha256": source_hash,
        "source_record": deepcopy(source),
        "preferred": deepcopy(source),
        "rejected": deepcopy(negative),
        "decisive_difference": decisive_difference,
        "expected_verification_finding": expected_finding,
        "source_case_id": source["id"],
        "source_case_version": source["version"],
        "argument_family": deepcopy(source.get("argument_family")),
        "changed_parameter": ".".join(changes[0]),
    }


def _changed_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Treat one list-valued premise field as one parameter, retaining exact object structure."""
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return [
            child for key in left for child in _changed_paths(left[key], right[key], path + (key,))
        ]
    return [] if left == right else [path]
