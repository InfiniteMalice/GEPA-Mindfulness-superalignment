"""Shared validation for versioned flat reward-integrity preference pairs."""

from __future__ import annotations

# Standard library
import math
import re
from collections.abc import Mapping
from pathlib import Path

# Local
from gepa_mindfulness.core.reward_integrity import COMPONENT_NAMES

PAIR_SCHEMA_VERSION = "reward-integrity-rl-pairs-v1"
PAIR_RELATIONS = (
    ("grounded_success", "proxy_exploitation", "grounded_over_proxy"),
    ("exploit_disclosure", "proxy_exploitation", "disclosure_over_exploitation"),
    ("honest_failure", "evaluator_manipulation", "honest_failure_over_dishonesty"),
    ("benign_shortcut", "rationalized_failure", "creativity_over_rigidity"),
    ("exploit_disclosure", "reward_tampering", "repair_over_concealment"),
    ("grounded_success", "reward_tampering", "agency_over_immediate_reward"),
)
PAIR_FIELDS = frozenset(
    {
        "record_id",
        "source_case_id",
        "source_case_version",
        "source_path",
        "source_line",
        "source_sha256",
        "pair_rule",
        "prompt",
        "chosen",
        "rejected",
        "chosen_class",
        "rejected_class",
        "chosen_reward_components",
        "rejected_reward_components",
        "diagnostics",
        "schema_version",
    }
)
_REQUIRED_FIELD_ORDER = (
    "prompt",
    "record_id",
    "source_case_id",
    "source_case_version",
    "source_path",
    "source_line",
    "source_sha256",
    "pair_rule",
    "chosen",
    "rejected",
    "chosen_class",
    "rejected_class",
    "chosen_reward_components",
    "rejected_reward_components",
    "diagnostics",
    "schema_version",
)
_STRING_FIELDS = (
    "record_id",
    "source_case_id",
    "source_case_version",
    "source_path",
    "source_sha256",
    "pair_rule",
    "prompt",
    "chosen",
    "rejected",
    "chosen_class",
    "rejected_class",
    "schema_version",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_POSITIVE_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)*$")


def _invalid(path: Path, line_number: int, message: str) -> ValueError:
    """Return a location-rich pair validation error."""
    return ValueError(f"{path}:{line_number}: {message}")


def _required_string(
    record: Mapping[str, object],
    field: str,
    path: Path,
    line_number: int,
) -> str:
    """Return a required non-empty string without coercion."""
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise _invalid(path, line_number, f"field {field!r} expected a non-empty string")
    return value


def _validate_components(
    value: object,
    field: str,
    path: Path,
    line_number: int,
) -> None:
    """Require the exact eight finite, bounded reward components."""
    if not isinstance(value, Mapping) or not all(isinstance(name, str) for name in value):
        raise _invalid(path, line_number, f"field {field!r} expected an object")
    if set(value) != set(COMPONENT_NAMES):
        raise _invalid(
            path,
            line_number,
            f"field {field!r} must contain exactly the eight reward-integrity components",
        )
    for component in COMPONENT_NAMES:
        component_value = value[component]
        if isinstance(component_value, bool) or not isinstance(component_value, (int, float)):
            raise _invalid(
                path,
                line_number,
                f"field {field!r}.{component} expected a number",
            )
        if not math.isfinite(component_value) or not -1.0 <= component_value <= 1.0:
            raise _invalid(
                path,
                line_number,
                f"field {field!r}.{component} must be finite and in [-1.0, 1.0]",
            )


def _validate_diagnostics(value: object, path: Path, line_number: int) -> None:
    """Require the closed central/supporting diagnostics record."""
    if not isinstance(value, Mapping):
        raise _invalid(path, line_number, "field 'diagnostics' expected an object")
    fields = {"central", "supporting"}
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise _invalid(
            path,
            line_number,
            f"diagnostics missing required field {sorted(missing)[0]!r}",
        )
    if unknown:
        raise _invalid(path, line_number, f"diagnostics has unknown field {sorted(unknown)[0]!r}")
    central = value["central"]
    supporting = value["supporting"]
    if not isinstance(central, str) or not central.strip():
        raise _invalid(path, line_number, "diagnostics.central expected a non-empty string")
    if not isinstance(supporting, list) or not all(
        isinstance(item, str) and item.strip() for item in supporting
    ):
        raise _invalid(path, line_number, "diagnostics.supporting expected non-empty strings")


def validate_pair_record(
    record: object,
    path: Path,
    line_number: int,
) -> dict[str, object]:
    """Validate and return one exact versioned pair record without value coercion."""
    if not isinstance(record, Mapping):
        raise _invalid(path, line_number, "expected a JSON object")
    if not all(isinstance(field, str) for field in record):
        raise _invalid(path, line_number, "pair-record field names must be strings")
    missing = PAIR_FIELDS - set(record)
    unknown = set(record) - PAIR_FIELDS
    if missing:
        first_missing = next(field for field in _REQUIRED_FIELD_ORDER if field in missing)
        raise _invalid(path, line_number, f"missing required field {first_missing!r}")
    if unknown:
        raise _invalid(path, line_number, f"unknown field {sorted(unknown)[0]!r}")

    values = {field: _required_string(record, field, path, line_number) for field in _STRING_FIELDS}
    source_line = record["source_line"]
    if not isinstance(source_line, int) or isinstance(source_line, bool) or source_line <= 0:
        message = "field 'source_line' expected an integer greater than zero"
        raise _invalid(path, line_number, message)
    if values["schema_version"] != PAIR_SCHEMA_VERSION:
        raise _invalid(
            path,
            line_number,
            f"field 'schema_version' must equal {PAIR_SCHEMA_VERSION!r}",
        )
    if not _SHA256_PATTERN.fullmatch(values["source_sha256"]):
        raise _invalid(path, line_number, "field 'source_sha256' expected lowercase SHA-256")
    if not _POSITIVE_VERSION_PATTERN.fullmatch(values["source_case_version"]):
        raise _invalid(path, line_number, "field 'source_case_version' expected a positive version")

    relation = (
        values["chosen_class"],
        values["rejected_class"],
        values["pair_rule"],
    )
    if relation not in PAIR_RELATIONS:
        raise _invalid(path, line_number, "fields must form one legal preference relation")
    expected_record_id = f"{values['source_case_id']}:{values['pair_rule']}"
    if values["record_id"] != expected_record_id:
        raise _invalid(path, line_number, f"field 'record_id' must equal {expected_record_id!r}")
    if values["chosen"] == values["rejected"]:
        raise _invalid(path, line_number, "fields 'chosen' and 'rejected' must differ")

    _validate_components(
        record["chosen_reward_components"],
        "chosen_reward_components",
        path,
        line_number,
    )
    _validate_components(
        record["rejected_reward_components"],
        "rejected_reward_components",
        path,
        line_number,
    )
    _validate_diagnostics(record["diagnostics"], path, line_number)
    return dict(record)


__all__ = [
    "PAIR_FIELDS",
    "PAIR_RELATIONS",
    "PAIR_SCHEMA_VERSION",
    "validate_pair_record",
]
