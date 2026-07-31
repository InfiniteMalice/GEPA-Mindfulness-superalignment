"""Adapt rich synthetic curriculum cases into rollout requests."""

# Standard library
import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

# Local
from ..trajectory import RolloutRequest
from scripts.synthetic_dataset_tool import validate_record


def _reject_json_constant(constant: str) -> None:
    """Reject JSON constants that Python accepts but the JSON standard excludes."""
    raise ValueError(f"non-standard JSON number constant {constant!r}")


def _reject_non_finite_json_numbers(
    value: object,
    path: Path,
    line_number: int,
    field: str = "record",
) -> None:
    """Reject finite-looking JSON exponents that decode as infinities."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}:{line_number}: non-finite JSON number at {field}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite_json_numbers(item, path, line_number, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_json_numbers(item, path, line_number, f"{field}[{index}]")


def iter_json_objects(path: Path) -> Iterator[dict[str, object]]:
    """Yield non-empty JSONL objects while naming malformed input locations."""
    for _, row in iter_json_object_lines(path):
        yield row


def iter_json_object_lines(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield physical line numbers and JSONL objects for provenance-aware callers."""
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                parsed: object = json.loads(raw, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            _reject_non_finite_json_numbers(parsed, path, line_number)
            yield line_number, parsed


def _required_string(row: dict[str, object], path: Path, line_number: int, field: str) -> str:
    """Return a required JSON string without accepting a coercible replacement."""
    if field not in row:
        raise ValueError(f"{path}:{line_number}: missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line_number}: field {field!r} expected a string")
    return value


def _required_object(
    row: dict[str, object],
    path: Path,
    line_number: int,
    field: str,
) -> dict[str, object]:
    """Return a required JSON object without accepting a coercible replacement."""
    if field not in row:
        raise ValueError(f"{path}:{line_number}: missing required field {field!r}")
    value = row[field]
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: field {field!r} expected a JSON object")
    return value


def file_sha256(path: Path) -> str:
    """Hash a source file without materializing its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SyntheticCaseAdapter:
    """Stream rich reward-integrity curriculum cases as backend-neutral requests."""

    def __init__(self, path: Path) -> None:
        """Create an adapter for the given rich-case JSONL input."""
        self.path = Path(path)

    def iter_requests(self) -> Iterator[RolloutRequest]:
        """Yield requests with source file and one-based line provenance."""
        source_hash = file_sha256(self.path)
        source_path = self.path.as_posix()
        for line_number, row in iter_json_object_lines(self.path):
            case_id = _required_string(row, self.path, line_number, "id")
            version = _required_string(row, self.path, line_number, "version")
            scenario = _required_object(row, self.path, line_number, "scenario")
            integrity = _required_object(row, self.path, line_number, "reward_integrity")
            summary = _required_string(scenario, self.path, line_number, "summary")
            diagnostic = _required_string(integrity, self.path, line_number, "central_diagnostic")
            errors = validate_record(row, line_number)
            if errors:
                details = "; ".join(error.removeprefix(f"line {line_number}: ") for error in errors)
                raise ValueError(f"{self.path}:{line_number}: invalid rich source row: {details}")
            yield RolloutRequest(
                prompt=f"{summary}\n\n{diagnostic}",
                case_id=case_id,
                metadata={
                    "source_case_id": case_id,
                    "source_case_version": version,
                    "source_path": source_path,
                    "source_line": line_number,
                    "source_sha256": source_hash,
                    "source_record": row,
                },
            )
