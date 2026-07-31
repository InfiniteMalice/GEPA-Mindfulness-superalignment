"""Adapt rich synthetic curriculum cases into rollout requests."""

# Standard library
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

# Local
from ..trajectory import RolloutRequest


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
                parsed: object = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
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
