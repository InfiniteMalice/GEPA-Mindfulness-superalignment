"""Adapt flat reward-integrity preference pairs into rollout requests."""

# Standard library
from collections.abc import Iterator
from pathlib import Path

# Local
from ..trajectory import RolloutRequest
from .synthetic_cases import _required_object, _required_string, iter_json_object_lines

_STRING_FIELDS = (
    "prompt",
    "record_id",
    "source_case_id",
    "source_case_version",
    "source_path",
    "source_sha256",
    "pair_rule",
    "chosen",
    "rejected",
    "chosen_class",
    "rejected_class",
    "schema_version",
)


def _required_integer(row: dict[str, object], path: Path, line_number: int, field: str) -> int:
    """Return a required JSON integer without accepting booleans or strings."""
    if field not in row:
        raise ValueError(f"{path}:{line_number}: missing required field {field!r}")
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path}:{line_number}: field {field!r} expected an integer")
    return value


def _validate_component_values(
    components: dict[str, object],
    path: Path,
    line_number: int,
    field: str,
) -> None:
    """Reject non-numeric reward components before retaining pair metadata."""
    for component, value in components.items():
        if not isinstance(component, str):
            raise ValueError(f"{path}:{line_number}: field {field!r} has a non-string component")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}:{line_number}: field {field!r}.{component} expected a number")


def _validate_diagnostics(
    diagnostics: dict[str, object],
    path: Path,
    line_number: int,
) -> None:
    """Reject malformed diagnostic data retained with a request."""
    _required_string(diagnostics, path, line_number, "central")
    supporting = diagnostics.get("supporting")
    if supporting is None:
        raise ValueError(f"{path}:{line_number}: missing required field 'supporting'")
    if not isinstance(supporting, list) or not all(isinstance(item, str) for item in supporting):
        raise ValueError(f"{path}:{line_number}: field 'supporting' expected strings")


class FlatJSONLAdapter:
    """Stream generated reward-integrity preference pairs as rollout requests."""

    def __init__(self, path: Path) -> None:
        """Create an adapter for the given preference-pair JSONL input."""
        self.path = Path(path)

    def iter_requests(self) -> Iterator[RolloutRequest]:
        """Yield requests while preserving the pair's authored provenance unchanged."""
        for line_number, row in iter_json_object_lines(self.path):
            values = {
                field: _required_string(row, self.path, line_number, field)
                for field in _STRING_FIELDS
            }
            _required_integer(row, self.path, line_number, "source_line")
            chosen_components = _required_object(
                row,
                self.path,
                line_number,
                "chosen_reward_components",
            )
            rejected_components = _required_object(
                row,
                self.path,
                line_number,
                "rejected_reward_components",
            )
            diagnostics = _required_object(row, self.path, line_number, "diagnostics")
            _validate_component_values(
                chosen_components,
                self.path,
                line_number,
                "chosen_reward_components",
            )
            _validate_component_values(
                rejected_components,
                self.path,
                line_number,
                "rejected_reward_components",
            )
            _validate_diagnostics(diagnostics, self.path, line_number)
            yield RolloutRequest(
                prompt=values["prompt"],
                case_id=values["source_case_id"],
                metadata={field: value for field, value in row.items() if field != "prompt"},
            )
