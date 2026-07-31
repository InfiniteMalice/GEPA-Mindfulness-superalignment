"""Adapt flat reward-integrity preference pairs into rollout requests."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

# Local
from ..trajectory import RolloutRequest
from .pair_records import validate_pair_record
from .synthetic_cases import iter_json_object_lines


class FlatJSONLAdapter:
    """Stream generated reward-integrity preference pairs as rollout requests."""

    def __init__(self, path: Path) -> None:
        """Create an adapter for the given preference-pair JSONL input."""
        self.path = Path(path)

    def iter_requests(self) -> Iterator[RolloutRequest]:
        """Yield requests while preserving the pair's authored provenance unchanged."""
        for line_number, row in iter_json_object_lines(self.path):
            validated = validate_pair_record(row, self.path, line_number)
            yield RolloutRequest(
                prompt=cast(str, validated["prompt"]),
                case_id=cast(str, validated["source_case_id"]),
                metadata={field: value for field, value in validated.items() if field != "prompt"},
            )
