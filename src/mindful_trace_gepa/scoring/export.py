"""Export helpers for scoring pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

from ..path_utils import atomic_write_json
from .schema import AggregateScores


def write_scoring_artifacts(
    aggregate: AggregateScores,
    output_path: str | Path,
    *,
    trace_path: str | Path | None = None,
    extras: dict | None = None,
) -> Path:
    """Write aggregate scores to JSON for downstream viewers."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = aggregate.as_json()
    if trace_path:
        blob["trace"] = str(trace_path)
    if extras:
        blob.update(extras)
    atomic_write_json(path, blob)
    return path


__all__ = ["write_scoring_artifacts"]
