"""Export helpers for scoring pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

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
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


__all__ = ["write_scoring_artifacts"]
