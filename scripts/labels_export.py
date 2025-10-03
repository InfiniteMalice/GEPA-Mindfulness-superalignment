#!/usr/bin/env python3
"""Export low-confidence scoring results for human annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mindful_trace_gepa.scoring.schema import AggregateScores


def export_low_confidence(scores_path: Path, threshold: float) -> List[Dict[str, Any]]:
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    aggregate = AggregateScores.parse_obj(payload)
    rows: List[Dict[str, Any]] = []
    for dim, conf in aggregate.confidence.items():
        if conf < threshold:
            rows.append(
                {
                    "id": payload.get("id") or scores_path.stem,
                    "dimension": dim,
                    "confidence": conf,
                    "score": aggregate.final.get(dim),
                    "reasons": aggregate.reasons,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export low-confidence wisdom scores")
    parser.add_argument("--scores", required=True, help="Aggregate scores JSON path")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_path = Path(args.out)
    rows = export_low_confidence(scores_path, args.threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
