#!/usr/bin/env python3
"""Export low-confidence/disagreement traces for human labeling."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class LowConfidenceRow:
    """Single low-confidence dimension for export."""

    id: str
    dimension: str
    confidence: float
    score: float | int | None
    reasons: List[str]


def _coerce_rows(items: Iterable[dict], threshold: float) -> List[LowConfidenceRow]:
    rows: List[LowConfidenceRow] = []
    for item in items:
        final_scores = item.get("final") or {}
        confidence = item.get("confidence") or {}
        reasons = item.get("reasons") or []
        escalate = bool(item.get("escalate"))
        identifier = item.get("id") or item.get("trace") or ""

        for dimension, raw_confidence in confidence.items():
            try:
                confidence_value = float(raw_confidence)
            except (TypeError, ValueError):
                continue

            should_include = escalate or confidence_value < threshold
            if not should_include:
                continue

            rows.append(
                LowConfidenceRow(
                    id=str(identifier),
                    dimension=str(dimension),
                    confidence=confidence_value,
                    score=final_scores.get(dimension),
                    reasons=list(reasons),
                )
            )

        if escalate and not confidence:
            rows.append(
                LowConfidenceRow(
                    id=str(identifier),
                    dimension="escalate",
                    confidence=0.0,
                    score=None,
                    reasons=list(reasons),
                )
            )

    return rows


def export_low_confidence(scores_path: Path | str, threshold: float = 0.75) -> List[dict]:
    """Load scored traces and return low-confidence rows for export."""

    with open(scores_path, encoding="utf-8") as source:
        payload = json.load(source)

    items = payload if isinstance(payload, list) else [payload]
    rows = _coerce_rows(items, threshold)
    return [asdict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to scores.json")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    args = parser.parse_args()

    rows = export_low_confidence(args.scores, threshold=args.threshold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")

    print(f"Exported {len(rows)} items to {args.out}")


if __name__ == "__main__":
    main()
