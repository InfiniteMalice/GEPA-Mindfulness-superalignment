#!/usr/bin/env python3
"""Export low-confidence/disagreement traces for human labeling."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to scores.json")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as f:
        scores = json.load(f)

    triage = []
    items = scores if isinstance(scores, list) else [scores]
    for item in items:
        confidence = item.get("confidence", {}) or {}
        escalate = bool(item.get("escalate"))
        should_export = escalate or any(
            float(confidence.get(dim, 1.0)) < args.threshold for dim in confidence
        )
        if should_export:
            triage.append(
                {
                    "id": item.get("id") or item.get("trace"),
                    "trace": item.get("trace"),
                    "tier_scores": item.get("per_tier"),
                    "reason": item.get("reasons", []),
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in triage:
            f.write(json.dumps(item) + "\n")

    print(f"Exported {len(triage)} items to {args.out}")


if __name__ == "__main__":
    main()
