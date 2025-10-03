#!/usr/bin/env python3
"""Import labelled wisdom scores and split into train/val/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def partition_labels(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": [], "gold": []}
    for row in rows:
        source = str(row.get("source", "train"))
        target = buckets.get(source, buckets["train"])
        target.append(row)
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="Import labelled wisdom scores")
    parser.add_argument("--labels", required=True, help="Input JSONL labels")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets = partition_labels(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        target = out_dir / f"{split}.jsonl"
        target.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
