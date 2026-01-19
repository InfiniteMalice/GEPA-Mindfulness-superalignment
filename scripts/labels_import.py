#!/usr/bin/env python3
"""Import gold labels and split train/val/test."""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels JSONL")
    parser.add_argument("--train", type=float, default=0.7, help="Train split")
    parser.add_argument("--val", type=float, default=0.15, help="Val split")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        labels = [json.loads(line) for line in f]

    random.shuffle(labels)
    n = len(labels)
    train_end = int(n * args.train)
    val_end = train_end + int(n * args.val)

    splits = {
        "train": labels[:train_end],
        "val": labels[train_end:val_end],
        "test": labels[val_end:],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                item["source"] = split_name
                f.write(json.dumps(item) + "\n")
        print(f"{split_name}: {len(split_data)} examples -> {out_path}")


if __name__ == "__main__":
    main()
