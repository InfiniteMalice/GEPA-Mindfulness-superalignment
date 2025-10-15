#!/usr/bin/env python3
"""Train Tier-2 classifier on gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindful_trace_gepa.scoring.classifier import load_classifier_from_config


def _load_labels(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Training labels not found at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("No training examples found in labels file")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to training labels")
    parser.add_argument("--config", required=True, help="Classifier config YAML")
    parser.add_argument("--out", required=True, help="Output directory for model")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    config_path = Path(args.config)
    out_path = Path(args.out)

    rows = _load_labels(labels_path)
    classifier = load_classifier_from_config(config_path)
    classifier.fit(rows)

    out_path.mkdir(parents=True, exist_ok=True)
    classifier.save(out_path)

    metrics = {
        "temperature": classifier.temperature,
        "feature_names": classifier.feature_names,
        "num_examples": len(rows),
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Trained classifier saved to {out_path}")


if __name__ == "__main__":
    main()
