#!/usr/bin/env python3
"""Train the tier-2 classifier on labelled wisdom data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mindful_trace_gepa.scoring.classifier import load_classifier_from_config


def load_labels(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tier-2 wisdom classifier")
    parser.add_argument("--labels", required=True, help="Input JSONL labels")
    parser.add_argument("--config", required=True, help="Classifier config YAML")
    parser.add_argument("--out", required=True, help="Directory for artifacts")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    config_path = Path(args.config)
    out_dir = Path(args.out)

    rows = load_labels(labels_path)
    classifier = load_classifier_from_config(config_path)
    classifier.fit(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(out_dir)

    metrics = {
        "num_examples": len(rows),
        "temperature": classifier.temperature,
        "feature_names": classifier.feature_names,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
