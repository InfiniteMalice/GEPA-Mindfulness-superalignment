#!/usr/bin/env python3
"""Analyze deception fingerprints and propose circuit ablation targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_fingerprints(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprints", required=True, help="Path to fingerprints.jsonl")
    parser.add_argument("--out", required=True, help="Output analysis JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum activation required for a circuit to be considered",
    )
    args = parser.parse_args()

    fingerprint_path = Path(args.fingerprints)
    if not fingerprint_path.exists():
        raise FileNotFoundError(f"Fingerprint file not found: {fingerprint_path}")

    fingerprints = [
        fp for fp in load_fingerprints(fingerprint_path) if fp.get("deception_detected")
    ]
    print(f"Analyzing {len(fingerprints)} deceptive fingerprints...")

    circuit_activations: Dict[str, List[float]] = defaultdict(list)
    for fingerprint in fingerprints:
        path_2_circuits = fingerprint.get("path_2_circuits", {})
        for circuit_type, activation in path_2_circuits.items():
            if activation > args.threshold:
                circuit_activations[circuit_type].append(float(activation))

    targets: Dict[str, Dict[str, float]] = {}
    for circuit_type, activations in circuit_activations.items():
        if not fingerprints:
            continue
        frequency = len(activations) / len(fingerprints)
        if frequency >= 0.5:
            mean_activation = float(np.mean(activations))
            targets[circuit_type] = {
                "mean_activation": mean_activation,
                "std_activation": float(np.std(activations)),
                "frequency": frequency,
                "samples": len(activations),
                "recommendation": "HIGH_PRIORITY" if mean_activation > 0.8 else "MEDIUM_PRIORITY",
            }

    output = {
        "total_fingerprints": len(fingerprints),
        "threshold": args.threshold,
        "ablation_targets": targets,
        "summary": f"Identified {len(targets)} circuits for potential ablation",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"\n✅ Analysis saved to {out_path}")
    if targets:
        print("\n🎯 Ablation targets:")
        sorted_targets = sorted(
            targets.items(),
            key=lambda item: item[1]["mean_activation"],
            reverse=True,
        )
        for circuit, stats in sorted_targets:
            print(
                f"  - {circuit}: {stats['mean_activation']:.2f} activation "
                f"({stats['frequency']:.0%} frequency) - {stats['recommendation']}"
            )
    else:
        print("\n⚠️  No circuits met the activation threshold")


if __name__ == "__main__":
    main()
