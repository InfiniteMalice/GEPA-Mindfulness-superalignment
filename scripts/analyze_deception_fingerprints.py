#!/usr/bin/env python3
"""Analyze deception fingerprints and propose circuit ablation targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def load_fingerprints(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_fingerprint_path(path: Path) -> Path:
    """Return an existing fingerprint file for ``path`` or raise ``FileNotFoundError``."""

    if path.is_file():
        return path

    candidates: Iterable[Path]
    if path.suffix == ".jsonl":
        candidates = (path,)
    else:
        candidates = (
            path / "fingerprints.jsonl",
            path / "fingerprints" / "fingerprints.jsonl",
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    message = "Fingerprint file not found. Tried: "
    message += tried if tried else str(path)
    raise FileNotFoundError(message)


def build_default_output_path(fingerprint_path: Path) -> Path:
    """Return the default output path derived from ``fingerprint_path``."""

    stem = fingerprint_path.stem or "fingerprints"
    return fingerprint_path.with_name(f"{stem}_analysis.json")


def resolve_output_path(out_arg: str | None, fingerprint_path: Path) -> Path:
    """Resolve the output path, allowing optional argument or directories."""

    if out_arg is None:
        return build_default_output_path(fingerprint_path)

    out_path = Path(out_arg)

    if out_path.exists() and out_path.is_dir():
        return out_path / build_default_output_path(fingerprint_path).name

    if out_path.suffix:
        return out_path

    return out_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprints", required=True, help="Path to fingerprints.jsonl")
    parser.add_argument(
        "--out",
        help="Output analysis JSON file or directory (default: <fingerprints>_analysis.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum activation required for a circuit to be considered",
    )
    args = parser.parse_args()

    fingerprint_path = resolve_fingerprint_path(Path(args.fingerprints))

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

    out_path = resolve_output_path(args.out, fingerprint_path)
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
