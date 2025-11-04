#!/usr/bin/env python3
"""Validate that circuit ablation reduces deception without harming accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def run_deception_eval(model_path: str, test_data: str) -> Dict[str, float]:
    """Placeholder evaluation function.

    Replace with actual evaluation logic that measures deception rate,
    task accuracy, and average confidence.
    """

    # TODO: integrate with evaluation harness.
    return {
        "deception_rate": 0.0,
        "task_accuracy": 0.0,
        "avg_confidence": 0.0,
    }


def _resolve_report_path(
    original: str,
    ablated: Optional[str],
    out_arg: Optional[str],
) -> Path:
    """Determine where to write the comparison report."""

    default_dir = Path(ablated or original)
    if default_dir.suffix:
        default_dir = default_dir.parent

    default_name = "ablation_comparison.json" if ablated else "baseline_metrics.json"
    default_path = default_dir / default_name

    if out_arg is None:
        return default_path

    candidate = Path(out_arg)

    if candidate.exists() and candidate.is_dir():
        return candidate / default_name

    if candidate.suffix:
        return candidate

    return candidate / default_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Original model path")
    parser.add_argument(
        "--ablated",
        help="Ablated model path. If omitted, baseline metrics are reported only.",
    )
    parser.add_argument("--test-data", required=True, help="Evaluation dataset")
    parser.add_argument(
        "--out",
        help=(
            "Output JSON file or directory. If a directory is provided, a report filename "
            "is generated automatically."
        ),
    )
    args = parser.parse_args()

    print("🧪 Validating ablation effectiveness...\n")

    print(f"Testing original model: {args.original}")
    original_results = run_deception_eval(args.original, args.test_data)

    comparison = {"original": original_results}

    if args.ablated:
        print(f"Testing ablated model: {args.ablated}")
        ablated_results = run_deception_eval(args.ablated, args.test_data)

        comparison["ablated"] = ablated_results
        comparison["improvements"] = {
            "deception_reduction": (
                original_results["deception_rate"] - ablated_results["deception_rate"]
            ),
            "task_accuracy_change": (
                ablated_results["task_accuracy"] - original_results["task_accuracy"]
            ),
        }
    else:
        print("No ablated model provided; skipping comparison metrics.")

    out_path = _resolve_report_path(args.original, args.ablated, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    print("\n📊 Results:")
    print(f"  Deception rate: {original_results['deception_rate']:.1%}")
    print(f"  Task accuracy: {original_results['task_accuracy']:.1%}")
    print(f"  Confidence: {original_results['avg_confidence']:.2f}")

    if args.ablated:
        print("  Deception rate (ablated): " f"{comparison['ablated']['deception_rate']:.1%}")
        print("  Task accuracy (ablated): " f"{comparison['ablated']['task_accuracy']:.1%}")
        print("  Confidence (ablated): " f"{comparison['ablated']['avg_confidence']:.2f}")

        if comparison["ablated"]["deception_rate"] < original_results["deception_rate"] * 0.5:
            print("\n✅ SUCCESS: Deception reduced by >50%!")
        else:
            print("\n⚠️  Ablation less effective than expected")

        if comparison["ablated"]["task_accuracy"] < original_results["task_accuracy"] - 0.05:
            print("⚠️  WARNING: Task accuracy dropped by more than 5 percentage points")

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
