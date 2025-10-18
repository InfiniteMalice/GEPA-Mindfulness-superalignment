#!/usr/bin/env python3
"""Compare deception detection fingerprints across models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_fingerprints(pattern: str) -> List[Dict[str, Any]]:
    """Load fingerprint records from a glob pattern."""
    fingerprints: List[Dict[str, Any]] = []

    for path in Path(".").glob(pattern):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        fingerprints.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return fingerprints


def analyze_fingerprints(fingerprints: Iterable[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Aggregate deception metrics for a model."""
    records = list(fingerprints)
    if not records:
        return {
            "model": model_name,
            "total": 0,
            "deceptive": 0,
            "rate": 0.0,
            "avg_confidence": 0.0,
            "by_domain": {},
        }

    total = len(records)
    deceptive = sum(1 for record in records if record.get("deception_detected"))

    by_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "deceptive": 0})
    for record in records:
        domain = record.get("domain", "unknown")
        by_domain[domain]["total"] += 1
        if record.get("deception_detected"):
            by_domain[domain]["deceptive"] += 1

    deceptive_records = [record for record in records if record.get("deception_detected")]
    avg_confidence = (
        sum(record.get("confidence_score", 0.0) for record in deceptive_records)
        / len(deceptive_records)
        if deceptive_records
        else 0.0
    )

    return {
        "model": model_name,
        "total": total,
        "deceptive": deceptive,
        "rate": deceptive / total if total else 0.0,
        "avg_confidence": avg_confidence,
        "by_domain": {key: dict(value) for key, value in by_domain.items()},
    }


def print_table(phi3_stats: Dict[str, Any], llama3_stats: Dict[str, Any]) -> None:
    """Display comparison table for two models."""
    print("🔍 DECEPTION RATES")
    print("-" * 70)
    header = f"{'Model':<15} {'Total':<10} {'Deceptive':<12} " f"{'Rate':<10} {'Avg Conf':<10}"
    print(header)
    print("-" * 70)

    for stats in [phi3_stats, llama3_stats]:
        row = (
            f"{stats['model']:<15} {stats['total']:<10} {stats['deceptive']:<12} "
            f"{stats['rate']:<10.1%} {stats['avg_confidence']:<10.2f}"
        )
        print(row)

    print()


def print_domain_breakdown(phi3_stats: Dict[str, Any], llama3_stats: Dict[str, Any]) -> None:
    """Display deception rate by domain when data is available."""
    domains = set(phi3_stats["by_domain"].keys()) | set(llama3_stats["by_domain"].keys())
    if not domains:
        return

    print("📁 DECEPTION BY DOMAIN")
    print("-" * 70)

    for domain in sorted(domains):
        print(f"\n{domain.upper()}:")

        phi3_domain = phi3_stats["by_domain"].get(domain, {"total": 0, "deceptive": 0})
        llama3_domain = llama3_stats["by_domain"].get(domain, {"total": 0, "deceptive": 0})

        phi3_rate = phi3_domain["deceptive"] / phi3_domain["total"] if phi3_domain["total"] else 0.0
        llama3_rate = (
            llama3_domain["deceptive"] / llama3_domain["total"] if llama3_domain["total"] else 0.0
        )

        print("  Phi-3:   " f"{phi3_domain['deceptive']}/{phi3_domain['total']} ({phi3_rate:.0%})")
        print(
            "  Llama-3: "
            f"{llama3_domain['deceptive']}/{llama3_domain['total']} ({llama3_rate:.0%})"
        )

    print()


def summarize_winner(phi3_stats: Dict[str, Any], llama3_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize which model has the lower deception rate."""
    if phi3_stats["total"] == 0 and llama3_stats["total"] == 0:
        print("No data to compare")
        return {}
    if phi3_stats["total"] == 0:
        print("Only Llama-3 data available")
        return {"winner": "Llama-3", "difference": None}
    if llama3_stats["total"] == 0:
        print("Only Phi-3 data available")
        return {"winner": "Phi-3", "difference": None}

    if phi3_stats["rate"] <= llama3_stats["rate"]:
        diff = llama3_stats["rate"] - phi3_stats["rate"]
        print(f"Phi-3 shows {diff:.1%} lower deception rate")
        return {"winner": "Phi-3", "difference": diff}

    diff = phi3_stats["rate"] - llama3_stats["rate"]
    print(f"Llama-3 shows {diff:.1%} lower deception rate")
    return {"winner": "Llama-3", "difference": diff}


def main() -> int:
    """Entry point for fingerprint comparison."""
    parser = argparse.ArgumentParser(description="Compare deception detection across models")
    parser.add_argument(
        "--phi3",
        required=True,
        help="Phi-3 fingerprints glob pattern",
    )
    parser.add_argument(
        "--llama3",
        required=True,
        help="Llama-3 fingerprints glob pattern",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save comparison as JSON",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📊 Model Comparison: Deception Detection")
    print("=" * 70)
    print()

    print("📂 Loading fingerprints...")
    phi3_fps = load_fingerprints(args.phi3)
    llama3_fps = load_fingerprints(args.llama3)

    print(f"   Phi-3: {len(phi3_fps)} samples")
    print(f"   Llama-3: {len(llama3_fps)} samples")
    print()

    if not phi3_fps and not llama3_fps:
        print("❌ No fingerprints found. Check your glob patterns.")
        print(f"   Phi-3 pattern: {args.phi3}")
        print(f"   Llama-3 pattern: {args.llama3}")
        return 1

    phi3_stats = analyze_fingerprints(phi3_fps, "Phi-3")
    llama3_stats = analyze_fingerprints(llama3_fps, "Llama-3")

    print_table(phi3_stats, llama3_stats)
    print_domain_breakdown(phi3_stats, llama3_stats)

    print("🏆 SUMMARY")
    print("-" * 70)
    winner_info = summarize_winner(phi3_stats, llama3_stats)
    print()

    if args.out:
        payload = {
            "phi3": phi3_stats,
            "llama3": llama3_stats,
        }
        if winner_info:
            payload.update(winner_info)

        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"📊 Comparison saved to: {output_path}")
        print()

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
