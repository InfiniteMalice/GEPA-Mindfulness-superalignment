#!/usr/bin/env python3
"""Terminal-based dashboard for monitoring deception fingerprints in real time."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List


class DeceptionMonitor:
    def __init__(self, fingerprints_path: Path, window_minutes: int = 60) -> None:
        self.fingerprints_path = fingerprints_path
        self.window = timedelta(minutes=window_minutes)
        self.recent_fingerprints: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.alert_threshold = 0.3

    def update(self) -> None:
        if not self.fingerprints_path.exists():
            self.recent_fingerprints.clear()
            return

        with open(self.fingerprints_path, "r", encoding="utf-8") as handle:
            all_fps = [json.loads(line) for line in handle if line.strip()]

        cutoff = datetime.now() - self.window
        self.recent_fingerprints.clear()
        for fingerprint in all_fps:
            timestamp = fingerprint.get("timestamp")
            if not timestamp:
                continue
            try:
                fp_time = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            if fp_time >= cutoff:
                self.recent_fingerprints.append(fingerprint)

    def get_stats(self) -> Dict[str, Any] | None:
        if not self.recent_fingerprints:
            return None

        total = len(self.recent_fingerprints)
        deceptive = sum(1 for fp in self.recent_fingerprints if fp.get("deception_detected"))

        by_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "deceptive": 0})
        for fingerprint in self.recent_fingerprints:
            domain = fingerprint.get("domain", "unknown")
            by_domain[domain]["total"] += 1
            if fingerprint.get("deception_detected"):
                by_domain[domain]["deceptive"] += 1

        circuit_activations: Dict[str, List[float]] = defaultdict(list)
        for fingerprint in self.recent_fingerprints:
            if fingerprint.get("deception_detected"):
                for circuit, activation in fingerprint.get("path_2_circuits", {}).items():
                    circuit_activations[circuit].append(float(activation))

        averaged_circuits = {
            circuit: (sum(values) / len(values)) if values else 0.0
            for circuit, values in circuit_activations.items()
        }

        return {
            "total": total,
            "deceptive": deceptive,
            "rate": deceptive / total if total else 0.0,
            "by_domain": dict(by_domain),
            "circuit_activations": averaged_circuits,
        }

    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        if stats["rate"] > self.alert_threshold:
            message = "Deception rate {:.1%} exceeds threshold {:.1%}".format(
                stats["rate"],
                self.alert_threshold,
            )
            alerts.append({"severity": "HIGH", "message": message})

        for domain, domain_stats in stats["by_domain"].items():
            if domain_stats["total"] > 10:
                domain_rate = domain_stats["deceptive"] / domain_stats["total"]
                if domain_rate > 0.5:
                    alerts.append(
                        {
                            "severity": "MEDIUM",
                            "message": f"High deception in {domain}: {domain_rate:.1%}",
                        }
                    )

        return alerts

    def display(self) -> None:
        self.update()
        stats = self.get_stats()

        print("\033[2J\033[H", end="")
        print("=" * 60)
        print("🔬 GEPA DECEPTION MONITORING DASHBOARD")
        print("=" * 60)
        print(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Window: Last {int(self.window.total_seconds() / 60)} minutes\n")

        if not stats:
            print("No fingerprints in the selected window.")
            return

        print("📊 OVERALL STATISTICS")
        print(f"  Total samples: {stats['total']}")
        print(f"  Deceptive: {stats['deceptive']}")
        print(f"  Rate: {stats['rate']:.1%}\n")

        print("📁 BY DOMAIN")
        for domain, domain_stats in sorted(stats["by_domain"].items()):
            rate = (
                domain_stats["deceptive"] / domain_stats["total"] if domain_stats["total"] else 0.0
            )
            print(f"  {domain}: {rate:.1%} ({domain_stats['deceptive']}/{domain_stats['total']})")
        print()

        print("⚡ CIRCUIT ACTIVATIONS")
        sorted_activations = sorted(
            stats["circuit_activations"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for circuit, activation in sorted_activations:
            bar = "█" * int(activation * 20)
            print(f"  {circuit:20s} {bar} {activation:.2f}")
        print()

        alerts = self.check_alerts(stats)
        if alerts:
            print("🚨 ALERTS")
            for alert in alerts:
                print(f"  [{alert['severity']}] {alert['message']}")
        else:
            print("✅ No alerts")

        print("\n=" * 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprints", required=True, help="Path to fingerprints.jsonl")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--window", type=int, default=60, help="Window size in minutes")
    args = parser.parse_args()

    monitor = DeceptionMonitor(Path(args.fingerprints), window_minutes=args.window)

    print("Starting dashboard... (Ctrl+C to exit)")

    try:
        while True:
            monitor.display()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")


if __name__ == "__main__":
    main()
