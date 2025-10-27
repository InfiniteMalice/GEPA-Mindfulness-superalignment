"""Merge evaluator, scratchpad, and tracer artefacts into a single report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional


def load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def collect_scenario_data(
    run_dir: Path,
    scenario_summary: Mapping[str, object],
    tracer_payload: Mapping[str, object],
    ablation_payload: Mapping[str, object],
) -> Dict[str, object]:
    scenario_id = str(scenario_summary.get("id"))
    scenario_dir = run_dir / scenario_id

    fingerprint_lookup = {entry.get("id"): entry for entry in tracer_payload.get("scenarios", [])}
    ablation_lookup = ablation_payload.get("scenarios", {})

    fingerprint_entry = fingerprint_lookup.get(scenario_id, {})
    ablation_entry = ablation_lookup.get(scenario_id, {})

    ledger = read_text(scenario_dir / "ledger.txt")
    scratch_a = read_text(scenario_dir / "scratch_A.txt")
    scratch_b = read_text(scenario_dir / "scratch_B.txt")

    return {
        "id": scenario_id,
        "metrics": scenario_summary.get("metrics", {}),
        "ledger": ledger,
        "public_a": read_text(scenario_dir / "public_a.txt"),
        "public_b": read_text(scenario_dir / "public_b.txt"),
        "scratchpad_a": scratch_a,
        "scratchpad_b": scratch_b,
        "fingerprint": fingerprint_entry.get("fingerprint", {}),
        "spans": fingerprint_entry.get("spans", []),
        "ablation_mask": ablation_entry,
    }


def merge_inspection(run_dir: Path) -> Path:
    summary = load_json(run_dir / "eval.json")
    tracer_payload = load_json(run_dir / "tracer" / "circuit_fingerprint.json")
    ablation_payload = load_json(run_dir / "tracer" / "ablation_mask.json")

    scenarios: List[Dict[str, object]] = []
    for scenario_summary in summary.get("scenarios", []):
        scenarios.append(
            collect_scenario_data(run_dir, scenario_summary, tracer_payload, ablation_payload)
        )

    merged = {
        "run": run_dir.name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scenarios": scenarios,
        "means": summary.get("means", {}),
        "artifacts": {
            "eval": str((run_dir / "eval.json").resolve()),
            "fingerprint": str((run_dir / "tracer" / "circuit_fingerprint.json").resolve()),
            "ablation": str((run_dir / "tracer" / "ablation_mask.json").resolve()),
        },
    }

    merged_path = run_dir / "merged_inspection.json"
    merged_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return merged_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge run inspection artefacts")
    parser.add_argument("run", help="Run directory to merge")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    merged_path = merge_inspection(run_dir)
    print(f"Merged inspection saved to {merged_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
