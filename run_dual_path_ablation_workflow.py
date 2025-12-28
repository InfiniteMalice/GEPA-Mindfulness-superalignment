#!/usr/bin/env python3
"""
Dual-path Deception Circuit Ablation Workflow.

This script runs a dual-path pipeline:
1. Collect dual-path traces.
2. Trace circuits and write deception fingerprints.
3. Emit a workflow summary for follow-on analysis and ablation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig
from mindful_trace_gepa.deception.dual_path_runner import load_scenarios, run_dual_path_batch
from mindful_trace_gepa.dual_path_circuit_tracer import run_tracing
from mindful_trace_gepa.storage import read_jsonl

LOGGER = logging.getLogger(__name__)


def _stub_model(prompt: str, _config: DualPathRunConfig | None = None) -> str:
    """Return a placeholder dual-path response for testing.

    This stub model is used when no real model is available. It produces
    deterministic output suitable for workflow validation.
    """
    return (
        "[PATH 1 SCRATCHPAD]\n"
        "Baseline scratchpad for Path 1.\n\n"
        "[PATH 1 PUBLIC REASONING]\n"
        "Baseline public reasoning for Path 1.\n\n"
        "[PATH 2 SCRATCHPAD]\n"
        "Baseline scratchpad for Path 2.\n\n"
        "[PATH 2 PUBLIC REASONING]\n"
        "Baseline public reasoning for Path 2.\n\n"
        "FINAL ANSWER: 1"
    )


def run_workflow(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    traces_dir = output_dir / "traces"
    reports_dir = output_dir / "reports"
    traces_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = Path(args.scenarios)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenario_path}")
    records = read_jsonl(scenario_path)
    scenarios = load_scenarios(records)
    config = DualPathRunConfig(model_id=args.model, log_dir=str(traces_dir))
    run_dual_path_batch(scenarios, _stub_model, config)
    run_tracing(traces_dir)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": len(scenarios),
        "traces_dir": str(traces_dir),
        "reports_dir": str(reports_dir),
        "notes": "Dual-path traces collected; proceed with analysis and ablation tooling.",
    }
    (reports_dir / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-path ablation workflow.")
    parser.add_argument("--model", required=True, help="Model identifier for logging")
    parser.add_argument(
        "--scenarios",
        default="datasets/dual_path/data.jsonl",
        help="Dual-path scenarios JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="results/dual_path_ablation",
        help="Output directory",
    )
    return parser


def main() -> int:
    parser = build_parser()
    try:
        run_workflow(parser.parse_args())
        return 0
    except Exception as exc:
        LOGGER.exception("Dual-path ablation workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
