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
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig
from mindful_trace_gepa.deception.dual_path_runner import load_scenarios, run_dual_path_batch
from mindful_trace_gepa.storage import read_jsonl


def _load_tracer() -> Callable[[Path], None]:
    tracer_path = Path(__file__).resolve().parent / "src" / "dual_path_circuit_tracer.py"
    spec = importlib.util.spec_from_file_location("dual_path_circuit_tracer", tracer_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load tracer module at {tracer_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_tracing = getattr(module, "run_tracing", None)
    if not callable(run_tracing):
        raise ValueError("run_tracing callable not found in dual_path_circuit_tracer module.")
    return run_tracing


def _stub_model(prompt: str, _config: DualPathRunConfig | None = None) -> str:
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
    records = read_jsonl(scenario_path)
    scenarios = load_scenarios(records)
    config = DualPathRunConfig(model_id=args.model, log_dir=str(traces_dir))
    run_dual_path_batch(scenarios, _stub_model, config)
    run_tracing = _load_tracer()
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


def main() -> None:
    parser = build_parser()
    run_workflow(parser.parse_args())


if __name__ == "__main__":
    main()
