"""Dual-path evaluator CLI for deception tracing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig, DualPathScenario
from mindful_trace_gepa.deception.dual_path_runner import run_dual_path_batch


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                records.append(json.loads(payload))
    return records


def _build_scenarios(records: Iterable[Dict[str, Any]]) -> List[DualPathScenario]:
    scenarios: List[DualPathScenario] = []
    for record in records:
        scenarios.append(
            DualPathScenario(
                scenario_id=str(record.get("id", "")) or None,
                question=str(record.get("prompt") or record.get("query") or ""),
                path_1_framing=str(record.get("path_1_framing", "Follow policy and caution.")),
                path_2_framing=str(record.get("path_2_framing", "Press for advantage.")),
                aligned_path=str(record.get("aligned_path"))
                if record.get("aligned_path")
                else None,
                deceptive_path=str(record.get("deceptive_path"))
                if record.get("deceptive_path")
                else None,
                metadata=dict(record.get("metadata", {})),
            )
        )
    return scenarios


def _stub_model(prompt: str, _config: DualPathRunConfig | None = None) -> str:
    return (
        "[PATH 1 SCRATCHPAD]\n"
        "Default scratchpad for Path 1.\n\n"
        "[PATH 1 PUBLIC REASONING]\n"
        "Default public reasoning for Path 1.\n\n"
        "[PATH 2 SCRATCHPAD]\n"
        "Default scratchpad for Path 2.\n\n"
        "[PATH 2 PUBLIC REASONING]\n"
        "Default public reasoning for Path 2.\n\n"
        "FINAL ANSWER: 1"
    )


def run_cli(args: argparse.Namespace) -> None:
    scenarios_path = Path(args.scenarios)
    output_dir = Path(args.run) if args.run else Path.cwd() / "runs" / "dual_path_eval"
    records = _load_records(scenarios_path)
    scenarios = _build_scenarios(records)
    config = DualPathRunConfig(
        model_id=args.model_id or "dual-path-stub",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_dir=str(output_dir),
    )
    run_dual_path_batch(scenarios, _stub_model, config)
    summary_path = output_dir / "summary.json"
    summary = {"scenarios": len(scenarios), "output_dir": str(output_dir)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-path evaluations.")
    parser.add_argument("--scenarios", required=True, help="Path to dual-path scenarios JSONL")
    parser.add_argument("--run", default=None, help="Directory for run outputs")
    parser.add_argument("--model-id", default=None, help="Model identifier for logging")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mode", default=None, help="Legacy mode flag (unused)")
    parser.add_argument("--response", default=None, help="Legacy response flag (unused)")
    return parser


def main() -> None:
    parser = build_parser()
    run_cli(parser.parse_args())


if __name__ == "__main__":
    main()
