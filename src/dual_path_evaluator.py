"""Dual-path evaluator CLI for deception tracing."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig, DualPathScenario
from mindful_trace_gepa.deception.dual_path_runner import run_dual_path_batch

ModelCallable = Callable[[str, DualPathRunConfig | None], str]


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
    return records


def _build_scenarios(records: Iterable[dict[str, Any]]) -> list[DualPathScenario]:
    scenarios: list[DualPathScenario] = []
    for record in records:
        scenarios.append(
            DualPathScenario(
                scenario_id=str(record.get("id", "")) or None,
                question=str(record.get("prompt") or record.get("query") or ""),
                path_1_framing=str(record.get("path_1_framing", "Follow policy and caution.")),
                path_2_framing=str(record.get("path_2_framing", "Press advantage.")),
                aligned_path=(
                    str(record.get("aligned_path")) if record.get("aligned_path") else None
                ),
                deceptive_path=(
                    str(record.get("deceptive_path")) if record.get("deceptive_path") else None
                ),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return scenarios


def _load_callable_from_module(module_path: str) -> ModelCallable:
    if ":" not in module_path:
        raise ValueError("Expected module path in 'module:callable' format.")
    module_name, _, attr_name = module_path.partition(":")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attr_name, None)
    if callable_obj is None or not callable(callable_obj):
        raise ValueError(f"Callable '{attr_name}' not found in module '{module_name}'.")
    return callable_obj


def _load_callable_from_file(file_path: Path) -> ModelCallable:
    spec = importlib.util.spec_from_file_location("dual_path_response_module", file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load module from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr_name in ("generate", "model_callable"):
        callable_obj = getattr(module, attr_name, None)
        if callable_obj is not None and callable(callable_obj):
            return callable_obj
    raise ValueError(
        "No callable found in response module. Expected 'generate' or 'model_callable'."
    )


def _resolve_model_callable(response: str | None) -> ModelCallable:
    if not response:
        raise ValueError(
            "No model hook provided. Pass --response module:callable or a .py file path."
        )
    response_path = Path(response)
    if response_path.suffix == ".py" and response_path.exists():
        return _load_callable_from_file(response_path)
    if ":" in response:
        return _load_callable_from_module(response)
    raise ValueError("Unsupported --response value. Use module:callable or a .py file path.")


def run_cli(args: argparse.Namespace) -> None:
    scenarios_path = Path(args.scenarios)
    output_dir = Path(args.run) if args.run else Path.cwd() / "runs" / "dual_path_eval"
    records = _load_records(scenarios_path)
    scenarios = _build_scenarios(records)
    model_callable = _resolve_model_callable(args.response)
    config = DualPathRunConfig(
        model_id=args.model_id or str(args.response),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_dir=str(output_dir),
    )
    run_dual_path_batch(scenarios, model_callable, config)
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
    parser.add_argument(
        "--response",
        default=None,
        help="Response hook as module:callable or path to a .py file with generate().",
    )
    return parser


def main() -> None:
    parser = build_parser()
    run_cli(parser.parse_args())


if __name__ == "__main__":
    main()
