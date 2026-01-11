"""Dual-path evaluator CLI for deception tracing."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig
from mindful_trace_gepa.deception.dual_path_runner import (
    ModelCallable,
    load_scenarios,
    run_dual_path_batch,
)
from mindful_trace_gepa.path_utils import ensure_dir, require_file


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
            if not isinstance(record, dict):
                type_name = type(record).__name__
                raise ValueError(f"Expected JSON object at line {line_num}; got {type_name}.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


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
    if not file_path.exists():
        raise ValueError(f"Response hook file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Response hook path is not a file: {file_path}")
    module_name = f"dual_path_response_module_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load module from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
    if ":" in response:
        return _load_callable_from_module(response)
    response_path = Path(response)
    if response_path.suffix == ".py":
        return _load_callable_from_file(response_path)
    raise ValueError("Unsupported --response value. Use module:callable or a .py file path.")


def run_cli(args: argparse.Namespace) -> None:
    scenarios_path = require_file(Path(args.scenarios), "Scenarios file")
    output_dir = Path(args.run) if args.run else Path.cwd() / "runs" / "dual_path_eval"
    output_dir = ensure_dir(output_dir, "Output directory")
    records = _load_records(scenarios_path)
    scenarios = load_scenarios(records)
    if not scenarios:
        raise ValueError(f"No valid scenarios loaded from: {scenarios_path}")
    model_callable = _resolve_model_callable(args.response)
    model_id_source = args.model_id if args.model_id is not None else args.response
    model_id = str(model_id_source) if model_id_source is not None else "unknown"
    config = DualPathRunConfig(
        model_id=model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_dir=str(output_dir),
    )
    traces = run_dual_path_batch(scenarios, model_callable, config)
    summary_path = output_dir / "summary.json"
    summary = {
        "scenarios": len(scenarios),
        "traces": len(traces),
        "output_dir": str(output_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-path evaluations.")
    parser.add_argument("--scenarios", required=True, help="Path to dual-path scenarios JSONL")
    parser.add_argument("--run", default=None, help="Directory for run outputs")
    parser.add_argument("--model-id", default=None, help="Model identifier for logging")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--response",
        default=None,
        help="Response hook as module:callable or path to a .py file with generate().",
    )
    return parser


def main() -> int:
    parser = build_parser()
    try:
        run_cli(parser.parse_args())
        return 0
    except FileNotFoundError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
