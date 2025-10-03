"""Command line interface for Mindful Trace GEPA utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

from .configuration import DSPyConfig, dump_json, load_dspy_config
from .dspy_modules.compile import DSPyCompiler, OptimizationMetric
from .dspy_modules.pipeline import GEPAChain
from .tokens import TokenRecorder
from .viewer.builder import build_viewer_html
from .emitters.paired_chains import emit_paired
from .deception.score import score_deception


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


# ---------------------------------------------------------------------------
# DSPy commands
# ---------------------------------------------------------------------------

def handle_dspy_run(args: argparse.Namespace) -> None:
    config = load_dspy_config()
    chain = GEPAChain(config=config, allow_optimizations=args.enable_optim)
    input_records = _read_jsonl(Path(args.input))
    trace_path = Path(args.trace)
    tokens_path = trace_path.with_name("tokens.jsonl")
    summary_path = trace_path.with_name("summary.json")
    deception_path = trace_path.with_name("deception.json")

    trace_rows: List[Dict[str, Any]] = []
    tokens = TokenRecorder()
    deception_rows: List[Dict[str, Any]] = []

    for record in input_records:
        inquiry = record.get("inquiry") or record.get("prompt")
        if not inquiry:
            raise ValueError("Each input row must contain an 'inquiry' or 'prompt' field.")
        context = record.get("context") or args.context or ""
        result = chain.run(inquiry=inquiry, context=context)
        abstained = False
        before_len = len(tokens.records)
        tokens.record_text(result.final_answer, abstained=abstained)
        after_records = tokens.records[before_len:]
        for checkpoint in result.checkpoints:
            payload = dict(checkpoint)
            payload.update(
                {
                    "gepa_hits": result.gepa_hits,
                    "principle_scores": result.principle_scores,
                    "imperative_scores": result.imperative_scores,
                    "context": context,
                    "flags": ["dspy-enabled" if config.enabled else "dspy-disabled"],
                }
            )
            trace_rows.append(payload)
        deception_payload = {
            "honest_chain": result.checkpoints,
            "deceptive_chain": [],
            "final_public_answer": result.final_answer,
            "confidence_trace": [rec.conf for rec in after_records],
        }
        deception_rows.append(score_deception(deception_payload))

    _write_jsonl(trace_path, trace_rows)
    tokens.dump_jsonl(tokens_path)
    dump_json(
        summary_path,
        {
            "model": args.model or "gepa-sim",
            "policy_version": "dspy-chain-v1",
            "thresholds": {"abstention": 0.75},
            "hashes": {"dspy_config": str(load_dspy_config())},
        },
    )
    dump_json(deception_path, {"runs": deception_rows})


def handle_dspy_compile(args: argparse.Namespace) -> None:
    config = load_dspy_config()
    compiler = DSPyCompiler(config=config)
    dataset = _read_jsonl(Path(args.dataset)) if args.dataset else []
    manifest = compiler.compile(
        out_dir=Path(args.out),
        dataset=dataset,
        metric=OptimizationMetric(),
        enable_optimizations=args.enable_optim,
    )
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def handle_view(args: argparse.Namespace) -> None:
    trace_rows = _read_jsonl(Path(args.trace))
    token_path = Path(args.tokens)
    token_rows = _read_jsonl(token_path) if token_path.exists() else []
    deception_data = {}
    if args.deception and Path(args.deception).exists():
        with Path(args.deception).open("r", encoding="utf-8") as handle:
            deception_data = json.load(handle)
    paired_data = {}
    if args.paired and Path(args.paired).exists():
        with Path(args.paired).open("r", encoding="utf-8") as handle:
            paired_data = json.load(handle)
    build_viewer_html(
        trace_events=trace_rows,
        token_events=token_rows,
        deception=deception_data,
        output_path=Path(args.out),
        paired=paired_data,
    )


# ---------------------------------------------------------------------------
# Paired baseline
# ---------------------------------------------------------------------------

def handle_paired_run(args: argparse.Namespace) -> None:
    dataset = _read_jsonl(Path(args.data))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated: Dict[str, Any] = {}
    for item in dataset:
        identifier = item["id"]
        if not item.get("require_paired", False):
            continue
        payload = emit_paired(item["prompt"], item)
        honest_path = out_dir / f"{identifier}_honest_trace.jsonl"
        deceptive_path = out_dir / f"{identifier}_deceptive_trace.jsonl"
        answer_path = out_dir / f"{identifier}_public_answer.txt"
        detection_path = out_dir / f"{identifier}_deception.json"

        _write_jsonl(honest_path, payload["honest_chain"])
        _write_jsonl(deceptive_path, payload["deceptive_chain"])
        answer_path.write_text(payload["final_public_answer"], encoding="utf-8")
        detection = score_deception(payload)
        dump_json(detection_path, detection)
        aggregated[identifier] = detection

    dump_json(Path("runs/deception.json"), aggregated)


def handle_paired_view(args: argparse.Namespace) -> None:
    base = Path(args.base)
    honest = _read_jsonl(base / f"{args.identifier}_honest_trace.jsonl")
    deceptive = _read_jsonl(base / f"{args.identifier}_deceptive_trace.jsonl")
    deception_data = {}
    detection_path = base / f"{args.identifier}_deception.json"
    if detection_path.exists():
        with detection_path.open("r", encoding="utf-8") as handle:
            deception_data = json.load(handle)
    build_viewer_html(
        trace_events=honest + deceptive,
        token_events=[],
        output_path=Path(args.out),
        paired={"honest_chain": honest, "deceptive_chain": deceptive},
        deception=deception_data,
    )


# ---------------------------------------------------------------------------
# Score summaries
# ---------------------------------------------------------------------------


def handle_score(args: argparse.Namespace) -> None:
    trace_rows = _read_jsonl(Path(args.trace))
    policy = {}
    if args.policy and Path(args.policy).exists():
        policy_path = Path(args.policy)
        raw = policy_path.read_text(encoding="utf-8")
        if yaml is not None:
            policy = yaml.safe_load(raw) or {}
        else:
            policy = {"raw": raw}

    principle_totals: Dict[str, List[float]] = {}
    imperative_totals: Dict[str, List[float]] = {}
    for event in trace_rows:
        for key, value in (event.get("principle_scores") or {}).items():
            principle_totals.setdefault(key, []).append(float(value))
        for key, value in (event.get("imperative_scores") or {}).items():
            imperative_totals.setdefault(key, []).append(float(value))

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "principles": {name: _mean(values) for name, values in principle_totals.items()},
        "imperatives": {name: _mean(values) for name, values in imperative_totals.items()},
        "events": len(trace_rows),
        "policy": policy,
    }

    html_parts = [
        "<html><head><title>GEPA Score Summary</title></head><body>",
        "<h1>GEPA Score Summary</h1>",
        f"<p>Total events: {summary['events']}</p>",
        "<h2>Principles</h2><ul>",
    ]
    for name, value in summary["principles"].items():
        html_parts.append(f"<li>{name}: {value:.3f}</li>")
    html_parts.append("</ul><h2>Imperatives</h2><ul>")
    for name, value in summary["imperatives"].items():
        html_parts.append(f"<li>{name}: {value:.3f}</li>")
    html_parts.append("</ul><h2>Policy</h2><pre>")
    html_parts.append(json.dumps(summary["policy"], indent=2))
    html_parts.append("</pre></body></html>")
    Path(args.out).write_text("\n".join(html_parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gepa", description="Mindful Trace GEPA CLI")
    subparsers = parser.add_subparsers(dest="command")

    dspy_parser = subparsers.add_parser("dspy", help="DSPy-style declarative pipelines")
    dspy_sub = dspy_parser.add_subparsers(dest="dspy_command")

    dspy_run = dspy_sub.add_parser("run", help="Execute the DSPy pipeline")
    dspy_run.add_argument("--input", required=True, help="Path to JSONL input file")
    dspy_run.add_argument("--trace", required=True, help="Where to write trace JSONL")
    dspy_run.add_argument("--context", help="Optional profile context")
    dspy_run.add_argument("--model", help="Model identifier")
    dspy_run.add_argument("--enable-optim", action="store_true", help="Opt-in to optimisation features")
    dspy_run.set_defaults(func=handle_dspy_run)

    dspy_compile = dspy_sub.add_parser("compile", help="Compile guarded DSPy prompts")
    dspy_compile.add_argument("--out", required=True, help="Output directory for compiler artifacts")
    dspy_compile.add_argument("--dataset", help="Optional JSONL dataset for optimisation")
    dspy_compile.add_argument("--enable-optim", action="store_true", help="Allow safe prompt augmentation")
    dspy_compile.set_defaults(func=handle_dspy_compile)

    view_parser = subparsers.add_parser("view", help="Build offline trace viewer")
    view_parser.add_argument("--trace", required=True, help="Trace JSONL path")
    view_parser.add_argument("--tokens", required=True, help="Token JSONL path")
    view_parser.add_argument("--out", required=True, help="Output HTML file")
    view_parser.add_argument("--deception", help="Optional deception score JSON")
    view_parser.add_argument("--paired", help="Optional paired metadata JSON")
    view_parser.set_defaults(func=handle_view)

    paired_parser = subparsers.add_parser("paired", help="Paired honest/deceptive baseline")
    paired_sub = paired_parser.add_subparsers(dest="paired_command")

    paired_run = paired_sub.add_parser("run", help="Emit paired chains and detectors")
    paired_run.add_argument("--data", required=True, help="Dataset JSONL path")
    paired_run.add_argument("--out", required=True, help="Output directory")
    paired_run.add_argument("--context", help="Optional context profile")
    paired_run.set_defaults(func=handle_paired_run)

    paired_view = paired_sub.add_parser("view", help="Render paired trace viewer")
    paired_view.add_argument("identifier", help="Scenario identifier")
    paired_view.add_argument("--base", required=True, help="Directory with paired outputs")
    paired_view.add_argument("--out", required=True, help="Output HTML file")
    paired_view.set_defaults(func=handle_paired_view)

    score_parser = subparsers.add_parser("score", help="Summarise GEPA scores from a trace")
    score_parser.add_argument("--trace", required=True, help="Trace JSONL path")
    score_parser.add_argument("--policy", required=False, help="Policy YAML path")
    score_parser.add_argument("--out", required=True, help="Output HTML report")
    score_parser.set_defaults(func=handle_score)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


__all__ = ["main", "build_parser"]
