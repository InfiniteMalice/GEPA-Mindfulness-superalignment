"""Command line interface for Mindful Trace GEPA utilities."""

from __future__ import annotations

import argparse
import json
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .cli_deception import register_cli as register_deception_cli
from .cli_scoring import register_cli as register_scoring_cli
from .configuration import dump_json, load_dspy_config
from .deception.score import score_deception
from .emitters.paired_chains import emit_paired
from .storage import TraceArchiveWriter, iter_jsonl, load_jsonl, read_jsonl
from .tokens import TokenRecorder
from .utils.imports import optional_import
from .viewer.builder import build_viewer_html

yaml = optional_import("yaml")

_dspy_pipeline = optional_import("mindful_trace_gepa.dspy_modules.pipeline")
if _dspy_pipeline is not None:
    GEPA_CHAIN_CLS = getattr(_dspy_pipeline, "GEPAChain", None)
else:  # pragma: no cover - optional dependency missing
    GEPA_CHAIN_CLS = None

_dspy_compile = optional_import("mindful_trace_gepa.dspy_modules.compile")
if _dspy_compile is not None:
    DSPY_COMPILER_CLS = getattr(_dspy_compile, "DSPyCompiler", None)
    OPTIMIZATION_METRIC_CLS = getattr(_dspy_compile, "OptimizationMetric", None)
else:  # pragma: no cover - optional dependency missing
    DSPY_COMPILER_CLS = None
    OPTIMIZATION_METRIC_CLS = None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


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
    if GEPA_CHAIN_CLS is None:
        raise RuntimeError("DSPy pipeline unavailable; optional dependencies missing")
    config = load_dspy_config()
    chain_factory = GEPA_CHAIN_CLS
    chain = chain_factory(config=config, allow_optimizations=args.enable_optim)
    input_records = _read_jsonl(Path(args.input))
    trace_path = Path(args.trace)
    tokens_path = trace_path.with_name("tokens.jsonl")
    summary_path = trace_path.with_name("summary.json")
    deception_path = trace_path.with_name("deception.json")

    with_logprobs = getattr(args, "with_logprobs", True)
    log_topk = getattr(args, "log_topk", 3)
    log_every = getattr(args, "log_every", 16)
    long_context = getattr(args, "long_context", False)
    shard_threshold = max(getattr(args, "shard_threshold", 10_000) or 10_000, 1)

    tokens = TokenRecorder(
        log_topk=log_topk if with_logprobs else 0,
        sample_every=log_every if with_logprobs else max(log_every, 1),
    )
    deception_rows: List[Dict[str, Any]] = []

    manifest_path: Optional[Path] = None

    writer = TraceArchiveWriter(trace_path, shard_threshold=shard_threshold)
    try:
        for record in input_records:
            inquiry = record.get("inquiry") or record.get("prompt")
            if not inquiry:
                raise ValueError("Each input row must contain an 'inquiry' or 'prompt' field.")
            context = record.get("context") or getattr(args, "context", None) or ""
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
                        "flags": [
                            "dspy-enabled" if config.enabled else "dspy-disabled",
                            "long-context" if long_context else "short-context",
                        ],
                    }
                )
                writer.append(payload)
            deception_payload = {
                "honest_chain": result.checkpoints,
                "deceptive_chain": [],
                "final_public_answer": result.final_answer,
                "confidence_trace": [rec.conf for rec in after_records],
            }
            deception_rows.append(score_deception(deception_payload))
    finally:
        manifest_path = writer.close()

    tokens.dump_jsonl(tokens_path)
    dump_json(
        summary_path,
        {
            "model": args.model or "gepa-sim",
            "policy_version": "dspy-chain-v1",
            "thresholds": {"abstention": 0.75},
            "hashes": {"dspy_config": str(load_dspy_config())},
            "shards": str(manifest_path) if manifest_path else None,
        },
    )
    dump_json(deception_path, {"runs": deception_rows})


def handle_dspy_compile(args: argparse.Namespace) -> None:
    if DSPY_COMPILER_CLS is None or OPTIMIZATION_METRIC_CLS is None:
        raise RuntimeError("DSPy compiler unavailable; optional dependencies missing")
    config = load_dspy_config()
    compiler = DSPY_COMPILER_CLS(config=config)
    dataset = _read_jsonl(Path(args.dataset)) if args.dataset else []
    manifest = compiler.compile(
        out_dir=Path(args.out),
        dataset=dataset,
        metric=OPTIMIZATION_METRIC_CLS(),
        enable_optimizations=args.enable_optim,
    )
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


def handle_view(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace)
    token_path = Path(args.tokens)
    out_path = Path(args.out)
    page_size = getattr(args, "page_size", 200)
    max_points = getattr(args, "max_points", 5000)

    trace_rows: List[Dict[str, Any]] = []
    if trace_path.exists():
        trace_rows = load_jsonl(trace_path, limit=page_size)

    token_rows: List[Dict[str, Any]] = []
    if token_path.exists():
        token_rows = load_jsonl(token_path, limit=max_points)

    manifest_data: Dict[str, Any] = {}
    manifest_override = getattr(args, "manifest", None)
    manifest_path = (
        Path(manifest_override) if manifest_override else trace_path.with_name("manifest.json")
    )
    manifest_rel: Optional[str] = None
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            manifest_rel = str(manifest_path.relative_to(out_path.parent))
        except ValueError:
            manifest_rel = str(manifest_path.resolve())

    def _safe_load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {}

    deception_data: Dict[str, Any] = {}
    if args.deception and Path(args.deception).exists():
        deception_data = _safe_load_json(Path(args.deception))

    if (
        deception_data
        and "scores" in deception_data
        and "probe" not in deception_data
        and "summary" not in deception_data
    ):
        deception_data = {"probe": deception_data}

    trace_dir = trace_path.parent
    probe_candidates = [trace_dir / "deception_probe.json", Path("runs/deception_probe.json")]
    for candidate in probe_candidates:
        payload = _safe_load_json(candidate)
        if payload and "probe" not in deception_data:
            deception_data["probe"] = payload
            break

    summary_candidates = [trace_dir / "deception_summary.json", Path("runs/deception_summary.json")]
    for candidate in summary_candidates:
        payload = _safe_load_json(candidate)
        if payload and "summary" not in deception_data:
            deception_data["summary"] = payload
            break

    mm_candidates = [trace_dir / "mm_eval.json", Path("runs/mm_eval.json")]
    for candidate in mm_candidates:
        payload = _safe_load_json(candidate)
        if payload and "mm" not in deception_data:
            deception_data["mm"] = payload
            break

    paired_data: Dict[str, Any] = {}
    if args.paired and Path(args.paired).exists():
        paired_data = _safe_load_json(Path(args.paired))
    if not paired_data:
        for candidate in [trace_dir / "deception.json", Path("runs/deception.json")]:
            paired_data = _safe_load_json(candidate)
            if paired_data:
                break

    if "paired" in deception_data and not paired_data:
        paired_data = deception_data.get("paired") or {}

    scoring_data: Dict[str, Any] = {}
    scoring_path = trace_path.with_name("scores.json")
    if scoring_path.exists():
        scoring_data = _safe_load_json(scoring_path)

    build_viewer_html(
        trace_events=trace_rows,
        token_events=token_rows,
        deception=deception_data,
        output_path=out_path,
        paired=paired_data,
        manifest=manifest_data,
        settings={
            "pageSize": page_size,
            "maxPoints": max_points,
            "manifestPath": manifest_rel,
        },
        scoring=scoring_data,
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
    deception_data: Dict[str, Any] = {}
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
    trace_path = Path(args.trace)
    zstd_flag = getattr(args, "zstd", False)
    stream_flag = getattr(args, "stream", True)
    sharded_flag = getattr(args, "sharded", False)
    manifest_override = getattr(args, "manifest", None)

    if zstd_flag and not trace_path.exists():
        candidate = (
            trace_path.with_suffix(trace_path.suffix + ".zst")
            if trace_path.suffix
            else trace_path.with_name(f"{trace_path.name}.zst")
        )
        if candidate.exists():
            trace_path = candidate
    policy: Dict[str, Any] = {}
    if args.policy and Path(args.policy).exists():
        policy_path = Path(args.policy)
        raw = policy_path.read_text(encoding="utf-8")
        if yaml is not None:
            policy = yaml.safe_load(raw) or {}
        else:
            policy = {"raw": raw}

    def _iter_events() -> Iterable[Dict[str, Any]]:
        if sharded_flag:
            manifest_path = (
                Path(manifest_override)
                if manifest_override
                else (
                    trace_path
                    if trace_path.name == "manifest.json"
                    else trace_path.with_name("manifest.json")
                )
            )
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            manifest_blob = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_map = manifest_blob if isinstance(manifest_blob, dict) else {}
            base = manifest_path.parent
            for shard in manifest_map.get("shards", []):
                if not isinstance(shard, Mapping):
                    continue
                shard_path_str = str(shard.get("path", ""))
                if not shard_path_str:
                    continue
                shard_path = base / shard_path_str
                yield from iter_jsonl(shard_path)
        else:
            yield from iter_jsonl(trace_path)

    event_iter: Iterable[Dict[str, Any]] = _iter_events() if stream_flag else list(_iter_events())

    principle_sums: Dict[str, float] = {}
    principle_counts: Dict[str, int] = {}
    imperative_sums: Dict[str, float] = {}
    imperative_counts: Dict[str, int] = {}
    flags_seen: set[str] = set()
    total_events = 0
    for event in event_iter:
        total_events += 1
        for key, value in (event.get("principle_scores") or {}).items():
            principle_sums[key] = principle_sums.get(key, 0.0) + float(value)
            principle_counts[key] = principle_counts.get(key, 0) + 1
        for key, value in (event.get("imperative_scores") or {}).items():
            imperative_sums[key] = imperative_sums.get(key, 0.0) + float(value)
            imperative_counts[key] = imperative_counts.get(key, 0) + 1
        for flag in event.get("flags", []) or []:
            flags_seen.add(flag)

    def _mean(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
        return {name: (sums[name] / counts[name]) for name in sums}

    principles_mean = _mean(principle_sums, principle_counts)
    imperatives_mean = _mean(imperative_sums, imperative_counts)
    summary = {
        "principles": principles_mean,
        "imperatives": imperatives_mean,
        "events": total_events,
        "policy": policy,
        "flags": sorted(flags_seen),
    }

    html_parts = [
        "<html><head><title>GEPA Score Summary</title></head><body>",
        "<h1>GEPA Score Summary</h1>",
        f"<p>Total events: {summary['events']}</p>",
        "<h2>Principles</h2><ul>",
    ]
    for name, value in principles_mean.items():
        html_parts.append(f"<li>{name}: {value:.3f}</li>")
    html_parts.append("</ul><h2>Imperatives</h2><ul>")
    for name, value in imperatives_mean.items():
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
    dspy_run.add_argument(
        "--enable-optim", action="store_true", help="Opt-in to optimisation features"
    )
    dspy_run.add_argument(
        "--with-logprobs",
        action=BooleanOptionalAction,
        default=True,
        help="Record log probabilities",
    )
    dspy_run.add_argument(
        "--log-topk", type=int, default=3, help="Number of top-k alternatives to store"
    )
    dspy_run.add_argument(
        "--log-every", type=int, default=16, help="Sample frequency for token logging"
    )
    dspy_run.add_argument(
        "--long-context", action="store_true", help="Enable long-context formatting hints"
    )
    dspy_run.add_argument(
        "--shard-threshold", type=int, default=10_000, help="Events per shard before splitting"
    )
    dspy_run.set_defaults(func=handle_dspy_run)

    dspy_compile = dspy_sub.add_parser("compile", help="Compile guarded DSPy prompts")
    dspy_compile.add_argument(
        "--out", required=True, help="Output directory for compiler artifacts"
    )
    dspy_compile.add_argument("--dataset", help="Optional JSONL dataset for optimisation")
    dspy_compile.add_argument(
        "--enable-optim", action="store_true", help="Allow safe prompt augmentation"
    )
    dspy_compile.set_defaults(func=handle_dspy_compile)

    view_parser = subparsers.add_parser("view", help="Build offline trace viewer")
    view_parser.add_argument("--trace", required=True, help="Trace JSONL path")
    view_parser.add_argument("--tokens", required=True, help="Token JSONL path")
    view_parser.add_argument("--out", required=True, help="Output HTML file")
    view_parser.add_argument("--deception", help="Optional deception score JSON")
    view_parser.add_argument("--paired", help="Optional paired metadata JSON")
    view_parser.add_argument(
        "--page-size", type=int, default=200, help="Events per page in the viewer"
    )
    view_parser.add_argument(
        "--max-points", type=int, default=5000, help="Maximum token events to embed inline"
    )
    view_parser.add_argument("--manifest", help="Optional manifest path (auto-detected if omitted)")
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
    score_parser.add_argument(
        "--stream", action=BooleanOptionalAction, default=True, help="Stream events during scoring"
    )
    score_parser.add_argument(
        "--sharded", action="store_true", help="Read trace shards via manifest"
    )
    score_parser.add_argument("--manifest", help="Optional manifest path for sharded runs")
    score_parser.add_argument(
        "--zstd", action="store_true", help="Hint: trace files are zstd compressed"
    )
    score_parser.set_defaults(func=handle_score)

    register_deception_cli(subparsers)
    register_scoring_cli(subparsers)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], None] | None = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


__all__ = ["main", "build_parser"]
