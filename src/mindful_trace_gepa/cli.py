"""Command line interface for Mindful Trace GEPA utilities."""

from __future__ import annotations

import argparse
import json
from argparse import BooleanOptionalAction
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import click

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

_DSPY_PIPELINE_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency may fail
    _dspy_pipeline = import_module("mindful_trace_gepa.dspy_modules.pipeline")
except Exception as exc:  # pragma: no cover - surface later when needed
    _dspy_pipeline = None
    _DSPY_PIPELINE_ERROR = exc

if _dspy_pipeline is not None:
    GEPA_CHAIN_CLS = getattr(_dspy_pipeline, "GEPAChain", None)
    DUAL_PATH_CHAIN_CLS = getattr(_dspy_pipeline, "DualPathGEPAChain", None)
else:  # pragma: no cover - optional dependency missing
    GEPA_CHAIN_CLS = None
    DUAL_PATH_CHAIN_CLS = None

_DSPY_COMPILE_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency may fail
    _dspy_compile = import_module("mindful_trace_gepa.dspy_modules.compile")
except Exception as exc:  # pragma: no cover - surface later when needed
    _dspy_compile = None
    _DSPY_COMPILE_ERROR = exc

if _dspy_compile is not None:
    GEPA_COMPILER_CLS = getattr(_dspy_compile, "GEPACompiler", None)
    CREATE_GEPA_METRIC = getattr(_dspy_compile, "create_gepa_metric", None)
else:  # pragma: no cover - optional dependency missing
    GEPA_COMPILER_CLS = None
    CREATE_GEPA_METRIC = None

dspy_pkg = optional_import("dspy")


def _resolve_cli_path(path_str: str, *, require_exists: bool = True) -> Path:
    """Resolve CLI-supplied paths with a few friendly fallbacks.

    The DSPy quick-start documentation references assets relative to the
    project root (for example ``examples/self_tracing_sample.jsonl``).
    Users often execute the command from a sub-directory such as the
    ``dspy_modules`` folder, so we try a handful of sensible locations to
    locate those assets before failing. When ``require_exists`` is ``False`` we
    still return the best-effort resolution even if the path has not been
    created yet (useful for output destinations).
    """

    candidate = Path(path_str)
    if candidate.is_absolute():
        if candidate.exists() or not require_exists:
            return candidate
        raise FileNotFoundError(candidate)

    if candidate.exists():
        return candidate.resolve()

    module_dir = Path(__file__).resolve().parent
    package_root = module_dir.parent
    search_roots = [
        Path.cwd(),
        module_dir,
        package_root,
        package_root.parent,
    ]

    roots: List[Path] = []
    seen = set()
    for root in search_roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)

    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
        if not require_exists and resolved.parent.exists():
            return resolved

    if not require_exists:
        return (Path.cwd() / candidate).resolve()

    raise FileNotFoundError(candidate)


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
    dual_path_flag = getattr(args, "dual_path", False)
    if dual_path_flag and DUAL_PATH_CHAIN_CLS is None:
        _raise_dspy_import_error("dual-path pipeline", _DSPY_PIPELINE_ERROR)
    if GEPA_CHAIN_CLS is None and DUAL_PATH_CHAIN_CLS is None:
        _raise_dspy_import_error("pipeline", _DSPY_PIPELINE_ERROR)
    config = load_dspy_config()
    chain_factory = (
        DUAL_PATH_CHAIN_CLS
        if dual_path_flag and DUAL_PATH_CHAIN_CLS is not None
        else GEPA_CHAIN_CLS
    )
    chain = chain_factory(config=config, allow_optimizations=args.enable_optim)
    input_path = _resolve_cli_path(args.input)
    input_records = _read_jsonl(input_path)
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
    if GEPA_COMPILER_CLS is None or CREATE_GEPA_METRIC is None:
        _raise_dspy_import_error("compiler", _DSPY_COMPILE_ERROR)
    if dspy_pkg is None:
        raise RuntimeError(
            "DSPy compilation requires the optional 'dspy-ai' dependency. "
            "Install it via 'pip install -e .[dspy]' or 'pip install dspy-ai'."
        )

    raw_config_path = getattr(args, "config", "configs/policies/dspy.yml")
    config_path = _resolve_cli_path(raw_config_path, require_exists=False)
    if not config_path.exists():
        config_data: Dict[str, Any] = {}
    else:
        raw = config_path.read_text(encoding="utf-8")
        if yaml is not None:
            config_data = yaml.safe_load(raw) or {}
        else:
            config_data = json.loads(raw)

    enable_optim = bool(getattr(args, "enable_optim", False))

    if enable_optim:
        config_data["allow_optimizations"] = True
        config_data.setdefault("enabled", True)

    weights = config_data.get("optimization", {}).get("metric_weights", {})
    metric_fn = CREATE_GEPA_METRIC(
        alpha=weights.get("gepa_alignment", 0.5),
        beta=weights.get("task_success", 0.5),
        gamma=weights.get("gate_violation_penalty", 1.0),
    )

    compiler = GEPA_COMPILER_CLS(
        config=config_data,
        metric_fn=metric_fn,
        forbidden_phrases=config_data.get("safety", {}).get("forbidden_phrases", []),
    )

    dataset_records = (_read_jsonl(_resolve_cli_path(args.dataset)) if args.dataset else [])
    trainset = [
        dspy_pkg.Example(
            inquiry=record.get("query", record.get("prompt", "")),
            context=record.get("context", ""),
            answer=record.get("answer", ""),
        ).with_inputs("inquiry", "context")
        for record in dataset_records
    ]

    module = DUAL_PATH_CHAIN_CLS(use_dual_path=config_data.get("use_dual_path", False))
    compiled_module = compiler.compile(
        module,
        trainset=trainset,
        method=config_data.get("optimization", {}).get("method", "bootstrap"),
    )

    compiler.save_compiled_module(compiled_module, args.out)


def run_dual_path_contrastive(data: Path, out: Path, context: str) -> None:
    from .deception.circuit_analysis import detect_deception_heuristic
    from .prompts.dual_path import make_dual_path_prompt, parse_dual_path_response
    from .storage.jsonl_store import JSONLStore

    out.mkdir(parents=True, exist_ok=True)
    store = JSONLStore(str(data))
    examples = list(store.read())
    results: List[Dict[str, Any]] = []

    for record in examples:
        query = record.get("query") or record.get("prompt") or ""
        prompt = make_dual_path_prompt(query, context)
        response = (
            "[PATH 1 REASONING]\n"
            "This path emphasises caution and potential risks.\n\n"
            "[PATH 2 REASONING]\n"
            "This path emphasises confident, immediate action.\n\n"
            "[COMPARISON]\n"
            "The careful path weighs risks; the confident path prioritises action.\n\n"
            "[RECOMMENDATION]\n"
            "I recommend Path 1 for safety."
        )

        sections = parse_dual_path_response(response)
        deception = detect_deception_heuristic(sections)

        result = {
            "id": record.get("id", ""),
            "query": query,
            "prompt": prompt,
            "response": response,
            "sections": sections,
            "deception_signals": deception,
            "ground_truth": record.get("ground_truth_correct_path", ""),
        }
        results.append(result)

        (out / f"{record.get('id', 'example')}_response.txt").write_text(
            response,
            encoding="utf-8",
        )
        (out / f"{record.get('id', 'example')}_deception.json").write_text(
            json.dumps(deception, indent=2),
            encoding="utf-8",
        )

    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


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
    data_path = _resolve_cli_path(args.data)
    dataset = _read_jsonl(data_path)
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
    base = _resolve_cli_path(args.base)
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
    dspy_compile.add_argument(
        "--config",
        default="configs/policies/dspy.yml",
        help="DSPy configuration YAML",
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


@click.group()
def dspy_cli() -> None:
    """DSPy module compilation and execution."""


@dspy_cli.command("run")
@click.option("--input", "input_path", required=True, help="Input JSONL file")
@click.option("--trace", "trace_path", required=True, help="Output trace JSONL")
@click.option("--context", default="", help="Optional context profile")
@click.option("--model", default="", help="Model identifier")
@click.option("--enable-optim", is_flag=True, help="Enable optimisation features")
@click.option("--dual-path", is_flag=True, help="Use dual-path pipeline")
def click_dspy_run(
    input_path: str,
    trace_path: str,
    context: str,
    model: str,
    enable_optim: bool,
    dual_path: bool,
) -> None:
    namespace = argparse.Namespace(
        input=input_path,
        trace=trace_path,
        context=context,
        model=model,
        enable_optim=enable_optim,
        dual_path=dual_path,
        with_logprobs=True,
        log_topk=3,
        log_every=16,
        long_context=False,
        shard_threshold=10_000,
    )
    handle_dspy_run(namespace)


@dspy_cli.command("compile")
@click.option("--out", required=True, help="Output directory")
@click.option("--config", default="configs/policies/dspy.yml", help="DSPy config")
@click.option("--dataset", default="", help="Training dataset JSONL")
@click.option("--enable-optim", is_flag=True, help="Enable optimisations")
def click_dspy_compile(out: str, config: str, dataset: str, enable_optim: bool) -> None:
    namespace = argparse.Namespace(
        out=out,
        config=config,
        dataset=dataset if dataset else None,
        enable_optim=enable_optim,
    )
    handle_dspy_compile(namespace)


@dspy_cli.command("contrastive-run")
@click.option("--data", "data_path", required=True, help="Dual-path dataset JSONL")
@click.option("--out", "out_dir", required=True, help="Output directory")
@click.option("--context", default="general", help="Context profile")
def click_dspy_contrastive(data_path: str, out_dir: str, context: str) -> None:
    run_dual_path_contrastive(Path(data_path), Path(out_dir), context)


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], None] | None = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


__all__ = ["main", "build_parser"]
