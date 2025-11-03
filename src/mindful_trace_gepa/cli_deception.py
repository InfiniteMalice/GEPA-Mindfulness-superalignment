"""Command line utilities for deception probe and evaluation workflows."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .configuration import dump_json
from .deception.probes_linear import ProbeWeights, infer_probe, load_probe
from .deception.score import summarize_deception_sources
from .storage.jsonl_store import load_jsonl
from .utils.imports import optional_import

yaml = optional_import("yaml")

LOGGER = logging.getLogger(__name__)


@dataclass
class ActivationBundle:
    """Container describing extracted activations and their provenance."""

    activations: Dict[str, Any]
    source: str


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    if yaml is None:
        LOGGER.warning("PyYAML unavailable; attempting JSON parsing for %s", path)
        return json.loads(raw)
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} is not a mapping")
    return data


def _load_trace_events(trace_path: Path) -> List[Dict[str, Any]]:
    if not trace_path.exists():
        LOGGER.warning("Trace file %s missing; continuing with empty trace", trace_path)
        return []
    return load_jsonl(trace_path)


def _normalise_vector(values: Iterable[Any], dimension: Optional[int]) -> List[float]:
    floats = [float(v) for v in values]
    if dimension is not None and dimension > 0:
        if len(floats) > dimension:
            return floats[:dimension]
        if len(floats) < dimension:
            floats.extend([0.0] * (dimension - len(floats)))
    return floats


def _append_vectors(
    target: Dict[str, Dict[str, List[Any]]],
    layer_key: str,
    vectors: Iterable[Iterable[Any]],
    step_index: int,
    dimension: Optional[int],
) -> None:
    bucket = target.setdefault(layer_key, {"tokens": [], "token_to_step": []})
    for vector in vectors:
        try:
            floats = _normalise_vector(vector, dimension)
        except TypeError:
            continue
        bucket["tokens"].append(floats)
        bucket["token_to_step"].append(step_index)


def _collect_activations_from_trace(
    trace_events: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    dimension: Optional[int],
) -> Optional[ActivationBundle]:
    layers: Dict[str, Dict[str, List[Any]]] = {}
    layer_filter: Optional[set[str]] = None
    if layer_indices:
        layer_filter = {str(idx) for idx in layer_indices}
        layer_filter.update({str(int(idx)) for idx in layer_indices})
    for step_index, event in enumerate(trace_events):
        event_layers = None
        if isinstance(event.get("activations"), Mapping):
            event_layers = event.get("activations")
        elif isinstance(event.get("probe_activations"), Mapping):
            event_layers = event.get("probe_activations")
        if not isinstance(event_layers, Mapping):
            continue
        for raw_layer, vectors in event_layers.items():
            key = str(raw_layer)
            if layer_filter and key not in layer_filter:
                continue
            if isinstance(vectors, Mapping) and "tokens" in vectors:
                vectors_iter = vectors.get("tokens") or []
            else:
                vectors_iter = vectors
            if not isinstance(vectors_iter, Iterable):
                continue
            _append_vectors(layers, key, vectors_iter, step_index, dimension)
    if not layers:
        return None
    return ActivationBundle({"layers": layers, "pool": "mean"}, source="trace")


def _synthetic_activations(
    trace_events: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    dimension: Optional[int],
) -> ActivationBundle:
    events = list(trace_events)
    if not events:
        events = [{"content": "synthetic placeholder"}]
    dim = dimension or 1
    layers: Dict[str, Dict[str, List[Any]]] = {}
    for layer in layer_indices or [0]:
        layer_key = str(layer)
        tokens: List[List[float]] = []
        token_to_step: List[int] = []
        for step_idx, event in enumerate(events):
            text = str(event.get("content") or event.get("text") or event.get("final_answer") or "")
            base = float(len(text) + 1 + step_idx + abs(int(layer)))
            vector = [(base + (i * 0.17)) % 1.0 for i in range(dim)]
            tokens.append(vector)
            token_to_step.append(step_idx)
        layers[layer_key] = {"tokens": tokens, "token_to_step": token_to_step}
    return ActivationBundle({"layers": layers, "pool": "mean"}, source="synthetic")


def _prepare_activations(
    trace_events: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    probe: Optional[ProbeWeights],
) -> ActivationBundle:
    bundle = _collect_activations_from_trace(
        trace_events,
        layer_indices,
        probe.dimension if probe else None,
    )
    if bundle is not None:
        return bundle
    LOGGER.info("Falling back to synthetic activations for probe evaluation")
    return _synthetic_activations(trace_events, layer_indices, probe.dimension if probe else None)


def _collect_labels(trace_events: Sequence[Mapping[str, Any]]) -> Optional[List[int]]:
    labels: List[int] = []
    found = False
    for event in trace_events:
        label = event.get("deception_label") if isinstance(event, Mapping) else None
        if label is None:
            label = event.get("label") if isinstance(event, Mapping) else None
        if label is None:
            labels.append(0)
            continue
        try:
            labels.append(int(label))
            found = True
        except (TypeError, ValueError):
            labels.append(0)
    if not found:
        return None
    positives = sum(1 for value in labels if value == 1)
    negatives = sum(1 for value in labels if value == 0)
    if positives == 0 or negatives == 0:
        return None
    return labels


def handle_deception_probes(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = _load_yaml_config(config_path)
    trace_path = Path(args.trace)
    probe_path = Path(args.probe)
    trace_events = _load_trace_events(trace_path)
    probe = load_probe(probe_path)

    layers = config.get("model_layer_indices") or []
    pooling = config.get("pooling", "mean")
    threshold_config = config.get("threshold") or {}

    activations_bundle = _prepare_activations(trace_events, layers, probe)
    labels = _collect_labels(trace_events)
    result = infer_probe(
        activations_bundle.activations,
        probe,
        pooling=pooling,
        threshold_config=threshold_config,
        labels=labels,
    )

    output = dict(result)
    output.update(
        {
            "config": config,
            "model": args.model,
            "trace_path": str(trace_path),
            "probe_path": str(probe_path),
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "trace_events": len(trace_events),
            "activations_source": activations_bundle.source,
        }
    )

    output_config = config.get("output") or {}
    default_out = output_config.get("path", "runs/deception_probe.json")
    out_path = Path(args.out or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, output)
    LOGGER.info("Wrote deception probe analysis to %s", out_path)


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not path.exists():
        LOGGER.debug("Optional artifact %s missing", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse %s: %s", path, exc)
        return None


def handle_deception_summary(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    base_dir = Path(args.runs) if getattr(args, "runs", None) else out_path.parent
    if not base_dir.exists():
        base_dir = Path("runs")

    probe_candidates = [Path(args.probe)] if getattr(args, "probe", None) else []
    dual_path_candidates: List[Path] = []
    dual_path_arg = getattr(args, "dual_path", None) or getattr(args, "paired", None)
    if dual_path_arg:
        dual_path_candidates.append(Path(dual_path_arg))
    mm_candidates = [Path(args.mm)] if getattr(args, "mm", None) else []

    probe_candidates.append(base_dir / "deception_probe.json")
    dual_path_candidates.extend([base_dir / "deception.json", base_dir / "paired_deception.json"])
    mm_candidates.append(base_dir / "mm_eval.json")

    probe_data = next(
        (data for data in (_load_json(path) for path in probe_candidates) if data),
        None,
    )
    dual_path_data = next(
        (data for data in (_load_json(path) for path in dual_path_candidates) if data),
        None,
    )
    mm_data = next(
        (data for data in (_load_json(path) for path in mm_candidates) if data),
        None,
    )

    context = {
        "probe_path": str(probe_candidates[0]) if probe_candidates else None,
        "dual_path_path": str(dual_path_candidates[0]) if dual_path_candidates else None,
        "mm_path": str(mm_candidates[0]) if mm_candidates else None,
        "search_dir": str(base_dir),
    }

    summary = summarize_deception_sources(
        dual_path=dual_path_data,
        probe=probe_data,
        mm=mm_data,
        context=context,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, summary)
    LOGGER.info("Wrote deception summary to %s", out_path)


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    deception_parser = subparsers.add_parser("deception", help="Deception research utilities")
    deception_sub = deception_parser.add_subparsers(dest="deception_command")

    probes = deception_sub.add_parser("probes", help="Run linear probe deception analysis")
    probes.add_argument("--trace", required=True, help="Trace JSONL file with optional activations")
    probes.add_argument("--model", required=True, help="Model identifier or endpoint")
    probes.add_argument("--probe", required=True, help="Path to probe weight file")
    probes.add_argument("--config", required=True, help="Configuration YAML for the probe")
    probes.add_argument("--out", help="Output JSON path for probe scores")
    probes.set_defaults(func=handle_deception_probes)

    summary = deception_sub.add_parser("summary", help="Merge deception artifacts into a summary")
    summary.add_argument("--out", required=True, help="Where to write the deception summary JSON")
    summary.add_argument("--probe", help="Optional override path for probe results")
    summary.add_argument(
        "--dual-path",
        dest="dual_path",
        help="Optional override for dual-path analysis results",
    )
    summary.add_argument("--paired", dest="dual_path", help=argparse.SUPPRESS)
    summary.add_argument("--mm", help="Optional override for multimodal evaluation metrics")
    summary.add_argument("--runs", help="Directory to search for deception artifacts")
    summary.set_defaults(func=handle_deception_summary)

    deception_parser.set_defaults(func=lambda args: deception_parser.print_help())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gepa-deception",
        description="Mindful Trace GEPA deception utilities",
    )
    subparsers = parser.add_subparsers(dest="command")
    register_cli(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


__all__ = [
    "handle_deception_probes",
    "handle_deception_summary",
    "register_cli",
    "build_parser",
    "main",
]
