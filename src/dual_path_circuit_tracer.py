"""Dual-path circuit tracing stub for attribution and fingerprint logging."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindful_trace_gepa.deception.circuit_analysis import detect_deception_heuristic
from mindful_trace_gepa.deception.fingerprints import DeceptionFingerprint, FingerprintCollector


def _load_traces(trace_path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                traces.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {trace_path} at line {line_num}: {payload}"
                ) from exc
    return traces


def _write_trace_report(trace_dir: Path, total: int) -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traces_processed": total,
        "circuits_identified": 0,
    }
    (trace_dir / "trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run_tracing(run_dir: Path) -> None:
    trace_path = run_dir / "dual_path_traces.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    traces = _load_traces(trace_path)
    collector = FingerprintCollector(str(run_dir))
    for trace in traces:
        sections = {
            "path_1": trace.get("path_1_public_reasoning", ""),
            "path_2": trace.get("path_2_public_reasoning", ""),
            "recommended_path": trace.get("recommended_path", "unclear"),
        }
        deception = detect_deception_heuristic(sections)
        fingerprint = DeceptionFingerprint(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=str(trace.get("prompt", "")),
            domain=str(trace.get("metadata", {}).get("domain", "general")),
            path_1_text=str(sections.get("path_1", "")),
            path_2_text=str(sections.get("path_2", "")),
            comparison="",
            recommendation="",
            recommended_path=str(sections.get("recommended_path", "unclear")),
            path_1_circuits={},
            path_2_circuits={},
            deception_detected=bool(deception.get("deception_detected")),
            confidence_score=float(deception.get("confidence_score", 0.0)),
            signals=dict(deception.get("signals", {})),
            reasons=list(deception.get("reasons", [])),
            model_checkpoint=str(trace.get("metadata", {}).get("model_id", "unknown")),
            training_step=_safe_int(trace.get("metadata", {}).get("training_step", 0)),
        )
        collector.add(fingerprint)
    _write_trace_report(run_dir, len(traces))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-path circuit tracing.")
    parser.add_argument("run_dir", help="Run directory containing dual_path_traces.jsonl")
    parser.add_argument(
        "--tokenizer", default=None, help="[DEPRECATED] Legacy flag, no longer used."
    )
    parser.add_argument(
        "--apply-ablation",
        action="store_true",
        help="[DEPRECATED] Legacy flag, no longer used.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_tracing(Path(args.run_dir))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
