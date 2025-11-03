"""Scratchpad-aware circuit tracing for dual-public adversarial runs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - fallback when transformers absent
    AutoTokenizer = None  # type: ignore[assignment]

LEDGER_REFERENCE_PATTERN = re.compile(r"\((?:L|l)(\d+)\)|\[(?:L|l)(\d+)\]")


@dataclass
class TokenSpan:
    """Token span referencing a ledger line."""

    reference: str
    token_start: int
    token_end: int
    text: str


@dataclass
class ScratchpadTrace:
    """Trace information derived from a scratchpad."""

    scratchpad: str
    spans: List[TokenSpan]


@dataclass
class ScenarioTrace:
    """Circuit tracing artefacts for a scenario."""

    scenario_id: str
    scratch_a: ScratchpadTrace
    scratch_b: ScratchpadTrace
    fingerprint: Mapping[str, float]


def load_eval_summary(run_dir: Path) -> Mapping[str, object]:
    """Load the evaluation summary file."""

    summary_path = run_dir / "eval.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Evaluation summary missing at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_scratchpads(run_dir: Path, scenario_id: str) -> Dict[str, str]:
    scenario_dir = run_dir / scenario_id
    return {
        "A": load_text(scenario_dir / "scratch_A.txt"),
        "B": load_text(scenario_dir / "scratch_B.txt"),
    }


def load_ledger(run_dir: Path, scenario_id: str) -> Dict[str, str]:
    scenario_dir = run_dir / scenario_id
    ledger_path = scenario_dir / "ledger.txt"
    ledger: Dict[str, str] = {}
    if not ledger_path.exists():
        return ledger
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^L(\d+):\s*(.+)$", line.strip())
        if not match:
            continue
        ledger[match.group(1)] = match.group(2)
    return ledger


def build_tokenizer(name: Optional[str]) -> Optional[object]:
    if name and AutoTokenizer is not None:
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception:  # pragma: no cover - best effort
            return None
    return None


def tokenize(text: str, tokenizer: Optional[object]) -> List[Dict[str, int]]:
    if tokenizer is None:
        tokens = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            tokens.append({"start": start, "end": end})
            cursor = end
        return tokens

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    offsets = encoded.get("offset_mapping", [])
    return [
        {"start": int(start), "end": int(end)}
        for start, end in offsets
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
    ]


def spans_from_references(text: str, tokenizer: Optional[object]) -> List[TokenSpan]:
    tokens = tokenize(text, tokenizer)
    spans: List[TokenSpan] = []
    for match in LEDGER_REFERENCE_PATTERN.finditer(text):
        index = match.group(1) or match.group(2)
        if not index:
            continue
        start_char, end_char = match.span()
        token_start = 0
        token_end = 0
        for idx, token in enumerate(tokens):
            if token_start == 0 and token["start"] <= start_char <= token["end"]:
                token_start = idx
            if token["start"] <= end_char <= token["end"]:
                token_end = idx + 1
                break
        snippet = text[start_char:end_char]
        spans.append(
            TokenSpan(
                reference=f"L{index}",
                token_start=token_start,
                token_end=token_end,
                text=snippet,
            )
        )
    return spans


def fingerprint_from_spans(spans: Iterable[TokenSpan]) -> Dict[str, float]:
    fingerprint: Dict[str, float] = {}
    for span in spans:
        width = max(span.token_end - span.token_start, 1)
        fingerprint[span.reference] = fingerprint.get(span.reference, 0.0) + 1.0 / width
    return fingerprint


def merge_fingerprints(traces: Iterable[ScratchpadTrace]) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for trace in traces:
        for key, value in fingerprint_from_spans(trace.spans).items():
            combined[key] = combined.get(key, 0.0) + value
    return combined


def build_ablation_mask(
    fingerprint: Mapping[str, float], threshold: float = 1.0
) -> Dict[str, float]:
    return {reference: score for reference, score in fingerprint.items() if score >= threshold}


def trace_scenario(
    run_dir: Path,
    scenario_id: str,
    tokenizer: Optional[object],
) -> ScenarioTrace:
    scratchpads = load_scratchpads(run_dir, scenario_id)
    trace_a = ScratchpadTrace(
        scratchpad=scratchpads.get("A", ""),
        spans=spans_from_references(scratchpads.get("A", ""), tokenizer),
    )
    trace_b = ScratchpadTrace(
        scratchpad=scratchpads.get("B", ""),
        spans=spans_from_references(scratchpads.get("B", ""), tokenizer),
    )
    fingerprint = merge_fingerprints([trace_a, trace_b])
    return ScenarioTrace(
        scenario_id=scenario_id,
        scratch_a=trace_a,
        scratch_b=trace_b,
        fingerprint=fingerprint,
    )


def save_traces(run_dir: Path, traces: Iterable[ScenarioTrace], apply_ablation: bool) -> None:
    tracer_dir = run_dir / "tracer"
    tracer_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scenarios": [
            {
                "id": trace.scenario_id,
                "fingerprint": dict(trace.fingerprint),
                "spans": [
                    {
                        "scratchpad": "A",
                        "reference": span.reference,
                        "token_start": span.token_start,
                        "token_end": span.token_end,
                        "text": span.text,
                    }
                    for span in trace.scratch_a.spans
                ]
                + [
                    {
                        "scratchpad": "B",
                        "reference": span.reference,
                        "token_start": span.token_start,
                        "token_end": span.token_end,
                        "text": span.text,
                    }
                    for span in trace.scratch_b.spans
                ],
            }
            for trace in traces
        ],
    }
    (tracer_dir / "circuit_fingerprint.json").write_text(
        json.dumps(fingerprint_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    ablation_payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scenarios": {
            trace.scenario_id: build_ablation_mask(trace.fingerprint) for trace in traces
        },
    }
    (tracer_dir / "ablation_mask.json").write_text(
        json.dumps(ablation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if apply_ablation:
        print("--apply-ablation requested; integrate with ablation workflow as needed.")
        # If circuit_tracer package available:
        # from circuit_tracer import fingerprint
        # fingerprint(spans, model=..., topk=20) -> dict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Trace dual-public adversarial runs")
    parser.add_argument(
        "run",
        nargs="?",
        help="Run directory produced by the evaluator",
    )
    parser.add_argument(
        "--run",
        dest="run_dir",
        help="Run directory produced by the evaluator",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Limit tracing to specific scenario identifiers",
    )
    parser.add_argument(
        "--tokenizer",
        help="Optional Hugging Face tokenizer name",
    )
    parser.add_argument(
        "--apply-ablation",
        action="store_true",
        help="Apply ablation mask after tracing",
    )
    args = parser.parse_args(argv)

    run_arg = args.run_dir or args.run
    if not run_arg:
        parser.error("run directory required (pass RUN or --run <path>)")

    run_dir = Path(run_arg)
    summary = load_eval_summary(run_dir)
    available = [scenario["id"] for scenario in summary.get("scenarios", [])]

    scenario_ids = args.scenario or available
    tokenizer = build_tokenizer(args.tokenizer)

    traces = [trace_scenario(run_dir, scenario_id, tokenizer) for scenario_id in scenario_ids]
    save_traces(run_dir, traces, args.apply_ablation)
    print(f"Saved circuit tracing artefacts in {run_dir / 'tracer'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
