from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindful_trace_gepa.cli import handle_score


def _make_event(idx: int) -> dict:
    return {
        "content": f"Event {idx}",
        "principle_scores": {"mindfulness": float(idx % 5)},
        "imperative_scores": {"care": float(idx % 3)},
        "flags": ["stream-test"],
    }


def test_streaming_score_large_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for idx in range(20_000):
            handle.write(json.dumps(_make_event(idx)) + "\n")

    out_path = tmp_path / "report.html"
    args = argparse.Namespace(
        trace=str(trace_path),
        policy=None,
        out=str(out_path),
        stream=True,
        sharded=False,
        manifest=None,
        zstd=False,
    )

    handle_score(args)

    html = out_path.read_text(encoding="utf-8")
    assert "Total events: 20000" in html or "20000" in html
