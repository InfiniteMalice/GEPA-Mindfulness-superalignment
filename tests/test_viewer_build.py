from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mindful_trace_gepa.cli import handle_view


def test_viewer_cli_build(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    tokens_path = tmp_path / "tokens.jsonl"
    trace_path.write_text('{"stage": "framing", "content": "hello"}\n', encoding="utf-8")
    tokens_path.write_text(
        (
            '{"idx": 0, "token": "hello", "logprob": -0.1, "topk": [], '
            '"conf": 0.9, "abstained": false, "ts": "2024"}\n'
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "report.html"
    args = SimpleNamespace(
        trace=str(trace_path),
        tokens=str(tokens_path),
        out=str(out_path),
        deception=None,
        paired=None,
    )
    handle_view(args)
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "window.__GEPA__" in html
    assert "hello" in html
