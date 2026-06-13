from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mindful_trace_gepa.cli import handle_view
from mindful_trace_gepa.viewer import builder
from mindful_trace_gepa.viewer.builder import build_viewer_html, load_static_asset


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
        dual_path=None,
    )
    handle_view(args)
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert 'id="gepa-data"' in html
    assert "hello" in html


def test_static_asset_loader_ignores_staging_new_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "VIEWER_DIR", tmp_path)
    (tmp_path / "viewer.js").write_text("canonical", encoding="utf-8")
    (tmp_path / "viewer.js.new").write_text("staging", encoding="utf-8")

    assert load_static_asset("viewer.js") == "canonical"


def test_viewer_html_serializes_hostile_payloads_safely(tmp_path: Path) -> None:
    hostile_script = "</script><script>alert(1)</script>"
    hostile_img = "<img src=x onerror=alert(1)>"
    out_path = tmp_path / "report.html"

    build_viewer_html(
        trace_events=[
            {
                "stage": "reasoning",
                "content": hostile_script,
                "context": hostile_img,
                "gepa_hits": [hostile_img],
                "payload": {"dataset": hostile_script},
            }
        ],
        token_events=[{"token": hostile_img, "conf": 0.8}],
        output_path=out_path,
        deception={"score": 0.2, "reasons": [hostile_script]},
        scoring={
            "final": {hostile_img: 3},
            "confidence": {hostile_img: 0.7},
            "reasons": [hostile_script],
            "per_tier": [
                {
                    "tier": hostile_img,
                    "scores": {hostile_img: 3},
                    "confidence": {hostile_img: 0.7},
                    "meta": {
                        "rationales": {hostile_img: hostile_script},
                        "spans": {hostile_img: [{"start": 0, "end": len(hostile_script)}]},
                    },
                }
            ],
        },
        dual_path={"honest_chain": [hostile_img], "deceptive_chain": [hostile_script]},
        manifest={"dataset": hostile_img},
        settings={"model": hostile_script},
    )

    html = out_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "\\u003cimg src=x onerror=alert(1)\\u003e" in html
    assert 'type="application/json"' in html
