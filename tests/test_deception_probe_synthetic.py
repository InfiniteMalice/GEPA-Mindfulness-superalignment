import json
from pathlib import Path
from types import SimpleNamespace

from mindful_trace_gepa.cli_deception import handle_deception_probes, handle_deception_summary
from mindful_trace_gepa.deception.probes_linear import ProbeWeights, infer_probe
from mindful_trace_gepa.viewer.builder import build_viewer_html


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_linear_probe_cli_and_summary(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {
            "content": "Honest disclosure",
            "activations": {
                "-1": {
                    "tokens": [[0.2, 0.4], [0.5, 0.6]],
                    "token_to_step": [0, 0],
                }
            },
            "deception_label": 0,
        },
        {
            "content": "Potentially deceptive reply",
            "activations": {
                "-1": {
                    "tokens": [[0.6, 0.2], [0.4, 0.1]],
                    "token_to_step": [1, 1],
                }
            },
            "deception_label": 1,
        },
    ]
    _write_jsonl(trace_path, rows)

    weights_path = tmp_path / "weights.pt"
    weights_path.write_text(json.dumps({"weights": [0.9, -0.2], "bias": 0.05}), encoding="utf-8")

    probe_out = tmp_path / "deception_probe.json"
    args = SimpleNamespace(
        trace=str(trace_path),
        model="dummy",
        probe=str(weights_path),
        config=str(Path("configs/deception/probes_linear.yaml")),
        out=str(probe_out),
    )
    handle_deception_probes(args)
    probe_payload = json.loads(probe_out.read_text(encoding="utf-8"))
    assert probe_payload["status"] == "ok"
    assert probe_payload["scores"]["per_step"], "expected per-step scores"

    paired_path = tmp_path / "deception.json"
    paired_path.write_text(json.dumps({"score": 0.6, "reasons": ["synthetic"]}), encoding="utf-8")
    mm_path = tmp_path / "mm_eval.json"
    mm_path.write_text(json.dumps({"metrics": {"test": {"accuracy": 0.75}}}), encoding="utf-8")
    summary_out = tmp_path / "deception_summary.json"
    summary_args = SimpleNamespace(
        out=str(summary_out),
        probe=str(probe_out),
        paired=str(paired_path),
        mm=str(mm_path),
        runs=None,
    )
    handle_deception_summary(summary_args)
    summary_payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert "sources" in summary_payload
    assert summary_payload["sources"]["linear_probe"]["status"] == "ok"

    html_out = tmp_path / "viewer.html"
    deception_blob = {"probe": probe_payload, "summary": summary_payload}
    build_viewer_html(
        trace_events=rows,
        token_events=[],
        deception=deception_blob,
        output_path=html_out,
        paired={},
        manifest={},
        settings={},
        scoring={},
    )
    html_text = html_out.read_text(encoding="utf-8")
    assert "Deception Probe" in html_text


def test_infer_probe_basic():
    probe = ProbeWeights(weights=[0.5, -0.3])
    activations = {
        "layers": {
            "-1": {
                "tokens": [[0.1, 0.2], [0.3, 0.4]],
                "token_to_step": [0, 1],
            }
        },
        "pool": "mean",
    }
    result = infer_probe(
        activations,
        probe,
        pooling="mean",
        threshold_config={"type": "fixed_fpr", "fpr": 0.5},
        labels=[0, 1],
    )
    assert result["status"] == "ok"
    assert result["scores"]["per_step"], "expected per-step aggregation"
