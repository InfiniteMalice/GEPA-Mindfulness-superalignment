import json
import json
import argparse
import os
from pathlib import Path

from mindful_trace_gepa.cli_scoring import handle_score_auto


def test_score_auto_generates_output(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {
            "content": "We assume 30% uptake with monitoring; stakeholders include users and regulators.",
            "gepa_hits": ["monitor"],
        },
        {
            "content": "Policy GEPA-1 forbids deception; compare option A vs contingency B.",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(evt) for evt in events), encoding="utf-8")

    out_path = tmp_path / "scores.json"
    monkeypatch.setenv("GEPA_JUDGE_MOCK", "1")
    args = argparse.Namespace(
        trace=str(trace_path),
        policy=None,
        out=str(out_path),
        config="configs/scoring.yml",
        judge=True,
        classifier=False,
        classifier_config=None,
        classifier_artifacts=None,
        print=False,
    )
    handle_score_auto(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "final" in payload and "confidence" in payload
    assert payload["per_tier"]
    assert any(tier["tier"] == "judge" for tier in payload["per_tier"])
