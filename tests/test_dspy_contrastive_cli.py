from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindful_trace_gepa import cli


def test_dspy_contrastive_subcommand_invokes_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data.jsonl"
    data_payload = json.dumps({"id": "example", "query": "How to proceed?"}) + "\n"
    data_path.write_text(data_payload)
    out_dir = tmp_path / "runs"

    called: dict[str, object] = {}

    def fake_runner(data: Path, out: Path, context: str) -> None:
        called["data"] = data
        called["out"] = out
        called["context"] = context

    monkeypatch.setattr(cli, "run_dual_path_contrastive", fake_runner)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "dspy",
            "contrastive-run",
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
            "--context",
            "general",
        ]
    )

    args.func(args)

    assert called == {"data": data_path, "out": out_dir, "context": "general"}


def test_run_dual_path_contrastive_reads_jsonl_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data.jsonl"
    record = {"id": "example", "query": "Proceed?", "ground_truth_correct_path": "one"}
    data_path.write_text(json.dumps(record) + "\n")
    out_dir = tmp_path / "runs"

    def fake_make_prompt(query: str, context: str) -> str:
        return f"prompt::{query}::{context}"

    def fake_parse_response(response: str) -> dict[str, str]:
        return {"raw": response}

    def fake_detect_deception(sections: dict[str, str]) -> dict[str, int]:
        return {"signals": len(sections)}

    monkeypatch.setattr(cli, "make_dual_path_prompt", fake_make_prompt)
    monkeypatch.setattr(cli, "parse_dual_path_response", fake_parse_response)
    monkeypatch.setattr(cli, "detect_deception_heuristic", fake_detect_deception)

    cli.run_dual_path_contrastive(data_path, out_dir, "general")

    summary_path = out_dir / "summary.json"
    response_path = out_dir / "example_response.txt"
    deception_path = out_dir / "example_deception.json"

    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary[0]["prompt"] == "prompt::Proceed?::general"
    assert summary[0]["deception_signals"] == {"signals": 1}

    assert response_path.exists()
    assert deception_path.exists()
