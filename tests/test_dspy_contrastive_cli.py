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

    def fake_runner(data: Path, out: Path, context: str, probes_path: Path | None = None) -> None:
        called["data"] = data
        called["out"] = out
        called["context"] = context
        called["probes_path"] = probes_path

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

    assert called == {
        "data": data_path,
        "out": out_dir,
        "context": "general",
        "probes_path": None,
    }


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
        return {"raw": response, "recommended_path": "path_2"}

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
    assert summary["results"][0]["prompt"] == "prompt::Proceed?::general"
    assert summary["results"][0]["deception_signals"] == {"signals": 2}
    assert summary["fingerprint_summary"]["total"] == 1

    assert response_path.exists()
    assert deception_path.exists()


def test_run_dual_path_contrastive_includes_dual_path_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "example", "query": "Proceed?"}) + "\n")

    probes_path = tmp_path / "probes.jsonl"
    probes_payload = {"id": "probe_1", "query": "Disclose incident?", "domain": "safety"}
    probes_path.write_text(json.dumps(probes_payload) + "\n")

    out_dir = tmp_path / "runs"

    cli.run_dual_path_contrastive(data_path, out_dir, "general", probes_path)

    summary = json.loads((out_dir / "summary.json").read_text())

    assert summary["counts"] == {"dataset_records": 1, "dual_path_probes": 1}
    assert summary["results"][1]["id"] == "probe_1"
    assert summary["results"][1]["source"] == "dual_path_probe"


def test_run_dual_path_contrastive_sanitizes_result_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "../escape", "query": "Proceed?"}) + "\n")
    out_dir = tmp_path / "runs"

    cli.run_dual_path_contrastive(data_path, out_dir, "general")

    sanitized_id = "escape"
    response_path = out_dir / f"{sanitized_id}_response.txt"
    deception_path = out_dir / f"{sanitized_id}_deception.json"
    summary = json.loads((out_dir / "summary.json").read_text())

    assert response_path.exists()
    assert deception_path.exists()
    assert summary["results"][0]["id"] == sanitized_id


def test_contrastive_parser_rejects_missing_probes(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "example", "query": "Proceed?"}) + "\n")
    out_dir = tmp_path / "runs"
    missing_probes = tmp_path / "missing.jsonl"

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "dspy",
            "contrastive-run",
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
            "--probes",
            str(missing_probes),
        ]
    )

    with pytest.raises(FileNotFoundError):
        args.func(args)
