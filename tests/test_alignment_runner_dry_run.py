import json
import subprocess

import pytest

from evaluation.run_alignment_battery import _run_model_command, main


def test_alignment_runner_dry_run_writes_results_and_summary(tmp_path) -> None:
    output = tmp_path / "dry_run.jsonl"

    exit_code = main(["--suite", "simpleqa", "--dry-run", "--output-path", str(output)])

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "dry_run_summary.json").read_text(encoding="utf-8"))
    assert rows
    assert rows[0]["outcome"] == "needs_manual_review"
    assert rows[0]["metadata"]["dry_run"] is True
    assert summary["by_suite"]["simpleqa"] == len(rows)


def test_alignment_runner_category_dry_run_includes_calibration(tmp_path) -> None:
    output = tmp_path / "calibration_dry_run.jsonl"

    exit_code = main(["--category", "calibration", "--dry-run", "--output-path", str(output)])

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["category"] for row in rows} == {"calibration"}


def test_alignment_runner_scores_precomputed_responses(tmp_path) -> None:
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        '{"eval_id":"simpleqa-toy-1","model_answer":"Paris","confidence":0.9}\n',
        encoding="utf-8",
    )
    output = tmp_path / "scored.jsonl"

    exit_code = main(
        [
            "--suite",
            "simpleqa",
            "--responses-path",
            str(responses),
            "--output-path",
            str(output),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["outcome"] == "correct"
    assert row["gepa_score"] == 4


def test_alignment_runner_errors_when_requested_suite_has_no_fixture(tmp_path) -> None:
    output = tmp_path / "fever.jsonl"

    with pytest.raises(ValueError, match="fever has no input dataset"):
        main(["--suite", "fever", "--dry-run", "--output-path", str(output)])


def test_run_model_command_uses_no_shell_and_timeout(monkeypatch) -> None:
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _run_model_command("model --flag value", "prompt", timeout_seconds=3) == "ok"
    assert observed["args"] == ["model", "--flag", "value"]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["timeout"] == 3


def test_run_model_command_timeout_mentions_partial_output(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1, output="partial out", stderr="partial err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        _run_model_command("model --slow", "prompt", timeout_seconds=1)
