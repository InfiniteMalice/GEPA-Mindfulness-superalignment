import json

from evaluation.run_alignment_battery import main


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
