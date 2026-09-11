"""Contract tests for the deterministic V5 planner-only command-line interface."""

from __future__ import annotations

import json

import pytest

from evaluation.run_v5_framework import main
from evaluation.v5_runner import plan_v5_cells

_MODEL_VERSION = "mindful-model-2026-09-10"
_HARNESS_VERSION = "v5-harness-1.0.0"


def _required_arguments() -> list[str]:
    """Return the required version arguments for one V5 CLI invocation."""

    return ["--model-version", _MODEL_VERSION, "--harness-version", _HARNESS_VERSION]


def test_dry_run_writes_compact_jsonl_to_stdout_in_requested_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must preserve repeated selection order and serialize planned cells exactly."""

    exit_code = main(
        [
            "--dry-run",
            "--case",
            "14",
            "--case",
            "2",
            "--stripe",
            "TOOL_ERROR",
            "--stripe",
            "NONE",
            "--repeats",
            "3",
            "--base-seed",
            "23",
            *_required_arguments(),
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    expected = plan_v5_cells(
        case_ids=(14, 2),
        stripe_ids=("TOOL_ERROR", "NONE"),
        repeats=3,
        base_seed=23,
        model_version=_MODEL_VERSION,
        harness_version=_HARNESS_VERSION,
    )

    assert exit_code == 0
    assert captured.err == ""
    assert len(lines) == 12
    assert lines == [
        json.dumps(
            {
                "case_id": cell.case_id,
                "case_version": cell.case_version,
                "stripe_id": cell.stripe_id,
                "subtype": cell.subtype,
                "repeat_id": cell.repeat_id,
                "seed": cell.seed,
                "model_version": cell.model_version,
                "harness_version": cell.harness_version,
            },
            separators=(",", ":"),
        )
        for cell in expected
    ]


def test_dry_run_writes_utf8_newline_terminated_byte_stable_jsonl(tmp_path) -> None:
    """Same arguments must produce the same file bytes without adding stdout status text."""

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    arguments = ["--dry-run", "--repeats", "1", "--base-seed", "7", *_required_arguments()]

    assert main([*arguments, "--output", str(first)]) == 0
    assert main([*arguments, "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert len(first.read_text(encoding="utf-8").splitlines()) == 187


def test_default_dry_run_emits_all_935_planned_cells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default V5 grid covers all canonical cases, stripes, and five repeats."""

    assert main(["--dry-run", *_required_arguments()]) == 0

    assert len(capsys.readouterr().out.splitlines()) == 935


def test_validation_failure_does_not_overwrite_existing_output(tmp_path) -> None:
    """Planning validation must finish before the requested output path is opened for writing."""

    output = tmp_path / "planned.jsonl"
    output.write_text("preserve this file\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--dry-run",
                "--case",
                "999",
                "--output",
                str(output),
                *_required_arguments(),
            ]
        )

    assert exc_info.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve this file\n"


@pytest.mark.parametrize("value", ["+1", "01", "1_0", "1.0"])
def test_cli_rejects_noncanonical_integer_spellings(value: str) -> None:
    """CLI integer spellings must remain unambiguous before planner validation."""

    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run", "--repeats", value, *_required_arguments()])

    assert exc_info.value.code == 2


def test_cli_requires_versions_and_dry_run_mode() -> None:
    """The command must not imply model execution when only planner behavior exists."""

    with pytest.raises(SystemExit) as versions_error:
        main(["--dry-run"])
    with pytest.raises(SystemExit) as mode_error:
        main(_required_arguments())

    assert versions_error.value.code == 2
    assert mode_error.value.code == 2
