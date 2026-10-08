"""Ablation CLI reports invalid inputs and output failures without tracebacks."""

import json
from pathlib import Path

import pytest

from scripts.run_epistemic_ablations import main


@pytest.mark.parametrize(
    "contents,detail",
    [
        (None, "cannot summarize"),
        (b"\n{}\nnot-json\n", ":3:"),
        (b"{}\n", "trial has missing or unknown fields"),
        (b"null\n", "JSON object"),
        (b"\xff\n", "cannot summarize"),
    ],
)
def test_input_errors_are_usage_errors(
    tmp_path: Path, monkeypatch, capsys, contents: bytes | None, detail: str
) -> None:
    source = tmp_path / "trials.jsonl"
    if contents is not None:
        source.write_bytes(contents)
    monkeypatch.setattr("sys.argv", ["ablations", "--input", str(source)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert str(source) in captured.err
    assert detail in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("field,bad_value", [("ablation", []), ("split", {})])
def test_invalid_label_types_are_usage_errors(
    tmp_path: Path, monkeypatch, capsys, field: str, bad_value: object
) -> None:
    source = tmp_path / "trials.jsonl"
    row = {
        "run_id": "r1",
        "ablation": "A",
        "family_id": "f1",
        "split": "DEVELOPMENT",
        "metrics": {"correctness": 0.5},
        "cost": {"tool_calls": 1, "latency_seconds": 1, "verification_cost": 1},
    }
    source.write_text(json.dumps(row | {field: bad_value}) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["ablations", "--input", str(source)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert str(source) in captured.err
    assert field in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("failure_kind", ["parent_file", "output_directory"])
def test_output_errors_are_usage_errors(
    tmp_path: Path, monkeypatch, capsys, failure_kind: str
) -> None:
    if failure_kind == "parent_file":
        parent = tmp_path / "blocked"
        parent.write_text("keep", encoding="utf-8")
        output = parent / "summary.json"
    else:
        output = tmp_path / "directory"
        output.mkdir()
    monkeypatch.setattr("sys.argv", ["ablations", "--output", str(output)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert str(output) in captured.err
    assert "cannot write" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("mode", ["matrix", "fixture", "input"])
def test_successful_cli_operations(tmp_path: Path, monkeypatch, capsys, mode: str) -> None:
    arguments = ["ablations"]
    output = tmp_path / "nested" / "result.json"
    if mode == "fixture":
        arguments += ["--fixture-smoke", "--output", str(output)]
    elif mode == "input":
        source = tmp_path / "trials.jsonl"
        row = {
            "run_id": "r1",
            "ablation": "A",
            "family_id": "f1",
            "split": "DEVELOPMENT",
            "metrics": {"correctness": 0.5},
            "cost": {"tool_calls": 1, "latency_seconds": 1, "verification_cost": 1},
        }
        source.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")
        arguments += ["--input", str(source)]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(output.read_text(encoding="utf-8") if mode == "fixture" else captured.out)
    if mode == "matrix":
        assert tuple(result["matrix"]) == tuple("ABCDEFGHIJK")
    elif mode == "fixture":
        assert result["selected_check_ids"] == ["discriminate"]
    else:
        assert result["A"]["metrics"]["correctness"]["mean"] == 0.5
