"""An explicitly enabled CPU runner never changes evidence, rewards or permissions."""

import json
import subprocess
import sys
from importlib.resources import files

import pytest

from evaluation.run_metacognitive import main, run_demo


def test_disabled_runner_does_not_read_spec():
    assert run_demo(object()) == {"enabled": False, "status": "DISABLED"}
    with pytest.raises(ValueError):
        run_demo(object(), enabled=1)


@pytest.mark.parametrize(
    "name,kind", [("unsupported", "verify_assumption"), ("healthy", "continue")]
)
def test_packaged_cpu_demos_are_deterministic_and_preserve_authority(name, kind):
    path = files("gepa_mindfulness").joinpath(f"data/metacognitive_{name}_demo.json")
    spec = json.loads(path.read_text())
    a = run_demo(spec, enabled=True)
    assert a == run_demo(spec, enabled=True)
    assert a["proposal"]["kind"] == kind
    assert a["invariants"] == {
        "accepted_evidence_unchanged": True,
        "reward_unchanged": True,
        "authority_granted": False,
        "execution_performed": False,
    }
    if name == "healthy":
        assert a["report"]["assessment"]["findings"] == []


def test_disabled_cli_ignores_even_missing_input(tmp_path, capsys):
    assert main(["--input", str(tmp_path / "missing.json")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DISABLED"


def test_malformed_demo_schema_is_rejected():
    spec = json.loads(
        files("gepa_mindfulness").joinpath("data/metacognitive_healthy_demo.json").read_text()
    )
    for changed in (
        spec | {"grant_authority": True},
        spec | {"repeat_id": True},
        spec | {"schema_version": "future"},
    ):
        with pytest.raises(ValueError):
            run_demo(changed, enabled=True)


def test_duplicate_json_spec_fields_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    resource = files("gepa_mindfulness").joinpath("data/metacognitive_healthy_demo.json")
    path.write_text(resource.read_text().replace('"case_id": 1', '"case_id": 17, "case_id": 1'))
    with pytest.raises(SystemExit) as error:
        main(["--enable", "--input", str(path)])
    assert error.value.code == 2


def test_cli_battery_exports_raw_results_and_unknown_outcomes(tmp_path, capsys):
    out = tmp_path / "results.jsonl"
    assert (
        main(
            [
                "--enable",
                "--battery",
                "--case-id",
                "1",
                "--stripe-id",
                "NONE",
                "--repeats",
                "1",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["unit_count"] == 11
    assert summary["arm_count"] == 44
    assert summary["final_behavior_status"] == "NOT_RUN"
    assert summary["verified_task_success"] is None
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 11 and all(
        r["conditions"][3]["execution_status"] == "SIMULATION_ONLY" for r in rows
    )


def test_cpu_module_import_does_not_load_optional_model_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import evaluation.run_metacognitive; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
