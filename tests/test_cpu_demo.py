from __future__ import annotations

from pathlib import Path

from gepa_mindfulness.examples.cpu_demo import run_cpu_demo


def test_cpu_demo_honors_output_and_root(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, cwd, check):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check})

        class Completed:
            returncode = 0

        return Completed()

    output_dir = tmp_path / "custom-output"
    root_dir = tmp_path / "custom-root"
    monkeypatch.setattr(run_cpu_demo.subprocess, "run", fake_run)

    return_code = run_cpu_demo._run_demo("ppo", repo_root=root_dir, output_dir=output_dir)

    assert return_code == 0
    assert calls[0]["cwd"] == root_dir
    assert str(output_dir) in calls[0]["cmd"]
    config_path = Path(calls[0]["cmd"][calls[0]["cmd"].index("--config") + 1])
    config_text = config_path.read_text(encoding="utf-8")
    assert str(run_cpu_demo.DATASET_PATH) in config_text
    assert str(output_dir) in config_text


def test_cpu_demo_relative_root_passes_absolute_config(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, cwd, check):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check})

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_cpu_demo.subprocess, "run", fake_run)

    return_code = run_cpu_demo._run_demo("ppo", repo_root=Path("runs/demo-root"))

    assert return_code == 0
    assert calls[0]["cwd"] == (tmp_path / "runs/demo-root").resolve()
    config_path = Path(calls[0]["cmd"][calls[0]["cmd"].index("--config") + 1])
    assert config_path.is_absolute()
    assert config_path.exists()


def test_cpu_demo_parser_accepts_output_and_root(tmp_path: Path) -> None:
    args = run_cpu_demo._parse_args(
        ["--trainer", "grpo", "--output", str(tmp_path / "out"), "--root", str(tmp_path)]
    )

    assert args.trainer == "grpo"
    assert args.output == tmp_path / "out"
    assert args.root == tmp_path
