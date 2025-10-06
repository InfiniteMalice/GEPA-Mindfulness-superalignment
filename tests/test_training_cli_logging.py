from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = textwrap.dedent(
    """
    import sys
    import types
    from dataclasses import dataclass

    def install_pipeline_stub() -> None:
        pipeline = types.ModuleType("gepa_mindfulness.training.pipeline")

        @dataclass
        class RolloutResult:
            prompt: str
            response: str
            reward: float
            trace_summary: dict
            contradiction_report: dict

        class TrainingOrchestrator:
            def __init__(self, config=None, **_kwargs) -> None:
                self._results = [
                    RolloutResult(
                        prompt="prompt one",
                        response="stub response",
                        reward=0.42,
                        trace_summary={"step": 1},
                        contradiction_report={"issues": []},
                    )
                ]

            def run(self, _prompts):
                return list(self._results)

            def run_adversarial_eval(self):
                return list(self._results)

        pipeline.RolloutResult = RolloutResult
        pipeline.TrainingOrchestrator = TrainingOrchestrator
        sys.modules["gepa_mindfulness.training.pipeline"] = pipeline

    def install_configs_stub() -> None:
        configs = types.ModuleType("gepa_mindfulness.training.configs")

        @dataclass
        class TrainingConfig:
            seed: int = 0

        def load_training_config(_path):
            return TrainingConfig()

        configs.TrainingConfig = TrainingConfig
        configs.load_training_config = load_training_config
        sys.modules["gepa_mindfulness.training.configs"] = configs

    def install_jinja_stub() -> None:
        if "jinja2" in sys.modules:
            return
        jinja = types.ModuleType("jinja2")

        class Template:
            def __init__(self, *_args, **_kwargs) -> None:
                return

            def render(self, **_kwargs) -> str:
                return ""

        jinja.Template = Template
        sys.modules["jinja2"] = jinja

    install_pipeline_stub()
    install_configs_stub()
    install_jinja_stub()

    import gepa_mindfulness.training.cli as cli

    cli.iterate_adversarial_pool = lambda: []

    cli.main()
    """
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_stub_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: value\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("prompt one\n", encoding="utf-8")
    log_dir = tmp_path / "logs"
    return config_path, dataset_path, log_dir


def _run_cli(
    args: list[str],
    env: dict[str, str],
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-c", SCRIPT, *args]
    return subprocess.run(
        command,
        env=env,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )


def test_training_cli_logs_via_subprocess(tmp_path: Path) -> None:
    config_path, dataset_path, log_dir = _write_stub_files(tmp_path)
    env = os.environ.copy()
    project_root = _project_root()
    pythonpath_parts = [str(project_root), str(project_root / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--log-dir",
            str(log_dir),
        ],
        env,
    )

    assert result.returncode == 0, result.stderr

    rollout_log = log_dir / "rollouts.jsonl"
    training_log = log_dir / "training.log"

    assert rollout_log.exists()
    assert training_log.exists()

    payloads = [json.loads(line) for line in rollout_log.read_text(encoding="utf-8").splitlines()]
    assert payloads
    assert payloads[0]["response"] == "stub response"
