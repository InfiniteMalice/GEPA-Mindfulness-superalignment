from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

# Provide a lightweight pydantic stub so the training CLI can be imported without
# the optional dependency installed in the test environment.
if "pydantic" not in sys.modules:
    stub = types.ModuleType("pydantic")

    class _StubBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def parse_obj(cls, payload):  # type: ignore[override]
            return cls(**payload)

        def dict(self):  # type: ignore[override]
            return self.__dict__.copy()

    def _stub_field(*_args, default=None, **_kwargs):
        return default

    def _stub_validator(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    stub.BaseModel = _StubBaseModel
    stub.Field = _stub_field
    stub.validator = _stub_validator
    sys.modules["pydantic"] = stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    def _stub_device(value):
        return value

    def _stub_manual_seed(_seed):
        return None

    def _stub_tensor(values):
        return values

    torch_stub.device = _stub_device
    torch_stub.manual_seed = _stub_manual_seed
    torch_stub.tensor = _stub_tensor
    sys.modules["torch"] = torch_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device):
            return self

        def generate(self, **_kwargs):
            return [[0]]

    class _TokenizerCall(dict):
        def to(self, _device):
            return self

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, _prompt, **_kwargs):
            return _TokenizerCall({"input_ids": [[0]]})

        def decode(self, _tokens, **_kwargs):
            return ""

    transformers_stub.AutoModelForCausalLM = _AutoModel
    transformers_stub.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = transformers_stub

if "trl" not in sys.modules:
    trl_stub = types.ModuleType("trl")

    class _PPOConfig:
        def __init__(self, **_kwargs):
            return

    class _PPOTrainer:
        def __init__(self, **_kwargs):
            return

        def step(self, *_args, **_kwargs):
            return None

    trl_stub.PPOConfig = _PPOConfig
    trl_stub.PPOTrainer = _PPOTrainer
    sys.modules["trl"] = trl_stub

if "jinja2" not in sys.modules:
    jinja_stub = types.ModuleType("jinja2")

    class _Template:
        def __init__(self, *_args, **_kwargs):
            return

        def render(self, **_kwargs):
            return ""

    jinja_stub.Template = _Template
    sys.modules["jinja2"] = jinja_stub

cli = importlib.import_module("gepa_mindfulness.training.cli")
RolloutResult = importlib.import_module("gepa_mindfulness.training.pipeline").RolloutResult


class _StubOrchestrator:
    def __init__(self, results: list[RolloutResult]):
        self._results = results
        self.run_calls: list[list[str]] = []
        self.adversarial_calls = 0

    def run(self, prompts: list[str]) -> list[RolloutResult]:
        self.run_calls.append(list(prompts))
        return self._results

    def run_adversarial_eval(self) -> list[RolloutResult]:
        self.adversarial_calls += 1
        return self._results


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    root_logger = logging.getLogger()
    # Remove any file handlers attached by previous tests to avoid side effects.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def _write_stub_files(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: value\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("prompt one\n", encoding="utf-8")
    return config_path, dataset_path


def test_training_cli_writes_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, dataset_path = _write_stub_files(tmp_path)
    log_dir = tmp_path / "logs"

    results = [
        RolloutResult(
            prompt="prompt one",
            response="response",
            reward=0.5,
            trace_summary={"step": 1},
            contradiction_report={"issues": []},
        )
    ]
    orchestrator = _StubOrchestrator(results)

    monkeypatch.setattr(cli, "load_training_config", lambda _: object())
    monkeypatch.setattr(cli, "TrainingOrchestrator", lambda config: orchestrator, raising=False)
    monkeypatch.setattr(cli, "_resolve_orchestrator_factory", lambda: lambda config: orchestrator)
    monkeypatch.setattr(cli, "iterate_adversarial_pool", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gepa-train",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--log-dir",
            str(log_dir),
        ],
    )

    cli.main()

    rollout_log = log_dir / "rollouts.jsonl"
    training_log = log_dir / "training.log"

    assert rollout_log.exists()
    assert training_log.exists()

    payloads = [json.loads(line) for line in rollout_log.read_text(encoding="utf-8").splitlines()]
    assert payloads == [asdict(result) for result in results]


def test_training_cli_prompts_for_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, dataset_path = _write_stub_files(tmp_path)
    chosen_dir = tmp_path / "prompted"

    results = [
        RolloutResult(
            prompt="prompt one",
            response="response",
            reward=1.0,
            trace_summary={},
            contradiction_report={},
        )
    ]
    orchestrator = _StubOrchestrator(results)

    monkeypatch.setattr(cli, "load_training_config", lambda _: object())
    monkeypatch.setattr(cli, "TrainingOrchestrator", lambda config: orchestrator, raising=False)
    monkeypatch.setattr(cli, "_resolve_orchestrator_factory", lambda: lambda config: orchestrator)
    monkeypatch.setattr(cli, "iterate_adversarial_pool", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gepa-train",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
        ],
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: str(chosen_dir))

    cli.main()

    assert (chosen_dir / "rollouts.jsonl").exists()
    assert (chosen_dir / "training.log").exists()


def test_training_cli_uses_default_log_dir_when_non_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path, dataset_path = _write_stub_files(tmp_path)

    results = [
        RolloutResult(
            prompt="prompt one",
            response="response",
            reward=0.25,
            trace_summary={},
            contradiction_report={},
        )
    ]
    orchestrator = _StubOrchestrator(results)

    monkeypatch.setattr(cli, "load_training_config", lambda _: object())
    monkeypatch.setattr(cli, "TrainingOrchestrator", lambda config: orchestrator, raising=False)
    monkeypatch.setattr(cli, "_resolve_orchestrator_factory", lambda: lambda config: orchestrator)
    monkeypatch.setattr(cli, "iterate_adversarial_pool", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["gepa-train", "--config", str(config_path), "--dataset", str(dataset_path)],
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("GEPA_MINDFULNESS_TRAINING_ASSUME_TTY", raising=False)

    cli.main()

    default_dir = tmp_path / "training_logs"
    assert (default_dir / "rollouts.jsonl").exists()
    assert (default_dir / "training.log").exists()
