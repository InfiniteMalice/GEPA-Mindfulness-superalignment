from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

STUB_MODULE = """
from dataclasses import dataclass
from typing import Iterable


@dataclass
class StubRollout:
    prompt: str
    response: str
    reward: float
    trace_summary: dict
    contradiction_report: dict


class StubTrainingOrchestrator:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, prompts: Iterable[str]) -> list[StubRollout]:
        prompts_list = list(prompts)
        prompt = prompts_list[0] if prompts_list else ""
        return [
            StubRollout(
                prompt=prompt,
                response="stub-response",
                reward=1.23,
                trace_summary={"trace": True},
                contradiction_report={"conflict": False},
            )
        ]

    def run_adversarial_eval(self) -> list[StubRollout]:
        return []


def create_orchestrator(config) -> StubTrainingOrchestrator:
    return StubTrainingOrchestrator(config)
"""


PYDANTIC_STUB = """
_MISSING = object()


class _FieldInfo:
    def __init__(self, default=_MISSING, default_factory=None, **_kwargs):
        self.default = default
        self.default_factory = default_factory


def Field(default=_MISSING, *, default_factory=None, **_kwargs):
    return _FieldInfo(default=default, default_factory=default_factory)


def validator(*_args, **_kwargs):
    def decorator(func):
        return func

    return decorator


class BaseModel:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__fields__ = {}
        cls.__annotations__ = getattr(cls, "__annotations__", {})
        for name in cls.__annotations__:
            cls.__fields__[name] = getattr(cls, name, _MISSING)

    def __init__(self, **data):
        for name, default in self.__class__.__fields__.items():
            if name in data:
                value = data[name]
            else:
                value = self._resolve_default(default)

            annotation = self.__class__.__annotations__.get(name)
            if isinstance(value, dict) and isinstance(annotation, type) and issubclass(
                annotation, BaseModel
            ):
                value = annotation.parse_obj(value)
            setattr(self, name, value)

    @staticmethod
    def _resolve_default(default):
        if isinstance(default, _FieldInfo):
            if default.default_factory is not None:
                return default.default_factory()
            if default.default is not _MISSING:
                return default.default
            return None
        if default is _MISSING:
            return None
        if isinstance(default, type) and issubclass(default, BaseModel):
            return default()
        return default

    @classmethod
    def parse_obj(cls, obj):
        return cls(**obj)

    def dict(self):
        result = {}
        for name in self.__class__.__fields__:
            value = getattr(self, name)
            if isinstance(value, BaseModel):
                result[name] = value.dict()
            else:
                result[name] = value
        return result
"""


def _prepare_environment(tmp_path: Path) -> dict[str, str]:
    stub_path = tmp_path / "training_stub.py"
    stub_path.write_text(STUB_MODULE, encoding="utf-8")
    (tmp_path / "pydantic.py").write_text(PYDANTIC_STUB, encoding="utf-8")

    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    extras = [env.get("PYTHONPATH", ""), str(project_root), str(src_dir), str(tmp_path)]
    env["PYTHONPATH"] = os.pathsep.join(path for path in extras if path)
    env["GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR"] = "training_stub:create_orchestrator"
    return env


def _drain_master_fd(master_fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if master_fd not in ready:
            break
        try:
            data = os.read(master_fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _run_cli(
    args: list[str],
    env: dict[str, str],
    input_text: str | None = None,
    *,
    use_tty: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "gepa_mindfulness.training.cli", *args]
    if not use_tty:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout, stderr = proc.communicate(input_text)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    pending_input = input_text.encode() if input_text is not None else None
    sent_input = False
    start = time.monotonic()
    stdout_chunks: list[bytes] = []
    while True:
        chunk = _drain_master_fd(master_fd)
        if chunk:
            stdout_chunks.append(chunk)
        if pending_input and not sent_input:
            buffered = b"".join(stdout_chunks)
            if b"Enter log output directory path" in buffered:
                os.write(master_fd, pending_input)
                sent_input = True
            elif time.monotonic() - start > 1.0:
                os.write(master_fd, pending_input)
                sent_input = True
        if proc.poll() is not None:
            tail = _drain_master_fd(master_fd)
            if tail:
                stdout_chunks.append(tail)
            break
        time.sleep(0.05)

    if pending_input and not sent_input:
        os.write(master_fd, pending_input)

    os.close(master_fd)
    stderr = proc.stderr.read().decode()
    proc.stderr.close()
    stdout = b"".join(stdout_chunks).decode()
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def test_training_cli_prompts_and_logs(tmp_path: Path) -> None:
    env = _prepare_environment(tmp_path)
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("test prompt\n", encoding="utf-8")
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")

    # When --log-dir is omitted the CLI should request one interactively.
    interactive_log_dir = tmp_path / "interactive_logs"
    result = _run_cli(
        ["--config", str(config), "--dataset", str(dataset)],
        env=env,
        input_text=f"{interactive_log_dir}\n",
        use_tty=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Enter log output directory path" in result.stderr
    training_log = interactive_log_dir / "training.log"
    rollouts_log = interactive_log_dir / "rollouts.jsonl"
    assert training_log.exists()
    assert rollouts_log.exists()

    with rollouts_log.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert lines and lines[0]["response"] == "stub-response"

    # When --log-dir is provided the CLI should write the logs without prompting.
    provided_log_dir = tmp_path / "provided_logs"
    result_with_flag = _run_cli(
        [
            "--config",
            str(config),
            "--dataset",
            str(dataset),
            "--log-dir",
            str(provided_log_dir),
        ],
        env=env,
    )
    assert result_with_flag.returncode == 0, result_with_flag.stderr
    assert "Enter log output directory path" not in result_with_flag.stderr
    file_log = provided_log_dir / "training.log"
    jsonl_log = provided_log_dir / "rollouts.jsonl"
    assert file_log.exists()
    assert jsonl_log.exists()
    contents = file_log.read_text(encoding="utf-8")
    assert "Serialized 1 rollouts" in contents
