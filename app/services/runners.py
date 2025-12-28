"""Utility helpers for running CLI commands from the Textual app."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandRequest:
    """Specification for a subprocess command."""

    argv: Sequence[str]
    cwd: Path | None = None


async def run_command(request: CommandRequest, log: list[str] | Callable[[str], None]) -> int:
    """Run a subprocess and stream output to the provided log sink."""

    argv = list(request.argv)
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=request.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert process.stdout is not None  # pragma: no cover - contract
    async for chunk in process.stdout:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if isinstance(log, list):
                log.append(line)
            else:
                log(line)
    return await process.wait()


def build_dual_path_command(
    model_path: str,
    dataset_path: str,
    run_dir: str,
) -> CommandRequest:
    argv: list[str] = [
        "python",
        "src/dual_path_evaluator.py",
        "--scenarios",
        dataset_path,
        "--run",
        run_dir,
    ]
    if model_path:
        argv.extend(["--response", model_path])
    return CommandRequest(argv=argv)


def build_tracer_command(run_dir: str, tokenizer: str, apply_ablation: bool) -> CommandRequest:
    argv: list[str] = ["python", "src/dual_path_circuit_tracer.py", run_dir]
    if tokenizer:
        argv.extend(["--tokenizer", tokenizer])
    if apply_ablation:
        argv.append("--apply-ablation")
    return CommandRequest(argv=argv)


def build_merge_command(run_dir: str) -> CommandRequest:
    return CommandRequest(argv=["python", "tools/merge_run_inspection.py", run_dir])


def build_finetune_command(config_path: str) -> CommandRequest:
    argv: list[str] = ["python", "train_model.py"]
    if config_path:
        argv.extend(["--config", config_path])
    return CommandRequest(argv=argv)
