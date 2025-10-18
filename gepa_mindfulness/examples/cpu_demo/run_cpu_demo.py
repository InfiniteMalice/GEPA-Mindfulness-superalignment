"""Convenience wrapper for running the CPU demo from the examples directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DATASET_PATH = Path(__file__).parent / "prompts.txt"

CONFIG_MAP = {
    "ppo": CONFIG_ROOT / "ppo" / "cpu_demo.yaml",
    "grpo": CONFIG_ROOT / "grpo" / "cpu_demo.yaml",
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight CPU demo")
    parser.add_argument(
        "--trainer",
        choices=["ppo", "grpo"],
        default="ppo",
        help="Trainer to execute (default: ppo)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_demo(trainer: str, repo_root: Path = REPO_ROOT) -> int:
    config_path = CONFIG_MAP[trainer]
    output_dir = REPO_ROOT / "runs" / f"{trainer}_cpu_demo"
    cmd = [
        sys.executable,
        "-m",
        "gepa_mindfulness.training.cli",
        "train",
        "--trainer",
        trainer,
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
    ]
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    return_code = _run_demo(args.trainer)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
