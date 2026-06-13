"""Convenience wrapper for running the CPU demo from the examples directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./runs/<trainer>_cpu_demo)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Working directory for the training command (default: current directory)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_demo(
    trainer: str,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    config_path = CONFIG_MAP[trainer]
    run_root = (repo_root or Path.cwd()).resolve()
    resolved_output = output_dir or (Path.cwd() / "runs" / f"{trainer}_cpu_demo")
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_output = resolved_output.resolve()
    adjusted_config = _materialize_config(
        config_path=config_path,
        trainer=trainer,
        run_root=run_root,
        output_dir=resolved_output,
    )
    cmd = [
        sys.executable,
        "-m",
        "gepa_mindfulness.training.cli",
        "train",
        "--trainer",
        trainer,
        "--config",
        str(adjusted_config),
        "--output",
        str(resolved_output),
    ]
    completed = subprocess.run(cmd, cwd=run_root, check=False)
    return completed.returncode


def _materialize_config(
    *,
    config_path: Path,
    trainer: str,
    run_root: Path,
    output_dir: Path,
) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload["dataset_path"] = str(DATASET_PATH.resolve())
    payload["output_dir"] = str(output_dir)
    target = run_root / f".gepa_{trainer}_cpu_demo.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    return_code = _run_demo(args.trainer, repo_root=args.root, output_dir=args.output)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
