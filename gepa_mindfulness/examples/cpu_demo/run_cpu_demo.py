"""Convenience wrapper for running the CPU demo from the examples directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

# ``gepa_mindfulness/examples/cpu_demo`` lives two directories beneath the package root
# (``gepa_mindfulness``) and three beneath the repository root. When we spawn the
# subprocess below we want to run it from the top-level repository directory so that
# ``gepa_mindfulness`` is importable as a normal package.
REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
DATASET_PATH = Path(__file__).parent / "prompts.txt"


def _run_demo(args: Sequence[str], repo_root: Path = REPO_ROOT) -> int:
    """Execute the canonical training CLI command in a subprocess."""

    cmd = [
        sys.executable,
        "-m",
        "gepa_mindfulness.training.cli",
        "--config",
        str(CONFIG_PATH),
        "--dataset",
        str(DATASET_PATH),
        *args,
    ]
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint that mirrors ``python -m gepa_mindfulness.examples.cpu_demo``."""

    args = list(argv if argv is not None else sys.argv[1:])
    return_code = _run_demo(args)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
