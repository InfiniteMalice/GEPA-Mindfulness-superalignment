"""Convenience wrapper for running the CPU demo from the examples directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_demo(args: Sequence[str], repo_root: Path = REPO_ROOT) -> int:
    """Execute the canonical module-based CPU demo command in a subprocess."""

    cmd = [sys.executable, "-m", "gepa_mindfulness.examples.cpu_demo.run_cpu_demo", *args]
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
