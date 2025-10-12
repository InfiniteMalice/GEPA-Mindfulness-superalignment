"""Convenience wrapper for launching the CPU demo from the examples directory."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Sequence

_DEMO_PATH = Path(__file__).resolve().parent / "cpu_demo" / "run_cpu_demo.py"


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the CPU demo script located under ``examples/cpu_demo``."""

    if argv is None:
        runpy.run_path(str(_DEMO_PATH), run_name="__main__")
        return

    # ``run_path`` does not accept argv directly, so simulate ``sys.argv``.
    import sys

    original_argv = sys.argv[:]
    sys.argv = [str(_DEMO_PATH), *argv]
    try:
        runpy.run_path(str(_DEMO_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
