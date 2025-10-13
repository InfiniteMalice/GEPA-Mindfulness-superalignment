"""Standalone entry point for the CPU demo from the top-level examples tree."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    """Guarantee that the repository root is importable."""

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def main(argv: Sequence[str] | None = None) -> None:
    """Mirror ``python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo``."""

    _ensure_repo_on_path()
    from gepa_mindfulness.examples.cpu_demo.run_cpu_demo import main as demo_main

    if argv is None:
        demo_main()
        return

    original_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        demo_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
