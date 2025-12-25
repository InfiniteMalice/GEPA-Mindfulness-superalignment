from __future__ import annotations

import runpy
from pathlib import Path

# Deprecated entry point for dual-path evaluation.
# NOTE: New implementation lives in src/dual_path_evaluator.py; keep this file as a thin shim.


def main() -> None:
    target = Path(__file__).resolve().parent / "src" / "dual_path_evaluator.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
