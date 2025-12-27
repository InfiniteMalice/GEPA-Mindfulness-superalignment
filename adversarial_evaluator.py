from __future__ import annotations

import runpy
import warnings
from pathlib import Path

# Deprecated entry point for dual-path evaluation.
# NOTE: New implementation lives in src/dual_path_evaluator.py; keep this file as a thin shim.


def main() -> int:
    warnings.warn(
        "adversarial_evaluator is deprecated; use src/dual_path_evaluator.py instead",
        DeprecationWarning,
        stacklevel=2,
    )
    target = Path(__file__).resolve().parent / "src" / "dual_path_evaluator.py"
    if not target.exists():
        raise FileNotFoundError(f"Dual-path evaluator not found at {target}")
    module_globals = runpy.run_path(str(target), run_name="dual_path_evaluator")
    entrypoint = module_globals.get("main")
    if not callable(entrypoint):
        raise RuntimeError(f"Dual-path evaluator missing callable main in {target}")
    return entrypoint()


if __name__ == "__main__":
    raise SystemExit(main())
