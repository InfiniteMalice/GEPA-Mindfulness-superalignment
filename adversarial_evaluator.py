from __future__ import annotations

import runpy
from pathlib import Path

# Deprecated entry point for dual-path evaluation.
# NOTE: New implementation lives in src/dual_path_evaluator.py; keep this file as a thin shim.


def main() -> int:
    target = Path(__file__).resolve().parent / "src" / "dual_path_evaluator.py"
    if not target.exists():
        raise FileNotFoundError(f"Dual-path evaluator not found at {target}")
    module_globals = runpy.run_path(str(target), run_name="__main__")
    entrypoint = module_globals.get("main")
    if not callable(entrypoint):
        raise SystemExit(f"Dual-path evaluator missing callable main in {target}")
    result = entrypoint()
    if not isinstance(result, int):
        raise SystemExit("Dual-path evaluator returned non-integer exit code")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
