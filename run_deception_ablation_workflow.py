#!/usr/bin/env python3
"""Deprecated dual-path ablation workflow entry point."""

# NOTE: New implementation lives in run_dual_path_ablation_workflow.py; keep this file as a
# thin shim.

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


def main() -> int:
    warnings.warn(
        "run_deception_ablation_workflow.py is deprecated; "
        "use run_dual_path_ablation_workflow.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    target = Path(__file__).resolve().parent / "run_dual_path_ablation_workflow.py"
    if not target.exists():
        raise FileNotFoundError(f"Dual-path ablation workflow not found at {target}")
    module_globals = runpy.run_path(str(target), run_name="dual_path_ablation_workflow")
    entrypoint = module_globals.get("main")
    if not callable(entrypoint):
        raise SystemExit(f"Dual-path ablation workflow missing callable main in {target}")
    try:
        result = entrypoint()
    except SystemExit as exc:
        raise SystemExit(exc.code) from exc
    if not isinstance(result, int):
        raise SystemExit("Dual-path ablation workflow returned non-integer exit code")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
