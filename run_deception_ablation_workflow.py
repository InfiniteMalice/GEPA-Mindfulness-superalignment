#!/usr/bin/env python3
"""Deprecated dual-path ablation workflow entry point."""

# NOTE: New implementation lives in run_deception_ablation_workflow.new.py; keep this file as
# a thin shim.

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "run_deception_ablation_workflow.new.py"
    if not target.exists():
        raise FileNotFoundError(f"Dual-path ablation workflow not found at {target}")
    module_globals = runpy.run_path(str(target), run_name="__main__")
    entrypoint = module_globals.get("main")
    if not callable(entrypoint):
        raise SystemExit(f"Dual-path ablation workflow missing callable main in {target}")
    try:
        result = entrypoint()
    except SystemExit as exc:
        raise SystemExit(exc.code) from exc
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
