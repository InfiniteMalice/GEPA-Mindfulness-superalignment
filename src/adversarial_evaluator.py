"""Deprecated entry point for dual-path evaluation."""

# NOTE: New implementation lives in dual_path_evaluator.py; keep this file as a thin shim.

from __future__ import annotations

import runpy


def main() -> int:
    module_globals = runpy.run_path("src/dual_path_evaluator.py", run_name="__main__")
    entrypoint = module_globals.get("main")
    if callable(entrypoint):
        result = entrypoint()
        return result if isinstance(result, int) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
