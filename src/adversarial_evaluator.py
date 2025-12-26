"""Deprecated entry point for dual-path evaluation."""

from __future__ import annotations

from src.dual_path_evaluator import main

# NOTE: New implementation lives in dual_path_evaluator.py; keep this file as a thin shim.


if __name__ == "__main__":
    raise SystemExit(main())
