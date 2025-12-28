"""Deprecated entry point for dual-path evaluation."""

from __future__ import annotations

import warnings

from src.dual_path_evaluator import main

# NOTE: New implementation lives in dual_path_evaluator.py; keep this file as a thin shim.

__all__ = ["main"]


if __name__ == "__main__":
    warnings.warn(
        "src.adversarial_evaluator is deprecated; use src.dual_path_evaluator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())
