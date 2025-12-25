"""Deprecated entry point for dual-path evaluation."""

# NOTE: New implementation lives in dual_path_evaluator.py; keep this file as a thin shim.

from __future__ import annotations

import sys

from dual_path_evaluator import main


if __name__ == "__main__":
    sys.exit(main())
