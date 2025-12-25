#!/usr/bin/env python3
"""Deprecated dual-path ablation workflow entry point."""

# NOTE: New implementation lives in run_deception_ablation_workflow.new.py; keep this file as
# a thin shim.

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    target = Path(__file__).resolve().parent / "run_deception_ablation_workflow.new.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
