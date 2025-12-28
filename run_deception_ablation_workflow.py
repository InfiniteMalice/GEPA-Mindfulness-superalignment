#!/usr/bin/env python3
"""Legacy deception ablation entry point (deprecated; use run_dual_path_ablation_workflow.py)."""

# NOTE: New implementation lives in run_dual_path_ablation_workflow.py; keep this file as a
# thin shim.

from __future__ import annotations

import sys
import warnings
from pathlib import Path


def main() -> int:
    warnings.warn(
        "run_deception_ablation_workflow.py is deprecated; "
        "use run_dual_path_ablation_workflow.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from run_dual_path_ablation_workflow import main as dual_path_main

    return dual_path_main()


if __name__ == "__main__":
    raise SystemExit(main())
