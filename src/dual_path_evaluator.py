"""Deprecated entry point for dual-path evaluation."""

from __future__ import annotations

import sys
import warnings


def main() -> int:
    """Run the canonical package-owned dual-path evaluator."""

    warnings.warn(
        "src/dual_path_evaluator.py is deprecated. "
        "Use python -m mindful_trace_gepa.dual_path_evaluator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from mindful_trace_gepa.dual_path_evaluator import main as dual_main
    except ImportError as exc:
        print(
            "Failed to import mindful_trace_gepa.dual_path_evaluator: "
            f"{exc}. Run from the repo root or install the package.",
            file=sys.stderr,
        )
        return 1
    result = dual_main()
    if isinstance(result, int):
        return result
    print("Dual-path evaluator returned non-integer exit code", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
