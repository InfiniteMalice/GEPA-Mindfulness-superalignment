"""Deprecated entry point for dual-path evaluation."""

from __future__ import annotations

import sys
import warnings

# NOTE: New implementation lives in dual_path_evaluator.py; keep this file as a thin shim.


def main() -> int:
    warnings.warn(
        "src.adversarial_evaluator is deprecated; use src.dual_path_evaluator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from src.dual_path_evaluator import main as dual_main
    except ImportError as exc:
        print(f"Failed to import dual-path evaluator: {exc}", file=sys.stderr)
        return 1
    result = dual_main()
    if isinstance(result, int):
        return result
    print("Dual-path evaluator returned non-integer exit code", file=sys.stderr)
    return 1


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
