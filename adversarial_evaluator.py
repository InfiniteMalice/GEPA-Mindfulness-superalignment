from __future__ import annotations

import warnings

# Deprecated entry point for dual-path evaluation.
# NOTE: New implementation lives in src/dual_path_evaluator.py; keep this file as a thin shim.


def main() -> int:
    warnings.warn(
        "adversarial_evaluator is deprecated; use src/dual_path_evaluator.py instead",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from src.dual_path_evaluator import main as dual_main
    except ImportError as exc:
        raise FileNotFoundError(f"Cannot import dual-path evaluator: {exc}") from exc
    if not callable(dual_main):
        raise RuntimeError("Dual-path evaluator main is not callable")
    return dual_main()


if __name__ == "__main__":
    raise SystemExit(main())
