"""Deprecated entry point for dual-path circuit tracing."""

# NOTE: New implementation lives in src/dual_path_circuit_tracer.py; keep this file as a thin shim.

from __future__ import annotations

import sys
import warnings


def main() -> int:
    warnings.warn(
        "adversarial_circuit_tracer is deprecated; "
        "use mindful_trace_gepa.dual_path_circuit_tracer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from src.dual_path_circuit_tracer import main as circuit_main
    except ImportError as exc:
        print(f"Failed to import dual-path circuit tracer: {exc}", file=sys.stderr)
        return 1
    result = circuit_main()
    if isinstance(result, int):
        return result
    print("Dual-path circuit tracer returned non-integer exit code", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
