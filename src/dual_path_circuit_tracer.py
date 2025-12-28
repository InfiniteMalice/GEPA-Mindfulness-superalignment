"""Deprecated entry point for dual-path circuit tracing."""

from __future__ import annotations

import sys
import warnings

# NOTE: New implementation lives in mindful_trace_gepa.dual_path_circuit_tracer.


def main() -> int:
    warnings.warn(
        "src/dual_path_circuit_tracer.py is deprecated. "
        "Use mindful_trace_gepa.dual_path_circuit_tracer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from mindful_trace_gepa.dual_path_circuit_tracer import main as circuit_main
    except ImportError as exc:
        print(
            "Failed to import mindful_trace_gepa.dual_path_circuit_tracer: "
            f"{exc}. Run from the repo root or install the package.",
            file=sys.stderr,
        )
        return 1
    result = circuit_main()
    if isinstance(result, int):
        return result
    print("Dual-path circuit tracer returned non-integer exit code", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
