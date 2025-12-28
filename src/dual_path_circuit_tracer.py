"""Deprecated entry point for dual-path circuit tracing."""

from __future__ import annotations

import sys
import warnings

# NOTE: New implementation lives in mindful_trace_gepa.dual_path_circuit_tracer.


if __name__ == "__main__":
    try:
        from mindful_trace_gepa.dual_path_circuit_tracer import main
    except ImportError as exc:
        print(
            "Failed to import mindful_trace_gepa.dual_path_circuit_tracer: "
            f"{exc}. Run from the repo root or install the package.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    warnings.warn(
        "src/dual_path_circuit_tracer.py is deprecated. "
        "Use mindful_trace_gepa.dual_path_circuit_tracer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
    print("Dual-path circuit tracer returned non-integer exit code", file=sys.stderr)
    raise SystemExit(1)
