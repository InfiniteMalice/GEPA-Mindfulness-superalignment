"""Deprecated entry point for dual-path circuit tracing."""

from __future__ import annotations

# NOTE: New implementation lives in dual_path_circuit_tracer.py; keep this file as a thin shim.


def main() -> int:
    try:
        from src.dual_path_circuit_tracer import main as circuit_main
    except ImportError as exc:
        raise FileNotFoundError(f"Dual-path circuit tracer not found: {exc}") from exc
    result = circuit_main()
    if not isinstance(result, int):
        raise SystemExit("Dual-path circuit tracer returned non-integer exit code")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
