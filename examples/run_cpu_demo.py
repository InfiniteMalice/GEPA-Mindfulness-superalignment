"""Convenience wrapper for running the CPU demo from the examples directory."""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    """Temporarily place ``path`` at the front of ``sys.path``."""

    path_str = str(path)
    if sys.path and sys.path[0] == path_str:
        yield
        return

    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _run_demo(args: Sequence[str]) -> None:
    """Execute the CPU demo module with ``args`` injected into ``sys.argv``."""

    original_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).resolve()), *args]
    try:
        runpy.run_module(
            "gepa_mindfulness.examples.cpu_demo.run_cpu_demo",
            run_name="__main__",
            alter_sys=True,
        )
    finally:
        sys.argv = original_argv


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint that mirrors ``python -m gepa_mindfulness.examples.cpu_demo``."""

    args = list(argv if argv is not None else sys.argv[1:])
    with _temporary_sys_path(REPO_ROOT):
        _run_demo(args)


if __name__ == "__main__":
    main()
