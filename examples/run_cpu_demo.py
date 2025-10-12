"""Convenience wrapper for running the CPU demo from the examples directory."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _configure_path(repo_root: Path = REPO_ROOT) -> None:
    """Ensure the repository root is available on ``sys.path`` for imports."""
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    """Entrypoint that configures the path and dispatches to the real demo."""
    _configure_path()
    from gepa_mindfulness.examples.cpu_demo.run_cpu_demo import main as run_main

    run_main()


if __name__ == "__main__":
    main()
