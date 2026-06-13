"""Tests for GUI command construction helpers."""

from app.services.runners import build_dual_path_command


def test_dual_path_command_uses_package_module_entrypoint() -> None:
    """Dual-path GUI runs should use the installed package module path."""

    command = build_dual_path_command(
        model_path="hooks.example:generate",
        dataset_path="datasets/dual_path/data.jsonl",
        run_dir="runs/demo",
    )

    assert list(command.argv) == [
        "python",
        "-m",
        "mindful_trace_gepa.dual_path_evaluator",
        "--scenarios",
        "datasets/dual_path/data.jsonl",
        "--run",
        "runs/demo",
        "--response",
        "hooks.example:generate",
    ]
