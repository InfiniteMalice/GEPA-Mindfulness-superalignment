"""Dual-path execution view."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, TextLog

from app.services.runners import build_dual_path_command, run_command

DEFAULT_DATASET_PATH = "datasets/dual_path/data.jsonl"


class DualPathView(Container):
    """Allow researchers to launch dual-path evaluations."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label("Model / response path"),
            Input(placeholder="path/to/model_or_response.txt", id="model-path"),
        )
        yield Horizontal(
            Label("Scenario dataset"),
            Input(placeholder=DEFAULT_DATASET_PATH, id="dataset-path"),
        )
        yield Horizontal(
            Label("Run directory"),
            Input(placeholder="runs/latest", id="run-path"),
        )
        yield Button("Run Dual-Path", id="run-dual-path", variant="success")
        yield TextLog(id="dual-log", highlight=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run-dual-path":
            return
        model_path = self.query_one("#model-path", Input).value
        dataset_widget = self.query_one("#dataset-path", Input)
        dataset_path = dataset_widget.value or DEFAULT_DATASET_PATH
        run_path = self.query_one("#run-path", Input).value or "runs/latest"

        log_widget = self.query_one("#dual-log", TextLog)
        log_widget.clear()
        log_widget.write("Launching dual-path evaluation...")

        command = build_dual_path_command(
            model_path=model_path,
            dataset_path=dataset_path,
            run_dir=run_path,
        )

        def sink(line: str) -> None:
            log_widget.write(line)

        await run_command(command, sink)
        log_widget.write("Dual-path evaluation completed.")
