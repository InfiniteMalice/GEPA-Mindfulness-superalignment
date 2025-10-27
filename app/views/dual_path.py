"""Dual-path execution view."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, Switch, TextLog

from app.services.runners import build_dual_path_command, run_command


class DualPathView(Container):
    """Allow researchers to launch dual-path evaluations."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label("Model / response path"),
            Input(placeholder="path/to/model_or_response.txt", id="model-path"),
        )
        yield Horizontal(
            Label("Config path"),
            Input(placeholder="configs/training/phi3_dual_path.yml", id="config-path"),
        )
        yield Horizontal(
            Label("Scenario dataset"),
            Input(placeholder="adversarial_scenarios.jsonl", id="dataset-path"),
        )
        yield Horizontal(
            Label("Run directory"),
            Input(placeholder="runs/latest", id="run-path"),
        )
        yield Horizontal(
            Label("Capture scratchpads"),
            Switch(value=True, id="scratchpad-toggle"),
            Label("Enable DSPy"),
            Switch(value=False, id="dspy-toggle"),
        )
        yield Button("Run Dual-Path", id="run-dual-path", variant="success")
        yield TextLog(id="dual-log", highlight=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run-dual-path":
            return
        model_path = self.query_one("#model-path", Input).value
        config_path = self.query_one("#config-path", Input).value
        dataset_widget = self.query_one("#dataset-path", Input)
        dataset_path = dataset_widget.value or "adversarial_scenarios.jsonl"
        run_path = self.query_one("#run-path", Input).value or "runs/latest"
        scratchpad = self.query_one("#scratchpad-toggle", Switch).value
        dspy = self.query_one("#dspy-toggle", Switch).value

        log_widget = self.query_one("#dual-log", TextLog)
        log_widget.clear()
        log_widget.write("Launching dual-path evaluation...")

        command = build_dual_path_command(
            model_path=model_path,
            config_path=config_path,
            dataset_path=dataset_path,
            run_dir=run_path,
            scratchpad=scratchpad,
            dspy=dspy,
        )

        async def sink(line: str) -> None:
            log_widget.write(line)

        await asyncio.create_task(run_command(command, sink))
        log_widget.write("Dual-path evaluation completed.")
