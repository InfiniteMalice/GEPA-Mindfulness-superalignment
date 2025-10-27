"""Circuit tracer view."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, Switch, TextLog

from app.services.runners import build_merge_command, build_tracer_command, run_command


class TracerView(Container):
    """Run circuit tracing and merge inspection artefacts."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label("Run directory"),
            Input(placeholder="runs/latest", id="trace-run"),
        )
        yield Horizontal(
            Label("Tokenizer"),
            Input(placeholder="mistral-instruct", id="trace-tokenizer"),
        )
        yield Horizontal(
            Label("Apply ablation"),
            Switch(value=False, id="trace-ablation"),
        )
        yield Horizontal(
            Button("Run Tracer", id="trace-run-btn", variant="primary"),
            Button("Merge Inspection", id="trace-merge-btn", variant="warning"),
        )
        yield TextLog(id="trace-log", highlight=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        log_widget = self.query_one("#trace-log", TextLog)
        log_widget.clear()

        run_dir = self.query_one("#trace-run", Input).value or "runs/latest"
        tokenizer = self.query_one("#trace-tokenizer", Input).value
        apply_ablation = self.query_one("#trace-ablation", Switch).value

        if event.button.id == "trace-run-btn":
            command = build_tracer_command(run_dir, tokenizer, apply_ablation)
            log_widget.write("Running circuit tracer...")
        elif event.button.id == "trace-merge-btn":
            command = build_merge_command(run_dir)
            log_widget.write("Merging inspection artefacts...")
        else:
            return

        async def sink(line: str) -> None:
            log_widget.write(line)

        await asyncio.create_task(run_command(command, sink))
        log_widget.write("Operation completed.")
