"""Ablation and fine-tuning controls."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, TextLog

from app.services.runners import build_finetune_command, run_command


class AblationView(Container):
    """Launch fine-tuning workflows such as GRPO or LoRA."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label("Training config"),
            Input(placeholder="configs/training/phi3_dual_path.yml", id="train-config"),
        )
        yield Horizontal(
            Label("Notes"),
            Input(placeholder="Optional notes", id="train-notes"),
        )
        yield Button("Fine-Tune", id="train-run", variant="success")
        yield TextLog(id="train-log", highlight=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "train-run":
            return
        config_path = self.query_one("#train-config", Input).value
        log_widget = self.query_one("#train-log", TextLog)
        log_widget.clear()
        log_widget.write("Starting fine-tuning job...")

        command = build_finetune_command(config_path)

        async def sink(line: str) -> None:
            log_widget.write(line)

        await asyncio.create_task(run_command(command, sink))
        log_widget.write("Fine-tuning command finished.")
