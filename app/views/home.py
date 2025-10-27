"""Home view for the Textual GEPA control surface."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Markdown


class HomeView(Container):
    """Landing view describing the workflow."""

    def compose(self) -> ComposeResult:
        yield Markdown(
            """
# GEPA Dual-Prompt + Adversarial Suite

Use the tabs to run evaluations, trace circuits, review merged artefacts, and
kick off fine-tuning loops. All commands use local scripts, so ensure the
required Python dependencies are installed in the current environment.
            """
        )
