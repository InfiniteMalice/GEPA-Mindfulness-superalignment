"""Entry point for the Textual dual-path control surface."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from app.views.ablation import AblationView
from app.views.dual_path import DualPathView
from app.views.home import HomeView
from app.views.tracer import TracerView


class DualPromptApp(App[None]):
    """Minimal Textual application for GEPA workflows."""

    BINDINGS = [Binding("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TabbedContent(
            TabPane(HomeView(), title="Home"),
            TabPane(DualPathView(), title="Dual Path"),
            TabPane(TracerView(), title="Tracer"),
            TabPane(AblationView(), title="Fine-Tune"),
        )
        yield Footer()


if __name__ == "__main__":  # pragma: no cover
    DualPromptApp().run()
