"""NetworkX compatibility loader with an explicit project-owned fallback."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def load_networkx() -> Any:
    """Return real NetworkX when installed, otherwise the internal fallback."""

    if importlib.util.find_spec("networkx") is not None:
        return importlib.import_module("networkx")
    from gepa_mindfulness import internal_graph

    return internal_graph


nx = load_networkx()

__all__ = ["load_networkx", "nx"]
