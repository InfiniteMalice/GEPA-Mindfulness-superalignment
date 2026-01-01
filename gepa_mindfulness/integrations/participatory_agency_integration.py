"""Optional integration helpers for participatory agency value heads."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

from ..participatory_agency import ParticipatoryAgencyConfig, ParticipatoryValueHead


def build_participatory_value_head(
    hidden_size: int,
    config: ParticipatoryAgencyConfig | None = None,
) -> ParticipatoryValueHead:
    """Construct a participatory agency value head without mutating any models."""

    return ParticipatoryValueHead(hidden_size=hidden_size, config=config)


def attach_participatory_value_head(
    model: Any,
    hidden_size: int,
    config: ParticipatoryAgencyConfig | None = None,
) -> ParticipatoryValueHead:
    """Attach a value head to *model* without altering training defaults."""

    head = build_participatory_value_head(hidden_size=hidden_size, config=config)
    if hasattr(model, "participatory_value_head"):
        raise AttributeError(
            "Model already has a participatory_value_head. "
            "Remove it first or use build_participatory_value_head() directly."
        )
    if torch is not None and isinstance(model, torch.nn.Module):
        model.add_module("participatory_value_head", head)
        return head
    if hasattr(model, "add_module"):
        model.add_module("participatory_value_head", head)
        return head
    setattr(model, "participatory_value_head", head)
    return head
