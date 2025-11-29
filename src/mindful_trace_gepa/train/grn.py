"""Global Response Normalization utilities shared across training modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..utils.imports import optional_import

LOGGER = logging.getLogger(__name__)

torch = optional_import("torch")
nn_module = optional_import("torch.nn") if torch is not None else None

if nn_module is not None:
    BaseModule = nn_module.Module
else:
    BaseModule = object


@dataclass
class GRNSettings:
    """Configuration container for :class:`GlobalResponseNorm`."""

    enabled: bool = False
    dim: int | tuple[int, ...] = -1
    eps: float = 1e-6
    learnable: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GRNSettings":
        payload = payload or {}
        dim_value = payload.get("dim", -1)
        if isinstance(dim_value, list):
            dim_value = tuple(int(item) for item in dim_value)
        return cls(
            enabled=bool(payload.get("enabled", False)),
            dim=dim_value if isinstance(dim_value, (int, tuple)) else -1,
            eps=float(payload.get("eps", 1e-6)),
            learnable=bool(payload.get("learnable", False)),
        )


class GlobalResponseNorm(BaseModule):
    """Applies global response normalisation across the specified dimensions."""

    def __init__(self, dim: int | tuple[int, ...] = -1, eps: float = 1e-6, learnable: bool = False):
        if torch is None or nn_module is None:
            raise ImportError("torch is required to construct GlobalResponseNorm")
        super().__init__()
        self.dim = dim
        self.eps = eps
        gamma = torch.ones(1)
        beta = torch.zeros(1)
        if learnable:
            self.gamma = nn_module.Parameter(gamma)
            self.beta = nn_module.Parameter(beta)
        else:
            self.register_buffer("gamma", gamma)
            self.register_buffer("beta", beta)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            raise ImportError("torch is required to execute GlobalResponseNorm")
        if inputs.dim() not in (2, 3):
            raise ValueError("GlobalResponseNorm expects 2D or 3D inputs")
        norm = inputs.norm(p=2, dim=self.dim, keepdim=True)
        scaled = inputs / (norm + self.eps)
        return inputs * (self.gamma * scaled + self.beta)


def build_grn(settings: GRNSettings | Mapping[str, Any] | None) -> GlobalResponseNorm | None:
    """Instantiate a :class:`GlobalResponseNorm` if ``enabled`` is set."""

    parsed = settings
    if isinstance(settings, Mapping):
        parsed = GRNSettings.from_mapping(settings)
    if parsed is None or not parsed.enabled:
        return None
    if torch is None or nn_module is None:
        LOGGER.warning("GRN enabled but torch is unavailable; skipping normalisation")
        return None
    return GlobalResponseNorm(dim=parsed.dim, eps=parsed.eps, learnable=parsed.learnable)


__all__ = ["GRNSettings", "GlobalResponseNorm", "build_grn"]
