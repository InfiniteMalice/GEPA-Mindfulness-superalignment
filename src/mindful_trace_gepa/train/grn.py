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
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Expected mapping or None for GRNSettings, got {type(payload).__name__}"
            )
        dim_value = payload.get("dim", -1)
        if isinstance(dim_value, list):
            dim_value = tuple(int(item) for item in dim_value)
        elif not isinstance(dim_value, (int, tuple)):
            raise ValueError(f"Invalid type for 'dim': {type(dim_value).__name__}")
        return cls(
            enabled=bool(payload.get("enabled", False)),
            dim=dim_value,
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
        gamma = torch.ones(1, dtype=torch.float32)
        beta = torch.zeros(1, dtype=torch.float32)
        if learnable:
            self.gamma = nn_module.Parameter(gamma)
            self.beta = nn_module.Parameter(beta)
        else:
            self.register_buffer("gamma", gamma)
            self.register_buffer("beta", beta)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply GRN with residual connection.

        Args:
            inputs: 2D (batch, features) or 3D (batch, seq, features) tensor

        Returns:
            Normalized tensor with residual connection
        """

        if torch is None:
            raise ImportError("torch is required to execute GlobalResponseNorm")
        if inputs.dim() not in (2, 3):
            raise ValueError("GRN expects 2D (batch, feat) or 3D (batch, seq, feat) inputs")
        gx = inputs.norm(p=2, dim=self.dim, keepdim=True)
        dim_indices = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
        total_dims = inputs.dim()
        canonical_dims = {dim if dim >= 0 else total_dims + dim for dim in dim_indices}
        reduced = []
        for idx in range(total_dims):
            canonical = idx
            if canonical not in canonical_dims:
                reduced.append(idx)
        denom = gx.mean(dim=reduced, keepdim=True) if reduced else gx
        nx = gx / (denom + self.eps)
        # Apply GRN normalization with learnable scale/bias and residual connection
        return self.gamma * (inputs * nx) + self.beta + inputs


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
