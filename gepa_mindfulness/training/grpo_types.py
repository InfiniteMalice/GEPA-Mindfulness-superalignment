"""Shared dataclasses for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - hints only
    from torch import Tensor
else:  # pragma: no cover - fallback when torch unavailable
    Tensor = object  # type: ignore[misc,assignment]

from ..core.circuit_tracer_adapter import TraceAnalysis


@dataclass
class GRPOGroupSample:
    prompt: str
    samples: List["Sample"] = field(default_factory=list)

    @dataclass
    class Sample:
        response: str
        tokens: Sequence[int]
        log_prob: Tensor
        ref_log_prob: Tensor
        trace: TraceAnalysis
        reward: float = 0.0
        advantage: float = 0.0


__all__ = [
    "GRPOGroupSample",
]
