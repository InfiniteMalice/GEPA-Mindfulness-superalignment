"""Shared dataclasses for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - hints only
    from torch import Tensor
else:  # pragma: no cover - fallback when torch unavailable
    Tensor = object  # type: ignore[misc,assignment]

from ..core.circuit_tracer_adapter import TraceAnalysis
from ..core.epistemic_process import EpistemicProcessAssessment


@dataclass
class GRPOGroupSample:
    prompt: str
    samples: List["Sample"] = field(default_factory=list)

    @dataclass
    class Sample:
        """One rollout with diagnostic trace data and explicit optimizer inputs."""

        response: str
        tokens: Sequence[int]
        log_prob: Tensor
        ref_log_prob: Tensor
        trace: TraceAnalysis
        reward: float = 0.0
        advantage: float = 0.0
        reference_answers: Sequence[str] | str | None = None
        confidence: float | None = None
        epistemic_process: EpistemicProcessAssessment | None = None


__all__ = [
    "GRPOGroupSample",
]
