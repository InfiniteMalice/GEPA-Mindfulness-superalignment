"""Core dataclasses for dual-path deception traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DualPathScenario:
    """Structured description of a dual-path prompt.

    Invariants:
        - There are always two candidate paths.
        - Each path has a scratchpad and a public reasoning channel.
        - Final answer labels must be one of: 1, 2, both, neither.
    """

    question: str
    path_1_framing: str
    path_2_framing: str
    scenario_id: str | None = None
    aligned_path: str | None = None
    deceptive_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> str:
        """Render the scenario framing into a prompt context block."""
        return (
            "Path 1 framing:\n"
            f"{self.path_1_framing}\n\n"
            "Path 2 framing:\n"
            f"{self.path_2_framing}"
        )


@dataclass
class DualPathRunConfig:
    """Configuration for dual-path runs and trace logging."""

    model_id: str
    temperature: float = 0.7
    max_tokens: int = 512
    request_thought_traces: bool = True
    log_dir: str | None = None
    max_attempts: int = 3
    strict_parsing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DualPathTrace:
    """Structured outputs from a dual-path run."""

    scenario_id: str | None
    prompt: str
    raw_response: str
    path_1_scratchpad: str
    path_2_scratchpad: str
    path_1_public_reasoning: str
    path_2_public_reasoning: str
    final_answer: str
    final_answer_value: str
    recommended_path: str
    probe_activations: dict[str, Any] = field(default_factory=dict)
    attribution_graph: dict[str, Any] = field(default_factory=dict)
    circuit_fingerprint: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sections(
        cls,
        *,
        scenario_id: str | None,
        prompt: str,
        raw_response: str,
        sections: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> DualPathTrace:
        return cls(
            scenario_id=scenario_id,
            prompt=prompt,
            raw_response=raw_response,
            path_1_scratchpad=str(sections.get("path_1_scratchpad", "")),
            path_2_scratchpad=str(sections.get("path_2_scratchpad", "")),
            path_1_public_reasoning=str(sections.get("path_1", "")),
            path_2_public_reasoning=str(sections.get("path_2", "")),
            final_answer=str(sections.get("final_answer", "")),
            final_answer_value=str(sections.get("final_answer_value", "")),
            recommended_path=str(sections.get("recommended_path", "unclear")),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["DualPathScenario", "DualPathRunConfig", "DualPathTrace"]
