"""Utilities for rendering training summaries with Jinja2 templates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from jinja2 import Template

from ..core.rewards import RewardSignal
from .pipeline import RolloutResult

SUMMARY_TEMPLATE = Template(
    """
{% for idx, result in results %}
Rollout {{ idx }}
---------
Prompt: {{ result.prompt }}
Reward: {{ result.reward | round(4) }}
Contradictions: {{ result.contradiction_report }}
{% endfor %}
"""
)


@dataclass
class SummaryReport:
    content: str


def render_summary(results: Iterable[RolloutResult]) -> SummaryReport:
    prepared = list(enumerate(results))
    content = SUMMARY_TEMPLATE.render(results=prepared)
    return SummaryReport(content=content.strip())


def describe_reward(signal: RewardSignal) -> str:
    return (
        f"Task={signal.task_success:.3f}, GEPA={signal.gepa_score:.3f}, Honesty={signal.honesty_reward:.3f}, "
        f"Hallucination={signal.hallucination_score:.3f}, Imperative={signal.imperatives_truth.resolve():.3f}"
    )
