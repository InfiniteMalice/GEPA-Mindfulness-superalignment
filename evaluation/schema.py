"""Common schema for trace-aware alignment benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Outcome = Literal[
    "correct",
    "incorrect",
    "abstained",
    "unsafe",
    "overrefused",
    "leaked",
    "sycophantic",
    "deceptive",
    "sandbagging",
    "instruction_hierarchy_failure",
    "privacy_boundary_failure",
    "unsupported_claim",
    "needs_manual_review",
]

TRACE_FLAG_KEYS = (
    "unsupported_claim",
    "user_belief_over_truth",
    "strategic_deception",
    "sandbagging_suspected",
    "privacy_boundary_failure",
    "instruction_hierarchy_failure",
    "unsafe_compliance",
    "overrefusal",
    "abstention_appropriate",
    "abstention_inappropriate",
    "ood_failure",
    "bias_or_stereotype",
    "toxic_content",
)

OUTCOMES: set[str] = {
    "correct",
    "incorrect",
    "abstained",
    "unsafe",
    "overrefused",
    "leaked",
    "sycophantic",
    "deceptive",
    "sandbagging",
    "instruction_hierarchy_failure",
    "privacy_boundary_failure",
    "unsupported_claim",
    "needs_manual_review",
}


@dataclass(slots=True)
class EvalCase:
    """A benchmark row normalized for a model runner or response scorer."""

    eval_id: str
    suite: str
    category: str
    prompt: str
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalResult:
    """A trace-aware outcome for one alignment evaluation example."""

    eval_id: str
    suite: str
    category: str
    prompt: str
    model_answer: str
    gold_answer: str | None
    outcome: Outcome
    confidence: float | None = None
    gepa_score: int | None = None
    trace_flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Unsupported alignment outcome: {self.outcome}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.gepa_score is not None:
            if isinstance(self.gepa_score, bool):
                raise ValueError("gepa_score must be an integer in [0, 4] or None")
            if isinstance(self.gepa_score, str):
                gepa_score = int(self.gepa_score) if self.gepa_score.isdigit() else None
            elif isinstance(self.gepa_score, int):
                gepa_score = self.gepa_score
            else:
                gepa_score = None
            if gepa_score is None or not 0 <= gepa_score <= 4:
                raise ValueError("gepa_score must be an integer in [0, 4] or None")
            self.gepa_score = gepa_score
        self.confidence = None if self.confidence is None else float(self.confidence)
        self.gepa_score = None if self.gepa_score is None else int(self.gepa_score)
        self.trace_flags = normalize_trace_flags(self.trace_flags)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_trace_flags(flags: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return all known trace flags with missing values set to False."""

    raw = flags or {}
    return {key: bool(raw.get(key, False)) for key in TRACE_FLAG_KEYS}


def result_to_jsonl(result: EvalResult) -> str:
    """Serialize one result to a JSONL line."""

    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def result_from_jsonl(line: str) -> EvalResult:
    """Parse one JSONL line into an EvalResult."""

    payload = json.loads(line)
    return EvalResult(**payload)


def write_results_jsonl(results: list[EvalResult], path: str | Path) -> None:
    """Write alignment results to JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result_to_jsonl(result) + "\n")


def read_results_jsonl(path: str | Path) -> list[EvalResult]:
    """Read alignment results from JSONL."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [result_from_jsonl(line) for line in handle if line.strip()]
