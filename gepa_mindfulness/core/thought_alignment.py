"""Thought alignment scoring based on epistemic grounding signals."""

from __future__ import annotations

import re
from typing import Iterable


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _split_segments(trace: str) -> list[str]:
    candidates = re.split(r"(?:\n+|(?<=[.!?])\s+)", trace)
    return [candidate.strip() for candidate in candidates if candidate.strip()]


def _segment_weight(index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    numerator = index + 1
    denominator = total * (total + 1) / 2
    return numerator / denominator


def _numeric_alignment(answer: str, segment: str) -> float:
    if not answer:
        return 0.0
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    if not numbers:
        return 0.0
    boost = 0.0
    has_operator = "+" in segment or "-" in segment or "*" in segment or "/" in segment
    for number in numbers:
        if re.search(rf"\b{re.escape(number)}\b", segment):
            boost += 0.2
        if f"= {number}" in segment or f"= {number}." in segment:
            boost += 0.2
    if has_operator:
        boost += 0.1
    return boost


def _conflict_penalty(trace: str) -> float:
    penalty = 0.0
    if re.search(r"\w+\?\s+\w+\?", trace):
        penalty += 0.2
    if trace.count("?") > 1 and " or " in trace:
        penalty += 0.1
    if "maybe" in trace and "?" in trace:
        penalty += 0.05
    return penalty


def compute_match_score(trace: str, answer: str, context: str) -> float:
    """Return how tightly the trace endorses the final answer.

    The score prioritises segments near the end of the trace and looks for explicit
    derivations or endorsements of the answer. Conflicting unresolved candidates reduce
    the score.
    """

    if not trace:
        return 0.0

    trace_lower = trace.lower()
    answer_lower = (answer or "").strip().lower()
    context_lower = (context or "").lower()

    segments = _split_segments(trace_lower)
    if not segments:
        return 0.0

    weights = [_segment_weight(idx, len(segments)) for idx in range(len(segments))]
    score = 0.0

    conclusion_markers: Iterable[str] = ("therefore", "so", "thus", "implies", "hence")

    for segment, weight in zip(segments, weights):
        local = 0.0
        if answer_lower and answer_lower in segment:
            local += 0.7
        if any(marker in segment for marker in conclusion_markers):
            local += 0.15
        local += _numeric_alignment(answer_lower, segment)

        if context_lower:
            overlap = set(context_lower.split()) & set(segment.split())
            if overlap:
                local += min(0.1, 0.02 * len(overlap))

        local = min(local, 1.0)
        score += local * weight

    if segments and answer_lower and answer_lower in segments[-1]:
        score += 0.1
    if segments and any(marker in segments[-1] for marker in conclusion_markers):
        score += 0.05
    if answer_lower and trace_lower.count(answer_lower) > 1:
        score += 0.1

    score -= _conflict_penalty(trace_lower)
    return _clamp(score)


def compute_epistemic_score(trace: str) -> float:
    """Return an epistemic robustness score based on justification cues."""

    if not trace:
        return 0.0

    lowered = trace.lower()
    positive_markers = (
        "because",
        "therefore",
        "thus",
        "so",
        "as a result",
        "this implies",
        "since",
        "hence",
        "which means",
        "calculation",
        "compute",
        "sum",
        "equals",
        "first",
        "second",
        "finally",
    )
    limited_uncertainty = (
        "not sure about",
        "uncertain about",
        "assume",
        "approximate",
        "estimate",
    )
    negative_markers = (
        "just guessing",
        "random",
        "no idea",
        "wild guess",
        "flip a coin",
        "i have no idea",
    )
    contradiction_markers = ("but then", "changed my mind", "contradict", "however, maybe")

    score = 0.15
    score += 0.15 * min(sum(marker in lowered for marker in positive_markers), 4)
    score += 0.18 * min(sum(marker in lowered for marker in limited_uncertainty), 2)

    enumerated_steps = len(re.findall(r"\b\d+\.\s", lowered))
    calculation_cues = len(re.findall(r"=|\bcompute\b|\bresult\b", lowered))
    structured_cues = enumerated_steps + calculation_cues
    score += 0.08 * min(structured_cues, 3)

    arithmetic_chains = len(re.findall(r"\d+\s*[+\-/*]\s*\d+", lowered))
    score += 0.06 * min(arithmetic_chains, 3)

    score -= 0.18 * sum(marker in lowered for marker in negative_markers)
    score -= 0.1 * sum(marker in lowered for marker in contradiction_markers)

    return _clamp(score)


def classify_thought_alignment(
    trace: str,
    answer: str,
    context: str,
    theta_match: float = 0.8,
    theta_epistemic: float = 0.5,
) -> tuple[bool, float, float]:
    """Return whether the trace is aligned with the answer and its component scores.

    Args:
        trace: The reasoning trace to evaluate.
        answer: The final answer to check alignment against.
        context: Additional context for scoring.
        theta_match: Threshold for match score (default 0.8).
        theta_epistemic: Threshold for epistemic score (default 0.5).

    Returns:
        Tuple of (is_aligned, match_score, epistemic_score).
    """

    s_match = compute_match_score(trace, answer, context)
    s_epistemic = compute_epistemic_score(trace)
    is_logically_connected = s_match >= theta_match and s_epistemic >= theta_epistemic
    return is_logically_connected, s_match, s_epistemic


__all__ = [
    "classify_thought_alignment",
    "compute_epistemic_score",
    "compute_match_score",
]
