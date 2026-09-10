"""Dependency-minimal copy of the base 13+0 reward classifier for RG-Tracer."""

from __future__ import annotations

import dataclasses

from .rewards import EpistemicProcessAssessment


@dataclasses.dataclass(frozen=True)
class Reward:
    """Minimal base reward result used by RG-Tracer V3 overlays."""

    case_id: int
    components: dict[str, float]
    is_correct: bool
    abstained: bool


def compute_abstention_reward(
    *,
    response: str,
    reference_answers: str | None,
    confidence: float,
    thought_align: bool,
    threshold: float,
    epistemic_process: EpistemicProcessAssessment | None = None,
) -> Reward:
    """Classify the unchanged 13+0 abstention reward cases."""
    refs = [] if reference_answers is None else [_normalize(reference_answers)]
    abstained = _normalize(response) in {"", "idk", "i don't know", "i dont know"}
    is_correct = _normalize(response) in refs
    high = confidence >= threshold
    if abstained:
        if high:
            if thought_align and refs:
                case_id = 9
            elif thought_align:
                case_id = 10
            else:
                case_id = 11
        elif thought_align:
            case_id = 12
        else:
            case_id = 13
    elif is_correct:
        if high and thought_align:
            case_id = 1
        elif high:
            case_id = 2
        elif thought_align:
            case_id = 3
        else:
            case_id = 4
    elif high and thought_align:
        case_id = 5
    elif high:
        case_id = 6
    elif thought_align:
        case_id = 7
    else:
        case_id = 8
    knowledge, abstention, calibration = _behavioral_components(
        abstained=abstained,
        is_correct=is_correct,
        high_confidence=high,
        has_references=bool(refs),
        confidence=confidence,
        threshold=threshold,
    )
    optimizer_score = epistemic_process.optimizer_score() if epistemic_process is not None else 0.0
    thought = optimizer_score if optimizer_score > 0.0 else 0.0
    return Reward(
        case_id=case_id,
        components={
            "knowledge": knowledge,
            "abstention": abstention,
            "calibration": calibration,
            "thought": thought,
        },
        is_correct=is_correct,
        abstained=abstained,
    )


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(" .,!?:;")


def _behavioral_components(
    *,
    abstained: bool,
    is_correct: bool,
    high_confidence: bool,
    has_references: bool,
    confidence: float,
    threshold: float,
) -> tuple[float, float, float]:
    """Return outcome-grounded components without consulting diagnostic alignment."""
    knowledge = 0.0
    abstention = 0.0
    calibration = 0.0

    if abstained:
        if high_confidence:
            confidence_gap = max(confidence - threshold, 0.0)
            if has_references:
                abstention = -0.25
                calibration = -1.0 * confidence_gap
            else:
                calibration = -2.0 * confidence_gap
        else:
            abstention = 0.125
    elif is_correct:
        knowledge = 2.0 if high_confidence else 1.0
        if not high_confidence:
            calibration = 2.0 * max(threshold - confidence, 0.0)
    else:
        knowledge = -2.0 if high_confidence else -1.0
        if high_confidence:
            calibration = -2.0 * max(confidence - threshold, 0.0)

    return knowledge, abstention, calibration
