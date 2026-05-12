"""Dependency-minimal copy of the base 13+0 reward classifier for RG-Tracer."""

from __future__ import annotations

import dataclasses


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
) -> Reward:
    """Classify the unchanged 13+0 abstention reward cases."""
    refs = [] if reference_answers is None else [_normalize(reference_answers)]
    abstained = _normalize(response) in {"", "idk", "i don't know", "i dont know"}
    is_correct = _normalize(response) in refs
    high = confidence >= threshold
    knowledge = 0.0
    abstention = 0.0
    calibration = 0.0
    if abstained:
        if high:
            if thought_align and refs:
                case_id = 9
                abstention = -0.25
                calibration = -1.0 * max(confidence - threshold, 0.0)
            elif thought_align:
                case_id = 10
                calibration = -2.0 * max(confidence - threshold, 0.0)
            else:
                case_id = 11
                calibration = -2.0 * max(confidence - threshold, 0.0)
        elif thought_align:
            case_id = 12
            abstention = 0.25
        else:
            case_id = 13
            abstention = 0.125
    elif is_correct:
        knowledge = 2.0 if high else 1.0
        if high and thought_align:
            case_id = 1
        elif high:
            case_id = 2
        elif thought_align:
            case_id = 3
            calibration = 2.0 * max(threshold - confidence, 0.0)
        else:
            case_id = 4
    elif high and thought_align:
        case_id = 5
        knowledge = -2.0
        calibration = -2.0 * max(confidence - threshold, 0.0)
    elif high:
        case_id = 6
        knowledge = -2.0
        calibration = -2.0 * max(confidence - threshold, 0.0)
    elif thought_align:
        case_id = 7
        knowledge = -0.5
    else:
        case_id = 8
        knowledge = -1.0
    thought = 1.0 if case_id in {1, 3, 5, 7, 10, 12} and thought_align else 0.0
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
