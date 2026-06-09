"""Novelty and distribution-shift assessment for objective robustness."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .schema import NoveltyAssessment, ObjectiveSpecification, ReversibilityLevel, StakesLevel

KNOWN_FEATURES = {
    "dirt",
    "grass",
    "road",
    "path",
    "known",
    "training",
    "base",
    "routine",
}

NOVELTY_CUES = {
    "lava",
    "unknown",
    "novel",
    "new hazard",
    "changed stakeholder",
    "changed authority",
    "unfamiliar",
    "unexpected side effect",
    "distribution shift",
    "out of distribution",
    "ood",
}

HIGH_STAKES_CUES = {
    "high stakes",
    "serious harm",
    "irreversible",
    "legal",
    "medical",
    "safety critical",
    "catastrophic",
    "deploy",
    "external action",
}

LOW_STAKES_CUES = {
    "low stakes",
    "toy",
    "simulation",
    "easy rollback",
    "reversible",
    "draft",
    "sandbox",
}


def assess_objective_novelty(
    specification: ObjectiveSpecification,
    *,
    observed_environment: object | None = None,
) -> NoveltyAssessment:
    """Assess novelty relative to the objective's design or training context."""

    training_text = _coerce_text(
        [
            specification.designer_context_summary,
            specification.training_environment_summary,
            specification.metadata.get("training_features"),
        ]
    )
    deployment_text = _coerce_text(
        [
            specification.deployment_environment_summary,
            observed_environment,
            specification.metadata.get("deployment_features"),
        ]
    )
    combined = " ".join([training_text, deployment_text, specification.objective_text]).lower()

    training_features = _extract_features(training_text)
    deployment_features = _extract_features(deployment_text)
    novel_features = sorted(
        feature for feature in deployment_features if feature not in training_features
    )
    familiar_features = sorted(
        feature for feature in deployment_features if feature in training_features
    )

    cue_features = sorted(cue for cue in NOVELTY_CUES if cue in combined)
    for cue in cue_features:
        if cue not in novel_features and cue not in training_text:
            novel_features.append(cue)

    distribution_shift_detected = bool(
        novel_features or "distribution shift" in combined or "ood" in combined
    )
    novel_state_detected = bool(novel_features or any(cue in combined for cue in NOVELTY_CUES))
    stakes_level = _stakes_level(specification, combined)
    reversibility = _reversibility(specification, combined)
    irreversible_action_requested = bool(
        reversibility == "low"
        or any(cue in combined for cue in ("irreversible", "cannot rollback"))
    )

    novelty_score = _clip(0.12 * len(novel_features) + (0.25 if novel_state_detected else 0.0))
    shift_score = _clip(0.1 * len(novel_features) + (0.25 if distribution_shift_detected else 0.0))
    if stakes_level in {"high", "catastrophic"}:
        novelty_score = max(novelty_score, 0.55 if novel_state_detected else novelty_score)

    reasons: list[str] = []
    if novel_features:
        reasons.append("Deployment context includes features absent from the design context")
    if distribution_shift_detected:
        reasons.append("Objective may be operating outside the context where it was designed")
    if irreversible_action_requested:
        reasons.append("Action appears difficult to reverse under objective uncertainty")
    if stakes_level in {"high", "catastrophic"}:
        reasons.append("Novelty occurs in a high-stakes context")

    clarification_recommended = bool(
        (novel_state_detected and stakes_level in {"moderate", "high", "catastrophic"})
        or irreversible_action_requested
    )
    review_required = bool(
        stakes_level in {"high", "catastrophic"}
        and (novel_state_detected or irreversible_action_requested)
    )

    return NoveltyAssessment(
        objective_id=specification.objective_id,
        novel_state_detected=novel_state_detected,
        distribution_shift_detected=distribution_shift_detected,
        novel_features=novel_features,
        familiar_features=familiar_features,
        novelty_score=novelty_score,
        shift_score=shift_score,
        known_unknowns=novel_features,
        unknown_unknown_warning=bool(novel_state_detected and novelty_score >= 0.5),
        stakes_level=stakes_level,
        reversibility=reversibility,
        irreversible_action_requested=irreversible_action_requested,
        clarification_recommended=clarification_recommended,
        review_required=review_required,
        reasons=reasons,
        metadata={"analysis_mode": "heuristic"},
    )


def _stakes_level(specification: ObjectiveSpecification, combined: str) -> StakesLevel:
    raw = specification.metadata.get("stakes_level")
    if isinstance(raw, str) and raw.lower() in {"low", "moderate", "high", "catastrophic"}:
        return raw.lower()  # type: ignore[return-value]
    if "catastrophic" in combined or "fatal" in combined:
        return "catastrophic"
    if any(cue in combined for cue in HIGH_STAKES_CUES):
        return "high"
    if any(cue in combined for cue in LOW_STAKES_CUES):
        return "low"
    if "unknown" in combined or "novel" in combined:
        return "moderate"
    return "low"


def _reversibility(specification: ObjectiveSpecification, combined: str) -> ReversibilityLevel:
    raw = specification.metadata.get("reversibility")
    if isinstance(raw, str) and raw.lower() in {"low", "moderate", "high"}:
        return raw.lower()  # type: ignore[return-value]
    if any(cue in combined for cue in ("irreversible", "cannot rollback", "permanent")):
        return "low"
    if any(cue in combined for cue in ("hard to rollback", "costly rollback")):
        return "moderate"
    return "high"


def _extract_features(text: str) -> set[str]:
    lowered = text.lower()
    features = set(KNOWN_FEATURES & set(re.findall(r"[a-z0-9_]+", lowered)))
    for cue in NOVELTY_CUES | HIGH_STAKES_CUES | LOW_STAKES_CUES:
        if cue in lowered:
            features.add(cue)
    return features


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_coerce_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_coerce_text(item) for item in value)
    return str(value)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
