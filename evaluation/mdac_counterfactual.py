"""Opt-in counterfactual evaluation for compatible scalar outcomes and discrete PEO replays."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from gepa_mindfulness.verification.epistemic_state import EpistemicContext, _number, _text
from synthetic_data.world_peo import EpisodeStep, build_episode
from synthetic_data.worlds import SyntheticWorld, _enabled


@dataclass(frozen=True, slots=True)
class ScalarOutcome:
    initial_state: str
    action: str
    unit: str
    value: float
    source: str
    reference: str

    def __post_init__(self) -> None:
        for name in ("initial_state", "action", "unit", "reference"):
            _text(getattr(self, name), name)
        _number(self.value, "value")
        if self.source not in ("observed", "simulated", "predicted"):
            raise ValueError("unknown outcome provenance")


@dataclass(frozen=True, slots=True)
class StructuralAssumptions:
    disturbance_model: str
    shared_disturbance: str
    validation_reference: str

    def __post_init__(self) -> None:
        if self.disturbance_model not in ("additive", "approximately_separable"):
            raise ValueError("residual reuse requires compatible disturbance structure")
        _text(self.shared_disturbance, "shared_disturbance")
        _text(self.validation_reference, "validation_reference")


@dataclass(frozen=True, slots=True)
class CounterfactualEstimate:
    value: float
    anchor: ScalarOutcome
    factual_prediction: ScalarOutcome
    alternative_prediction: ScalarOutcome
    assumptions: StructuralAssumptions
    source = "model_estimated_counterfactual"

    @property
    def references(self) -> tuple[str, str, str]:
        return (
            self.anchor.reference,
            self.factual_prediction.reference,
            self.alternative_prediction.reference,
        )


def estimate_counterfactual(
    anchor: ScalarOutcome,
    factual: ScalarOutcome,
    alternative: ScalarOutcome,
    assumptions: StructuralAssumptions,
    *,
    enabled: bool = False,
) -> CounterfactualEstimate:
    """Estimate the individual alternative under a declared shared additive disturbance."""
    _enabled(enabled)
    if type(assumptions) is not StructuralAssumptions:
        raise ValueError("explicit structural assumptions are required")
    for item in (anchor, factual, alternative):
        if type(item) is not ScalarOutcome:
            raise ValueError("scalar outcomes required")
        if item.unit != anchor.unit or item.initial_state != anchor.initial_state:
            raise ValueError("initial state and units must match")
    if factual.source != "predicted" or alternative.source != "predicted":
        raise ValueError("both predictions require model provenance")
    if anchor.source == "predicted" or anchor.action != factual.action:
        raise ValueError("anchor must record the factual action")
    if alternative.action == factual.action:
        raise ValueError("provide a distinct alternative")
    estimate = anchor.value + (alternative.value - factual.value)
    if not isfinite(estimate):
        raise ValueError("counterfactual estimate exceeds finite scalar range")
    return CounterfactualEstimate(estimate, anchor, factual, alternative, assumptions)


def compare_actions(
    world: SyntheticWorld,
    factual: EpisodeStep,
    alternative: EpisodeStep,
    *,
    context: EpistemicContext,
    start_timestamp: str,
    comparison_id: str,
    enabled: bool = False,
) -> dict[str, Any]:
    """Replay two candidate actions against identical simulator state and permissions."""
    _enabled(enabled)
    _text(comparison_id, "comparison_id")
    if factual.action_id == alternative.action_id:
        raise ValueError("actions must differ")
    reports = {
        name: build_episode(
            world,
            (step,),
            context=context,
            episode_id=f"{comparison_id}:{name}",
            start_timestamp=start_timestamp,
            enabled=True,
        )
        for name, step in (("factual", factual), ("alternative", alternative))
    }
    effect = int(reports["alternative"]["steps"][0]["success"]) - int(
        reports["factual"]["steps"][0]["success"]
    )
    return {
        "schema_version": "mdac-counterfactual-replay-v1",
        "comparison_id": comparison_id,
        "initial_state": world.digest,
        "source": "sandbox_paired_replay",
        "individual_simulated_effect": effect,
        **reports,
    }


def summarize_outcome_pairs(
    pairs: tuple[tuple[str, float, float], ...], *, sampling: str, enabled: bool = False
) -> dict[str, Any]:
    """Summarize evaluator-matched scalar observations in a common outcome unit."""
    from statistics import fmean

    _enabled(enabled)
    estimands = {
        "observational": "observational_association",
        "matched_intervention": "sample_average_interventional_effect",
    }
    if sampling not in estimands:
        raise ValueError("declare an observational or matched interventional design")
    differences: dict[str, float] = {}
    for identity, factual, alternative in pairs:
        _text(identity, "identity")
        if identity in differences:
            raise ValueError("repeated pair identity")
        delta = _number(alternative, "alternative") - _number(factual, "factual")
        differences[identity] = _number(delta, "difference")
    return {
        "estimand": estimands[sampling],
        "n_pairs": len(differences),
        "difference": fmean(differences.values()) if differences else None,
        "paired_differences": differences,
    }
