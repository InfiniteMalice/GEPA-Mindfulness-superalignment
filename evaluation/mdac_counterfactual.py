"""Opt-in counterfactual evaluation for compatible scalar outcomes and discrete PEO replays."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

# Third-party
# Local
from evaluation.improvement_records import ImprovementRecord
from gepa_mindfulness.verification.epistemic_state import EpistemicContext, _number, _text
from synthetic_data.lesson_curation import authenticate_payload
from synthetic_data.world_peo import EpisodeStep, build_episode
from synthetic_data.worlds import SyntheticWorld, _enabled


@dataclass(frozen=True, slots=True)
class OutcomeContext(ImprovementRecord):
    """Pinned prediction or intervention protocol for one matched disturbance."""

    model_version: str
    configuration_id: str
    seed: int
    disturbance_id: str


@dataclass(frozen=True, slots=True)
class ScalarOutcome(ImprovementRecord):
    initial_state: str
    action: str
    unit: str
    value: float
    source: str
    reference: str
    context: OutcomeContext | None = None

    def _validate(self) -> None:
        if self.source not in ("observed", "simulated", "predicted"):
            raise ValueError("unknown outcome provenance")


@dataclass(frozen=True, slots=True)
class InterventionPair(ImprovementRecord):
    """Evaluator-owned manipulation and its two outcome captures in a matched block."""

    pair_id: str
    design_id: str
    evaluator_id: str
    factual: ScalarOutcome
    alternative: ScalarOutcome

    def _validate(self) -> None:
        f, a = self.factual, self.alternative
        if f.context is None or f.context != a.context or f.initial_state != a.initial_state:
            raise ValueError("intervention arms need a matched initial state and protocol")
        if f.source == "predicted" or f.source != a.source or f.unit != a.unit:
            raise ValueError("intervention arms need a common observed or simulated outcome unit")
        if f.reference == a.reference or f.action == a.action:
            raise ValueError("intervention arms need distinct actions and evidence references")

    def contrast(self) -> dict[str, str]:
        f = self.factual
        assert f.context is not None
        return dict(
            design_id=self.design_id,
            evaluator_id=self.evaluator_id,
            unit=f.unit,
            source=f.source,
            factual_action=f.action,
            alternative_action=self.alternative.action,
            model_version=f.context.model_version,
            configuration_id=f.context.configuration_id,
        )


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
    if (
        factual.context is None
        or factual.context != alternative.context
        or anchor.context is None
        or anchor.context.disturbance_id != factual.context.disturbance_id
        or anchor.context.seed != factual.context.seed
        or assumptions.shared_disturbance != factual.context.disturbance_id
    ):
        raise ValueError(
            "residual reuse requires matched prediction context and factual disturbance"
        )
    try:
        estimate = float(
            Fraction(alternative.value) + Fraction(anchor.value) - Fraction(factual.value)
        )
    except OverflowError as exc:
        raise ValueError("counterfactual estimate exceeds finite scalar range") from exc
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
    capture = OutcomeContext(
        context.system.model_version, context.system.harness_version, world.seed, world.digest
    )

    def outcome(arm: str) -> ScalarOutcome:
        report = reports[arm]
        step = report["steps"][0]
        observation = next(e for e in report["events"] if e["event_type"] == "outcome_observed")
        return ScalarOutcome(
            world.digest,
            step["action_id"],
            "simulated_binary_success",
            int(step["success"]),
            "simulated",
            observation["event_id"],
            capture,
        )

    pair = InterventionPair(
        comparison_id,
        "synthetic-world-action-pair-v1",
        "world-peo-v1",
        outcome("factual"),
        outcome("alternative"),
    )
    effect = int(pair.alternative.value - pair.factual.value)
    return {
        "schema_version": "mdac-counterfactual-replay-v1",
        "comparison_id": comparison_id,
        "initial_state": world.digest,
        "source": "sandbox_paired_replay",
        "individual_simulated_effect": effect,
        "intervention_pair": pair.to_dict(),
        **reports,
    }


def summarize_outcome_pairs(
    pairs: tuple[tuple[str, float, float] | InterventionPair, ...],
    *,
    sampling: str,
    authenticate: Callable[[dict], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Summarize observations, or matched interventions authenticated by their host evaluator."""
    _enabled(enabled)
    estimands = {
        "observational": "observational_association",
        "matched_intervention": "sample_average_interventional_effect",
    }
    if sampling not in estimands or type(pairs) is not tuple:
        raise ValueError("declare a sampling design and tuple of outcome pairs")
    differences: dict[str, float] = {}
    captures = []
    contrast = None
    refs: set[str] = set()
    for item in pairs:
        if sampling == "matched_intervention":
            if type(item) is not InterventionPair:
                raise ValueError("matched intervention requires typed intervention captures")
            wire = item.to_dict()
            pair = cast(InterventionPair, InterventionPair.from_dict(wire))
            current = pair.contrast()
            if contrast is not None and contrast != current:
                raise ValueError(
                    "cannot pool different intervention directions, units or protocols"
                )
            if not authenticate_payload(wire, authenticate):
                raise ValueError("intervention evidence requires independent host authentication")
            new_refs = {pair.factual.reference, pair.alternative.reference}
            if new_refs & refs:
                raise ValueError("intervention outcome evidence reused across pairs")
            refs.update(new_refs)
            contrast = current
            captures.append(wire)
            identity, factual, alternative = (
                pair.pair_id,
                pair.factual.value,
                pair.alternative.value,
            )
        else:
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("observational samples require an identity and two scalar values")
            identity, factual, alternative = item
        _text(identity, "identity")
        if identity in differences:
            raise ValueError("repeated pair identity")
        delta = _number(alternative, "alternative") - _number(factual, "factual")
        differences[identity] = _number(delta, "difference")
    mean_difference = (
        float(sum(map(Fraction, differences.values()), Fraction()) / len(differences))
        if differences
        else None
    )
    return {
        "estimand": estimands[sampling],
        "n_pairs": len(differences),
        "difference": mean_difference,
        "paired_differences": differences,
        "contrast": contrast,
        "pairs": captures,
    }
