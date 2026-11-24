from dataclasses import replace

import pytest

from evaluation.mdac_counterfactual import (
    ScalarOutcome,
    StructuralAssumptions,
    compare_actions,
    estimate_counterfactual,
)
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from synthetic_data.world_peo import EpisodeStep
from synthetic_data.worlds import generate_world


def test_reuse_known_additive_disturbance():
    factual = ScalarOutcome("initial", "a", "metres", 13, "observed", "sensor-3")
    prediction = replace(factual, value=10, source="predicted", reference="model-1")
    alternative = replace(prediction, action="b", value=16)
    assumptions = StructuralAssumptions("additive", "fixed wind", "bench:paired-v1")
    result = estimate_counterfactual(factual, prediction, alternative, assumptions, enabled=True)
    assert result.value == 19
    assert result.source == "model_estimated_counterfactual"
    assert result.references == ("sensor-3", "model-1", "model-1")
    for changed in (
        replace(alternative, unit="feet"),
        replace(alternative, initial_state="different"),
        replace(alternative, action="a"),
        replace(alternative, source="observed"),
    ):
        with pytest.raises(ValueError):
            estimate_counterfactual(factual, prediction, changed, assumptions, enabled=True)
    with pytest.raises(ValueError):
        estimate_counterfactual(factual, prediction, alternative, assumptions)


def test_matched_sandbox_peo_replay_preserves_authorization_boundary():
    world = generate_world(seed=7, enabled=True)
    actions = world.actions
    context = EpistemicContext("trial", 1, EvaluatedSystemVersion("fixture", "simulator"))
    result = compare_actions(
        world,
        EpisodeStep(actions[0].action_id, 1.0, 1.0),
        EpisodeStep(actions[1].action_id, 1.0, 1.0),
        context=context,
        start_timestamp="2026-01-01T00:00:00Z",
        comparison_id="pair:1",
        enabled=True,
    )
    assert result["source"] == "sandbox_paired_replay"
    first, second = (result[k]["steps"][0] for k in ("factual", "alternative"))
    assert first["before_digest"] == second["before_digest"] == world.digest
    assert result["individual_simulated_effect"] == int(second["success"]) - int(first["success"])


def test_discrete_replay_enforces_denied_permission_despite_confidence():
    from synthetic_data.worlds import counterfactual_permission

    world = generate_world(seed=3, enabled=True)
    denied = counterfactual_permission(
        world, world.actions[0].action_id, allowed=False, enabled=True
    )
    result = compare_actions(
        denied,
        EpisodeStep(world.actions[0].action_id, 1, 1),
        EpisodeStep(world.actions[1].action_id, 0, 0),
        comparison_id="denied",
        context=EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness")),
        start_timestamp="2026-01-01T00:00:00Z",
        enabled=True,
    )
    assert result["factual"]["steps"][0]["success"] is False


def test_population_estimands():
    from evaluation.mdac_counterfactual import summarize_outcome_pairs

    assert summarize_outcome_pairs((), sampling="observational", enabled=True)["difference"] is None
    for sampling, target in (
        ("observational", "observational_association"),
        ("matched_intervention", "sample_average_interventional_effect"),
    ):
        result = summarize_outcome_pairs(
            (("unit-1", 0, 1), ("unit-2", 1, 1)), sampling=sampling, enabled=True
        )
        assert result["estimand"] == target
        assert result["difference"] == 0.5
        assert result["n_pairs"] == 2
