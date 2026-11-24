# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.mdac_counterfactual import (
    InterventionPair,
    OutcomeContext,
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
    factual = ScalarOutcome(
        "initial",
        "a",
        "metres",
        13,
        "observed",
        "sensor-3",
        OutcomeContext("model", "v1", 3, "fixed wind"),
    )
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
    from evaluation.mdac_counterfactual import summarize_outcome_pairs

    record = result["intervention_pair"]
    summary = summarize_outcome_pairs(
        (InterventionPair.from_dict(record),),
        sampling="matched_intervention",
        authenticate=lambda p: p == record,
        enabled=True,
    )
    assert summary["difference"] == result["individual_simulated_effect"]
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
    result = summarize_outcome_pairs(
        (("unit-1", 0, 1), ("unit-2", 1, 1)), sampling="observational", enabled=True
    )
    assert result["estimand"] == "observational_association"
    assert result["difference"] == 0.5
    assert result["n_pairs"] == 2


def test_local_model_predictions_enter_simulated_replays():
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL to run CPU model-backed replay")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).eval()
    world = generate_world(seed=5, enabled=True)
    steps = []
    for action in world.actions[:2]:
        tokens = tokenizer(action.action_id, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**tokens).logits[0, -1]
        # A real model score for smoke coverage only; no calibration is assumed.
        confidence = float(logits.softmax(dim=-1).max())
        steps.append(EpisodeStep(action.action_id, confidence, confidence))
    result = compare_actions(
        world,
        *steps,
        context=EpistemicContext("smoke", 0, EvaluatedSystemVersion(checkpoint, "v1")),
        start_timestamp="2026-01-01T00:00:00Z",
        comparison_id="smoke",
        enabled=True,
    )
    assert result["factual"]["steps"][0]["predicted_success"] == steps[0].predicted_success


def test_zero_residual_preserves_small_alternative():
    anchor = ScalarOutcome(
        "initial",
        "a",
        "metres",
        1e16,
        "observed",
        "sensor",
        OutcomeContext("model", "v1", 0, "fixed wind"),
    )
    factual = replace(anchor, source="predicted", reference="model")
    alternative = replace(factual, action="b", value=1.0)
    assumptions = StructuralAssumptions("additive", "fixed wind", "paired experiment")
    result = estimate_counterfactual(anchor, factual, alternative, assumptions, enabled=True)
    assert result.value == 1.0


def test_predictions_must_match_model_configuration_seed_and_named_disturbance():
    context = OutcomeContext("forecast-v1", "config-1", 77, "fixed-wind")
    anchor = ScalarOutcome("initial", "a", "metres", 4, "observed", "obs", context)
    factual = replace(anchor, value=2, source="predicted", reference="forecast:a")
    alt = replace(factual, action="b", reference="forecast:b")
    assumptions = StructuralAssumptions("additive", "fixed-wind", "validation:wind")
    for changed in (
        replace(context, model_version="other"),
        replace(context, configuration_id="other"),
        replace(context, seed=88),
        replace(context, disturbance_id="new-wind"),
        None,
    ):
        with pytest.raises(ValueError):
            estimate_counterfactual(
                anchor, factual, replace(alt, context=changed), assumptions, enabled=True
            )
    with pytest.raises(ValueError):
        estimate_counterfactual(
            anchor, factual, alt, replace(assumptions, shared_disturbance="fog"), enabled=True
        )


def test_interventional_population_requires_bound_and_authenticated_common_contrast():
    from evaluation.mdac_counterfactual import summarize_outcome_pairs

    ctx = OutcomeContext("system-v1", "harness-v1", 1, "block-1")
    first = ScalarOutcome("x", "a", "metres", 0, "observed", "group1/pre", ctx)
    second = replace(first, action="b", value=1, reference="group1/post")
    pair = InterventionPair("s1", "design1", "judge1", first, second)
    args = dict(sampling="matched_intervention", enabled=True)
    with pytest.raises(ValueError):
        summarize_outcome_pairs((("s1", 0, 1),), **args)
    with pytest.raises(ValueError):
        summarize_outcome_pairs((pair,), **args)
    with pytest.raises(ValueError):
        summarize_outcome_pairs((pair,), authenticate=lambda payload: False, **args)
    for changed in (
        replace(second, unit="seconds"),
        replace(second, initial_state="other"),
        replace(second, source="predicted"),
        replace(second, context=replace(ctx, seed=2)),
    ):
        with pytest.raises(ValueError):
            replace(pair, alternative=changed)
    reverse = replace(pair, pair_id="s2", factual=second, alternative=first)
    with pytest.raises(ValueError):
        summarize_outcome_pairs((pair, reverse), authenticate=lambda _: True, **args)
    result = summarize_outcome_pairs((pair,), authenticate=lambda p: p == pair.to_dict(), **args)
    assert result["difference"] == 1
    assert result["contrast"]["factual_action"] == "a"
    assert result["pairs"][0]["alternative"]["reference"] == "group1/post"


def test_large_finite_contrasts_retain_finite_answers():
    from evaluation.mdac_counterfactual import summarize_outcome_pairs

    ctx = OutcomeContext("model", "config", 0, "noise")
    x = ScalarOutcome("x", "a", "m", 1e308, "observed", "o", ctx)
    f = replace(x, source="predicted", value=-1e308)
    a = replace(f, action="b")
    assert (
        estimate_counterfactual(
            x, f, a, StructuralAssumptions("additive", "noise", "validation"), enabled=True
        ).value
        == 1e308
    )
    ps = (("p", 0.0, 1e308), ("q", 0.0, 1e308))
    assert (
        summarize_outcome_pairs(ps, sampling="observational", enabled=True)["difference"] == 1e308
    )


def test_population_pools_distinct_disturbances_with_same_design_and_unit():
    from evaluation.mdac_counterfactual import summarize_outcome_pairs

    def sample(i, x, y):
        ctx = OutcomeContext("subject", "config-v1", i, f"disturbance-{i}")
        factual = ScalarOutcome(f"state-{i}", "a", "watts", x, "observed", f"{i}/a", ctx)
        return InterventionPair(
            str(i),
            "controlled-switch",
            "meter-v1",
            factual,
            replace(factual, action="b", value=y, reference=f"{i}/b"),
        )

    left, right = sample(1, 0, 1), sample(2, 3, 1)
    args = dict(sampling="matched_intervention", authenticate=lambda _: True, enabled=True)
    assert summarize_outcome_pairs((left, right), **args)["difference"] == -0.5
    for changed in (
        replace(right, evaluator_id="different-meter"),
        replace(right, design_id="another-design"),
        replace(
            right,
            factual=replace(right.factual, unit="metres"),
            alternative=replace(right.alternative, unit="metres"),
        ),
        replace(right, factual=replace(right.factual, reference=left.factual.reference)),
    ):
        with pytest.raises(ValueError):
            summarize_outcome_pairs((left, changed), **args)
