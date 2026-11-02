"""Analytical controls distinguish action accessibility from rollout consistency."""

# Standard library
import builtins
import json
from dataclasses import replace
from unittest.mock import patch

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_records import SystemConfig
from evaluation.intrinsic_dynamics import (
    capture_world_transition,
    compare_rollout_costs,
    run_action_probes,
)
from evaluation.intrinsic_dynamics_records import (
    DynamicsProbeConfig,
    ProbeTransition,
    RolloutCost,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from synthetic_data.worlds import generate_world, simulate

np = pytest.importorskip("numpy", reason="optional numerical extra is not installed")
SYSTEM = SystemConfig("fixture-encoder:v1", "fixture-probes:v1", content_digest("config"))
REFS = (EvidenceReference("fixture:record", EvidenceSourceKind.EXTERNAL_RECORD),)
CFG = DynamicsProbeConfig(enabled=True, seed=0)


def row(trajectory, index, x, action, **changes):
    """Create explicit fixture latents where displacement exposes the executed action."""
    values = dict(
        trajectory_id=trajectory,
        step_index=index,
        root_world_digest=content_digest(trajectory),
        transition_digest=content_digest((trajectory, index)),
        action_id="left" if action == -1 else "right",
        action_vocabulary=("left", "right"),
        before=(float(x),),
        after=(float(x + action),),
        evidence_refs=REFS,
    )
    values.update(changes)
    return ProbeTransition(**values)


def fixture_rows():
    """Held-out starts differ; both action classes occur at every start in each split."""
    return tuple(
        row(trajectory, i * 2 + j, x, action)
        for trajectory, starts in (("fit", (-3, -1, 1, 3)), ("test", (-4, -2, 2, 4)))
        for i, x in enumerate(starts)
        for j, action in enumerate((-1, 1))
    )


def run(rows, **changes):
    """Use one predeclared trajectory split, never a row-level random split."""
    return run_action_probes(
        rows,
        system=SYSTEM,
        fit_trajectories=("fit",),
        test_trajectories=("test",),
        config=CFG,
        **changes,
    )


def test_disabled_calls_do_not_import_numpy_or_inspect_inputs():
    """Disabled diagnostics cannot consume evidence or initialize a numerical backend."""
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "numpy":
            raise AssertionError("unexpected numerical import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", guarded):
        assert (
            run_action_probes(object(), system=None, fit_trajectories=(), test_trajectories=())
            is None
        )
        assert compare_rollout_costs(object(), system=None) is None
        assert (
            capture_world_transition(
                None, object(), None, trajectory_id=None, before=None, after=None
            )
            is None
        )


def test_pair_and_displacement_recover_action_with_matched_endpoint_controls():
    """Endpoint accuracy must not be reported as evidence of transition decoding."""
    result = run(fixture_rows())
    metrics = result["probes"]
    assert metrics["before"]["mse"] == pytest.approx(0.25)
    assert metrics["before"]["r2"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["after"]["mse"] == pytest.approx(35 / 144)
    assert metrics["after"]["r2"] == pytest.approx(1 / 36)
    for arm in ("delta", "pair"):
        assert metrics[arm]["mse"] < 1e-24
        assert metrics[arm]["r2"] == pytest.approx(1.0)
        assert metrics[arm]["accuracy"] == 1.0
        assert len(metrics[arm]["predictions"]) == 8
    assert metrics["pair_shuffled"]["r2"] < 0.9
    assert sorted(result["shuffled_fit_indices"]) == list(range(8))
    assert result == run(fixture_rows())
    assert result["optimizer_eligible"] is False
    with pytest.raises(ValueError):
        require_training_eligible(result)


def test_test_labels_do_not_change_fitted_predictions():
    """Holding features fixed while changing test targets cannot leak into fitting."""
    rows = fixture_rows()
    altered = tuple(replace(r, action_id="left") if r.trajectory_id == "test" else r for r in rows)
    before = run(rows)["probes"]["pair"]
    after = run(altered)["probes"]["pair"]
    assert [p["scores"] for p in before["predictions"]] == [
        p["scores"] for p in after["predictions"]
    ]
    assert after["r2"] is None and after["r2_status"] == "constant_targets"
    assert after["mse"] > 0


def test_negative_r2_is_not_clipped_and_forward_missingness_is_separate():
    """A constant wrong fit can score below zero; missing forward predictions stay missing."""
    rows = tuple(
        replace(r, action_id="left") if r.trajectory_id == "fit" else r for r in fixture_rows()
    )
    rows = tuple(
        (
            replace(r, predicted_after=r.after)
            if r.trajectory_id == "test" and r.step_index != 0
            else r
        )
        for r in rows
    )
    report = run(rows)
    assert report["probes"]["pair"]["r2"] == pytest.approx(-1.0)
    assert report["forward"]["mse"] == 0.0
    assert report["forward"]["observed"] == 7
    assert report["forward"]["missing_step_ids"] == [["test", 0]]


@pytest.mark.parametrize("kind", ("root_alias", "transition_alias", "duplicate_step"))
def test_source_aliases_cannot_cross_the_held_out_boundary(kind):
    """Renaming a trajectory does not remove source overlap or duplicate observations."""
    rows = list(fixture_rows())
    if kind == "root_alias":
        rows[-1] = replace(rows[-1], root_world_digest=rows[0].root_world_digest)
    elif kind == "transition_alias":
        rows[-1] = replace(rows[-1], transition_digest=rows[0].transition_digest)
    else:
        rows.append(rows[-1])
    with pytest.raises(ValueError):
        run(tuple(rows))


def test_rosters_must_be_disjoint_complete_and_exact():
    """A missing trajectory or overlapping fit/test roster cannot silently shrink a study."""
    for fit, test in ((("fit",), ("fit",)), (("fit",), ("absent",)), (("fit",), ())):
        with pytest.raises(ValueError):
            run_action_probes(
                fixture_rows(),
                system=SYSTEM,
                fit_trajectories=fit,
                test_trajectories=test,
                config=CFG,
            )


def test_records_round_trip_detach_vectors_and_reject_extensions():
    """Serialization must not add optimizer authority or retain mutable aliases."""
    vector = [1.0]
    original = row("fit", 0, 1, 1, before=vector)
    vector[0] = 99
    assert original.before == (1.0,)
    assert DynamicsProbeConfig.from_dict(CFG.to_dict()) == CFG
    assert ProbeTransition.from_dict(json.loads(json.dumps(original.to_dict()))) == original
    wire = original.to_dict()
    wire["authority_granted"] = True
    with pytest.raises(ValueError):
        ProbeTransition.from_dict(wire)
    for change in (
        {"before": (True,)},
        {"after": (float("nan"),)},
        {"after": (1.0, 2.0)},
        {"action_id": "absent"},
        {"predicted_after": (1.0, 2.0)},
        {"before": (0.0,) * 17},
        {"before": (1e7,)},
    ):
        with pytest.raises(ValueError):
            replace(original, **change)
    with pytest.raises(ValueError):
        DynamicsProbeConfig(enabled=1)


@pytest.mark.parametrize("vocabulary, action", (({"left": 99, "right": 0}, "left"), ("ab", "a")))
def test_wire_vocabulary_rejects_non_arrays(vocabulary, action):
    """Malformed JSON must not become action labels through generic iterable coercion."""
    wire = row("fit", 0, 1, 1).to_dict()
    wire.update(action_vocabulary=vocabulary, action_id=action)
    with pytest.raises(ValueError):
        ProbeTransition.from_dict(wire)


def test_capture_replays_existing_world_without_exposing_or_mutating_it():
    """The bridge binds actual simulated action/state hashes, not self-certified success."""
    world = generate_world(seed=4, enabled=True)
    digest = world.digest
    result = capture_world_transition(
        world,
        ("inspect", "release"),
        0,
        trajectory_id="fit",
        before=(0.0,),
        after=(1.0,),
        enabled=True,
    )
    assert result.action_id == "inspect"
    assert result.root_world_digest == digest
    assert result.action_vocabulary == ("inspect", "release")
    assert world.digest == digest
    changed = simulate(world, "inspect", enabled=True).after
    assert changed.digest != digest
    for label in (TrainingEligibility.REGRESSION, TrainingEligibility.HIDDEN_EVAL):
        with pytest.raises(ValueError):
            capture_world_transition(
                replace(world, training_eligibility=label),
                ("inspect",),
                0,
                trajectory_id="x",
                before=(0.0,),
                after=(1.0,),
                enabled=True,
            )


def test_synthetic_capture_feeds_the_matched_probe_without_policy_training():
    """Replay-bound fixture rows can run end to end without modifying either world."""
    captured = []
    for trajectory, seed in (("fit", 2), ("test", 3)):
        world = generate_world(seed=seed, enabled=True)
        for index, action in enumerate(("inspect", "release")):
            captured.append(
                capture_world_transition(
                    world,
                    ("inspect", "release"),
                    index,
                    trajectory_id=trajectory,
                    before=(0.0,),
                    after=(1.0 if action == "inspect" else -1.0,),
                    enabled=True,
                )
            )
    result = run(tuple(captured))
    assert result["probes"]["pair"]["accuracy"] == 1.0
    assert result["probes"]["before"]["accuracy"] == 0.5
    assert result["source_binding"] == "host_declared_not_authenticated"


def cost(candidate, predicted, actual, horizon=1):
    """Keep complete raw cost observations instead of inferring real outcomes from ranks."""
    return RolloutCost(
        "test", "decision", horizon, candidate, content_digest(candidate), predicted, actual, REFS
    )


def test_rollout_ranking_is_separate_by_horizon_and_handles_ties():
    """An inverse ranking must score negative, not be obscured by a good action decoder."""
    rows = (
        cost("a", 0, 2),
        cost("b", 1, 1),
        cost("c", 2, 0),
        cost("a", 1, 2, 2),
        cost("b", 1, 0, 2),
        cost("a", 0, 0, 3),
        cost("b", 0, 1, 3),
        cost("c", 1, 2, 3),
    )
    result = compare_rollout_costs(rows, system=SYSTEM, enabled=True)
    assert result["groups"][0]["spearman"] == pytest.approx(-1.0)
    assert result["groups"][1]["spearman"] is None
    assert result["groups"][1]["status"] == "constant_costs"
    assert result["groups"][2]["spearman"] == pytest.approx(3**0.5 / 2)
    assert len(result["rows"]) == 8
    with pytest.raises(ValueError):
        require_training_eligible(result)


def test_missing_costs_and_duplicate_candidates_are_not_silently_dropped():
    """Incomplete candidate cohorts remain explicitly unscored with their raw rows."""
    rows = (cost("a", None, 1), cost("b", 1, 0))
    result = compare_rollout_costs(rows, system=SYSTEM, enabled=True)
    assert result["groups"][0]["status"] == "missing_costs"
    assert result["groups"][0]["spearman"] is None
    assert len(result["rows"]) == 2
    with pytest.raises(ValueError):
        compare_rollout_costs((rows[0], rows[0]), system=SYSTEM, enabled=True)
