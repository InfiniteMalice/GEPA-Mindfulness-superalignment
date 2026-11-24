"""Matched interventions preserve trial identity, controls and attribution limits."""

import json
import os
from dataclasses import replace

import pytest

from evaluation.mdac_interventions import ReplayResult, run_interventions
from synthetic_data.mdac_interventions import (
    FAULTS,
    FixtureOracle,
    fixture_actor,
    known_fault_block,
)


def run(blocks, *, actor=fixture_actor, **kwargs):
    host = FixtureOracle()
    return run_interventions(
        blocks,
        execute=actor,
        adjudicate=host.adjudicate,
        authenticate=kwargs.pop("authenticate", host.authenticate),
        alternatives=("The intervention made the answer more salient.",),
        enabled=kwargs.pop("enabled", True),
        max_calls=kwargs.pop("max_calls", 100),
        **kwargs,
    )


@pytest.mark.parametrize("fault", list(FAULTS))
def test_known_fault_recovery_uses_public_actions_and_independent_sham(fault):
    block = known_fault_block(fault)
    result = run((block,))
    effects = result["effects"]
    assert effects["repair"]["control_pass_rate"] == 0
    assert effects["repair"]["intervention_pass_rate"] == 1
    assert effects["repair"]["drs"] == 1
    assert effects["sham"]["drs"] == 0
    assert effects["positive"]["drs"] == 1
    assert effects["repair"]["matched_trials"] == 1
    assert effects["repair"]["interval"] is None
    assert result["calls_used"] == 4
    assert result["alternatives"] and result["status"] == "observed"
    assert result["observations"][0]["captures"]
    assert "evaluator" in result["observations"][0]["adjudication"]


def test_roster_contains_distinct_faults():
    assert set(FAULTS) == {
        "missing_context",
        "stale_evidence",
        "missing_authorization",
        "chronology_corruption",
        "bad_reasoning",
        "ambiguous_task",
        "proxy_reward",
        "conflicting_memory",
        "lossy_compression",
        "bad_tool_report",
    }


def test_interaction_requires_complete_factorial_and_matches_within_block():
    blocks = tuple(
        known_fault_block("missing_context", secondary="stale_evidence", task_id=str(i), seed=i)
        for i in range(4)
    )
    kwargs = dict(interactions=(("repair", "second", "joint"),), independent_tasks=True)
    report = run(blocks, **kwargs)
    effect = report["interactions"]["joint"]
    assert effect["estimate"] == 1
    assert effect["matched_trials"] == 4
    assert effect["interval"][0] < 1 <= effect["interval"][1] <= 2
    assert report["effects"]["repair"]["drs"] == report["effects"]["second"]["drs"] == 0
    assert (
        report["effects"]["repair"]["interval"]
        == run(blocks, **kwargs)["effects"]["repair"]["interval"]
    )
    with pytest.raises(ValueError, match="factorial"):
        run(blocks, interactions=(("repair", "sham", "joint"),))


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint", "other"),
        ("task_id", "other"),
        ("configuration", "other"),
        ("initial_state", "other"),
    ],
)
def test_returned_identity_mismatch_invalidates_replay(field, value):
    def mismatch(identity, turns):
        outcome = fixture_actor(identity, turns)
        return replace(outcome, identity=replace(identity, **{field: value}))

    with pytest.raises(ValueError, match="identity"):
        run((known_fault_block("bad_reasoning"),), actor=mismatch)


def test_seed_mismatch_rejected_and_missing_authentication_excludes_trials():
    block = known_fault_block("bad_tool_report")

    def wrong_seed(identity, turns):
        system = replace(identity.system, seed=identity.system.seed + 1)
        return replace(fixture_actor(identity, turns), identity=replace(identity, system=system))

    with pytest.raises(ValueError, match="identity"):
        run((block,), actor=wrong_seed)
    report = run((block,), authenticate=None)
    assert report["effects"]["repair"]["drs"] is None
    assert report["effects"]["repair"]["matched_trials"] == 0
    assert report["status"] == "inconclusive"


def test_censored_actions_do_not_become_failures_and_failed_controls_stay_visible():
    block = known_fault_block("conflicting_memory")

    def constant(identity, turns):
        return replace(fixture_actor(identity, turns), actions=("withhold",))

    def censored(identity, turns):
        return replace(fixture_actor(identity, turns), actions=())

    report = run((block,), actor=constant)
    assert report["effects"]["repair"]["drs"] == 0
    assert report["status"] == "inconclusive" and "positive" in str(report["reasons"])
    assert run((block,), actor=censored)["effects"]["repair"]["drs"] is None


def test_disabled_budget_and_duplicate_trials_fail_before_execution():
    block = known_fault_block("chronology_corruption")

    def unexpected(*args):
        pytest.fail("executor was called")

    with pytest.raises(ValueError, match="enabled"):
        run((block,), enabled=False, actor=unexpected)
    with pytest.raises(ValueError, match="duplicate"):
        run((block, block), actor=unexpected)
    report = run((block,), max_calls=3, actor=unexpected)
    assert report["budget_exhausted"] and report["calls_used"] == 0
    assert report["effects"]["sham"]["drs"] is None
    assert report["status"] == "inconclusive"


def test_actor_receives_only_public_messages_and_observed_system_identity():
    block = known_fault_block("missing_authorization")

    def inspect(identity, turns):
        assert all(set(turn) == {"role", "content"} for turn in turns)
        assert "expected_actions" not in str(turns) and "missing_authorization" not in str(turns)
        return fixture_actor(identity, turns)

    assert run((block,), actor=inspect)["calls_used"] == 4


@pytest.mark.skipif(
    not os.environ.get("MDAC_LOCAL_TEST_MODEL"), reason="provide a local model path"
)
def test_optional_cpu_model_executes_all_arms():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = os.environ["MDAC_LOCAL_TEST_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).cpu().eval()
    block = known_fault_block("ambiguous_task", model=path, checkpoint=path)

    def execute(identity, turns):
        torch.manual_seed(identity.system.seed)
        prompt = "\n".join(turn["content"] for turn in turns)
        tokens = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**tokens, do_sample=False, max_new_tokens=8)
        answer = tokenizer.decode(output[0, tokens.input_ids.shape[1] :], skip_special_tokens=True)
        return ReplayResult.from_actions(identity, turns, (answer.strip() or "(empty)",))

    result = run((block,), actor=execute)
    assert result["calls_used"] == 4
    assert result["effects"]["repair"]["matched_trials"] == 1
    json.dumps(result, allow_nan=False)


def test_multiple_successful_repairs_remain_ambiguous():
    block = known_fault_block("missing_context", secondary="stale_evidence")

    def alternatives(identity, turns):
        card = json.loads(turns[-1]["content"])
        passed = bool(card["entries"]) or card["as_of"] == 2 or card["audit_copy"] is not None
        return ReplayResult.from_actions(identity, turns, ("9" if passed else "clarify",))

    result = run((block,), actor=alternatives)
    assert result["attribution"] == "ambiguous"
    assert set(result["effective_repairs"]) == {"repair", "second", "joint"}
    assert result["mechanism_recovery_established"] is False
    assert result["observations"][0]["source_record"]["family_id"] == "missing_context"


def test_no_target_recovery_and_failed_sham_are_inconclusive():
    block = known_fault_block("missing_context")

    def responds_to_audit(identity, turns):
        card = json.loads(turns[-1]["content"])
        answer = card["audit_copy"] or "clarify"
        return ReplayResult.from_actions(identity, turns, (answer,))

    def responds_to_sham(identity, turns):
        card = json.loads(turns[-1]["content"])
        if card["presentation"] == "compact":
            return ReplayResult.from_actions(identity, turns, ("9",))
        return fixture_actor(identity, turns)

    assert run((block,), actor=responds_to_audit)["attribution"] == "inconclusive"
    result = run((block,), actor=responds_to_sham)
    assert result["status"] == "inconclusive"
    assert result["attribution"] == "inconclusive"
    assert "negative control" in " ".join(result["reasons"])


def test_exact_config_runs_regression_and_reports_effect_provenance():
    from pathlib import Path

    from evaluation.causal_records import content_digest
    from synthetic_data.mdac_interventions import run_fixture_config

    path = Path(__file__).resolve().parents[1] / "configs/mdac/interventions.json"
    config = json.loads(path.read_text())
    report = run_fixture_config(config)
    digest = report.pop("result_digest")
    assert digest == content_digest(report)
    assert report["dataset_kind"] == "synthetic_known_fault"
    assert report["training_eligibility"] == "REGRESSION"
    assert report["effects"]["repair"]["matched_trials"] == len(config["seeds"])
    with pytest.raises(ValueError, match="enabled"):
        run_fixture_config(config | {"enabled": False})


def test_scoped_host_intervention_requires_registered_plan_and_bound_receipt():
    from evaluation.causal_records import FactorChange, canonical_json, content_digest

    block = known_fault_block("missing_context")
    repair, *controls = block.treatments
    plan = content_digest({"checkpoint": block.identity.checkpoint, "layer": 0, "positions": [1]})
    arms = []
    for t in (repair, *controls):
        before = replace(
            t.pair.before, factors=t.pair.before.factors + (("activation_plan", "null"),)
        )
        factors = t.pair.after.factors + (("activation_plan", "null"),)
        after = replace(t.pair.after, factors=factors)
        pair = replace(t.pair, before=before, after=after)
        if t is repair:
            after = replace(
                before,
                variant_id=after.variant_id,
                factors=before.factors[:-1] + (("activation_plan", canonical_json(plan)),),
            )
            pair = replace(
                pair,
                after=after,
                changes=(FactorChange("activation_plan", "null", canonical_json(plan)),),
            )
        arms.append(replace(t, pair=pair, procedure_id=plan if t is repair else None))
    block = replace(block, treatments=tuple(arms))
    with pytest.raises(ValueError, match="registered"):
        run((block,))
    with pytest.raises(ValueError, match="procedure"):
        run((block,), procedures={plan: fixture_actor})

    def scoped(identity, turns):
        return ReplayResult.from_actions(identity, turns, ("9",), procedure_id=plan)

    result = run((block,), procedures={plan: scoped})
    assert result["effects"]["repair"]["drs"] == 1
    assert result["observations"][0]["procedure_id"] == plan
