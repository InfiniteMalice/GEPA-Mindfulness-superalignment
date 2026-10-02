"""Contract and behavioral controls for matched offline world-model experiments."""

import json
from dataclasses import replace

import pytest

from evaluation.world_model_ablation import compare_world_models
from evaluation.world_model_contracts import (
    Arm,
    Budget,
    Decision,
    ModelContract,
    WorldBackend,
    WorldCase,
)
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from gepa_mindfulness.verification.state import EvidenceState
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope
from synthetic_data.worlds import generate_world


def case(seed=1, **changes):
    """Return a JSON snapshot of a generated inspect/release fixture."""
    return WorldCase(
        f"case-{seed}",
        json.dumps(replace(generate_world(seed=seed, enabled=True), **changes).to_dict()),
        "operator",
        "release",
        "inspect-release",
        "consequential",
    )


def backend(factory):
    """Declare one deterministic, untrained test backend."""
    return WorldBackend(ModelContract("fixture", "a" * 64, "v1", 0, 0, "tokens"), factory)


def sensible(request):
    """Read only public evidence; request inspection before deciding to release."""
    data = json.loads(request.payload_json)
    observation = data["observation"]
    if "safe=unknown" in observation:
        action = "inspect"
    else:
        action = "release" if "safe=True" in observation else None
    return Decision(action, 1, 1.0, 0.8)


def run(factory=lambda seed: sensible, cases=None, budget=Budget(3, 2, 10), **kwargs):
    """Run the same policy factory in all arms with common caps."""
    return compare_world_models(
        cases or (case(),), backend(factory), budget, enabled=True, **kwargs
    )


def test_no_benefit_control_keeps_ties_and_added_representation_cost():
    """All arms can solve the toy task; structured state need not add behavioral value."""
    result = run(cases=(case(1), case(0)))
    assert len(result["rows"]) == 6
    assert all(row["task_success"] for row in result["rows"])
    assert all(row["model_calls"] == 2 for row in result["rows"])
    assert all(row["compute_units"] == 2 for row in result["rows"])
    for paired in result["paired"]:
        assert (paired["wins"], paired["losses"], paired["ties"]) == (0, 0, 2)
        assert paired["success_rate_delta"] == 0
        assert paired["mean_cost_delta"]["representation_bytes"] > 0
    assert result["arms"]["direct"]["prediction_brier"] is None
    assert result["arms"]["peo"]["prediction_brier"] == 0
    assert result["training_eligibility"] == "DEVELOPMENT"
    assert result["authority_granted"] is False
    assert result["mechanism_recovery_established"] is False
    json.dumps(result, allow_nan=False)


def test_benefit_and_regression_controls_preserve_signed_deltas():
    """A synthetic representation-dependent policy proves sensitivity, not model value."""

    def factory(seed):
        def decide(request):
            if request.arm is Arm.DIRECT:
                return Decision(None, 1)
            return sensible(request)

        return decide

    result = run(factory)
    assert all(p["wins"] == 1 and p["success_rate_delta"] == 1 for p in result["paired"])

    def regressing(seed):
        return lambda r: sensible(r) if r.arm is Arm.DIRECT else Decision(None, 1)

    result = run(regressing)
    assert all(p["losses"] == 1 and p["success_rate_delta"] == -1 for p in result["paired"])
    assert len(result["failures"]) == 2


def test_public_projection_and_reconciliation_are_prospective():
    """PEO gets its validated past; no arm receives latent world export fields."""
    captured = []

    def factory(seed):
        def decide(request):
            captured.append((request.arm, json.loads(request.payload_json)))
            return sensible(request)

        return decide

    result = run(factory)
    for arm, data in captured:
        assert set(data) == {
            "observation",
            "history",
            "target_action",
            "available_actions",
            "state",
            "reconciliations",
        }
        if not data["history"]:
            assert "safe=True" not in data["observation"]
            assert data["reconciliations"] == []
        assert (data["state"] is None) == (arm is Arm.DIRECT)
        if arm is not Arm.PEO:
            assert data["reconciliations"] == []
        elif data["history"]:
            update = data["reconciliations"][0]
            assert update["predicted_success"] == 1
            assert update["actual_success"] == 1
            assert update["residual"] == 0
            assert update["world_uncertainty_before"] == 0.5
            assert update["world_uncertainty_after"] == 0
    peo = next(r for r in result["rows"] if r["arm"] == "peo")
    assert len(peo["episode"]["events"]) == 12
    assert peo["episode"]["verification_scope"] == "simulator_consistency_only"


def test_private_metadata_and_evidence_ids_cannot_change_actor_inputs():
    """Identical visible facts hide different seeds, labels, provenance and evidence IDs."""
    original = case()
    data = json.loads(original.world_json)
    data["world_id"] = "SECRET_WORLD"
    data["seed"] = 923412
    data["provenance"] = ["SECRET_PROVENANCE"]
    for fact in data["facts"]:
        fact["evidence"]["reference_id"] = "SECRET_EVIDENCE"
        fact["evidence"]["source_kind"] = "observable_output"
        if fact["fact_id"] == "safe":
            fact["value"] = not fact["value"]
    changed = replace(original, case_id="SECRET_CASE", world_json=json.dumps(data))

    def capture(item):
        seen = {}

        def factory(seed):
            def decide(request):
                seen[request.arm] = request.payload_json
                return Decision(None, 1)

            return decide

        run(factory, cases=(item,))
        return seen

    assert capture(original) == capture(changed)


def test_fresh_sessions_shared_seed_and_snapshot_before_callbacks():
    """Mutated source records and disposable requests cannot change later comparisons."""
    item = case()
    caps = Budget(3, 2, 10)
    seeds = []
    calls = []

    def factory(seed):
        seeds.append(seed)
        object.__setattr__(item, "world_json", "invalid")
        object.__setattr__(caps, "compute_units", 0)
        object.__setattr__(host.contract, "model_version", "MUTATED")
        object.__setattr__(host, "factory", lambda _: None)
        count = [0]

        def decide(request):
            count[0] += 1
            calls.append(count[0])
            object.__setattr__(request.remaining, "model_calls", 999)
            return sensible(request)

        return decide

    host = backend(factory)
    result = compare_world_models((item,), host, caps, enabled=True)
    assert len(seeds) == 3 and len(set(seeds)) == 1
    assert calls == [1, 2, 1, 2, 1, 2]
    assert result["matching"]["model"]["model_version"] == "fixture"
    assert result["matching"]["budget"]["compute_units"] == 10


@pytest.mark.parametrize(
    "caps,status",
    [
        (Budget(1, 2, 10), "model_budget_exhausted"),
        (Budget(3, 2, 1), "compute_budget_exhausted"),
        (Budget(3, 0, 10), "tool_budget_exhausted"),
    ],
)
def test_exhaustion_preserves_all_rows(caps, status):
    """Missing completion counts as failure in every denominator."""
    result = run(budget=caps)
    assert len(result["rows"]) == 3
    assert all(not r["task_success"] and r["status"] == status for r in result["rows"])
    assert all(r["tool_calls"] <= caps.tool_calls for r in result["rows"])


def test_abstention_after_last_tool_is_allowed():
    """A revealed unsafe target can be declined after the last allowed inspection."""
    result = run(cases=(case(0),), budget=Budget(2, 1, 10))
    assert all(r["task_success"] and r["tool_calls"] == 1 for r in result["rows"])


def test_denied_attempts_and_premature_abstention_remain_failures():
    """No oracle labels are used as policy decisions or erased from failure inventory."""
    result = run(lambda seed: lambda request: Decision("release", 1, 1, 1))
    assert len(result["failures"]) == 3
    assert all(r["unsuccessful_actions"] == 1 for r in result["rows"])
    assert all(r["task_success"] is False for r in result["rows"])
    result = run(lambda seed: lambda request: Decision(None, 1))
    assert all(r["status"] == "premature_abstention" for r in result["rows"])


@pytest.mark.parametrize(
    "choice",
    [
        Decision("unknown", 1, 1, 1),
        Decision("release", 11, 1, 1),
        Decision("inspect", 1),
        "inspect",
    ],
)
def test_invalid_policy_receipts_fail_closed(choice):
    """Invalid or unmetered decisions abort comparison instead of producing biased aggregates."""
    with pytest.raises(ValueError):
        run(lambda seed: lambda request: choice)


@pytest.mark.parametrize("mutation", ["json", "duplicate", "foreign", "target", "nan"])
def test_full_catalog_preflight_precedes_callbacks(mutation):
    """Malformed later cases cannot execute an earlier policy callback."""
    item = case(0)
    if mutation == "json":
        object.__setattr__(item, "world_json", "{}")
    elif mutation == "duplicate":
        item = replace(case(), case_id="alias")
    elif mutation == "foreign":
        item = replace(item, actor_id="supervisor")
    elif mutation == "target":
        item = replace(item, target_action="missing")
    else:
        object.__setattr__(item, "world_json", '{"invalid": NaN}')
    calls = []
    with pytest.raises(ValueError):
        run(lambda seed: calls.append(seed), cases=(case(), item))
    assert calls == []


@pytest.mark.parametrize("value", [True, -1, 1.5, 2**53, "1"])
def test_budgets_reject_nonexact_or_unsafe_integers(value):
    """Resource accounting uses bounded nonnegative exact integers."""
    with pytest.raises(ValueError):
        Budget(value, 1, 10)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_probabilities_reject_invalid_values(value):
    """Prospective prediction and confidence must be finite unit values."""
    with pytest.raises(ValueError):
        Decision("inspect", 1, value, 0.5)


def test_disabled_has_no_callbacks():
    """Opt-in is exact and checked before reading policy input."""
    for enabled in (False, 1, "yes"):
        with pytest.raises(ValueError):
            compare_world_models((), None, None, enabled=enabled)


def test_effectful_auxiliary_action_is_not_available():
    """An actor cannot sabotage target truth conditions through an auxiliary action."""
    data = json.loads(case().world_json)
    data["actions"][0]["effects"] = [{"fact_id": "safe", "value": False}]
    altered = replace(case(), world_json=json.dumps(data))
    with pytest.raises(ValueError, match="available"):
        run(cases=(altered,))


def test_severe_rows_include_success_and_failures():
    """Consequential and catastrophic observations stay individually inspectable."""
    result = run(cases=(replace(case(), severity="catastrophic"),))
    assert len(result["severe_rows"]) == 3
    assert all(r["severity"] == "catastrophic" for r in result["severe_rows"])
    assert result["strata"][0]["cases"] == 1


def test_public_state_roundtrips_and_episode_has_real_causal_links():
    """The treatment reuses canonical state and PEO contracts without extra state fields."""

    def factory(seed):
        def decide(request):
            data = json.loads(request.payload_json)
            if data["state"] is not None:
                assert EvidenceState.from_dict(data["state"]).to_dict() == data["state"]
            return sensible(request)

        return decide

    report = run(factory)
    peo = next(r for r in report["rows"] if r["arm"] == "peo")
    events = [EventEnvelope(**event) for event in peo["episode"]["events"]]
    validate_action_bound_sequence(events)
    assert events[6].parent_event_ids == (events[5].event_id,)


def test_decision_receipts_account_for_every_call_including_abstention():
    """The report preserves individual compute receipts and prospective declarations."""
    report = run(cases=(case(0),))
    for row in report["rows"]:
        assert len(row["decisions"]) == row["model_calls"]
        assert sum(d["compute_used"] for d in row["decisions"]) == row["compute_units"]
        assert row["decisions"][-1]["action_id"] is None
        assert row["decisions"][0]["confidence"] == 0.8


@pytest.mark.parametrize("label", [TrainingEligibility.REGRESSION, TrainingEligibility.HIDDEN_EVAL])
def test_strictest_eligibility_survives_mixed_cases(label):
    """Nontraining data remain forbidden to the existing admission gate."""
    report = run(cases=(case(), case(0, training_eligibility=label)))
    assert report["training_eligibility"] == label.value
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)


@pytest.mark.parametrize("change", ["train", "duplicate_key", "empty", "bad_budget", "bad_model"])
def test_invalid_contracts_abort_before_factory(change):
    """All static inputs are revalidated before any trusted host work begins."""
    item = case()
    cases = (item,)
    caps = Budget(3, 2, 10)
    calls = []
    host = backend(lambda seed: calls.append(seed))
    if change == "train":
        object.__setattr__(item, "world_json", item.world_json.replace("DEVELOPMENT", "TRAIN"))
    elif change == "duplicate_key":
        object.__setattr__(item, "world_json", '{"world_id":"x",' + item.world_json[1:])
    elif change == "empty":
        cases = ()
    elif change == "bad_budget":
        object.__setattr__(caps, "compute_units", True)
    else:
        object.__setattr__(host.contract, "checkpoint_sha256", "bad")
    with pytest.raises(ValueError):
        compare_world_models(cases, host, caps, enabled=True)
    assert calls == []


def test_changed_decision_is_revalidated_and_host_errors_propagate():
    """Frozen-record mutation cannot bypass numeric validation; failed runs are not summarized."""
    decision = Decision(None, 1)
    object.__setattr__(decision, "compute_used", True)
    with pytest.raises(ValueError):
        run(lambda seed: lambda request: decision)

    def broken(seed):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        run(broken)
    with pytest.raises(ValueError, match="callable"):
        run(lambda seed: None)


def test_permission_denial_over_refusal_and_failed_auxiliary_inventory():
    """Successful abstention never hides an earlier failed investigation."""
    data = json.loads(case().world_json)
    for permission in data["permissions"]:
        permission["allowed"] = False
    item = replace(case(), world_json=json.dumps(data))

    def factory(seed):
        return lambda r: (
            Decision(None, 1)
            if json.loads(r.payload_json)["history"]
            else Decision("inspect", 1, 0.5, 0.5)
        )

    report = run(factory, cases=(item,))
    assert all(r["task_success"] for r in report["rows"])
    assert len(report["failures"]) == 3
    assert report["arms"]["peo"]["prediction_brier"] == 0.25
    data = json.loads(case().world_json)
    for fact in data["facts"]:
        fact["visible_to"] = ["operator"]
    report = run(
        lambda seed: lambda r: Decision(None, 1),
        cases=(replace(case(), world_json=json.dumps(data)),),
    )
    assert all(r["status"] == "over_refusal" for r in report["rows"])


def test_scheduling_is_repeatable_but_not_fixed_arm_order():
    """Independent sampling seed controls schedule and is shared within each triplet."""
    a = run(seed=91, cases=(case(), case(0)))
    b = run(seed=91, cases=(case(), case(0)))
    c = run(seed=92, cases=(case(), case(0)))
    assert a["execution_order"] == b["execution_order"]
    assert a["execution_order"] != c["execution_order"]
    for case_id in ("case-0", "case-1"):
        assert len({r["episode_seed"] for r in a["rows"] if r["case_id"] == case_id}) == 1


@pytest.mark.parametrize("value", ["", "   ", 1, True])
def test_text_and_zero_compute_are_invalid(value):
    """Contract identifiers cannot be blank/coerced; callbacks must meter positive compute."""
    with pytest.raises(ValueError):
        replace(case(), actor_id=value)
    with pytest.raises(ValueError):
        Decision(None, 0)


def test_zero_model_budget_does_not_call_policy():
    """Zero model allowance returns three failed rows without requesting a decision."""
    report = run(
        lambda seed: lambda r: pytest.fail("unexpected model call"), budget=Budget(0, 0, 10)
    )
    assert all(r["model_calls"] == 0 for r in report["rows"])
