"""Three-condition comparison recomputes raw evidence and retains unmatched/missing runs."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_causal_diagnostics import captures, judgment, opportunity
from test_causal_records import pair
from test_debate_records import protocol as debate_protocol
from test_debate_records import snapshot
from test_perspective_protocol import candidate
from test_pluralistic_diagnostics import assessment
from test_pluralistic_records import protocol
from test_sensitive_debate import run_fixture

# Local
from evaluation.causal_records import PromptTurn
from evaluation.pluralistic_comparison import (
    CONDITIONS,
    ComparisonPlan,
    ComparisonRun,
    ComparisonSlot,
    DebateAttachment,
    compare_pluralistic_conditions,
)
from semantic_intent_robustness.perspective_generation import capture_perspectives


def comparison_fixture(family="family", unknown=False):
    """Different checkpoints of one declared model family share evaluation coordinates."""
    slots, runs = [], []
    for condition in CONDITIONS:
        base = pair()
        p = replace(
            base,
            family_id=family,
            pair_id=family,
            seed_policy="shared",
            before=replace(
                base.before, system=replace(base.before.system, model_version=condition)
            ),
            after=replace(
                base.after,
                system=replace(
                    base.after.system, model_version=condition, seed=base.before.system.seed
                ),
            ),
        )
        pp = protocol(p)
        cs = captures(p)
        pc = None
        debate = None
        if condition == "laundering_debate_pluralistic":
            pc = capture_perspectives(
                pp.plan, candidates=(candidate(), candidate("b")), enabled=True
            )
            dp = replace(debate_protocol(1), subject=p.after)
            session = run_fixture(dp, defend=lambda ctx: snapshot(action="proceed"))
            debate = DebateAttachment(dp, session, None)
        a = assessment(pp, cs, pc)
        if unknown:
            a = replace(a, verdicts=tuple(replace(v, value=None) for v in a.verdicts))
        op = opportunity(p, metric="semantic_laundering_susceptibility", cohort="benign")
        j = judgment(p, (p,), cs, (op,))
        run_id = f"{family}:{condition}"
        runs.append(ComparisonRun(run_id, pp, cs, pc, a, "fixture-v1", (op,), (j,), debate))
        slots.append(
            ComparisonSlot(
                run_id,
                condition,
                family,
                "heldout-v1",
                "curriculum-" + condition,
                "model-family",
                condition,
                p.after.system.harness_version,
                p.after.system.seed,
                p.after.system.repeat_id,
                p.digest,
            )
        )
    return ComparisonPlan("experiment", tuple(slots)), tuple(runs)


def compare(plan, runs, **kwargs):
    """Fixture authorities compare complete raw receipts, never saved acceptance flags."""
    receipts = [r.assessment.to_dict() for r in runs if r.assessment is not None]
    judgments = [j.to_dict() for r in runs for j in r.causal_judgments]
    return compare_pluralistic_conditions(
        plan,
        runs,
        enabled=True,
        authenticate_pluralistic=kwargs.pop(
            "authenticate_pluralistic", lambda a: a.to_dict() in receipts
        ),
        authenticate_causal=lambda j: j.to_dict() in judgments,
        **kwargs,
    )


def test_all_three_conditions_keep_treatments_and_raw_provenance():
    """Checkpoint/curriculum differences are retained treatments, not pairing exclusions."""
    plan, runs = comparison_fixture()
    report = compare(plan, runs)
    assert set(report["conditions"]) == set(CONDITIONS)
    assert len(report["paired"]) == 1 and report["unpaired"] == []
    assert report["paired"][0]["deltas"]["perspective_robustness"]["laundering_only"] == 0
    combined = report["run_rows"][-1]
    assert combined["combined_protocol_complete"]
    assert all("raw_run" in r and "pluralistic_report" in r for r in report["run_rows"])
    assert all(
        "benign_overrefusal" in c["causal_metrics"]["semantic_laundering_susceptibility"]
        for c in report["conditions"].values()
    )


def test_missing_runs_and_auxiliary_phases_are_visible():
    """A missing run cannot silently shrink the experiment; missing auxiliaries are separate."""
    plan, runs = comparison_fixture()
    result = compare(plan, runs[:-1])
    assert result["conditions"][CONDITIONS[-1]]["planned_runs"] == 1
    assert result["conditions"][CONDITIONS[-1]]["missing_runs"] == [runs[-1].run_id]
    assert result["paired"][0]["deltas"]["perspective_robustness"] is None
    r = runs[-1]
    a = assessment(r.protocol, r.captures)
    runs = runs[:-1] + (replace(r, perspective_capture=None, debate=None, assessment=a),)
    result = compare(plan, runs)
    row = result["run_rows"][-1]
    assert not row["combined_protocol_complete"]
    assert row["missing_phases"] == ["debate", "perspectives"]
    assert row["pluralistic_report"]["metrics"]["perspective_robustness"]["rate"] == 1


def test_eligible_unknown_aggregates_as_one_of_two_with_null_rate():
    """Aggregate counts rather than averaging per-run rates or dropping unknown outcomes."""
    one, first = comparison_fixture("one")
    two, second = comparison_fixture("two", unknown=True)
    plan = ComparisonPlan("experiment", one.slots + two.slots)
    result = compare(plan, first + second)
    for c in result["conditions"].values():
        metric = c["metrics"]["perspective_robustness"]
        assert (metric["numerator"], metric["denominator"], metric["rate"]) == (1, 2, None)
    # Unequal condition rosters are allowed and remain explicit.
    plan = ComparisonPlan("unequal", one.slots + (two.slots[0],))
    result = compare(plan, first + (second[0],))
    assert result["conditions"][CONDITIONS[0]]["planned_runs"] == 2


@pytest.mark.parametrize("field,value", [("split_id", "different"), ("model_family", "unrelated")])
def test_nonmatching_dimensions_do_not_pool(field, value):
    """Different holdouts or model families cannot establish a paired delta."""
    plan, runs = comparison_fixture()
    plan = replace(plan, slots=plan.slots[:2] + (replace(plan.slots[2], **{field: value}),))
    result = compare(plan, runs)
    assert all(p["deltas"]["perspective_robustness"] is None for p in result["paired"])


def test_per_arm_seed_difference_stays_unpaired():
    """A before/after seed mismatch is retained without pretending to be paired."""
    plan, runs = comparison_fixture()
    r = runs[0]
    p = replace(
        r.protocol.pair,
        before=replace(
            r.protocol.pair.before, system=replace(r.protocol.pair.before.system, seed=999)
        ),
        seed_policy="per_arm",
    )
    pp = protocol(p)
    cs = captures(p)
    r = replace(r, protocol=pp, captures=cs, assessment=assessment(pp, cs), causal_judgments=())
    plan = replace(plan, slots=(replace(plan.slots[0], pair_digest=p.digest),) + plan.slots[1:])
    result = compare(plan, (r,) + runs[1:])
    assert result["unpaired"][0]["run_id"] == r.run_id


def test_saved_reports_never_authenticate_and_bindings_reject_substitution():
    """Reanalysis without receipts resolves nothing, and stale slots are errors."""
    plan, runs = comparison_fixture()
    assert (
        compare(plan, runs)["conditions"][CONDITIONS[0]]["metrics"]["perspective_robustness"][
            "rate"
        ]
        == 1
    )
    fresh = compare(plan, runs, authenticate_pluralistic=None)
    assert fresh["conditions"][CONDITIONS[0]]["metrics"]["perspective_robustness"]["rate"] is None
    with pytest.raises(ValueError):
        compare_pluralistic_conditions(plan, ({"assessment_accepted": True},), enabled=True)
    with pytest.raises(ValueError):
        compare(replace(plan, slots=(replace(plan.slots[0], seed=999),) + plan.slots[1:]), runs)
    with pytest.raises(ValueError):
        ComparisonPlan("missing-condition", plan.slots[:2])


@pytest.mark.parametrize("change", ["subject", "action"])
def test_debate_binds_exact_subject_and_final_capture(change):
    """An unrelated subject or stale final action cannot be attached to a run."""
    plan, runs = comparison_fixture()
    r = runs[-1]
    dp = r.debate.protocol
    if change == "subject":
        dp = replace(dp, subject=replace(dp.subject, expected_actions=("other",)))
    session = run_fixture(
        dp, defend=lambda ctx: snapshot(action="other" if change == "action" else "proceed")
    )
    r = replace(r, debate=DebateAttachment(dp, session, None))
    with pytest.raises(ValueError):
        compare(plan, runs[:-1] + (r,))


def test_benign_label_is_not_silently_rewritten_under_a_receipt():
    """Hosts select PR-1's benign population before obtaining authenticated judgments."""
    _, runs = comparison_fixture()
    run = runs[0]
    with pytest.raises(ValueError, match="cohort 'benign'"):
        replace(
            run,
            causal_opportunities=(replace(run.causal_opportunities[0], cohort="benign_control"),),
        )


@pytest.mark.parametrize("drift", ["facts", "rubric", "oracle"])
def test_changed_evaluation_item_or_rubric_cannot_produce_paired_deltas(drift):
    """Fresh individual receipts do not establish cross-condition scenario equivalence."""
    plan, runs = comparison_fixture()
    run = runs[0]
    pair = run.protocol.pair
    if drift == "facts":
        pair = replace(
            pair,
            before=replace(
                pair.before,
                turns=(PromptTurn("user", "New material fact: the risk is resolved."),)
                + pair.before.turns,
            ),
            after=replace(
                pair.after,
                turns=(PromptTurn("user", "New material fact: the risk is resolved."),)
                + pair.after.turns,
            ),
        )
    elif drift == "oracle":
        pair = replace(pair, after=replace(pair.after, expected_actions=("defer",)))
    pp = protocol(pair)
    if drift == "rubric":
        pp = replace(pp, rubric_id="different-semantic-rubric")
    cs = captures(pair)
    cj = judgment(pair, (pair,), cs, run.causal_opportunities)
    run = replace(
        run, protocol=pp, captures=cs, assessment=assessment(pp, cs), causal_judgments=(cj,)
    )
    plan = replace(plan, slots=(replace(plan.slots[0], pair_digest=pair.digest),) + plan.slots[1:])
    result = compare(plan, (run,) + runs[1:])
    assert all(
        all(value is None for value in group["deltas"].values()) for group in result["paired"]
    )
    assert any(row["run_id"] == run.run_id for row in result["unpaired"])
