"""Content-matched three-condition comparisons retain missing opportunities and measured units."""

from dataclasses import replace

import pytest
from artifact_fixtures import make_assessment, make_capture, make_protocol, make_topology

from evaluation.evidence_topology_comparison import (
    TopologyComparisonPlan,
    TopologyComparisonRun,
    TopologyComparisonSlot,
    compare_evidence_topology,
)
from evaluation.evidence_topology_records import CONDITIONS


def experiment():
    p = make_protocol()
    slots = tuple(
        TopologyComparisonSlot(
            condition,
            condition,
            "family",
            f"index-{i}",
            replace(
                p,
                protocol_id=condition,
                query=replace(p.query, request_id=condition),
                subject=replace(
                    p.subject, system=replace(p.subject.system, model_version=f"model-{i}")
                ),
            ),
        )
        for i, condition in enumerate(CONDITIONS)
    )
    plan = TopologyComparisonPlan("experiment", slots)
    return plan, runs_for(plan)


def runs_for(plan):
    result = []
    for s in plan.slots:
        c = make_capture(s.protocol, s.condition)
        result.append(TopologyComparisonRun(s.run_id, c, make_assessment(s.protocol, c)))
    return tuple(result)


def compare(plan, runs, **kw):
    return compare_evidence_topology(
        plan,
        runs,
        enabled=True,
        authorize=kw.pop("authorize", lambda _: True),
        authenticate=kw.pop("authenticate", lambda _: True),
        **kw,
    )


def test_content_matching_and_treatment_versions():
    plan, runs = experiment()
    out = compare(plan, runs)
    assert len(out["paired"]) == 2
    assert all(row["deltas"]["correctness"] == 0 for row in out["paired"])
    assert out["training_effect_established"] is False
    assert len({s.protocol.subject.system.model_version for s in plan.slots}) == 3
    for record in (plan, plan.slots[0], runs[0]):
        assert type(record).from_dict(record.to_dict()) == record
    with pytest.raises(ValueError):
        compare_evidence_topology(plan, runs)


@pytest.mark.parametrize(
    "field",
    [
        "query",
        "snapshot",
        "entity",
        "policy",
        "rubric",
        "principal",
        "scope",
        "time",
        "evaluator",
        "seed",
    ],
)
def test_comparison_drift(field):
    from evaluation.causal_records import PromptTurn, canonical_json

    plan, _ = experiment()
    p = plan.slots[1].protocol
    if field == "query":
        turns = (PromptTurn("user", "Different question"),)
        p = replace(
            p,
            subject=replace(p.subject, turns=turns),
            query=replace(p.query, public_query=canonical_json([t.to_dict() for t in turns])),
        )
    elif field in ("snapshot", "entity"):
        s = p.snapshot
        s = (
            replace(s, entity_ids=s.entity_ids + ("other",))
            if field == "entity"
            else replace(
                s, artifacts=(replace(s.artifacts[0], availability="restricted"), s.artifacts[1])
            )
        )
        p = replace(p, snapshot=s, topology=make_topology(s, "redundant"))
    elif field == "policy":
        p = replace(p, query=replace(p.query, policy_version="new"))
    elif field == "rubric":
        p = replace(p, rubric_id="new")
    elif field == "evaluator":
        p = replace(p, evaluator=replace(p.evaluator, contract_id="new"))
    elif field == "seed":
        p = replace(p, subject=replace(p.subject, system=replace(p.subject.system, seed=99)))
    else:
        name = {"principal": "principal_id", "scope": "scope_id", "time": "assessed_at"}[field]
        p = replace(
            p,
            query=replace(p.query, **{name: "2026-10-09T12:00:01Z" if field == "time" else "new"}),
        )
    plan = replace(plan, slots=(plan.slots[0], replace(plan.slots[1], protocol=p), plan.slots[2]))
    out = compare(plan, runs_for(plan))
    assert out["paired"][0]["deltas"]["correctness"] is None
    assert all(row["deltas"]["correctness"] is None for row in out["paired"])


def test_missing_run_retains_roster():
    plan, runs = experiment()
    out = compare(plan, runs[1:])
    baseline = out["conditions"]["existing_retrieval"]
    assert baseline["planned_runs"] == 1
    assert baseline["metrics"]["correctness"]["missing"] == 1
    assert baseline["verification_coverage"]["denominator"] == 6
    assert all(row["deltas"]["correctness"] is None for row in out["paired"])


def test_duplicate_pairing_keys():
    plan, runs = experiment()
    with pytest.raises(ValueError):
        replace(plan, slots=plan.slots + (replace(plan.slots[0], run_id="duplicate"),))
    with pytest.raises(ValueError):
        compare(plan, runs + (runs[0],))
    with pytest.raises(ValueError):
        compare(plan, (replace(runs[0], run_id="foreign"),))
    with pytest.raises(ValueError):
        compare(plan, (replace(runs[0], capture=runs[1].capture),))
    with pytest.raises(ValueError):
        replace(plan, slots=plan.slots[:2])


def test_raw_receipt_reauthentication():
    plan, runs = experiment()
    assert compare(plan, runs)["paired"][0]["deltas"]["correctness"] == 0
    denied = compare(plan, runs, authenticate=lambda _: False)
    assert denied["paired"][0]["deltas"]["correctness"] is None

    def mutate(a):
        object.__setattr__(a, "reason", "forged")
        return True

    assert compare(plan, runs, authenticate=mutate)["paired"][0]["deltas"]["correctness"] is None
    with pytest.raises(ValueError):
        compare(plan, (replace(runs[0], assessment=runs[1].assessment),))


def test_unequal_population_counts():
    plan, _ = experiment()
    extra = []
    for i in range(2):
        slot = plan.slots[0]
        p = replace(
            slot.protocol,
            protocol_id=f"extra-{i}",
            subject=replace(
                slot.protocol.subject, system=replace(slot.protocol.subject.system, seed=50 + i)
            ),
        )
        extra.append(replace(slot, run_id=f"extra-{i}", protocol=p))
    plan = replace(plan, slots=plan.slots + tuple(extra))
    runs = list(runs_for(plan))
    for i in (3, 4):
        runs[i] = replace(
            runs[i],
            assessment=replace(
                runs[i].assessment,
                verdicts=tuple(replace(v, value=False) for v in runs[i].assessment.verdicts),
            ),
        )
    out = compare(plan, tuple(runs))
    metric = out["conditions"]["existing_retrieval"]["metrics"]["correctness"]
    assert (metric["numerator"], metric["denominator"], metric["rate"]) == (1, 3, 1 / 3)
    assert out["conditions"]["artifact_index"]["metrics"]["correctness"]["denominator"] == 1


def test_partial_zero_and_incompatible_costs():
    plan, runs = experiment()
    runs = list(runs)
    for i, run in enumerate(runs):
        c = replace(run.capture, latency_seconds=0.0, cost_unit="EUR" if i == 1 else "USD")
        runs[i] = replace(run, capture=c, assessment=make_assessment(plan.slots[i].protocol, c))
    out = compare(plan, tuple(runs))
    assert out["paired"][0]["cost_delta"] is None
    assert out["paired"][0]["latency_delta"] == 0.0
    extra = replace(
        plan.slots[0],
        run_id="missing",
        protocol=replace(
            plan.slots[0].protocol,
            subject=replace(
                plan.slots[0].protocol.subject,
                system=replace(plan.slots[0].protocol.subject.system, seed=75),
            ),
        ),
    )
    plan = replace(plan, slots=plan.slots + (extra,))
    partial = compare(plan, tuple(runs))["conditions"]["existing_retrieval"]
    assert partial["retrieval_cost"]["known_count"] == 1
    assert partial["retrieval_cost"]["missing_count"] == 1
    assert partial["retrieval_cost"]["partial"] is True
    assert partial["retrieval_cost"]["by_unit"]["USD"]["mean"] == 0.0


def test_incomplete_metric_does_not_suppress_complete_metric():
    plan, runs = experiment()
    run = runs[1]
    a = replace(
        run.assessment,
        verdicts=tuple(
            replace(v, value=None) if v.opportunity_id == "op:entity_error" else v
            for v in run.assessment.verdicts
        ),
    )
    runs = (runs[0], replace(run, assessment=a), runs[2])
    out = compare(plan, runs)
    assert out["paired"][0]["deltas"]["correctness"] == 0
    assert out["paired"][0]["deltas"]["entity_error"] is None
