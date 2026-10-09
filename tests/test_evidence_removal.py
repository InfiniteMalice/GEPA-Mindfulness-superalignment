"""Whole-version removal distinguishes necessity, redundant support and unknown outcomes."""

from dataclasses import replace

import pytest
from artifact_fixtures import make_assessment, make_capture, make_derived_snapshot, make_protocol

from evaluation.evidence_removal import (
    RemovalObservation,
    RemovalPlan,
    RemovalSlot,
    analyze_evidence_removals,
)


def removal_plan(layout="redundant", snapshot=None):
    baseline = make_protocol(layout, snapshot)
    removed = replace(
        baseline,
        protocol_id="after-removal",
        query=replace(baseline.query, request_id="after-request", excluded_artifacts=(("A", "1"),)),
    )
    return RemovalPlan("removal", baseline, (RemovalSlot("remove-a", ("A", "1"), removed),))


def analyze(plan, *, observed=True, judged=True, baseline=True):
    c = make_capture(plan.baseline) if baseline else None
    observations = ()
    if observed:
        p = plan.slots[0].protocol
        rc = replace(make_capture(p), retrieved_item_ids=("b",), attributed_item_ids=("b",))
        a = make_assessment(p, rc)
        a = replace(a, verdicts=tuple(replace(v, value=False) for v in a.verdicts))
        observations = (RemovalObservation("remove-a", rc, a if judged else None),)
    return analyze_evidence_removals(
        plan,
        c,
        observations,
        baseline_assessment=make_assessment(plan.baseline, c) if judged else None,
        authorize=lambda _: True,
        authenticate=lambda _: True,
        enabled=True,
    )


def test_single_source_removal():
    out = analyze(removal_plan("single"))
    assert out["slots"][0]["lost_route_ids"] == ["via-a"]
    assert out["slots"][0]["surviving_route_ids"] == []
    assert out["slots"][0]["observed_action_changed"] is False
    assert out["slots"][0]["metrics"]["missing_evidence_failure"]["rate"] == 0


@pytest.mark.parametrize("layout", ["synthesis", "bridge"])
def test_synthesis_and_bridge_removal(layout):
    out = analyze(removal_plan(layout))
    assert out["slots"][0]["surviving_route_ids"] == []
    assert out["slots"][0]["structure"]["available_conclusion_ids"] == []


def test_redundant_support_survives():
    plan = removal_plan()
    out = analyze(plan)
    slot = out["slots"][0]
    assert slot["surviving_route_ids"] == ["via-b"]
    assert slot["metrics"]["missing_evidence_failure"]["rate"] == 0
    assert (
        analyze(plan, judged=False)["slots"][0]["metrics"]["missing_evidence_failure"]["rate"]
        is None
    )
    for rec in (plan, plan.slots[0], RemovalObservation("remove-a", None, None)):
        assert type(rec).from_dict(rec.to_dict()) == rec


def test_derived_ancestry_removal():
    out = analyze(removal_plan(snapshot=make_derived_snapshot()))
    rows = {r["item_id"]: r for r in out["slots"][0]["retrieval"]["item_rows"]}
    assert rows["a"]["channel"] == rows["summary"]["channel"] == "withheld"
    assert rows["b"]["channel"] == "candidate_evidence"


def test_version_specific_removal():
    from artifact_fixtures import make_snapshot

    from gepa_mindfulness.verification.state import EvidenceState

    s = make_snapshot()
    # A second fragment of A/1 disappears with the first; A/2 survives independently.
    first = s.sources[0]
    claim = replace(first.claim, claim_id="extra")
    extra = replace(
        first, item_id="extra", claim=claim, memory=replace(first.memory, memory_id="extra")
    )
    second = replace(s.artifacts[1], artifact_id="A", version="2")
    source_b = replace(s.sources[1], artifact_key=("A", "2"))
    s = replace(
        s,
        artifacts=(s.artifacts[0], second),
        sources=(first, source_b, extra),
        state=EvidenceState(s.state.claims + (claim,)),
    )
    out = analyze(removal_plan(snapshot=s))
    rows = {r["item_id"]: r for r in out["slots"][0]["retrieval"]["item_rows"]}
    assert rows["a"]["channel"] == rows["extra"]["channel"] == "withheld"
    assert rows["b"]["channel"] == "candidate_evidence"


def test_missing_planned_observations():
    plan = removal_plan()
    out = analyze(plan, observed=False, baseline=False)
    assert out["missing_slot_ids"] == ["remove-a"]
    assert out["slots"][0]["observed_action_changed"] is None
    assert out["slots"][0]["metrics"]["correctness"]["missing"] == 1
    assert out["baseline"]["metrics"]["correctness"]["missing"] == 1


def test_removal_binding_replay():
    plan = removal_plan()
    c = make_capture(plan.baseline)
    for obs in (
        (RemovalObservation("foreign", None, None),),
        (RemovalObservation("remove-a", c, None),),
        (RemovalObservation("remove-a", None, make_assessment(plan.baseline, None)),),
    ):
        with pytest.raises(ValueError):
            analyze_evidence_removals(plan, c, obs, enabled=True)
    for p in (
        replace(plan.slots[0].protocol, rubric_id="drift"),
        replace(
            plan.slots[0].protocol,
            query=replace(plan.slots[0].protocol.query, principal_id="drift"),
        ),
    ):
        with pytest.raises(ValueError):
            replace(plan, slots=(replace(plan.slots[0], protocol=p),))
    with pytest.raises(ValueError):
        replace(plan, slots=plan.slots * 2)
    with pytest.raises(ValueError):
        analyze_evidence_removals(plan, c, ())


def test_removal_retains_independent_destination_case():
    from test_causal_records import variant

    plan = removal_plan()
    p = plan.slots[0].protocol
    p = replace(
        p, subject=replace(p.subject, case=variant(case_id=14).case, expected_actions=("abstain",))
    )
    plan = replace(plan, slots=(replace(plan.slots[0], protocol=p),))
    out = analyze(plan)
    assert out["slots"][0]["original_case"] == 1
    assert out["slots"][0]["destination_case"] == 14
