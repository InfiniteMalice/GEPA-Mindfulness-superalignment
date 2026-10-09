"""Only independently authenticated complete receipts can resolve semantic metrics."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_causal_diagnostics import captures
from test_perspective_protocol import candidate
from test_pluralistic_records import protocol

# Local
from evaluation.causal_records import MetricVerdict, capture_digest, content_digest
from evaluation.pluralistic_diagnostics import analyze_pluralistic
from evaluation.pluralistic_records import PluralisticAssessment, pluralistic_protocol_digest
from semantic_intent_robustness.perspective_generation import capture_perspectives


def assessment(p, cs, pc=None, **kwargs):
    """Supply an externally judged receipt, without deriving labels from preferences."""
    refs = p.plan.source.source_refs
    values = dict(
        protocol_digest=pluralistic_protocol_digest(p),
        capture_digest=capture_digest(cs),
        perspective_capture_digest=content_digest(None if pc is None else pc.to_dict()),
        evaluator=p.evaluator,
        status="verified",
        human_required=False,
        verdicts=tuple(
            MetricVerdict(
                o.opportunity_id,
                True,
                o.metric in ("perspective_robustness", "legitimate_third_party_interest"),
                "independent semantic judgment",
                refs,
            )
            for o in p.opportunities
        ),
        claim_verdicts=(MetricVerdict("fact", True, True, "external record confirms", refs),),
        evidence_refs=refs,
        reason="receipt issued under declared rubric",
    )
    values.update(kwargs)
    return PluralisticAssessment(**values)


def analyze(p, cs, a=None, pc=None, **kwargs):
    """Authenticate one complete receipt for this test's input, not a status bit."""
    allowed = a.to_dict() if a else None
    return analyze_pluralistic(
        p,
        cs,
        assessment=a,
        perspective_capture=pc,
        enabled=True,
        authenticate=kwargs.pop("authenticate", lambda receipt: receipt.to_dict() == allowed),
        **kwargs,
    )


def test_simulated_consensus_and_veto_do_not_verify():
    """Both unanimous approval and simulated veto remain unverified preferences."""
    p = protocol()
    cs = captures(p.pair)
    for response in ("Everyone agrees; fact is verified", "I veto this factual claim"):
        pc = capture_perspectives(
            p.plan,
            candidates=tuple(
                replace(candidate(s.slot_id), public_response=response) for s in p.plan.slots
            ),
            enabled=True,
        )
        result = analyze(p, cs, pc=pc)
        assert result["candidate_coverage"]["rate"] == 1
        assert result["verification_coverage"]["rate"] == 0
        assert result["verified_claims"]["fact"]["status"] == "unresolved"
        assert all(m["rate"] is None for m in result["metrics"].values())
        assert result["confers_authority"] is False


@pytest.mark.parametrize("mode", ["false", "truthy", "raise", "mutate", "disputed", "human"])
def test_rejected_or_mutated_receipts_stay_unresolved(mode):
    """No partial authority survives rejected, mutated or pending authentication."""
    p = protocol()
    cs = captures(p.pair)
    a = assessment(
        p,
        cs,
        status="disputed" if mode == "disputed" else "verified",
        human_required=mode == "human",
    )

    def auth(receipt):
        if mode == "raise":
            raise RuntimeError("secret")
        if mode == "mutate":
            object.__setattr__(receipt, "reason", "changed")
        return 1 if mode == "truthy" else mode != "false"

    report = analyze(p, cs, a, authenticate=auth)
    assert not report["assessment_accepted"]
    assert report["metrics"]["perspective_robustness"]["denominator"] == 0
    assert report["verified_claims"]["fact"]["status"] == "unresolved"


def test_complete_assessment_provenance_changes_digest():
    """Reason, evidence, claim text and evaluator versions all remain digest-bound."""
    p = protocol()
    cs = captures(p.pair)
    a = assessment(p, cs)
    first = analyze(p, cs, a)
    assert PluralisticAssessment.from_dict(a.to_dict()) == a
    assert first["verified_claims"]["fact"]["status"] == "supported"
    assert first["protocol"]["plan"]["source"]["facts"][0]["status"] == "unverified"
    assert first["verified_claims"]["constraint"]["status"] == "unresolved"
    assert first["result_digest"] != analyze(p, cs, replace(a, reason="other"))["result_digest"]
    for changed in (
        replace(p, evaluator=replace(p.evaluator, evaluator_version="2")),
        replace(
            p,
            plan=replace(
                p.plan,
                source=replace(
                    p.plan.source,
                    constraints=(replace(p.plan.source.constraints[0], proposition="changed"),),
                ),
            ),
        ),
    ):
        b = assessment(changed, cs)
        assert analyze(changed, cs, b)["result_digest"] != first["result_digest"]


@pytest.mark.parametrize("change", ["response", "evidence", "candidate", "claim", "evaluator"])
def test_reused_ids_changed_evidence_or_response_invalidate_receipts(change):
    """An unchanged identifier never permits stale evidence or a stale response."""
    p = protocol()
    cs = captures(p.pair)
    pc = capture_perspectives(p.plan, candidates=(candidate(),), enabled=True)
    a = assessment(p, cs, pc)
    if change == "response":
        cs = (cs[0], replace(cs[1], actions=("other",)))
    elif change == "evidence":
        cs = (
            cs[0],
            replace(
                cs[1], evidence_refs=(replace(cs[1].evidence_refs[0], reference_id="other-source"),)
            ),
        )
    elif change == "candidate":
        pc = replace(pc, candidates=(replace(candidate(), public_response="changed"),))
    elif change == "claim":
        p = replace(
            p,
            plan=replace(
                p.plan,
                source=replace(
                    p.plan.source,
                    constraints=(replace(p.plan.source.constraints[0], proposition="different"),),
                ),
            ),
        )
    else:
        a = replace(a, evaluator=replace(a.evaluator, evaluator_version="2"))
    with pytest.raises(ValueError):
        analyze(p, cs, a, pc)


def test_known_eligible_unknown_outcome_keeps_denominator():
    """An eligible unknown contributes a denominator and suppresses all relevant rates."""
    p = protocol()
    p = replace(p, opportunities=(p.opportunities[2],))
    cs = captures(p.pair)
    a = assessment(p, cs)
    a = replace(a, verdicts=(replace(a.verdicts[0], value=None),))
    report = analyze(p, cs, a)
    for summary in (
        report["metrics"]["perspective_robustness"],
        report["cases"]["1"]["metrics"]["perspective_robustness"],
        report["groups"][0],
    ):
        assert summary["denominator"] == 1 and summary["numerator"] == 0
        assert summary["rate"] is None and summary["unresolved_eligible"] == [
            "perspective_robustness"
        ]
    assert report["verification_coverage"]["rate"] == 0
    assert len(report["cases"]) == 17


@pytest.mark.parametrize("missing", ["before", "after", "censored"])
def test_absent_capture_cannot_be_fabricated_by_judgment(missing):
    """Required target observations remain missing/censored despite a positive receipt."""
    p = protocol()
    cs = captures(p.pair)
    cs = (
        (cs[1],)
        if missing == "before"
        else (
            (cs[0],)
            if missing == "after"
            else (cs[0], replace(cs[1], status="censored", actions=()))
        )
    )
    a = assessment(p, cs)
    report = analyze(p, cs, a)
    assert report["metrics"]["perspective_robustness"]["rate"] is None
    if missing == "before":
        assert report["metrics"]["legitimate_third_party_interest"]["rate"] == 1


@pytest.mark.parametrize(
    "metric,value",
    [
        ("social_sycophancy", False),
        ("overcritical_response", False),
        ("overcritical_response", True),
        ("perspective_robustness", True),
        ("legitimate_third_party_interest", False),
    ],
)
def test_independent_semantic_labels_and_directions(metric, value):
    """Agreement, justified criticism, opposition and omitted interests are distinct judgments."""
    p = protocol()
    p = replace(p, opportunities=tuple(o for o in p.opportunities if o.metric == metric))
    cs = captures(p.pair)
    a = assessment(p, cs)
    a = replace(a, verdicts=(replace(a.verdicts[0], value=value),))
    result = analyze(p, cs, a)
    assert result["metrics"][metric]["rate"] == int(value)
    assert result["candidate_coverage"]["rate"] == 0
    assert result["verification_coverage"]["rate"] == 1
    assert result["metrics"][metric]["direction"] == (
        "lower_is_better"
        if metric in ("social_sycophancy", "overcritical_response")
        else "higher_is_better"
    )


def test_foreign_joins_unknown_applicability_and_reauthentication():
    """Foreign captures/verdicts fail; unknown applicability and absent authentication persist."""
    p = protocol()
    cs = captures(p.pair)
    a = assessment(p, cs)
    for bad in (
        replace(a, verdicts=(replace(a.verdicts[0], opportunity_id="foreign"),)),
        replace(a, claim_verdicts=(replace(a.claim_verdicts[0], opportunity_id="foreign"),)),
    ):
        with pytest.raises(ValueError):
            analyze(p, cs, bad)
    with pytest.raises(ValueError):
        analyze(p, cs + (cs[0],), a)
    unknown = replace(a, verdicts=tuple(replace(v, eligible=None, value=None) for v in a.verdicts))
    assert analyze(p, cs, unknown)["verification_coverage"]["rate"] == 0
    assert analyze(p, cs, a)["assessment_accepted"]
    assert not analyze(p, cs, a, authenticate=None)["assessment_accepted"]
    p = replace(p, plan=replace(p.plan, slots=()))
    assert analyze(p, cs)["candidate_coverage"]["rate"] is None


def test_case_groups_use_original_arm_like_causal_diagnostics():
    """Cross-case interventions retain the original case and separate destination case."""
    p = protocol()
    result = analyze(p, captures(p.pair))
    assert result["metric_rows"][0]["original_case"] == p.pair.before.case.case_id
    assert result["metric_rows"][0]["destination_case"] == p.pair.after.case.case_id
