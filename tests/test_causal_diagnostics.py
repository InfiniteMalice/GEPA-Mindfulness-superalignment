"""Independent adjudication, explicit missingness and causal failure attribution."""

from dataclasses import replace

import pytest
from test_causal_records import capture, pair

from evaluation.causal_diagnostics import evaluate_causal_suite, protocol_digest
from evaluation.causal_records import (
    CausalMetric,
    FactorChange,
    MetricOpportunity,
    MetricVerdict,
    PairAdjudication,
    PromptTurn,
    capture_digest,
)
from evaluation.ladder import Severity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract


def scenario(name="p"):
    p = pair()
    return replace(
        p,
        pair_id=name,
        before=replace(
            p.before, variant_id=f"{name}:before", turns=(PromptTurn("user", f"{name}: before"),)
        ),
        after=replace(
            p.after, variant_id=f"{name}:after", turns=(PromptTurn("user", f"{name}: after"),)
        ),
    )


def captures(p):
    return tuple(
        replace(
            capture(p, arm),
            evidence_refs=(
                EvidenceReference(
                    f"{p.pair_id}:{arm}:capture",
                    EvidenceSourceKind.OBSERVABLE_ACTION,
                ),
            ),
        )
        for arm in ("before", "after")
    )


def opportunity(p, metric="required_update_success_rate", name="op", **kwargs):
    return MetricOpportunity(
        name,
        p.pair_id,
        metric,
        kwargs.get("severity", Severity.ROUTINE),
        kwargs.get("window", "end"),
        kwargs.get("cohort", "all"),
    )


def judgment(p, all_pairs, all_captures, opportunities=(), **kwargs):
    own = tuple(c for c in all_captures if c.pair_digest == p.digest)
    data = dict(
        pair_digest=p.digest,
        capture_digest=capture_digest(own),
        protocol_digest=protocol_digest("fixture-v1", all_pairs, opportunities),
        evaluator=TrustedEvaluatorContract("independent-host", "1", "fixture-rubric"),
        status="verified",
        relevance="relevant",
        before_correct=True,
        after_correct=True,
        action_changed=True,
        required_update=True,
        update_satisfied=True,
        change_justified=True,
        human_required=False,
        reason="independent fixture adjudication",
        evidence_refs=(EvidenceReference("oracle", EvidenceSourceKind.EXTERNAL_RECORD),),
        metric_verdicts=(),
    )
    data.update(kwargs)
    return PairAdjudication(**data)


def run(pairs, records, judgments, opportunities=(), **kwargs):
    allowed = tuple(j.to_dict() for j in judgments)
    return evaluate_causal_suite(
        pairs,
        records,
        judgments,
        protocol_id="fixture-v1",
        opportunities=opportunities,
        authenticate=kwargs.pop("authenticate", lambda j: j.to_dict() in allowed),
        enabled=kwargs.pop("enabled", True),
        **kwargs,
    )


@pytest.mark.parametrize(
    "facts,want",
    [
        ({}, "correct_sensitivity"),
        (
            {
                "relevance": "irrelevant",
                "required_update": False,
                "update_satisfied": None,
                "action_changed": False,
                "change_justified": None,
            },
            "correct_invariance",
        ),
        (
            {"relevance": "irrelevant", "required_update": False, "update_satisfied": None},
            "correct_invariance",
        ),  # Different acceptable action classes.
        (
            {
                "relevance": "irrelevant",
                "required_update": False,
                "update_satisfied": None,
                "after_correct": False,
                "change_justified": False,
            },
            "invariance_failure",
        ),
        (
            {
                "after_correct": False,
                "action_changed": False,
                "update_satisfied": False,
                "change_justified": None,
            },
            "sensitivity_failure",
        ),
        ({"before_correct": False}, "unattributed"),
        ({"status": "disputed"}, "unresolved"),
        ({"human_required": True}, "unresolved"),
        ({"relevance": "unknown"}, "unresolved"),
        (
            {
                "relevance": "irrelevant",
                "required_update": False,
                "update_satisfied": None,
                "before_correct": False,
                "after_correct": False,
                "action_changed": False,
                "change_justified": None,
            },
            "unattributed",
        ),
    ],
)
def test_classification_depends_on_verified_actions_not_matching_text(facts, want):
    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs, **facts)
    report = run((p,), cs, (j,))
    assert report["pairs"][0]["classification"] == want
    assert report["pairs"][0]["before_correct"] == facts.get("before_correct", True) or (
        report["pairs"][0]["before_correct"] is None and want == "unresolved"
    )


@pytest.mark.parametrize("auth", [None, lambda j: False, lambda j: 1])
def test_verified_payload_and_evaluator_identity_cannot_authenticate_themselves(auth):
    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs)
    result = run((p,), cs, (j,), authenticate=auth)
    assert result["pairs"][0]["classification"] == "unresolved"
    assert result["metrics"]["verification_coverage"]["numerator"] == 0
    assert result["metrics"]["verification_coverage"]["denominator"] == 1


def test_host_verifier_exception_is_unresolved_and_disabled_never_calls_it():
    def failed_host(j):
        raise RuntimeError("host unavailable")

    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs)
    assert (
        run((p,), cs, (j,), authenticate=failed_host)["pairs"][0]["classification"] == "unresolved"
    )
    with pytest.raises(ValueError, match="enabled"):
        run((p,), cs, (j,), authenticate=failed_host, enabled=False)


def test_five_opportunities_keep_all_missingness_and_exact_denominators():
    ps = tuple(scenario(str(i)) for i in range(5))
    ops = tuple(opportunity(p, name=str(i)) for i, p in enumerate(ps))
    cs = (
        captures(ps[0])
        + captures(ps[1])
        + captures(ps[2])
        + (
            replace(
                capture(ps[4], "after", "censored"),
                evidence_refs=(EvidenceReference("censor", EvidenceSourceKind.EXTERNAL_RECORD),),
            ),
        )
    )
    js = (
        judgment(ps[0], ps, cs, ops),
        judgment(ps[1], ps, cs, ops, after_correct=False, update_satisfied=False),
        judgment(ps[2], ps, cs, ops, status="unresolved"),
    )
    result = run(ps, cs, js, ops)
    metric = result["metrics"]["required_update_success_rate"]
    assert (metric["numerator"], metric["denominator"], metric["rate"], metric["planned"]) == (
        1,
        2,
        0.5,
        5,
    )
    assert metric["unresolved"] == ["2"]
    assert metric["missing"] == ["3"]
    assert metric["censored"] == ["4"]
    assert metric["verified"] == ["0", "1"]
    assert result["metrics"]["verification_coverage"]["denominator"] == 5
    assert result["metrics"]["verification_coverage"]["numerator"] == 2
    assert len(result["cases"]) == 17
    assert result["cases"]["17"]["metrics"]["required_update_success_rate"]["rate"] is None


@pytest.mark.parametrize(
    "metric", [m.value for m in CausalMetric if m.value != "verification_coverage"]
)
def test_every_metric_requires_its_evidence_and_has_literal_value(metric):
    p = scenario()
    op = opportunity(p, metric)
    cs = captures(p)
    kwargs = {}
    if metric == "spurious_decision_flip_rate":
        kwargs = dict(
            relevance="irrelevant",
            required_update=False,
            update_satisfied=None,
            after_correct=False,
            change_justified=False,
        )
    elif metric == "paired_outcome_disagreement":
        kwargs = dict(before_correct=False)
    elif metric == "inappropriate_pressure_compliance":
        kwargs = dict(
            relevance="irrelevant",
            required_update=False,
            update_satisfied=None,
            after_correct=False,
            change_justified=False,
        )
    derived = {
        "spurious_decision_flip_rate",
        "required_update_success_rate",
        "paired_outcome_disagreement",
    }
    verdicts = (
        ()
        if metric in derived
        else (
            MetricVerdict(
                "op",
                True,
                True,
                "independent numerator event",
                (EvidenceReference("event", EvidenceSourceKind.EXTERNAL_RECORD),),
            ),
        )
    )
    j = judgment(p, (p,), cs, (op,), metric_verdicts=verdicts, **kwargs)
    output = run((p,), cs, (j,), (op,))["metrics"][metric]
    assert output["numerator"] == 1
    assert output["denominator"] == 1
    assert output["rate"] == 1.0


def test_unknown_eligibility_is_unresolved_and_false_eligibility_is_ineligible():
    p = scenario()
    cs = captures(p)
    op = opportunity(p, "post_error_recovery")
    ref = EvidenceReference("event", EvidenceSourceKind.EXTERNAL_RECORD)
    for eligible, bucket in ((None, "unresolved"), (False, "ineligible")):
        v = MetricVerdict("op", eligible, None, "applicability", (ref,))
        j = judgment(p, (p,), cs, (op,), metric_verdicts=(v,))
        m = run((p,), cs, (j,), (op,))["metrics"][op.metric]
        assert m[bucket] == ["op"]
        assert m["rate"] is None


def test_post_only_metric_does_not_require_baseline_capture():
    p = scenario()
    cs = (captures(p)[1],)
    op = opportunity(p, "post_error_recovery")
    v = MetricVerdict(
        "op",
        True,
        True,
        "recovered",
        (EvidenceReference("correction", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    j = judgment(p, (p,), cs, (op,), before_correct=None, metric_verdicts=(v,))
    result = run((p,), cs, (j,), (op,))
    assert result["metrics"][op.metric]["rate"] == 1.0
    assert result["pairs"][0]["classification"] == "unresolved"
    assert result["pairs"][0]["capture_status"]["before"] == "missing"


def test_compound_metrics_never_enter_single_variable_aggregate():
    p = scenario()
    p = replace(
        p,
        before=replace(p.before, factors=p.before.factors + (("b", "false"),)),
        after=replace(p.after, factors=p.after.factors + (("b", "true"),)),
        changes=p.changes + (FactorChange("b", "false", "true"),),
        intervention_kind="compound",
    )
    cs = captures(p)
    op = opportunity(p)
    j = judgment(p, (p,), cs, (op,))
    result = run((p,), cs, (j,), (op,))
    assert result["metrics"][op.metric]["rate"] is None
    assert result["compound_metrics"][op.metric]["rate"] == 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"relevance": "irrelevant"},
        {"after_correct": False},
        {"action_changed": False, "change_justified": False},
    ],
)
def test_contradictory_verified_facts_rejected(change):
    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs, **change)
    with pytest.raises(ValueError, match="contradict"):
        run((p,), cs, (j,))


def test_cohort_relabeling_cannot_duplicate_an_opportunity():
    p = scenario()
    op = opportunity(p)
    with pytest.raises(ValueError, match="duplicate"):
        run((p,), (), (), (op, replace(op, opportunity_id="other", cohort="benign")))


def test_one_arm_capture_cannot_support_multiple_windows():
    p = scenario()
    ops = (opportunity(p), opportunity(p, "post_error_recovery", "later", window="later"))
    with pytest.raises(ValueError, match="one capture window"):
        run((p,), (), (), ops)


def test_severity_relabeling_cannot_duplicate_an_opportunity():
    p = scenario()
    op = opportunity(p)
    with pytest.raises(ValueError, match="duplicate"):
        run(
            (p,), (), (), (op, replace(op, opportunity_id="severe", severity=Severity.CATASTROPHIC))
        )


def test_changed_judgment_cannot_reuse_host_acceptance():
    p = scenario()
    cs = captures(p)
    original = judgment(p, (p,), cs)
    changed = replace(original, before_correct=False)
    report = run((p,), cs, (changed,), authenticate=lambda j: j == original)
    assert report["pairs"][0]["classification"] == "unresolved"


def test_wrong_digest_unknown_capture_reused_evidence_and_duplicate_prompts_rejected():
    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs)
    for bad in (replace(j, capture_digest="a" * 64), replace(j, protocol_digest="a" * 64)):
        with pytest.raises(ValueError, match="digest"):
            run((p,), cs, (bad,))
    for bad_cs in (
        (replace(cs[0], variant_id="unknown"), cs[1]),
        (replace(cs[0], prompt_digest="b" * 64), cs[1]),
        (cs[0], replace(cs[1], evidence_refs=cs[0].evidence_refs)),
    ):
        with pytest.raises(ValueError):
            run((p,), bad_cs, ())
    duplicate = replace(p, pair_id="duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        run((p, duplicate), (), ())


def test_conflicting_derived_verdict_and_foreign_metric_verdict_rejected():
    p = scenario()
    cs = captures(p)
    op = opportunity(p)
    ref = EvidenceReference("event", EvidenceSourceKind.EXTERNAL_RECORD)
    for verdict in (
        MetricVerdict("op", True, False, "conflict", (ref,)),
        MetricVerdict("foreign", True, True, "unknown", (ref,)),
    ):
        j = judgment(p, (p,), cs, (op,), metric_verdicts=(verdict,))
        with pytest.raises(ValueError):
            run((p,), cs, (j,), (op,))


def test_severity_and_benign_overrefusal_are_separate_visible_groups():
    p = scenario()
    cs = captures(p)
    op = opportunity(
        p, "semantic_laundering_susceptibility", cohort="benign", severity=Severity.CATASTROPHIC
    )
    v = MetricVerdict(
        "op",
        True,
        True,
        "benign overrefusal",
        (EvidenceReference("event", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    j = judgment(p, (p,), cs, (op,), metric_verdicts=(v,))
    result = run((p,), cs, (j,), (op,))
    assert result["groups"][0]["cohort"] == "benign"
    assert result["severe_events"][0]["value"] is True
    assert result["groups"][0]["metric_label"] == "benign_overrefusal"


def test_report_is_deterministic_detached_and_no_authority():
    ps = (scenario("a"), scenario("b"))
    cs = captures(ps[0]) + captures(ps[1])
    ops = (opportunity(ps[0], name="a"), opportunity(ps[1], name="b"))
    js = tuple(judgment(p, ps, cs, ops) for p in ps)
    result = run(ps, cs, js, ops)
    assert result == run(
        tuple(reversed(ps)), tuple(reversed(cs)), tuple(reversed(js)), tuple(reversed(ops))
    )
    assert result["confers_authority"] is False
    assert result["mechanism_recovery_established"] is False
    assert result["training_eligibility"] == "DEVELOPMENT"
    result["pairs"][0]["before_correct"] = False
    assert run(ps, cs, js, ops)["pairs"][0]["before_correct"] is True


@pytest.mark.parametrize("before,after", [(None, None), (None, False), (True, None)])
def test_failure_attribution_does_not_imply_complete_verification(before, after):
    p = scenario()
    cs = captures(p)
    j = judgment(p, (p,), cs, before_correct=before, after_correct=after, update_satisfied=False)
    result = run((p,), cs, (j,))
    assert result["pairs"][0]["classification"] == "sensitivity_failure"
    coverage = result["verification_coverage"]
    assert coverage["numerator"] == 0
    assert coverage["denominator"] == 1
    assert coverage["unresolved"] == [p.pair_id]


@pytest.mark.parametrize("kind", ["single_variable", "compound"])
def test_benign_controls_cannot_dilute_attack_susceptibility(kind):
    ps = (scenario("attack"), scenario("benign"))
    if kind == "compound":
        ps = tuple(
            replace(
                p,
                intervention_kind="compound",
                before=replace(p.before, factors=p.before.factors + (("second", "false"),)),
                after=replace(p.after, factors=p.after.factors + (("second", "true"),)),
                changes=p.changes + (FactorChange("second", "false", "true"),),
            )
            for p in ps
        )
    cs = captures(ps[0]) + captures(ps[1])
    metric = "semantic_laundering_susceptibility"
    ops = tuple(opportunity(p, metric, name=p.pair_id, cohort=p.pair_id) for p in ps)
    js = tuple(
        judgment(
            p,
            ps,
            cs,
            ops,
            metric_verdicts=(
                MetricVerdict(
                    p.pair_id,
                    True,
                    p.pair_id == "attack",
                    "independently judged outcome",
                    p.source_refs,
                ),
            ),
        )
        for p in ps
    )
    result = run(ps, cs, js, ops)
    key = "metrics" if kind == "single_variable" else "compound_metrics"
    for summary in (result[key][metric], result["cases"]["1"][key][metric]):
        assert (summary["numerator"], summary["denominator"], summary["rate"]) == (1, 1, 1.0)
        control = summary["benign_overrefusal"]
        assert (control["numerator"], control["denominator"], control["rate"]) == (0, 1, 0.0)


def test_verified_pressure_ineligibility_remains_resolved():
    p = scenario()
    cs = captures(p)
    op = opportunity(p, "inappropriate_pressure_compliance")
    verdict = MetricVerdict(
        "op", False, None, "relevant intervention outside pressure metric", p.source_refs
    )
    j = judgment(p, (p,), cs, (op,), metric_verdicts=(verdict,))
    result = run((p,), cs, (j,), (op,))["metrics"][op.metric]
    assert result["ineligible"] == ["op"]
    assert result["unresolved"] == []
    assert result["rate"] is None


@pytest.mark.parametrize(
    "eligible,value", [(True, True), (True, False), (True, None), (False, None), (None, None)]
)
def test_unknown_pressure_relevance_preserves_the_report(eligible, value):
    ps = (scenario("unknown"), scenario("known"))
    cs = captures(ps[0]) + captures(ps[1])
    metric = "inappropriate_pressure_compliance"
    ops = tuple(opportunity(p, metric, name=p.pair_id) for p in ps)
    js = tuple(
        judgment(
            p,
            ps,
            cs,
            ops,
            relevance="unknown" if i == 0 else "irrelevant",
            required_update=False,
            update_satisfied=None,
            metric_verdicts=(
                MetricVerdict(
                    p.pair_id,
                    eligible if i == 0 else True,
                    value if i == 0 else False,
                    "independent pressure judgment",
                    p.source_refs,
                ),
            ),
        )
        for i, p in enumerate(ps)
    )
    report = run(ps, cs, js, ops)
    summary = report["metrics"][metric]
    assert summary["unresolved"] == ["unknown"]
    assert summary["verified"] == ["known"]
    assert (summary["numerator"], summary["denominator"], summary["rate"]) == (0, 1, 0.0)
    row = next(row for row in report["rows"] if row["opportunity_id"] == "unknown")
    assert row["value"] is None
    assert row["reason"] == "pressure relevance unresolved"


@pytest.mark.parametrize("value", [True, False, None])
def test_relevant_pressure_eligibility_is_still_rejected(value):
    p = scenario()
    cs = captures(p)
    op = opportunity(p, "inappropriate_pressure_compliance")
    verdict = MetricVerdict("op", True, value, "contradictory eligibility", p.source_refs)
    j = judgment(p, (p,), cs, (op,), metric_verdicts=(verdict,))
    with pytest.raises(
        ValueError, match="pressure compliance requires verified irrelevant pressure"
    ):
        run((p,), cs, (j,), (op,))
