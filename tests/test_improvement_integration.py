"""Real diagnostic reports remain restricted under improvement accounting."""

# Standard library
import json
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest
from test_improvement_records import DIGEST, EVALUATOR, REF, case, protocol

# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import audit_improvement, partition_digest
from evaluation.improvement_records import (
    AttemptEvent,
    AttemptJournal,
    DiagnosticEvidence,
    EvaluationSlot,
    ExposureRecord,
    FinalTestAuthorization,
    MetricSpec,
    SystemConfig,
    record_digest,
)
from evaluation.ladder import Metric, Observation, Probe, Severity, evaluate_ladder
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible


def fixture():
    cases = (case("selection"), case("audit", "independent_audit"), case("final", "final_test"))
    slots, evidence, events = [], [], [AttemptEvent("proposal", 0, "candidate", "r1", "proposal")]
    rubric = None
    for c in cases:
        for arm, model, value in (
            ("baseline", "baseline", False),
            ("candidate", "candidate", True),
        ):
            sid = f"{c.case_id}-{arm}"
            slots.append(
                EvaluationSlot(
                    sid, "candidate", c.case_id, "accuracy", arm, 0, DIGEST, "base", 7, "shared"
                )
            )
            source = evaluate_ladder(
                (
                    Probe(
                        "p",
                        Metric.REPRESENTATION_ACCURACY,
                        "all",
                        Severity.CONSEQUENTIAL,
                        "fraction",
                    ),
                ),
                (Observation("p", value, (REF,)),),
                protocol_id="source",
                system=SystemIdentity(0, 7, model, "harness"),
                evaluator=EVALUATOR,
                training_eligibility=TrainingEligibility.HIDDEN_EVAL,
                enabled=True,
            )
            rubric = content_digest(source["evaluator"])
            evidence.append(
                DiagnosticEvidence(sid, canonical_json(source), "rows", "probe_id", "p", (REF,))
            )
            events.extend(
                (
                    AttemptEvent(
                        sid + "-start", len(events), "candidate", "r1", "evaluation_started", sid
                    ),
                    AttemptEvent(
                        sid + "-finish",
                        len(events) + 1,
                        "candidate",
                        "r1",
                        "evaluation_finished",
                        sid,
                    ),
                )
            )
    spec, manifest = protocol(cases, slots=tuple(slots))
    manifest = replace(manifest, dependencies_known=True)
    spec = replace(
        spec,
        manifest_digest=record_digest(manifest),
        metrics=(
            MetricSpec(
                "accuracy", "evaluation-ladder-v1", "representation_accuracy", rubric, "fraction"
            ),
        ),
    )
    journal = AttemptJournal("journal", record_digest(spec), tuple(events), (REF,))
    return spec, manifest, journal, tuple(evidence)


def run(inputs=None, **kwargs):
    return audit_improvement(
        *(inputs or fixture()),
        enabled=kwargs.pop("enabled", True),
        authenticate=kwargs.pop("authenticate", lambda request: True),
        **kwargs,
    )


def test_real_ladder_reports_are_paired_but_do_not_grant_authority():
    result = run()
    assert result["training_eligibility"] == "DEVELOPMENT"
    assert result["confers_authority"] is False
    assert result["deployment_eligibility"] == "not_assessed"
    assert len(result["failures"]) == 8
    rows = result["evaluated_behavior"]["comparisons"]
    selection = next(r for r in rows if r["purpose"] == "optimizer_selection")
    assert selection["estimate"]["raw_delta"] == 1
    assert selection["estimate"]["interval"] is None
    assert result["verified_improvement_evidence"][0]["purpose"] == "independent_audit"
    assert any(r["status"] == "unauthorized" for r in result["evaluated_behavior"]["rows"])
    assert result["severe_events"]
    with pytest.raises(ValueError):
        require_training_eligible(json.loads(canonical_json(result)))


@pytest.mark.parametrize("auth", [None, lambda request: False, lambda request: 1])
def test_unverified_or_non_bool_authentication_cannot_create_an_effect(auth):
    result = run(authenticate=auth)
    assert not result["verified_improvement_evidence"]
    assert all(
        r["estimate"]["raw_delta"] is None for r in result["evaluated_behavior"]["comparisons"]
    )


def test_callback_exception_and_mutation_fail_closed_without_exception_text():
    def error(request):
        raise RuntimeError("secret exception detail")

    assert "secret exception detail" not in canonical_json(run(authenticate=error))

    def mutate(request):
        object.__setattr__(request, "subject_digest", "b" * 64)
        return True

    assert not run(authenticate=mutate)["verified_improvement_evidence"]


def test_final_test_authorization_binds_candidate_protocol_partition_and_event():
    spec, manifest, journal, evidence = fixture()
    authorizations = tuple(
        FinalTestAuthorization(
            "candidate",
            record_digest(spec.candidates[0]),
            record_digest(spec),
            partition_digest(manifest, "final_test"),
            event.event_id,
            (REF,),
        )
        for event in journal.events
        if event.kind == "evaluation_started" and event.slot_id.startswith("final-")
    )
    result = run((spec, manifest, journal, evidence), final_authorizations=authorizations)
    assert any(r["purpose"] == "final_test" for r in result["verified_improvement_evidence"])
    wrong = tuple(replace(a, candidate_digest="b" * 64) for a in authorizations)
    result = run((spec, manifest, journal, evidence), final_authorizations=wrong)
    assert not any(r["purpose"] == "final_test" for r in result["verified_improvement_evidence"])


def test_reused_audit_is_descriptive_only():
    spec, manifest, journal, evidence = fixture()
    exposure = ExposureRecord(
        100,
        partition_digest(manifest, "independent_audit"),
        "independent_audit",
        ("candidate",),
        (),
        "selection",
        (REF,),
    )
    result = run((spec, manifest, journal, evidence), exposures=(exposure,))
    assert not result["verified_improvement_evidence"]
    assert result["overstatement"][0]["audit"]["overstatement"] is None


@pytest.mark.parametrize(
    "mutation", ["disabled", "duplicate", "source", "row", "model", "rubric", "training"]
)
def test_ambiguous_or_incompatible_input_is_rejected(mutation):
    spec, manifest, journal, evidence = fixture()
    if mutation == "disabled":
        with pytest.raises(ValueError):
            run(enabled=False)
        return
    if mutation == "duplicate":
        evidence += (evidence[0],)
    elif mutation == "source":
        source = json.loads(evidence[0].source_json)
        source["rows"][0]["value"] = True
        evidence = (replace(evidence[0], source_json=canonical_json(source)),) + evidence[1:]
    elif mutation == "row":
        evidence = (replace(evidence[0], row_id="absent"),) + evidence[1:]
    elif mutation == "model":
        spec = replace(
            spec, candidates=(replace(spec.candidates[0], baseline=spec.candidates[0].candidate),)
        )
    elif mutation == "rubric":
        spec = replace(spec, metrics=(replace(spec.metrics[0], rubric_digest=DIGEST),))
    else:
        manifest = replace(
            manifest,
            cases=(replace(manifest.cases[0], purpose="synthetic_training"),) + manifest.cases[1:],
        )
        spec = replace(spec, manifest_digest=record_digest(manifest))
    journal = replace(journal, protocol_digest=record_digest(spec))
    with pytest.raises(ValueError):
        run((spec, manifest, journal, evidence))


def wrap_source(source, metric, evaluator, model, harness, row_id, row_list="rows"):
    slot = EvaluationSlot(
        "one", "candidate", "case", metric.metric_id, "candidate", 0, DIGEST, "base"
    )
    spec, manifest = protocol(slots=(slot,))
    spec = replace(
        spec,
        evaluator=evaluator,
        metrics=(metric,),
        candidates=(replace(spec.candidates[0], candidate=SystemConfig(model, harness, DIGEST)),),
    )
    events = (
        AttemptEvent("proposal", 0, "candidate", "r", "proposal"),
        AttemptEvent("start", 1, "candidate", "r", "evaluation_started", "one"),
        AttemptEvent("finish", 2, "candidate", "r", "evaluation_finished", "one"),
    )
    journal = AttemptJournal("j", record_digest(spec), events, (REF,))
    key = "probe_id" if source["schema_version"] == "evaluation-ladder-v1" else "opportunity_id"
    evidence = (DiagnosticEvidence("one", canonical_json(source), row_list, key, row_id, (REF,)),)
    return spec, manifest, journal, evidence


@pytest.mark.parametrize(
    "metric_name,family,success",
    [
        ("spurious_decision_flip_rate", "causal_invariance", False),
        ("required_update_success_rate", "required_update", True),
        ("semantic_laundering_susceptibility", "laundering", False),
        ("unjustified_abstention_stability", "unjustified_abstention", False),
        ("clarification_resumption_correctness", "clarification_resumption", True),
        ("severe_event_frequency", "severe_safety_authorization", False),
    ],
)
def test_real_causal_failure_semantics_and_denominators(metric_name, family, success):
    from dataclasses import asdict

    from test_causal_diagnostics import captures, judgment, opportunity
    from test_causal_diagnostics import run as causal_run
    from test_causal_records import pair

    from evaluation.causal_records import MetricVerdict

    p = pair()
    records = captures(p)
    op = opportunity(p, metric_name, severity=Severity.CATASTROPHIC)
    facts = dict(after_correct=False, update_satisfied=False)
    if family == "causal_invariance":
        facts.update(relevance="irrelevant", required_update=False, change_justified=False)
    elif family != "required_update":
        facts["metric_verdicts"] = (MetricVerdict("op", True, not success, "oracle", (REF,)),)
    j = judgment(p, (p,), records, (op,), **facts)
    source = causal_run((p,), records, (j,), (op,))
    metric = MetricSpec(
        family,
        source["schema_version"],
        metric_name,
        content_digest(asdict(j.evaluator)),
        direction="lower",
        transform="one_minus" if success else "identity",
    )
    result = run(wrap_source(source, metric, j.evaluator, "model", "harness", "op"))
    failure = result["failures"][family]
    assert failure["rows"][0]["value"] == 1
    assert failure["groups"][0]["denominator"] == 1
    assert failure["groups"][0]["rate"] == 1
    assert result["severe_events"]


def test_real_debate_fault_localization_failure_is_not_success_rate():
    from dataclasses import asdict

    from test_debate_analysis import assessment
    from test_debate_analysis import report as debate_report
    from test_debate_records import protocol as debate_protocol
    from test_sensitive_debate import run_fixture

    from evaluation.causal_records import MetricVerdict
    from evaluation.debate_records import DebateOpportunity

    p = debate_protocol(1)
    p = replace(
        p,
        opportunities=(
            DebateOpportunity(
                "fault", 0, "premise_fault_localization", Severity.CATASTROPHIC, "all"
            ),
        ),
    )
    session = run_fixture(p)
    adjudication = assessment(p, session, (MetricVerdict("fault", True, False, "oracle", (REF,)),))
    source = debate_report(
        p, session, assessment=adjudication, authenticate_assessment=lambda a: True
    )
    metric = MetricSpec(
        "fault",
        source["schema_version"],
        "premise_fault_localization",
        content_digest(asdict(p.evaluator)),
        direction="lower",
        transform="one_minus",
    )
    result = run(wrap_source(source, metric, p.evaluator, "fixture", "v1", "fault", "metric_rows"))
    assert result["failures"]["debate_fault_localization"]["groups"][0]["rate"] == 1


def test_real_probability_capture_preserves_brier_and_ece():
    from dataclasses import asdict

    source = evaluate_ladder(
        (Probe("p", Metric.PREDICTION_CALIBRATION, "all", Severity.ROUTINE, "fraction"),),
        (Observation("p", 0.9, (REF,), False),),
        protocol_id="calibration",
        system=SystemIdentity(0, 7, "model", "harness"),
        evaluator=EVALUATOR,
        training_eligibility=TrainingEligibility.HIDDEN_EVAL,
        enabled=True,
    )
    metric = MetricSpec(
        "brier",
        source["schema_version"],
        "prediction_calibration",
        content_digest(asdict(EVALUATOR)),
        "fraction",
        "lower",
        calibration_digest=DIGEST,
    )
    result = run(wrap_source(source, metric, EVALUATOR, "model", "harness", "p"))
    assert result["failures"]["calibration"]["rows"][0]["value"] == pytest.approx(0.81)
    assert result["calibration_aggregates"][0]["groups"][0][
        "expected_calibration_error"
    ] == pytest.approx(0.9)


def test_callback_cannot_mutate_protocol_via_nested_identity():
    def mutate(request):
        object.__setattr__(request.evaluator, "contract_id", "tampered")
        return True

    assert not run(authenticate=mutate)["verified_improvement_evidence"]


def test_absent_final_result_is_explicitly_not_run():
    spec, manifest, journal, evidence = fixture()
    result = run((spec, manifest, journal, evidence[:-2]))
    assert all(
        r["status"] == "not_run"
        for r in result["evaluated_behavior"]["rows"]
        if r["purpose"] == "final_test"
    )


def test_documented_offline_example_exposes_overstatement_without_granting_authority():
    guide = Path(__file__).parents[1] / "docs" / "reliable_improvement_evaluation.md"
    program = guide.read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(program, str(guide), "exec"), namespace)
    result = namespace["report"]
    assert result["overstatement"][0]["audit"]["overstatement"] == 1
    assert result["severe_events"]
    assert result["deployment_eligibility"] == "not_assessed"
    assert namespace["contamination_rejected"] is True


def test_unauthorized_final_preserves_public_severe_metadata_without_numeric_values():
    spec, manifest, journal, evidence = fixture()
    result = run((spec, manifest, journal, evidence[-2:]))
    assert len(result["severe_events"]) == 2
    for row in result["evaluated_behavior"]["rows"]:
        if row["status"] == "unauthorized":
            assert "source_digest" not in row
    for event in result["severe_events"]:
        assert "source_digest" not in event
        assert event["status"] == "unauthorized"
        assert event["event"]["evidence_refs"]
        assert "value" not in event["event"]
        assert "outcome" not in event["event"]
        assert event["authentication"] is False
    assert result["evaluated_behavior"]["sources"] == []


@pytest.mark.parametrize("status", ["verified", "observed", None])
def test_causal_source_without_adjudication_cannot_supply_numeric_claims(status):
    """Host authentication cannot substitute for a missing source evaluator contract."""
    from dataclasses import asdict

    from test_causal_diagnostics import captures, opportunity
    from test_causal_diagnostics import run as causal_run
    from test_causal_records import pair

    p = pair()
    op = opportunity(p, "spurious_decision_flip_rate")
    source = causal_run((p,), captures(p), (), (op,))
    assert source["pairs"][0]["adjudication"] is None
    if status is not None:
        source["rows"][0].update(status=status, value=True)
        source["result_digest"] = content_digest(
            {k: v for k, v in source.items() if k != "result_digest"}
        )
    metric = MetricSpec(
        "causal",
        source["schema_version"],
        "spurious_decision_flip_rate",
        content_digest(asdict(EVALUATOR)),
        direction="lower",
    )
    inputs = wrap_source(source, metric, EVALUATOR, "model", "harness", "op")
    if status is None:
        result = run(inputs)
        assert result["evaluated_behavior"]["rows"][0]["value"] is None
        assert not result["verified_improvement_evidence"]
    else:
        with pytest.raises(ValueError, match="requires evaluator/rubric contract"):
            run(inputs)


def test_partial_audit_keeps_descriptive_overstatement_without_verified_claim():
    spec, manifest, journal, evidence = fixture()
    added = tuple(
        replace(s, slot_id=s.slot_id + "-again", repeat_id=1)
        for s in spec.slots
        if s.case_id == "audit"
    )
    spec = replace(spec, slots=spec.slots + added)
    journal = replace(journal, protocol_digest=record_digest(spec))
    result = run((spec, manifest, journal, evidence))
    delta = result["overstatement"][0]["audit"]
    assert delta["overstatement"] == 0
    assert delta["interval"] is None
    assert "incomplete_pairs" in delta["independent_reasons"]
    assert not result["verified_improvement_evidence"]


def test_unresolved_eligible_debate_outcome_prevents_complete_failure_rate():
    from dataclasses import asdict

    from test_debate_analysis import assessment
    from test_debate_analysis import report as debate_report
    from test_debate_records import protocol as debate_protocol
    from test_sensitive_debate import run_fixture

    from evaluation.causal_records import MetricVerdict
    from evaluation.debate_records import DebateOpportunity

    p = debate_protocol(1)
    p = replace(
        p,
        opportunities=(
            DebateOpportunity("fault", 0, "premise_fault_localization", Severity.ROUTINE, "all"),
        ),
    )
    session = run_fixture(p)
    reports = []
    for value in (True, None):
        a = assessment(p, session, (MetricVerdict("fault", True, value, "oracle", (REF,)),))
        reports.append(
            debate_report(p, session, assessment=a, authenticate_assessment=lambda a: True)
        )
    metric = MetricSpec(
        "fault",
        reports[0]["schema_version"],
        "premise_fault_localization",
        content_digest(asdict(p.evaluator)),
        direction="lower",
        transform="one_minus",
    )
    spec, manifest, journal, evidence = wrap_source(
        reports[0], metric, p.evaluator, "fixture", "v1", "fault", "metric_rows"
    )
    manifest = replace(manifest, cases=manifest.cases + (case("second"),))
    spec = replace(
        spec,
        manifest_digest=record_digest(manifest),
        slots=spec.slots + (replace(spec.slots[0], slot_id="two", case_id="second"),),
    )
    events = journal.events + (
        AttemptEvent("start2", 3, "candidate", "r", "evaluation_started", "two"),
        AttemptEvent("finish2", 4, "candidate", "r", "evaluation_finished", "two"),
    )
    journal = replace(journal, protocol_digest=record_digest(spec), events=events)
    evidence += (replace(evidence[0], slot_id="two", source_json=canonical_json(reports[1])),)
    result = run((spec, manifest, journal, evidence))
    group = result["failures"]["debate_fault_localization"]["groups"][0]
    assert group["denominator"] == 2
    assert group["known_count"] == 1
    assert group["unresolved_eligible"] == 1
    assert group["rate"] is None
    assert group["observed_rate"] == 0
