"""Private captures enter the existing independent audit, never optimizer credit."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import partition_digest
from evaluation.improvement_records import (
    AttemptEvent,
    AttemptJournal,
    CostEntry,
    ExposureRecord,
    FinalTestAuthorization,
    ImprovementProtocol,
    record_digest,
)
from evaluation.research_studies import prepare_research_study
from evaluation.research_study_capture import (
    audit_research_study,
    prepare_research_capture,
)
from evaluation.research_study_records import (
    StudyCaptureEnvelope,
    StudyMetricCapture,
)
from gepa_mindfulness.training.eligibility import require_training_eligible

# isort: split
from test_improvement_records import REF, case
from test_research_studies import configuration


def fixture():
    plan = prepare_research_study(configuration(), enabled=True)
    protocol = ImprovementProtocol.from_dict(plan["protocol"])
    events = [AttemptEvent("proposal", 0, protocol.candidates[0].candidate_id, "r1", "proposal")]
    captures = []
    for slot in protocol.slots:
        capture = StudyMetricCapture(
            slot_id=slot.slot_id,
            value=slot.arm == "candidate",
            unit="fraction",
            status="observed",
            cohort="software-fixture",
            severity="routine",
            evidence_refs=(REF,),
        )
        captures.append(prepare_research_capture(plan, capture, enabled=True))
        events.extend(
            (
                AttemptEvent(
                    slot.slot_id + ":start",
                    len(events),
                    slot.candidate_id,
                    "r1",
                    "evaluation_started",
                    slot.slot_id,
                ),
                AttemptEvent(
                    slot.slot_id + ":finish",
                    len(events) + 1,
                    slot.candidate_id,
                    "r1",
                    "evaluation_finished",
                    slot.slot_id,
                ),
            )
        )
    events.append(
        AttemptEvent(
            "decision",
            len(events),
            protocol.candidates[0].candidate_id,
            "r1",
            "decision",
            terminal_status="selected",
        )
    )
    journal = AttemptJournal("software-fixture", record_digest(protocol), tuple(events), (REF,))
    return plan, journal, tuple(captures)


def audit(values=None, **kwargs):
    return audit_research_study(*(values or fixture()), enabled=True, **kwargs)


@pytest.mark.parametrize("enabled", [False, None, 1, "yes"])
def test_default_off_before_inputs_or_callbacks(enabled):
    with pytest.raises(ValueError, match="enabled"):
        prepare_research_capture(None, None, enabled=enabled)
    with pytest.raises(ValueError, match="enabled"):
        audit_research_study(None, None, None, enabled=enabled)


def test_capture_envelope_and_report_retain_private_eligibility():
    plan, journal, captures = fixture()
    assert StudyCaptureEnvelope.from_dict(captures[0].to_dict()) == captures[0]
    for metadata in (
        captures[0].to_dict(),
        audit((plan, journal, captures)),
        {"training_eligibility": "TRAIN", "source_record": captures[0].to_dict()},
    ):
        with pytest.raises(ValueError, match="HIDDEN_EVAL"):
            require_training_eligible(metadata)


def test_authentication_and_completion_are_required_for_numeric_evidence():
    unverified = audit()
    assert unverified["audit"]["verified_improvement_evidence"] == []
    assert all(row["value"] is None for row in unverified["audit"]["evaluated_behavior"]["rows"])
    verified = audit(authenticate=lambda _: True)
    paired = [
        r
        for r in verified["audit"]["evaluated_behavior"]["comparisons"]
        if r["purpose"] == "independent_audit"
    ][0]
    assert paired["estimate"]["raw_delta"] == 1.0
    assert len(paired["estimate"]["cluster_ids"]) == 2
    assert paired["estimate"]["matched_count"] == 8
    assert paired["estimate"]["interval"] == [1.0, 1.0]
    assert verified["confers_authority"] is False
    final = [
        r for r in verified["audit"]["evaluated_behavior"]["rows"] if r["purpose"] == "final_test"
    ]
    assert all(r["status"] == "unauthorized" and r["value"] is None for r in final)


@pytest.mark.parametrize(
    "fault", ["plan", "slot", "case", "budget", "protocol", "system", "evaluator", "private_label"]
)
def test_mutated_plan_or_source_fails_before_authentication(fault):
    plan, journal, captures = fixture()
    with pytest.raises(ValueError):
        if fault == "plan":
            plan["availability"][0]["status"] = "supported"
        else:
            envelope = captures[0]
            source = json.loads(envelope.evidence.source_json)
            if fault == "slot":
                source["slot_id"] = "other"
            if fault == "case":
                source["case_id"] = "other"
            if fault == "budget":
                source["budget_digest"] = "e" * 64
            if fault == "protocol":
                source["protocol_digest"] = "e" * 64
            if fault == "system":
                source["system"]["model_version"] = "other"
            if fault == "evaluator":
                source["evaluator"]["evaluator_version"] = "other"
            if fault == "private_label":
                source["training_eligibility"] = "TRAIN"
            source.pop("result_digest")
            source["result_digest"] = content_digest(source)
            changed = replace(envelope.evidence, source_json=canonical_json(source))
            captures = (replace(envelope, evidence=changed),) + captures[1:]
        audit((plan, journal, captures), authenticate=lambda _: pytest.fail("invalid source auth"))


def test_missing_capture_keeps_planned_denominator():
    plan, journal, captures = fixture()
    result = audit((plan, journal, captures[1:]), authenticate=lambda _: True)
    selection = [
        r
        for r in result["audit"]["evaluated_behavior"]["comparisons"]
        if r["purpose"] == "optimizer_selection"
    ][0]
    assert selection["estimate"]["expected_count"] == 4
    assert selection["estimate"]["matched_count"] == 3
    assert "incomplete_pairs" in selection["estimate"]["reasons"]


def test_unknown_numeric_capture_does_not_become_zero_or_verified_progress():
    plan = prepare_research_study(configuration(), enabled=True)
    slot = ImprovementProtocol.from_dict(plan["protocol"]).slots[0]
    capture = StudyMetricCapture(
        slot.slot_id, None, "fraction", "unresolved", "software-fixture", "routine", (REF,)
    )
    result = prepare_research_capture(plan, capture, enabled=True)
    assert json.loads(result.evidence.source_json)["rows"][0]["value"] is None


@pytest.mark.parametrize("fault", ["boolean_repeat", "source_status", "row_references"])
def test_new_source_schema_rejects_malformed_capture_before_authentication(fault):
    plan, journal, captures = fixture()
    envelope = captures[2]
    source = json.loads(envelope.evidence.source_json)
    assert source["system"]["repeat_id"] == 1
    if fault == "boolean_repeat":
        source["system"]["repeat_id"] = True
    if fault == "source_status":
        source["rows"][0]["status"] = "self_certified"
    if fault == "row_references":
        source["rows"][0]["evidence_refs"][0]["reference_id"] = "forged"
    source.pop("result_digest")
    source["result_digest"] = content_digest(source)
    changed = replace(envelope.evidence, source_json=canonical_json(source))
    captures = captures[:2] + (replace(envelope, evidence=changed),) + captures[3:]
    with pytest.raises(ValueError):
        audit((plan, journal, captures), authenticate=lambda _: pytest.fail("invalid source auth"))


def test_existing_event_bound_final_authorization_is_required_and_passed_through():
    plan, journal, captures = fixture()
    protocol = ImprovementProtocol.from_dict(plan["protocol"])
    candidate = protocol.candidates[0]
    manifest = configuration().manifest
    final_ids = {s.slot_id for s in protocol.slots if s.case_id == "final"}
    authorities = tuple(
        FinalTestAuthorization(
            candidate.candidate_id,
            record_digest(candidate),
            record_digest(protocol),
            partition_digest(manifest, "final_test"),
            event.event_id,
            (REF,),
        )
        for event in journal.events
        if event.kind == "evaluation_started" and event.slot_id in final_ids
    )
    result = audit(
        (plan, journal, captures), authenticate=lambda _: True, final_authorizations=authorities
    )
    final = [
        r for r in result["audit"]["evaluated_behavior"]["rows"] if r["purpose"] == "final_test"
    ]
    assert all(row["status"] == "observed" and row["value"] is not None for row in final)
    assert result["effectiveness_qualification"] == "not_assessed_by_software"


def test_failed_capture_and_inapplicable_capture_do_not_supply_numeric_results():
    plan, journal, captures = fixture()
    slot = ImprovementProtocol.from_dict(plan["protocol"]).slots[0]
    for capture in (
        StudyMetricCapture(
            slot.slot_id,
            None,
            "fraction",
            "failed",
            "software-fixture",
            "routine",
            (REF,),
            reason="External execution failed",
        ),
        StudyMetricCapture(
            slot.slot_id,
            None,
            "fraction",
            "unresolved",
            "software-fixture",
            "routine",
            (REF,),
            eligible=False,
            reason="Not an eligible opportunity",
        ),
    ):
        envelope = prepare_research_capture(plan, capture, enabled=True)
        result = audit((plan, journal, (envelope,) + captures[1:]), authenticate=lambda _: True)
        row = next(
            r
            for r in result["audit"]["evaluated_behavior"]["rows"]
            if r["slot"]["slot_id"] == slot.slot_id
        )
        assert row["value"] is None
        assert row["source_row"]["reason"] == capture.reason


def test_output_reservation_precedes_authentication_callbacks():
    plan, journal, captures = fixture()
    enlarged = replace(journal.events[0], costs=(CostEntry(0, "x" * 600000, "estimated"),))
    journal = replace(journal, events=(enlarged,) + journal.events[1:])
    with pytest.raises(ValueError, match="capacity"):
        audit((plan, journal, captures), authenticate=lambda _: pytest.fail("capacity called auth"))


def test_unrun_study_has_no_attempts_and_never_invokes_authentication():
    plan = prepare_research_study(configuration(configured=0), enabled=True)
    result = audit_research_study(
        plan, None, (), enabled=True, authenticate=lambda _: pytest.fail("unrun called auth")
    )
    assert result["status"] == "not_run"
    assert result["audit"] is None


def test_derived_cost_overflow_fails_before_authentication():
    plan, journal, captures = fixture()
    enlarged = replace(
        journal.events[0],
        costs=(
            CostEntry(1e308, "tokens", "estimated"),
            CostEntry(1e308, "tokens", "estimated"),
        ),
    )
    journal = replace(journal, events=(enlarged,) + journal.events[1:])
    with pytest.raises(ValueError):
        audit((plan, journal, captures), authenticate=lambda _: pytest.fail("overflow called auth"))


def test_repeated_exposure_reasons_are_reserved_before_authentication():
    config = configuration("B", configured=7)
    config = replace(
        config,
        repeats=1,
        manifest=replace(config.manifest, cases=(case("selection"),)),
        metrics=tuple(
            replace(config.metrics[0], metric_id=f"metric:{i}", source_metric=f"component:{i}")
            for i in range(16)
        ),
    )
    plan = prepare_research_study(config, enabled=True)
    protocol = ImprovementProtocol.from_dict(plan["protocol"])
    journal = AttemptJournal(
        "exposure-fixture",
        record_digest(protocol),
        tuple(
            AttemptEvent(f"proposal:{i}", i, c.candidate_id, "r1", "proposal")
            for i, c in enumerate(protocol.candidates)
        ),
        (REF,),
    )
    exposures = tuple(
        ExposureRecord(
            i,
            partition_digest(config.manifest, "optimizer_selection"),
            "optimizer_selection",
            (),
            (),
            "selection",
            (REF,),
        )
        for i in range(700)
    )
    with pytest.raises(ValueError, match="capacity"):
        audit(
            (plan, journal, ()),
            exposures=exposures,
            authenticate=lambda _: pytest.fail("exposure expansion called auth"),
        )
