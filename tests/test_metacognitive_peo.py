"""Causal-prefix monitoring never consumes hindsight or rewrites accepted evidence."""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_epistemic_reconciliation import reconciliation, sequence
from test_metacognitive_records import assessment
from test_metacognitive_records import sources as sources

from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.metacognitive_monitoring import MonitoringInput
from gepa_mindfulness.verification.metacognitive_peo import (
    OutcomeBinding,
    RecordedEpisode,
    RecordedEpisodeReport,
    bind_source_snapshot,
    monitor_recorded_episode,
)
from gepa_mindfulness.verification.metacognitive_records import FindingKind, SourceSnapshot
from gepa_mindfulness.verification.state import (
    EvidenceClaim,
    EvidenceState,
    parse_rfc3339_datetime,
)


def captured(
    events=None,
    *,
    posterior=True,
    bindings=(),
    evidence_state=None,
    continuity_requests=None,
    evidence_use=None,
    failure_graph=None,
):
    """Capture a source digest at a causal cutoff, before any later suffix exists."""
    chain = tuple(replace(e, trace_id="trace-1") for e in (events or sequence()))
    state = (
        reconciliation().update.posterior_state
        if posterior
        else reconciliation().update.prior_state
    )
    context = state.context
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["NONE"],
        repeats=1,
        model_version=context.system.model_version,
        harness_version=context.system.harness_version,
    )[0]
    sources = SourceSnapshot(
        context=context,
        cell=cell,
        evidence_state=evidence_state
        or EvidenceState(
            (EvidenceClaim("sensor", "Sensor reading", state.evidence_refs, "observed"),)
        ),
        hypotheses=None,
        epistemic=state,
        assumptions=(),
        source_event_ids=tuple(e.event_id for e in chain) + ("capture",),
        provenance=("host",),
    )
    timestamp = (
        parse_rfc3339_datetime(chain[-1].timestamp, "timestamp") + timedelta(seconds=1)
    ).isoformat()
    capture = bind_source_snapshot(
        sources,
        chain,
        event_id="capture",
        timestamp=timestamp,
        outcome_bindings=bindings,
        continuity_requests=continuity_requests,
        evidence_use=evidence_use,
        failure_graph=failure_graph,
    )
    return RecordedEpisode(
        MonitoringInput(sources, "assessment"),
        chain + (capture,),
        "capture",
        outcome_bindings=bindings,
        continuity_requests=continuity_requests,
        evidence_use=evidence_use,
        failure_graph=failure_graph,
        label_evaluator=TrustedEvaluatorContract(
            "verifier", "v1", "metacognitive-observed-labels-v1"
        ),
        label_verification_event_ids=tuple(
            e.event_id
            for e in chain
            if e.event_type == "verification_result"
            and e.payload.get("verifier_id") == "verifier"
            and e.payload.get("verifier_version") == "v1"
        ),
    )


def test_disabled_does_not_read_episode() -> None:
    """No causal adapter runs unless explicitly enabled."""
    assert monitor_recorded_episode(object()) is None
    with pytest.raises(ValueError):
        monitor_recorded_episode(object(), enabled=1)


def test_existing_reconciliation_and_evidence_remain_unchanged() -> None:
    """Prediction error retains model/environment ambiguity and original evidence."""
    episode = captured()
    before = episode.inputs.sources.to_dict()
    result = monitor_recorded_episode(episode, enabled=True)
    assert result.assessment.sources.to_dict() == before
    assert (
        result.assessment.uncertainty.model
        == reconciliation().update.posterior_state.model_uncertainty
    )
    assert FindingKind.CONTRADICTED_ASSUMPTION not in {f.kind for f in result.assessment.findings}
    assert result.to_dict()["diagnostic_status"] == "diagnostic"
    assert result.to_dict()["diagnostics"]["reconciliation_event_ids"] == ["r"]


def test_future_suffix_is_not_read_or_used() -> None:
    """Even an unreadable future object cannot affect an earlier source-bound assessment."""
    episode = captured(sequence()[:1], posterior=False)
    with_future = replace(episode, events=episode.events + (object(),))
    assert (
        monitor_recorded_episode(episode, enabled=True).to_dict()
        == monitor_recorded_episode(with_future, enabled=True).to_dict()
    )


def test_future_verification_binding_is_rejected() -> None:
    """A later verifier cannot certify a proposal made at an earlier cutoff."""
    binding = OutcomeBinding("p", "action", "o", "v")
    episode = captured(sequence()[:2], posterior=False, bindings=(binding,))
    episode = replace(episode, events=episode.events + tuple(sequence()[2:]))
    with pytest.raises(ValueError, match="outcome must belong"):
        monitor_recorded_episode(episode, enabled=True)


def test_source_snapshot_cannot_be_replaced_with_hindsight() -> None:
    """Changing retained evidence after capture breaks the protected snapshot digest."""
    episode = captured()
    state = episode.inputs.sources.evidence_state
    changed = replace(
        episode.inputs.sources,
        evidence_state=EvidenceState((replace(state.claims[0], status="supported"),)),
    )
    with pytest.raises(ValueError):
        replace(episode, inputs=replace(episode.inputs, sources=changed))


def test_captured_event_bytes_cannot_be_rewritten() -> None:
    """A valid event ID is insufficient when the recorded payload bytes have changed."""
    episode = captured()
    events = list(episode.events)
    payload = events[0].to_dict()["payload"]
    payload["confidence"] = 0.99
    events[0] = replace(events[0], payload=payload)
    with pytest.raises(ValueError):
        replace(episode, events=tuple(events))


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "another-run"),
        ("repeat_id", 1),
        ("case_id", 2),
        ("stripe_id", "PARAPHRASE"),
        ("model_version", "another-model"),
    ],
)
def test_cross_unit_event_metadata_is_rejected(field, value) -> None:
    """No caller can relabel an earlier event into a different evaluation cell."""
    episode = captured()
    events = list(episode.events)
    events[0] = replace(events[0], **{field: value})
    with pytest.raises(ValueError):
        replace(episode, events=tuple(events))


def test_calibration_requires_a_bound_verified_outcome() -> None:
    """Accuracy labels inside an observed, verified record can support diagnostics."""
    events = sequence()
    payload = events[3].to_dict()["payload"]
    payload["actual_outcome"].update(correct=False, progress=False)
    events[3] = replace(events[3], payload=payload)
    episode = captured(events, bindings=(OutcomeBinding("p", "action", "o", "v"),))
    result = monitor_recorded_episode(episode, enabled=True, confidence_threshold=0.6)
    assert FindingKind.CONFIDENCE_GAP in {f.kind for f in result.assessment.findings}
    calibration = result.to_dict()["diagnostics"]["calibration"]
    assert calibration["accuracy"] == 0.0
    assert calibration["brier_score"] == pytest.approx(0.49)
    assert "calibration" not in result.assessment.unassessed_domains


def test_partial_outcome_labels_do_not_claim_full_calibration_coverage() -> None:
    """Unscored pending actions remain in the calibration denominator."""
    events = sequence()[:5]
    payload = events[3].to_dict()["payload"]
    payload["actual_outcome"]["correct"] = True
    events[3] = replace(events[3], payload=payload)
    prediction = replace(
        events[0],
        event_id="pending-p",
        timestamp="2026-09-30T12:00:05Z",
        payload=events[0].to_dict()["payload"] | {"prediction_commit_id": "pending-prediction"},
    )
    proposed = replace(
        events[1],
        event_id="pending-a",
        action_id="pending-action",
        timestamp="2026-09-30T12:00:06Z",
        parent_event_ids=("pending-p",),
        payload=events[1].to_dict()["payload"]
        | {"action_id": "pending-action", "prediction_commit_id": "pending-prediction"},
    )
    episode = captured(
        events + [prediction, proposed],
        posterior=False,
        bindings=(
            OutcomeBinding("p", "action", "o", "v"),
            OutcomeBinding("pending-p", "pending-action", None, None),
        ),
    )
    report = monitor_recorded_episode(episode, enabled=True)
    assert "calibration" in report.assessment.unassessed_domains
    coverage = report.to_dict()["diagnostics"]["calibration_coverage"]
    assert coverage == {"bound_outcomes": 2, "verified_labels": 1, "scored_confidences": 1}


def test_pending_action_cannot_claim_strategy_from_unrelated_prediction() -> None:
    """Proposal identity matters before any execution has occurred."""
    events = sequence()[:2]
    payload = events[0].to_dict()["payload"]
    payload["prediction_commit_id"] = "unrelated-prediction"
    payload["predicted_outcome"]["strategy_id"] = "evidence_verification"
    unrelated = replace(
        events[0], event_id="unrelated", timestamp="2026-09-30T12:00:02Z", payload=payload
    )
    episode = captured(
        events + [unrelated],
        posterior=False,
        bindings=(OutcomeBinding("unrelated", "action", None, None, "evidence_verification"),),
    )
    with pytest.raises(ValueError, match="original prediction"):
        monitor_recorded_episode(episode, enabled=True)


def test_unverified_accuracy_is_unavailable_and_false_verification_is_flagged() -> None:
    """An actor's verification claim cannot replace a separate bound verifier record."""
    events = sequence()[:4]
    payload = events[3].to_dict()["payload"]
    payload["actual_outcome"].update(correct=True, claimed_verified=True)
    events[3] = replace(events[3], payload=payload)
    episode = captured(
        events, posterior=False, bindings=(OutcomeBinding("p", "action", "o", None),)
    )
    result = monitor_recorded_episode(episode, enabled=True, confidence_threshold=0.6)
    assert "calibration" in result.assessment.unassessed_domains
    assert result.to_dict()["diagnostics"]["calibration"] is None
    assert FindingKind.STRATEGY_OUTCOME_UNVERIFIED in {f.kind for f in result.assessment.findings}


def test_selected_executed_observed_verified_and_progress_are_separate() -> None:
    """A verified observation need not establish that a chosen strategy made progress."""
    events = sequence()
    pred = events[0].to_dict()["payload"]
    pred["predicted_outcome"]["strategy_id"] = "evidence_verification"
    events[0] = replace(events[0], payload=pred)
    observed = events[3].to_dict()["payload"]
    observed["actual_outcome"].update(progress=False)
    events[3] = replace(events[3], payload=observed)
    episode = captured(
        events,
        bindings=(OutcomeBinding("p", "action", "o", "v", strategy_id="evidence_verification"),),
    )
    result = monitor_recorded_episode(episode, enabled=True)
    row = result.to_dict()["diagnostics"]["strategy_observations"][0]
    assert row["selected"] is True and row["executed"] is True
    assert row["observed"] is True and row["verified"] is True
    assert row["verified_progress"] is False
    assert FindingKind.REPEATED_FAILURE not in {f.kind for f in result.assessment.findings}


def test_conflicting_verifiers_are_retained_and_not_cherry_picked() -> None:
    """One favorable verifier cannot hide a conflicting result about the same observation."""
    events = sequence()[:5]
    observed = events[3].to_dict()["payload"]
    observed["actual_outcome"]["correct"] = False
    events[3] = replace(events[3], payload=observed)
    negative = events[4].to_dict()["payload"] | {"verified": False, "verifier_id": "other-verifier"}
    events.append(
        replace(events[4], event_id="negative", timestamp="2026-09-30T12:00:06Z", payload=negative)
    )
    result = monitor_recorded_episode(
        captured(events, posterior=False, bindings=(OutcomeBinding("p", "action", "o", "v"),)),
        enabled=True,
    )
    assert result.to_dict()["diagnostics"]["calibration"] is None
    assert FindingKind.CONTRADICTION in {f.kind for f in result.assessment.findings}


def test_existing_continuity_audit_preserves_omission_without_motive_claims() -> None:
    """The existing PEO use audit determines continuity classification, not text heuristics."""
    from test_peo_continuity import request

    original = request(after={"action_reflected": False})
    events = tuple(replace(e, trace_id="trace-1") for e in original.events)
    original = replace(original, events=events)
    episode = captured(
        events,
        evidence_state=original.evidence_use.state,
        continuity_requests=(original,),
        evidence_use=(original.evidence_use,),
    )
    result = monitor_recorded_episode(episode, enabled=True)
    audit = result.to_dict()["diagnostics"]["continuity"][0]
    assert audit["classification"] == "influence_failure"
    assert audit["causal_or_motive_claim"] is False
    assert FindingKind.EVIDENCE_CONTINUITY in {f.kind for f in result.assessment.findings}
    assert result.assessment.sources.evidence_state == original.evidence_use.state


def test_repetition_reuses_existing_deloop_audit_without_auto_stop() -> None:
    """Same-state repeated actions are diagnostics and cannot directly stop a branch."""
    from test_deloop import fixture

    from gepa_mindfulness.verification.epistemic_state import EpistemicContext
    from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

    events, steps = fixture()
    events = tuple(replace(e, trace_id="trace-1") for e in events)
    context = EpistemicContext("run", 0, EvaluatedSystemVersion("student-v1", "host-v1"))
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["NONE"],
        repeats=1,
        model_version="student-v1",
        harness_version="host-v1",
    )[0]
    refs = tuple(dict.fromkeys(r for step in steps for r in step.request.observable_references))
    source = SourceSnapshot(
        context=context,
        cell=cell,
        evidence_state=EvidenceState(
            (EvidenceClaim("public-observations", "Recorded outcomes", refs, "observed"),)
        ),
        hypotheses=None,
        epistemic=None,
        assumptions=(),
        source_event_ids=tuple(e.event_id for e in events) + ("capture",),
        provenance=("host",),
    )
    stamp = (
        parse_rfc3339_datetime(events[-1].timestamp, "timestamp") + timedelta(seconds=1)
    ).isoformat()
    capture = bind_source_snapshot(
        source, events, event_id="capture", timestamp=stamp, loop_steps=steps
    )
    episode = RecordedEpisode(
        MonitoringInput(source, "loops"), events + (capture,), "capture", loop_steps=steps
    )
    result = monitor_recorded_episode(episode, enabled=True)
    entries = result.to_dict()["diagnostics"]["repetition"]["entries"]
    assert entries[1]["classification"] == "unresolved_repeat"
    assert FindingKind.REPEATED_FAILURE in {f.kind for f in result.assessment.findings}
    assert all(f.next_step != "stop_branch" for f in result.assessment.findings)


def test_report_round_trip_and_capture_immutability() -> None:
    """Diagnostic exports are detached, strict and cannot rewrite the captured evidence."""
    episode = captured()
    with pytest.raises(TypeError):
        episode.events[-1].payload["source_digest"] = "forged"
    report = monitor_recorded_episode(episode, enabled=True)
    assert RecordedEpisodeReport.from_dict(report.to_dict()).to_dict() == report.to_dict()
    wire = report.to_dict()
    wire["authority"] = True
    with pytest.raises(ValueError):
        RecordedEpisodeReport.from_dict(wire)


def test_observation_verification_does_not_automatically_verify_accuracy_labels() -> None:
    """A local/capture verifier cannot silently become an independent accuracy evaluator."""
    events = sequence()
    observed = events[3].to_dict()["payload"]
    observed["actual_outcome"]["correct"] = True
    events[3] = replace(events[3], payload=observed)
    episode = captured(events, bindings=(OutcomeBinding("p", "action", "o", "v"),))
    episode = replace(episode, label_evaluator=None)
    result = monitor_recorded_episode(episode, enabled=True)
    assert result.to_dict()["diagnostics"]["calibration"] is None
    assert "calibration" in result.assessment.unassessed_domains


def test_missing_binding_does_not_invent_verifier_conflict() -> None:
    """An omitted verifier binding is incomplete coverage, not contradictory evidence."""
    result = monitor_recorded_episode(
        captured(bindings=(OutcomeBinding("p", "action", "o", None),)), enabled=True
    )
    assert FindingKind.CONTRADICTION not in {f.kind for f in result.assessment.findings}


def test_unknown_observation_evidence_is_rejected_explicitly() -> None:
    """A missing retained reference must fail closed with a contract validation error."""
    events = sequence()[:4]
    payload = events[3].to_dict()["payload"] | {"evidence_refs": ["unknown-source"]}
    events[3] = replace(events[3], payload=payload, evidence_refs=("unknown-source",))
    episode = captured(
        events, posterior=False, bindings=(OutcomeBinding("p", "action", "o", None),)
    )
    with pytest.raises(ValueError):
        monitor_recorded_episode(episode, enabled=True)


def test_outcome_binding_requires_prediction_and_action_ids() -> None:
    """Required causal links cannot be represented by null or boolean IDs."""
    for prediction, action in ((None, "action"), ("p", None), (True, "action")):
        with pytest.raises(ValueError):
            OutcomeBinding(prediction, action, None, None)


def test_captured_diagnostic_inputs_cannot_be_dropped_after_outcome() -> None:
    """An outcome binding cannot be silently removed to hide an unfavorable diagnostic."""
    episode = captured(bindings=(OutcomeBinding("p", "action", "o", "v"),))
    with pytest.raises(ValueError):
        replace(episode, outcome_bindings=())


def test_future_evidence_quality_cannot_influence_an_earlier_cutoff() -> None:
    """A later evidence-use assessment is not prospective evidence at the recorded cutoff."""
    from test_peo_continuity import request

    original = request()
    future_use = replace(original.evidence_use, assessed_at="2027-01-01T00:00:00Z")
    with pytest.raises(ValueError):
        episode = captured(
            original.events, evidence_state=future_use.state, evidence_use=(future_use,)
        )
        monitor_recorded_episode(episode, enabled=True)


def test_archive_growth_after_cutoff_never_changes_a_replay() -> None:
    """Only the bounded inspected prefix affects eligibility, regardless of archive size."""
    episode = captured(sequence()[:1], posterior=False)
    grown = replace(episode, events=episode.events + (object(),) * 10_001)
    assert monitor_recorded_episode(grown, enabled=True) == monitor_recorded_episode(
        episode, enabled=True
    )


def test_evidence_freshness_is_reassessed_at_cutoff_without_rewriting_history() -> None:
    """A historically fresh measurement can become stale at the diagnostic decision."""
    from test_peo_continuity import request

    original = request()
    use = replace(
        original.evidence_use, policy=replace(original.evidence_use.policy, max_age_seconds=2)
    )
    assert use.limitations == ()
    episode = captured(original.events, evidence_state=use.state, evidence_use=(use,))
    report = monitor_recorded_episode(episode, enabled=True)
    assert FindingKind.INSUFFICIENT_EVIDENCE in {f.kind for f in report.assessment.findings}
    diagnostics = report.to_dict()["diagnostics"]
    assert diagnostics["evidence_use"][0]["assessed_at"] == use.assessed_at
    assert diagnostics["evidence_use_at_cutoff"][0]["STATUS"] == "stale"


def test_multiple_evidence_use_assessments_keep_distinct_findings_and_originals() -> None:
    """Different host quality declarations about one claim must not crash the report."""
    from test_peo_continuity import request

    original = request()
    first = replace(
        original.evidence_use,
        quality=replace(original.evidence_use.quality, source_reliability=None),
    )
    second = replace(first, quality=replace(first.quality, source_reliability=0.1))
    report = monitor_recorded_episode(
        captured(original.events, evidence_state=first.state, evidence_use=(first, second)),
        enabled=True,
    )
    assert len(report.to_dict()["diagnostics"]["evidence_use"]) == 2
    warnings = [
        f for f in report.assessment.findings if f.kind is FindingKind.INSUFFICIENT_EVIDENCE
    ]
    assert len(warnings) == 2 and len({f.finding_id for f in warnings}) == 2


def test_continuity_warnings_for_successive_decisions_have_distinct_identity() -> None:
    """Repeated audits of one commitment must retain the decision that each audited."""
    from test_peo_continuity import request

    first = request(after={"action_reflected": False})
    decision_prediction = first.events[-2]
    decision_proposal = first.events[-1]
    next_prediction = replace(
        decision_prediction,
        event_id="next-prediction",
        checkpoint_step=3,
        timestamp="2026-09-30T12:00:08Z",
        payload=decision_prediction.to_dict()["payload"] | {"prediction_commit_id": "next-p"},
    )
    next_proposal = replace(
        decision_proposal,
        event_id="next-proposal",
        checkpoint_step=3,
        timestamp="2026-09-30T12:00:08Z",
        action_id="next-action",
        parent_event_ids=("next-prediction",),
        payload=decision_proposal.to_dict()["payload"]
        | {"prediction_commit_id": "next-p", "action_id": "next-action"},
    )
    events = tuple(
        replace(e, trace_id="trace-1") for e in first.events + (next_prediction, next_proposal)
    )
    first = replace(first, events=events)
    second = replace(first, decision_event_id="next-proposal")
    report = monitor_recorded_episode(
        captured(
            events,
            evidence_state=first.evidence_use.state,
            continuity_requests=(first, second),
        ),
        enabled=True,
    )
    findings = [f for f in report.assessment.findings if f.kind is FindingKind.EVIDENCE_CONTINUITY]
    assert len(findings) == 2 and len({f.finding_id for f in findings}) == 2


def test_specialized_findings_share_the_bounded_partial_view(sources) -> None:
    """Existing specialized diagnostics can exceed 512 findings without crashing assembly."""
    from gepa_mindfulness.verification.metacognitive_monitoring import _assemble

    base = assessment(sources).findings[0]
    extra = tuple(
        replace(base, finding_id=f"specialized-{i}", kind=FindingKind.CONFIDENCE_GAP)
        for i in range(513)
    )
    result = _assemble(MonitoringInput(sources, "many"), extra, ("calibration",))
    assert len(result.findings) == 512
    assert result.omitted_finding_count == 6
    assert "calibration" in result.unassessed_domains


def test_failure_graph_keeps_existing_localization_and_causal_status() -> None:
    """Failure localization is retained with its existing status, not promoted to proven cause."""
    from test_failure_graph import _localization

    from gepa_mindfulness.verification.failure_graph import FailureGraph, FailureNode

    state = reconciliation().update.posterior_state
    node = FailureNode(
        "anomaly", "o", "Recorded anomaly", "2026-09-30T12:00:03Z", state.evidence_refs
    )
    graph = FailureGraph(
        (node,),
        (),
        _localization(root_cause=None, decisive_failure=None, symptoms=(), recoverable_until=None),
    )
    result = monitor_recorded_episode(captured(failure_graph=graph), enabled=True)
    raw = result.to_dict()["diagnostics"]["failure_graph"]
    assert raw["root_cause_status"] is None
    assert raw["localization"]["first_anomaly"] == "anomaly"
    bad = replace(graph, nodes=(replace(node, event_id="future-event"),))
    with pytest.raises(ValueError):
        captured(failure_graph=bad)


@pytest.mark.parametrize("contradiction", ["none", "unknown", "contradicted"])
def test_local_success_and_unsupported_relational_outcome_are_not_contradictory(contradiction):
    """Execution, support and unknown contradiction status are separate questions."""
    from gepa_mindfulness.verification.interfaces import (
        LocalVerificationResult,
        RelationalVerificationResult,
        VerificationEvidenceBinding,
        make_local_verification_event,
        make_relational_verification_event,
    )

    ref = reconciliation().update.prior_state.evidence_refs
    local_names = (
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
    )
    local = LocalVerificationResult(
        "action",
        True,
        True,
        True,
        True,
        True,
        None,
        ref,
        tuple(VerificationEvidenceBinding(n, ref) for n in local_names),
    )
    relation_names = ("provenance_intact", "authorization_scope_valid")
    if contradiction != "unknown":
        relation_names += ("contradiction_status",)
    relational = RelationalVerificationResult(
        "action",
        False,
        False,
        contradiction,
        True,
        True,
        False,
        False,
        ref,
        tuple(VerificationEvidenceBinding(n, ref) for n in relation_names),
    )
    metadata = dict(
        timestamp="2026-09-30T12:00:04Z",
        run_id="run",
        repeat_id=0,
        model_version="model",
        harness_version="harness",
        parent_event_ids=("o",),
    )
    events = sequence()[:4] + [
        make_local_verification_event(
            local, verifier_refs=("verifier-log",), event_id="local", **metadata
        ),
        make_relational_verification_event(
            relational, verifier_refs=("verifier-log",), event_id="relational", **metadata
        ),
    ]
    report = monitor_recorded_episode(
        captured(events, posterior=False, bindings=(OutcomeBinding("p", "action", "o", "local"),)),
        enabled=True,
    )
    kinds = {f.kind for f in report.assessment.findings}
    assert (FindingKind.CONTRADICTION in kinds) is (contradiction == "contradicted")
    assert FindingKind.STRATEGY_OUTCOME_UNVERIFIED in kinds


def test_evaluator_name_alone_does_not_enroll_a_result_for_accuracy_labels():
    """Only host-bound record IDs, together with host policy, admit independent labels."""
    events = sequence()
    payload = events[3].to_dict()["payload"]
    payload["actual_outcome"]["correct"] = True
    events[3] = replace(events[3], payload=payload)
    episode = captured(events, bindings=(OutcomeBinding("p", "action", "o", "v"),))
    assert (
        monitor_recorded_episode(episode, enabled=True).to_dict()["diagnostics"]["calibration"][
            "accuracy"
        ]
        == 1.0
    )
    report = monitor_recorded_episode(
        replace(episode, label_verification_event_ids=()), enabled=True
    )
    assert report.to_dict()["diagnostics"]["calibration"] is None
    assert "calibration" in report.assessment.unassessed_domains


def test_two_result_ids_can_bind_one_host_owned_label_evaluator():
    """Distinct verification results from one evaluator must both contribute to calibration."""
    events = sequence()[:5]
    payload = events[3].to_dict()["payload"]
    payload["actual_outcome"]["correct"] = True
    events[3] = replace(events[3], payload=payload)
    events[4] = replace(
        events[4], payload=events[4].to_dict()["payload"] | {"verifier_id": "result-one"}
    )
    second = []
    ids = {"p": "p2", "a": "a2", "x": "x2", "o": "o2", "v": "v2"}
    links = {
        "prediction_commit_id": "prediction-two",
        "action_id": "action-two",
        "observation_id": "observation-two",
        "verifier_id": "result-two",
    }
    for offset, event in enumerate(events):
        payload = event.to_dict()["payload"]
        for field, value in links.items():
            if field in payload:
                payload[field] = value
        second.append(
            replace(
                event,
                event_id=ids[event.event_id],
                action_id="action-two" if event.action_id else None,
                timestamp=f"2026-09-30T12:00:{offset + 5:02d}Z",
                payload=payload,
                parent_event_ids=tuple(ids[p] for p in event.parent_event_ids),
            )
        )
    episode = captured(
        events + second,
        posterior=False,
        bindings=(
            OutcomeBinding("p", "action", "o", "v"),
            OutcomeBinding("p2", "action-two", "o2", "v2"),
        ),
    )
    episode = replace(episode, label_verification_event_ids=("v", "v2"))
    report = monitor_recorded_episode(episode, enabled=True)
    assert report.to_dict()["diagnostics"]["calibration_coverage"] == {
        "bound_outcomes": 2,
        "verified_labels": 2,
        "scored_confidences": 2,
    }
    assert "calibration" not in report.assessment.unassessed_domains
    with pytest.raises(ValueError):
        replace(episode, label_verification_event_ids=("o",))


def test_json_equivalent_continuity_payloads_do_not_look_like_rewritten_events():
    """Canonical capture equality must accept JSON-equivalent tuple/list representations."""
    from test_peo_continuity import request

    original = request(after={"action_reflected": False})
    events = [replace(e, trace_id="trace-1") for e in original.events]
    from mindful_trace_gepa.logging_schema import EventEnvelope

    note = EventEnvelope(
        "1.0",
        "note",
        "prompt",
        events[0].timestamp,
        payload={"markers": ("a", "b")},
        run_id="run",
        repeat_id=0,
        model_version="model",
        harness_version="harness",
        trace_id="trace-1",
        conversation_id="c",
    )
    events.insert(0, note)
    original = replace(original, events=tuple(events))
    report = monitor_recorded_episode(
        captured(
            events, evidence_state=original.evidence_use.state, continuity_requests=(original,)
        ),
        enabled=True,
    )
    assert FindingKind.EVIDENCE_CONTINUITY in {f.kind for f in report.assessment.findings}


def test_generic_research_event_builder_requires_the_specialized_capture_helper():
    """A source snapshot cannot be constructed without its three capture digests."""
    from evaluation.research_audits import make_research_event
    from mindful_trace_gepa.logging_schema import StructuredEventType

    with pytest.raises(ValueError, match="bind_source_snapshot"):
        make_research_event(
            StructuredEventType.METACOGNITIVE_SOURCE_SNAPSHOT,
            "record",
            run_id="run",
            trace_id="trace",
            evaluation_ref="run",
            model_version="model",
            harness_version="harness",
            parent_event_ids=("p",),
            evidence_refs=("e",),
            provenance_refs=("host",),
        )
