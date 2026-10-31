"""Host-private study captures and a bounded handoff to the existing independent audit."""

# Standard library
from __future__ import annotations

import json
from dataclasses import asdict

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import _source_row, audit_improvement, summarize_journal
from evaluation.improvement_records import (
    AttemptJournal,
    DiagnosticEvidence,
    ExposureRecord,
    FinalTestAuthorization,
    ImprovementProtocol,
    record_digest,
)
from evaluation.research_studies import prepare_research_study
from evaluation.research_study_records import (
    STUDY_SOURCE_SCHEMA,
    StudyCaptureEnvelope,
    StudyConfiguration,
    StudyMetricCapture,
)
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope


def _plan(value: object) -> dict:
    validate_input_envelope(value)
    if type(value) is not dict or type(value.get("configuration")) is not dict:
        raise ValueError("exact prepared study plan required")
    config = StudyConfiguration.from_dict(value["configuration"])
    prepared = prepare_research_study(config, enabled=True)
    if canonical_json(value) != canonical_json(prepared):
        raise ValueError("prepared plan or protocol binding changed")
    return prepared


def prepare_research_capture(
    plan: dict,
    capture: StudyMetricCapture,
    *,
    enabled: bool = False,
) -> StudyCaptureEnvelope:
    """Emit a private diagnostic, not verified credit; host authentication is still required."""
    if enabled is not True:
        raise ValueError("capture preparation requires enabled=True")
    prepared = _plan(plan)
    if prepared["protocol"] is None or type(capture) is not StudyMetricCapture:
        raise ValueError("capture requires an existing planned slot and exact capture")
    capture = StudyMetricCapture.from_dict(StudyMetricCapture.to_dict(capture))
    protocol = ImprovementProtocol.from_dict(prepared["protocol"])
    slots = {s.slot_id: s for s in protocol.slots}
    if capture.slot_id not in slots:
        raise ValueError("capture does not name a planned slot")
    slot = slots[capture.slot_id]
    candidate = next(c for c in protocol.candidates if c.candidate_id == slot.candidate_id)
    system = getattr(candidate, slot.arm)
    metric = next(m for m in protocol.metrics if m.metric_id == slot.metric_id)
    if capture.unit != metric.unit:
        raise ValueError("capture unit does not match registered component metric")
    source = dict(
        schema_version=STUDY_SOURCE_SCHEMA,
        training_eligibility="HIDDEN_EVAL",
        confers_authority=False,
        study_plan_digest=prepared["result_digest"],
        protocol_digest=record_digest(protocol),
        slot_id=slot.slot_id,
        case_id=slot.case_id,
        budget_digest=slot.budget_digest,
        system=dict(
            model_version=system.model_version,
            harness_version=system.harness_version,
            repeat_id=slot.repeat_id,
            seed=slot.seed,
        ),
        evaluator=asdict(protocol.evaluator),
        rows=[
            dict(
                opportunity_id=slot.slot_id,
                metric=metric.source_metric,
                unit=capture.unit,
                value=capture.value,
                outcome=capture.outcome,
                status=capture.status,
                eligible=capture.eligible,
                cohort=capture.cohort,
                severity=capture.severity,
                reason=capture.reason,
                evidence_refs=[r.to_dict() for r in capture.evidence_refs],
            )
        ],
    )
    source["result_digest"] = content_digest(source)
    validate_input_envelope(source)
    evidence = DiagnosticEvidence(
        slot.slot_id,
        canonical_json(source),
        "rows",
        "opportunity_id",
        slot.slot_id,
        capture.evidence_refs,
    )
    return StudyCaptureEnvelope(prepared["result_digest"], evidence)


def audit_research_study(
    plan: dict,
    journal: AttemptJournal | None,
    captures: tuple[StudyCaptureEnvelope, ...],
    *,
    exposures: tuple[ExposureRecord, ...] = (),
    final_authorizations: tuple[FinalTestAuthorization, ...] = (),
    authenticate=None,
    enabled: bool = False,
) -> dict:
    """Audit an external experiment without running it or granting acceptance authority.

    The existing host contract authenticates source facts, complete attempts/exposure history,
    evaluator independence and event-specific final-test authority. No capture label proves those
    facts. Unconfigured studies return not_run without callbacks. All output remains host-private.
    """
    if enabled is not True:
        raise ValueError("study audit requires enabled=True")
    prepared = _plan(plan)
    if type(captures) is not tuple or any(type(c) is not StudyCaptureEnvelope for c in captures):
        raise ValueError("exact private capture tuple required")
    result = dict(
        schema_version="research-study-audit-v1",
        training_eligibility="HIDDEN_EVAL",
        confers_authority=False,
        optimizer_input=False,
        execute_runtime=False,
        study_plan_digest=prepared["result_digest"],
        availability=prepared["availability"],
        status="not_run" if prepared["protocol"] is None else "reported",
        effectiveness_qualification="not_assessed_by_software",
        audit=None,
    )
    if prepared["protocol"] is None:
        if captures or journal is not None:
            raise ValueError("unrun study must not contain fabricated attempt evidence")
        result["result_digest"] = content_digest(result)
        return result
    if type(journal) is not AttemptJournal:
        raise ValueError("existing complete attempt journal required")
    journal = AttemptJournal.from_dict(AttemptJournal.to_dict(journal))
    for values, cls in (
        (exposures, ExposureRecord),
        (final_authorizations, FinalTestAuthorization),
    ):
        if type(values) is not tuple or any(type(v) is not cls for v in values):
            raise ValueError("exact existing exposure/authorization tuples required")
    exposures = tuple(ExposureRecord.from_dict(ExposureRecord.to_dict(v)) for v in exposures)
    final_authorizations = tuple(
        FinalTestAuthorization.from_dict(FinalTestAuthorization.to_dict(v))
        for v in final_authorizations
    )
    protocol = ImprovementProtocol.from_dict(prepared["protocol"])
    manifest = StudyConfiguration.from_dict(prepared["configuration"]).manifest
    slots = {s.slot_id: s for s in protocol.slots}
    if len(captures) > len(slots):
        raise ValueError("capture count exceeds declared slot capacity")
    validate_input_envelope(summarize_journal(protocol, journal))
    metrics = {m.metric_id: m for m in protocol.metrics}
    candidates = {c.candidate_id: c for c in protocol.candidates}
    rows, seen = [], set()
    for capture in captures:
        copy = StudyCaptureEnvelope.from_dict(StudyCaptureEnvelope.to_dict(capture))
        ev = copy.evidence
        if copy.study_plan_digest != prepared["result_digest"] or ev.slot_id in seen:
            raise ValueError("capture plan mismatch or duplicate slot")
        if ev.slot_id not in slots:
            raise ValueError("capture names an unplanned slot")
        source = json.loads(ev.source_json)
        if source.get("study_plan_digest") != prepared["result_digest"]:
            raise ValueError("source plan binding mismatch")
        s = slots[ev.slot_id]
        _source_row(
            ev,
            s,
            metrics[s.metric_id],
            candidates[s.candidate_id],
            protocol.evaluator,
            protocol_digest=record_digest(protocol),
        )
        seen.add(ev.slot_id)
        rows.append(ev)
    # The existing audit retains sources/rows/estimates more than once. Reserve conservatively
    # for those copies and bounded metadata before invoking any authentication callback.
    raw = dict(
        plan=prepared,
        journal=AttemptJournal.to_dict(journal),
        captures=[c.to_dict() for c in captures],
        exposures=[v.to_dict() for v in exposures],
        final_authorizations=[v.to_dict() for v in final_authorizations],
    )
    validate_input_envelope(raw)
    exposure_bytes = len(canonical_json(raw["exposures"]).encode("utf-8"))
    # Each exposure reason may recur for every candidate/metric. Include complete serialized
    # exposure records as an upper bound, rather than omitting that multiplicative output.
    reservation = (
        8 * len(canonical_json(raw).encode("utf-8"))
        + len(protocol.candidates) * len(protocol.metrics) * exposure_bytes
        + 262144
    )
    if reservation > 4 * 1024 * 1024:
        raise ValueError("study audit output capacity exceeded before authentication")
    validate_input_envelope(dict(output_capacity="x" * reservation))
    result["audit"] = audit_improvement(
        protocol,
        manifest,
        journal,
        tuple(rows),
        exposures=exposures,
        final_authorizations=final_authorizations,
        authenticate=authenticate,
        enabled=True,
    )
    result["result_digest"] = content_digest(result)
    validate_input_envelope(result)
    return result
