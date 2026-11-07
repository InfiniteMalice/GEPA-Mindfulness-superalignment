"""Behavior-first experimental alignment research; no training or promotion authority."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

# Local
from evaluation.causal_diagnostics import evaluate_causal_suite
from evaluation.causal_records import (
    CausalCapture,
    CausalPair,
    FactorChange,
    MetricOpportunity,
    PairAdjudication,
    canonical_json,
    content_digest,
)
from evaluation.cva_benchmark import (
    BenchmarkCapture,
    CaptureVerificationInput,
    evaluate_cva_benchmark,
    plan_cva_slots,
)
from evaluation.experimental_records import (
    DiagnosticStatus,
    ExperimentalMaturity,
    MechanisticAuditReference,
)
from evaluation.mechanistic_study_records import (
    AlignmentStudyPlan,
    PublicBehavior,
    PublicStudyInput,
    StudyBehaviorBatch,
    StudyCVAObservation,
    StudyInterventionPacket,
    StudyItem,
    StudyMechanismRecord,
    StudyRoles,
    StudyTemplate,
)
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.interpret.causal_interventions import execute_causal_plan
from gepa_mindfulness.interpret.mechanistic_evidence import _report_snapshot
from gepa_mindfulness.interpret.native_causal_records import CausalCommit
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.cva_records import _record, array
from gepa_mindfulness.verification.runtime_governance import TrustedClock
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from synthetic_data.causal_interventions import variant_from_cell


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("alignment studies require enabled=True")


def _now(clock: TrustedClock | None) -> str:
    value = datetime.now(timezone.utc) if clock is None else clock.now()
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("study clock requires an aware datetime")
    return value.astimezone(timezone.utc).isoformat()


def _authenticated(record: Any, callback: Callable | None) -> bool:
    if callback is None:
        return False
    if not callable(callback):
        raise ValueError("host authentication requires a callable")
    snapshot = type(record).from_dict(record.to_dict())
    expected = snapshot.to_dict()
    try:
        accepted = callback(snapshot)
        return accepted is True and snapshot.to_dict() == expected
    except Exception:
        return False


def bind_alignment_studies(
    templates: tuple[StudyTemplate, ...],
    cells: tuple[tuple[V5EvaluationCell, V5EvaluationCell], ...],
    *,
    plan_id: str,
    roles: StudyRoles,
    condition: str = "baseline",
    physical_model_digest: str | None = None,
    candidate_ref: EvidenceReference | None = None,
    enabled: bool = False,
) -> AlignmentStudyPlan:
    """Retain host-supplied V5 coordinates; no semantic case classification is inferred."""
    _enabled(enabled)
    array(templates, 64, required=True)
    array(cells, 64, required=True)
    if len(templates) != len(cells):
        raise ValueError("one independently classified cell pair per template is required")
    items = []
    for template, coordinates in zip(templates, cells):
        t = _record(template, StudyTemplate)
        array(coordinates, 2, required=True)
        if len(coordinates) != 2:
            raise ValueError("two existing V5 cells required")
        pair_id = plan_id + ":" + t.template_id
        variants = tuple(
            variant_from_cell(
                pair_id + ":" + arm,
                cell,
                turns=getattr(t, arm + "_turns"),
                factors=getattr(t, arm + "_factors"),
                expected_actions=getattr(t, arm + "_expected"),
            )
            for arm, cell in zip(("before", "after"), coordinates)
        )
        before, after = (dict(v.factors) for v in variants)
        changes = tuple(
            FactorChange(k, before[k], after[k]) for k in before if before[k] != after[k]
        )
        pair = CausalPair(
            pair_id,
            t.family,
            variants[0],
            variants[1],
            "single_variable" if len(changes) == 1 else "compound",
            changes,
            t.claimed_equivalence,
            "shared" if variants[0].system.seed == variants[1].system.seed else "per_arm",
            t.source_refs,
            TrainingEligibility.DEVELOPMENT,
        )
        items.append(StudyItem(t.template_id, t.family, pair, t.cva_before, t.cva_after))
    return AlignmentStudyPlan(
        plan_id, tuple(items), roles, condition, physical_model_digest, candidate_ref
    )


def public_study_input(item: StudyItem, arm: str, *, enabled: bool = False) -> PublicStudyInput:
    _enabled(enabled)
    snapshot = _record(item, StudyItem)
    if type(arm) is not str or arm not in ("before", "after"):
        raise ValueError("unknown study arm")
    variant = getattr(snapshot.pair, arm)
    return PublicStudyInput(
        variant.variant_id,
        snapshot.pair.digest,
        variant.prompt_digest,
        variant.system,
        variant.turns,
    )


def capture_study_behavior(
    plan: AlignmentStudyPlan,
    actor: Callable[[PublicStudyInput], PublicBehavior],
    *,
    clock: TrustedClock | None = None,
    backend: FrozenNativeModel | None = None,
    authorize_candidate: Callable[[AlignmentStudyPlan], bool] | None = None,
    enabled: bool = False,
) -> StudyBehaviorBatch:
    """Invoke a host actor adapter on detached public inputs before inspecting internals."""
    _enabled(enabled)
    p = _record(plan, AlignmentStudyPlan)
    if not callable(actor):
        raise ValueError("actor must be a host-configured callable")
    if p.condition == "intervention_informed" and not _authenticated(p, authorize_candidate):
        raise PermissionError(
            "intervention-informed candidate requires separate host authorization"
        )
    if p.condition == "diagnostic_instrumented" and type(backend) is not FrozenNativeModel:
        raise ValueError("diagnostic condition requires its exact frozen backend")
    if backend is not None:
        if (
            type(backend) is not FrozenNativeModel
            or p.physical_model_digest != backend.identity.digest
        ):
            raise ValueError("behavior backend differs from the planned physical model")
        backend._assert_frozen()
    start = _now(clock)
    _cva_slots(p)  # Reject duplicate observation slots before any actor executes.
    captures, outputs = [], []
    for item in p.items:
        for arm in ("before", "after"):
            view = public_study_input(item, arm, enabled=True)
            prior = view.to_dict()
            output, failure = None, None
            try:
                output = actor(view)
            except Exception as error:
                failure = type(error).__name__
            finally:
                if backend is not None:
                    backend._assert_frozen()
            if view.to_dict() != prior:
                raise ValueError("actor mutated its public study input")
            if failure is None:
                output = _record(output, PublicBehavior)
                if output.native_input is not None and (
                    backend is None or output.physical_model_digest != backend.identity.digest
                ):
                    raise ValueError("native capture requires its exact frozen model provenance")
                refs, actions, status = output.evidence_refs, output.actions, "observed"
                reason = "Host adapter captured this original public action sequence."
            else:
                refs = (
                    EvidenceReference(
                        "censored:" + content_digest(prior), EvidenceSourceKind.EXTERNAL_RECORD
                    ),
                )
                actions, status, reason = (), "censored", "Actor capture failed: " + failure
            captures.append(
                CausalCapture(
                    item.pair.digest,
                    view.variant_id,
                    view.prompt_digest,
                    view.system,
                    status,
                    actions,
                    refs,
                    reason,
                )
            )
            outputs.append(output if failure is None else None)
    result = StudyBehaviorBatch(p.digest, tuple(captures), tuple(outputs), start, _now(clock))
    _validate_behavior(p, result)
    return result


def _validate_behavior(
    plan: AlignmentStudyPlan, behavior: StudyBehaviorBatch
) -> StudyBehaviorBatch:
    batch = _record(behavior, StudyBehaviorBatch)
    if batch.plan_digest != plan.digest:
        raise ValueError("behavior batch belongs to a different study plan")
    expected = [(i.pair.digest, v) for i in plan.items for v in (i.pair.before, i.pair.after)]
    if len(batch.captures) != len(expected):
        raise ValueError("behavior roster must close before instrumentation")
    refs: list[str] = []
    for capture, (pair_digest, variant) in zip(batch.captures, expected):
        if (
            capture.pair_digest != pair_digest
            or capture.variant_id != variant.variant_id
            or capture.prompt_digest != variant.prompt_digest
            or capture.system != variant.system
        ):
            raise ValueError("capture does not match its exact planned public input")
        refs.extend(r.reference_id for r in capture.evidence_refs)
    if len(set(refs)) != len(refs):
        raise ValueError("capture evidence cannot be reused across observations")
    _, _, joins = _cva_slots(plan)
    slots_by_variant = {variant: slot for _, variant, slot in joins}
    for capture, output in zip(batch.captures, batch.outputs):
        if output is not None and output.cva_capture is not None:
            if output.cva_capture.slot_digest != slots_by_variant.get(capture.variant_id):
                raise ValueError("original CVA capture does not match its exact study arm")
    return batch


def _validate_intervention(packet: StudyInterventionPacket) -> None:
    p, item, behavior, commit = packet.plan, packet.item, packet.behavior, packet.commit
    _validate_behavior(p, behavior)
    if p.condition != "diagnostic_instrumented":
        raise ValueError("only the separately declared diagnostic condition may instrument")
    if item not in p.items or p.physical_model_digest != commit.plan.model_digest:
        raise ValueError("intervention item/model is not bound to this study")
    selected = {c.variant_id: o for c, o in zip(behavior.captures, behavior.outputs)}
    before, after = (selected[v.variant_id] for v in (item.pair.before, item.pair.after))
    if before is None or after is None or before.native_input is None or after.native_input is None:
        raise ValueError("both original public captures require complete native provenance")
    if (
        before.native_input != commit.plan.donor
        or after.native_input != commit.plan.recipient
        or before.physical_model_digest != p.physical_model_digest
        or after.physical_model_digest != p.physical_model_digest
        or (before.tokenizer_id, before.tokenizer_version)
        != (after.tokenizer_id, after.tokenizer_version)
    ):
        raise ValueError("native inputs/model/tokenizer differ from captured behavioral contexts")
    if parse_rfc3339_datetime(commit.created_at, "prediction") < parse_rfc3339_datetime(
        behavior.completed_at, "behavior completion"
    ):
        raise ValueError("study prediction must follow completed behavioral observation")


def instrument_study_pair(
    plan: AlignmentStudyPlan,
    item: StudyItem,
    behavior: StudyBehaviorBatch,
    commit: CausalCommit,
    *,
    backend: FrozenNativeModel,
    clock: TrustedClock,
    authenticate: Callable[[StudyInterventionPacket], bool] | None = None,
    enabled: bool = False,
) -> StudyMechanismRecord:
    """Execute actual stage-6 interventions only after authenticated behavior/tokenization."""
    _enabled(enabled)
    packet = StudyInterventionPacket(
        _record(plan, AlignmentStudyPlan),
        _record(item, StudyItem),
        _record(behavior, StudyBehaviorBatch),
        _record(commit, CausalCommit),
    )
    _validate_intervention(packet)
    if (
        type(backend) is not FrozenNativeModel
        or backend.identity.digest != packet.plan.physical_model_digest
    ):
        raise ValueError("instrumentation requires the exact frozen physical model")
    if not _authenticated(packet, authenticate):
        raise PermissionError("host must authenticate capture custody and original tokenization")
    backend._assert_frozen()
    report = execute_causal_plan(backend, packet.commit, clock=clock, enabled=True)
    backend._assert_frozen()
    return StudyMechanismRecord(
        packet.plan.digest, packet.item.digest, packet.behavior.digest, canonical_json(report)
    )


def cva_study_slots(plan: AlignmentStudyPlan, *, enabled: bool = False) -> tuple:
    """Return the original benchmark fixtures/slots plus exact study-arm join identities."""
    _enabled(enabled)
    p = _record(plan, AlignmentStudyPlan)
    return _cva_slots(p)


def _cva_slots(p: AlignmentStudyPlan) -> tuple:
    fixtures, slots, joins = [], [], []
    for item in p.items:
        if item.cva_before is None:
            continue
        for arm in ("before", "after"):
            fixture, variant = getattr(item, "cva_" + arm), getattr(item.pair, arm)
            cell = V5EvaluationCell(
                variant.case.case_id,
                variant.case.case_version,
                variant.robustness.stripe_id,
                variant.robustness.subtype,
                variant.system.repeat_id,
                variant.system.seed,
                variant.system.model_version,
                variant.system.harness_version,
            )
            slot = plan_cva_slots((fixture,), (cell,), enabled=True)[0]
            if slot.variant.prompt_digest != variant.prompt_digest:
                raise ValueError("CVA study prompt differs from its original benchmark fixture")
            fixtures.append(fixture)
            slots.append(slot)
            joins.append((item.pair.digest, variant.variant_id, slot.digest))
    if len({s.digest for s in slots}) != len(slots):
        raise ValueError("duplicate CVA slots require distinct V5 repeats or separate study plans")
    return tuple(fixtures), tuple(slots), tuple(joins)


def public_dual_path(response: str, *, enabled: bool = False) -> dict[str, str]:
    """Reuse strict parsing while returning only public answers and the explicit final choice."""
    _enabled(enabled)
    from gepa_mindfulness.verification.cva_records import text
    from mindful_trace_gepa.prompts.dual_path import parse_dual_path_response

    text(response, "dual-path response", 65536)
    parsed = parse_dual_path_response(response, strict=True)
    return {k: parsed[k] for k in ("path_1", "path_2", "official_answer", "final_answer_value")}


def report_alignment_studies(
    plan: AlignmentStudyPlan,
    behavior: StudyBehaviorBatch,
    adjudications: tuple[PairAdjudication, ...],
    *,
    opportunities: tuple[MetricOpportunity, ...],
    mechanisms: tuple[StudyMechanismRecord, ...] = (),
    authenticate: Callable[[PairAdjudication], bool] | None = None,
    cva_captures: tuple[BenchmarkCapture, ...] = (),
    authenticate_cva: Callable[[StudyCVAObservation], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Compose existing diagnostics; raw mechanism joins do not authenticate causal support."""
    _enabled(enabled)
    p = _record(plan, AlignmentStudyPlan)
    batch = _validate_behavior(p, behavior)
    array(adjudications, 64)
    array(opportunities, 128)
    array(mechanisms, 64)
    raw = tuple(_record(m, StudyMechanismRecord) for m in mechanisms)
    if len({m.item_digest for m in raw}) != len(raw):
        raise ValueError(
            "one raw intervention per study item per report; retain further runs separately"
        )
    indexed = {}
    for record in raw:
        item = next((i for i in p.items if i.digest == record.item_digest), None)
        if item is None or record.plan_digest != p.digest or record.behavior_digest != batch.digest:
            raise ValueError("mechanistic record has a foreign study/behavior binding")
        validated_report = _report_snapshot(json.loads(record.report_json))
        commit = CausalCommit.from_dict(validated_report["commit"])
        _validate_intervention(StudyInterventionPacket(p, item, batch, commit))
        indexed[item.digest] = validated_report
    diagnostic = evaluate_causal_suite(
        tuple(i.pair for i in p.items),
        batch.captures,
        adjudications,
        protocol_id=p.plan_id,
        opportunities=opportunities,
        authenticate=authenticate,
        enabled=True,
    )
    fixtures, slots, joins = _cva_slots(p)
    originals = tuple(
        o.cva_capture for o in batch.outputs if o is not None and o.cva_capture is not None
    )
    supplied = tuple(_record(c, BenchmarkCapture) for c in array(cva_captures, 128))
    if len({c.slot_digest for c in supplied}) != len(supplied):
        raise ValueError("duplicate original CVA source cross-check")
    if supplied and {c.slot_digest: c.digest for c in supplied} != {
        c.slot_digest: c.digest for c in originals
    }:
        raise ValueError(
            "CVA results require the complete original structured behavior observations"
        )
    if authenticate_cva is not None and not callable(authenticate_cva):
        raise ValueError("CVA binding authentication requires a host callable")
    by_variant = {c.variant_id: (c, o) for c, o in zip(batch.captures, batch.outputs)}
    items_by_pair = {i.pair.digest: i for i in p.items}
    slot_sources = {slot: (items_by_pair[pair], variant) for pair, variant, slot in joins}
    plan_digest, behavior_digest = p.digest, batch.digest

    def authenticate_original(packet: CaptureVerificationInput) -> bool:
        item, variant_id = slot_sources[packet.slot.digest]
        capture, output = by_variant[variant_id]
        if output is None or output.cva_capture != packet.capture:
            return False
        arm = "before" if item.pair.before.variant_id == variant_id else "after"
        bound = StudyCVAObservation(
            plan_digest,
            item.digest,
            behavior_digest,
            batch.completed_at,
            public_study_input(item, arm, enabled=True),
            output,
            capture,
            packet,
        )
        return _authenticated(bound, authenticate_cva)

    cva = (
        evaluate_cva_benchmark(
            fixtures, slots, originals, authenticate=authenticate_original, enabled=True
        )
        if slots
        else None
    )
    rows, audits = [], []
    for item in p.items:
        report = indexed.get(item.digest)
        rows.append(
            {
                "template_id": item.template_id,
                "family": item.family,
                "pair_digest": item.pair.digest,
                "original_case": item.pair.before.case.to_dict(),
                "intervention_case": item.pair.after.case.to_dict(),
                "mechanistic_status": "NOT_RUN" if report is None else "RAW_UNVERIFIED",
                "measurement_digest": None if report is None else report["measurement_digest"],
                "skeptical_checks": (
                    None
                    if report is None
                    else {
                        "prediction_falsified": report["prediction_result"] == "FALSIFIED",
                        "nonspecific_control": report["prediction_result"] == "NOT_SPECIFIC",
                        "off_manifold_possible": True,
                        "intent_established": False,
                        "independent_registration": "NOT_PERFORMED",
                    }
                ),
            }
        )
        if report is not None:
            ref = EvidenceReference(
                "mechanistic-run:" + report["measurement_digest"],
                EvidenceSourceKind.EXTERNAL_RECORD,
            )
            for variant in (item.pair.before, item.pair.after):
                audits.append(
                    MechanisticAuditReference(
                        "audit:" + variant.variant_id,
                        variant.case.case_id,
                        1.0,
                        (ref,),
                        "mechanistic_circuit_audit",
                        ExperimentalMaturity.EXPERIMENTAL,
                        DiagnosticStatus.DIAGNOSTIC,
                        (ref,),
                        "Unverified raw intervention; uncertainty is an unknown placeholder.",
                    ).to_dict()
                )
    join_rows = []
    for pair, variant, slot in joins:
        source_capture, source_output = by_variant[variant]
        join_rows.append(
            {
                "pair_digest": pair,
                "variant_id": variant,
                "slot_digest": slot,
                "behavior_capture_digest": content_digest(source_capture.to_dict()),
                "public_output_digest": None if source_output is None else source_output.digest,
                "binding_status": (
                    "ORIGINAL_STRUCTURED_CAPTURE"
                    if source_output is not None and source_output.cva_capture is not None
                    else "NO_ORIGINAL_STRUCTURED_CAPTURE"
                ),
            }
        )
    result = {
        "schema_version": "alignment-mechanistic-study-report-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "plan": p.to_dict(),
        "behavior_batch": batch.to_dict(),
        "behavior": diagnostic,
        "study_pairs": rows,
        "mechanisms": [m.to_dict() for m in raw],
        "mechanistic_audits": audits,
        "cva": cva,
        "cva_joins": join_rows,
        "role_independence": "NOT_ESTABLISHED_BY_ROLE_NAMES",
        "workflow": {
            "protocol": [
                "observe",
                "hypothesize",
                "predict",
                "intervene",
                "measure",
                "replicate",
                "revise",
            ],
            "executed": ["observe_public_behavior"]
            + (
                ["supplied_hypothesis", "committed_prediction", "intervene", "measure_native"]
                if raw
                else []
            ),
            "replication": "NOT_RUN",
            "revision": "NOT_RUN",
            "registration": "NOT_PERFORMED",
        },
        "alignment_improvement": "NOT_ESTABLISHED",
        "promotion": "NOT_AUTHORIZED",
        "limitations": [
            "Authored hypotheses and semantic equivalence require independent review.",
            "Public labels/token mappings do not establish trained semantic mechanisms.",
            "Raw interventions require stage-6 independent replay and protected registration.",
            "CVA captures are separately authenticated structured observations.",
            "Families, repeats, transformations and multi-turn contexts are dependent.",
        ],
    }
    validate_input_envelope(result)
    result["result_digest"] = content_digest(result)
    return result
