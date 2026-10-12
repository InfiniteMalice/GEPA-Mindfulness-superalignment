"""Offline seven-arm specifications composed with the existing improvement audit."""

# Standard library
from __future__ import annotations

from collections.abc import Callable

# Third-party
# Local
from gepa_mindfulness.learning_surfaces import LessonProposal
from synthetic_data.curriculum_records import (
    COMPONENTS,
    AblationArm,
    BankSnapshot,
    CurationExposure,
    CurationPolicy,
    EpisodeSource,
    JsonObject,
    LessonReview,
    read_curriculum_object,
    unique,
    utc_time,
    validate_curriculum_envelope,
)
from synthetic_data.lesson_curation import (
    HostCheck,
    authenticate_payload,
    curate_lessons,
    report_document,
    require_enabled,
    snapshot,
    snapshots,
)

from .causal_records import canonical_json, content_digest
from .improvement_audit import audit_improvement, partition_digest, validate_manifest
from .improvement_records import (
    AttemptJournal,
    AuthenticationRequest,
    DatasetManifest,
    DiagnosticEvidence,
    ExposureRecord,
    FinalTestAuthorization,
    ImprovementProtocol,
)

ARM_IDS = (
    "baseline",
    "synthetic",
    "causal",
    "causal_laundering",
    "causal_debate",
    "causal_pluralistic",
    "knowledge",
)
_CONTROLS = {
    "baseline": None,
    "synthetic": "baseline",
    "causal": "synthetic",
    "causal_laundering": "causal",
    "causal_debate": "causal",
    "causal_pluralistic": "causal",
}


def _derive_exposures(
    records: tuple[CurationExposure, ...],
    manifest: DatasetManifest,
    arms: tuple[AblationArm, ...],
) -> tuple[ExposureRecord, ...]:
    cases = {case.case_id: case for case in manifest.cases}
    if any(not record.case_ids for record in records):
        raise ValueError("curation exposure requires case identities")
    sources = [
        (r.case_ids, r.ordinal, r.candidate_ids, r.round_ids, r.evidence_refs) for r in records
    ]
    # Arm data are declared candidate inputs; conservatively account them from ordinal zero.
    sources.extend(
        (arm.data_case_ids, 0, (arm.candidate_id,), (), manifest.evidence_refs)
        for arm in arms
        if arm.data_case_ids
    )
    derived = []
    for case_ids, ordinal, candidate_ids, round_ids, refs in sources:
        visited: set[str] = set()
        pending = list(case_ids)
        while pending:
            identity = pending.pop()
            if identity not in cases:
                raise ValueError("unknown curation-exposed case")
            if identity not in visited:
                visited.add(identity)
                pending.extend(cases[identity].parent_ids)
        for purpose in sorted({cases[name].purpose for name in visited}):
            derived.append(
                ExposureRecord(
                    ordinal,
                    partition_digest(manifest, purpose),
                    purpose,
                    candidate_ids,
                    round_ids,
                    "proposal",
                    refs,
                )
            )
    return tuple(derived)


def _knowledge_reasons(
    arm: AblationArm,
    records: tuple[CurationExposure, ...],
    journal: AttemptJournal,
    as_of: str,
    authenticated: bool,
) -> list[str]:
    if arm.knowledge_report_json is None:
        return ["knowledge_report_missing"]
    report = read_curriculum_object(arm.knowledge_report_json)
    if (
        set(report)
        != {
            "schema_version",
            "training_eligibility",
            "optimizer_input",
            "confers_authority",
            "inputs",
            "lessons",
            "proposed_bank_additions",
            "result_digest",
        }
        or report["training_eligibility"] != "DEVELOPMENT"
        or report["optimizer_input"] is not False
        or report["confers_authority"] is not False
    ):
        raise ValueError("invalid knowledge report schema or authority fields")
    if report.get("schema_version") != "lesson-curation-v1" or report.get(
        "result_digest"
    ) != content_digest({k: v for k, v in report.items() if k != "result_digest"}):
        raise ValueError("invalid knowledge report integrity")
    values = report.get("inputs")
    required = {"episodes", "lessons", "bank", "reviews", "policy", "as_of"}
    if type(values) is not dict or set(values) != required:
        raise ValueError("knowledge report requires complete curation inputs")
    episodes = tuple(EpisodeSource.from_dict(e) for e in values["episodes"])
    fresh = curate_lessons(
        episodes=episodes,
        lessons=tuple(LessonProposal.from_dict(item) for item in values["lessons"]),
        bank=BankSnapshot.from_dict(values["bank"]),
        reviews=tuple(LessonReview.from_dict(item) for item in values["reviews"]),
        policy=CurationPolicy.from_dict(values["policy"]),
        as_of=as_of,
        authenticate=lambda payload: authenticated,
        enabled=True,
    )
    reasons = []
    if not fresh["proposed_bank_additions"]:
        reasons.append("knowledge_not_currently_curated")
    if canonical_json(fresh["lessons"]) != canonical_json(report["lessons"]):
        reasons.append("knowledge_lessons_mismatch")
    if canonical_json(fresh["proposed_bank_additions"]) != canonical_json(
        report.get("proposed_bank_additions")
    ):
        reasons.append("knowledge_additions_mismatch")
    needed = {case for episode in episodes for case in episode.case_ids}
    rounds = {event.round_id for event in journal.events if event.candidate_id == arm.candidate_id}
    covered: set[str] = set()
    for record in records:
        recipients = record.candidate_ids
        receives = (
            not recipients
            and not record.round_ids
            or arm.candidate_id in recipients
            or bool(rounds & set(record.round_ids))
        )
        if receives and record.source_json == arm.knowledge_report_json:
            covered.update(record.case_ids)
    if not needed or not needed <= covered:
        reasons.append("knowledge_exposure_incomplete")
    return reasons


def audit_curriculum_comparison(
    arms: tuple[AblationArm, ...],
    protocol: ImprovementProtocol,
    manifest: DatasetManifest,
    journal: AttemptJournal,
    evidence: tuple[DiagnosticEvidence, ...],
    *,
    as_of: str,
    curation_exposures: tuple[CurationExposure, ...] = (),
    exposures: tuple[ExposureRecord, ...] = (),
    final_authorizations: tuple[FinalTestAuthorization, ...] = (),
    authenticate_manifest: HostCheck | None = None,
    authenticate_audit: Callable[[AuthenticationRequest], bool] | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Audit declared ablations and preserve the full PR-6 result without running experiments.

    The manifest host attests current knowledge sources, component-specific data rosters,
    measured budgets and complete exposure recipients. The separate audit host retains its
    existing source, journal and final-test authority checks.
    """
    require_enabled(enabled)
    utc_time(as_of)
    # Bound embedded JSON before PR-6 constructors invoke their existing JSON decoders.
    if type(evidence) is not tuple or any(type(e) is not DiagnosticEvidence for e in evidence):
        raise ValueError("expected diagnostic evidence tuple")
    for item in evidence:
        read_curriculum_object(item.source_json)
    if type(manifest) is not DatasetManifest:
        raise ValueError("expected exact DatasetManifest")
    for definition in manifest.ood_definitions:
        read_curriculum_object(definition)
    arms = snapshots(arms, AblationArm)
    protocol, manifest = snapshot(protocol, ImprovementProtocol), snapshot(
        manifest, DatasetManifest
    )
    journal = snapshot(journal, AttemptJournal)
    evidence = snapshots(evidence, DiagnosticEvidence)
    curation_exposures = snapshots(curation_exposures, CurationExposure)
    exposures = snapshots(exposures, ExposureRecord)
    final_authorizations = snapshots(final_authorizations, FinalTestAuthorization)
    unique(arms, "arm_id")
    unique(arms, "candidate_id")
    if {arm.arm_id for arm in arms} != set(ARM_IDS):
        raise ValueError("comparison requires exactly seven arms")
    validate_manifest(manifest)
    candidates = {c.candidate_id: c for c in protocol.candidates}
    cases = {case.case_id for case in manifest.cases}
    for arm in arms:
        if arm.candidate_id not in candidates or not set(arm.data_case_ids) <= cases:
            raise ValueError("unknown arm candidate or data case")
        if arm.arm_id in _CONTROLS and arm.control_arm_id != _CONTROLS[arm.arm_id]:
            raise ValueError("arm requires its declared control")
        if arm.arm_id == "knowledge" and arm.control_arm_id not in (None, *ARM_IDS[:-1]):
            raise ValueError("knowledge control must be a preceding arm")
    derived = _derive_exposures(curation_exposures, manifest, arms)
    inputs = dict(
        arms=[a.to_dict() for a in arms],
        protocol=protocol.to_dict(),
        manifest=manifest.to_dict(),
        journal=journal.to_dict(),
        evidence=[e.to_dict() for e in evidence],
        curation_exposures=[e.to_dict() for e in curation_exposures],
        exposures=[e.to_dict() for e in exposures],
        final_authorizations=[a.to_dict() for a in final_authorizations],
        as_of=as_of,
    )
    validate_curriculum_envelope(inputs)
    authenticated = authenticate_payload(
        dict(purpose="curriculum_manifest", as_of=as_of, subject=inputs), authenticate_manifest
    )
    by_arm = {arm.arm_id: arm for arm in arms}
    expected_components: dict[str, set[str]] = {}
    for index, name in enumerate(ARM_IDS):
        control_name = by_arm[name].control_arm_id
        expected_components[name] = set(expected_components.get(control_name or "", set()))
        if index:
            expected_components[name].add(COMPONENTS[index - 1])
    rows = []
    for index, name in enumerate(ARM_IDS):
        arm = by_arm[name]
        control = by_arm.get(arm.control_arm_id or "")
        reasons = [] if authenticated else ["manifest_unauthenticated"]
        components = read_curriculum_object(arm.components_json)
        if {key for key, value in components.items() if value is not None} != expected_components[
            name
        ]:
            reasons.append("component_roster_mismatch")
        planned: bool | None = None
        actual: bool | None = None
        isolated: bool | None = None
        if name == "baseline":
            if any(v is not None for v in components.values()):
                reasons.append("baseline_components_present")
        elif control is None:
            reasons.append("control_missing")
        else:
            base = read_curriculum_object(control.components_json)
            changed = {key for key in COMPONENTS if base[key] != components[key]}
            intended = COMPONENTS[index - 1]
            if changed != {intended} or base[intended] is not None or components[intended] is None:
                reasons.append("component_difference_mismatch")
            for field in ("starting_system", "optimizer_digest", "reward_digest", "seed_policy"):
                if getattr(arm, field) != getattr(control, field):
                    reasons.append(field + "_mismatch")
            if any(
                item.seed_policy == "unknown" or not item.seeds or None in item.seeds
                for item in (arm, control)
            ):
                reasons.append("seed_policy_unverified")
            if arm.seed_policy == "shared" and arm.seeds != control.seeds:
                reasons.append("seeds_mismatch")
            if candidates[arm.candidate_id].baseline != candidates[control.candidate_id].candidate:
                reasons.append("control_result_mismatch")
            isolated = not reasons
            planned = (
                arm.planned_budget_json == control.planned_budget_json if authenticated else None
            )
            if arm.actual_budget_json is not None and control.actual_budget_json is not None:
                actual = (
                    arm.actual_budget_json == control.actual_budget_json if authenticated else None
                )
            if planned is False:
                reasons.append("planned_budget_mismatch")
            if actual is not True:
                # Missing actual measurements do not prevent preparing an offline specification.
                actual_reason = (
                    "actual_budget_unknown" if actual is None else "actual_budget_mismatch"
                )
            else:
                actual_reason = None
        ready = not reasons
        if name == "knowledge":
            knowledge = _knowledge_reasons(arm, curation_exposures, journal, as_of, authenticated)
            reasons.extend(knowledge)
            ready &= not knowledge
        if control is not None and actual_reason is not None:
            reasons.append(actual_reason)
        rows.append(
            dict(
                arm_id=name,
                candidate_id=arm.candidate_id,
                control_arm_id=arm.control_arm_id,
                planned_budget_match=planned,
                actual_budget_match=actual,
                isolated=isolated,
                ready=ready,
                matched_execution=ready and isolated is True and actual is True,
                reasons=sorted(set(reasons)),
            )
        )
    audit = audit_improvement(
        protocol,
        manifest,
        journal,
        evidence,
        exposures=exposures + derived,
        final_authorizations=final_authorizations,
        authenticate=authenticate_audit,
        enabled=True,
    )
    return report_document(
        "curriculum-comparison-v1",
        inputs,
        arms=rows,
        manifest_authenticated=authenticated,
        derived_exposures=[e.to_dict() for e in derived],
        improvement_audit=audit,
        deployment_eligibility="not_assessed",
    )
