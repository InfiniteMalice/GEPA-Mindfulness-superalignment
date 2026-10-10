"""Offline improvement accounting; hashes never authenticate host evidence."""

# Standard library
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Callable
from hashlib import sha256
from typing import Any

# Third-party
# Local
from .causal_records import canonical_json, content_digest
from .improvement_records import (
    PURPOSES,
    AttemptJournal,
    AuthenticationRequest,
    DatasetManifest,
    DiagnosticEvidence,
    ExposureRecord,
    FinalTestAuthorization,
    ImprovementProtocol,
    record_digest,
)
from .improvement_statistics import PairedObservation, estimate_overstatement, estimate_paired

FAILURE_METRICS = {
    "causal_invariance": ("spurious_decision_flip_rate", False),
    "required_update": ("required_update_success_rate", True),
    "laundering": ("semantic_laundering_susceptibility", False),
    "debate_fault_localization": ("premise_fault_localization", True),
    "unjustified_abstention": ("unjustified_abstention_stability", False),
    "clarification_resumption": ("clarification_resumption_correctness", True),
    "calibration": ("prediction_calibration", False),
    "severe_safety_authorization": ("severe_event_frequency", False),
}


def _authenticate(purpose, subject, refs, protocol, callback, statuses) -> bool:
    request = AuthenticationRequest(purpose, content_digest(subject), protocol.evaluator, refs)
    request = AuthenticationRequest.from_dict(request.to_dict())
    before = request.to_dict()
    status = "unverified"
    if callback is not None and refs:
        try:
            accepted = callback(request)
            status = (
                "verified" if accepted is True and request.to_dict() == before else "unverified"
            )
        except Exception:
            status = "authentication_error"
    statuses.append(dict(purpose=purpose, subject_digest=before["subject_digest"], status=status))
    return status == "verified"


def _source_row(evidence, slot, metric, candidate, evaluator):
    source = json.loads(evidence.source_json)
    schema = source.get("schema_version")
    schemas = {
        "evaluation-ladder-v1": ("rows", "probe_id"),
        "causal-diagnostics-v1": ("rows", "opportunity_id"),
        "sensitive-debate-report-v1": ("metric_rows", "opportunity_id"),
    }
    if schema not in schemas or schema != metric.source_schema:
        raise ValueError("unsupported or mismatched diagnostic schema")
    if (evidence.row_list, evidence.row_id_key) != schemas[schema]:
        raise ValueError("source selector/schema mismatch")
    body = {k: v for k, v in source.items() if k != "result_digest"}
    digest = sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=schema == "evaluation-ladder-v1",
        ).encode()
    ).hexdigest()
    if source.get("result_digest") != digest:
        raise ValueError("source result digest mismatch")
    rows = source.get(evidence.row_list)
    if type(rows) is not list or any(type(r) is not dict for r in rows):
        raise ValueError("invalid source rows")
    selected = [r for r in rows if r.get(evidence.row_id_key) == evidence.row_id]
    if len(selected) != 1 or selected[0].get("metric") != metric.source_metric:
        raise ValueError("source row or metric mismatch")
    row = selected[0]
    if row.get("unit", metric.unit) != metric.unit:
        raise ValueError("source metric unit mismatch")
    if schema == "evaluation-ladder-v1":
        systems, contract = [source["system"]], source["evaluator"]
    elif schema == "causal-diagnostics-v1":
        pairs = [p for p in source["pairs"] if p["pair_id"] == row["pair_id"]]
        if len(pairs) != 1:
            raise ValueError("source pair mismatch")
        pair = pairs[0]
        systems = [pair["source_record"][arm]["system"] for arm in ("before", "after")]
        contract = pair["adjudication"]["evaluator"] if pair["adjudication"] else None
    else:
        systems = [source["protocol"]["subject"]["system"]]
        contract = source["protocol"]["evaluator"]
    expected_contract = {
        k: getattr(evaluator, k) for k in ("evaluator_id", "evaluator_version", "contract_id")
    }
    if contract is None and _numeric(row, metric) is not None:
        raise ValueError("numeric source requires evaluator/rubric contract")
    if contract is not None and (
        contract != expected_contract or content_digest(contract) != metric.rubric_digest
    ):
        raise ValueError("source evaluator/rubric mismatch")
    config = getattr(candidate, slot.arm)
    for system in systems:
        if (
            system.get("model_version"),
            system.get("harness_version"),
            system.get("repeat_id"),
        ) != (config.model_version, config.harness_version, slot.repeat_id) or (
            slot.seed is not None and system.get("seed") != slot.seed
        ):
            raise ValueError("source system/repeat/seed mismatch")
    if source.get("confers_authority") is not False:
        raise ValueError("source must preserve diagnostic authority boundary")
    return source, row


def _numeric(row, metric):
    if row.get("status") not in ("verified", "observed") or row.get("eligible") is False:
        return None
    value = row.get("value")
    if value is None:
        return None
    if "calibration" in metric.source_metric and type(value) is not bool:
        if metric.calibration_digest is None or type(row.get("outcome")) is not bool:
            return None
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid calibration probability")
        value = (value - int(row["outcome"])) ** 2
    elif type(value) not in (bool, int, float) or not math.isfinite(value):
        raise ValueError("invalid diagnostic number")
    if metric.transform == "one_minus":
        successes = {m for m, success in FAILURE_METRICS.values() if success}
        if metric.source_metric not in successes or type(value) is not bool:
            raise ValueError("one_minus requires registered success semantics")
        value = 1 - value
    return float(value)


def audit_improvement(
    protocol: ImprovementProtocol,
    manifest: DatasetManifest,
    journal: AttemptJournal,
    evidence: tuple[DiagnosticEvidence, ...],
    *,
    exposures: tuple[ExposureRecord, ...] = (),
    final_authorizations: tuple[FinalTestAuthorization, ...] = (),
    authenticate: Callable[[AuthenticationRequest], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Audit host captures without loading cases, calling models, or mutating catalogs.

    Authentication must independently establish exact protocol, source-to-slot binding,
    complete provenance/journal/exposure history, and event-specific final-test permission.
    Returning a serialized verification flag does not meet this host contract.
    """
    if enabled is not True:
        raise ValueError("improvement audit requires enabled=True")
    if authenticate is not None and not callable(authenticate):
        raise ValueError("authenticate must be a host callable")
    protocol = ImprovementProtocol.from_dict(protocol.to_dict())
    manifest = DatasetManifest.from_dict(manifest.to_dict())
    journal = AttemptJournal.from_dict(journal.to_dict())
    for values, cls in (
        (evidence, DiagnosticEvidence),
        (exposures, ExposureRecord),
        (final_authorizations, FinalTestAuthorization),
    ):
        if type(values) is not tuple or any(type(v) is not cls for v in values):
            raise ValueError("expected exact record tuple")
    evidence = tuple(DiagnosticEvidence.from_dict(e.to_dict()) for e in evidence)
    exposures = tuple(ExposureRecord.from_dict(e.to_dict()) for e in exposures)
    final_authorizations = tuple(
        FinalTestAuthorization.from_dict(a.to_dict()) for a in final_authorizations
    )
    if protocol.manifest_digest != record_digest(manifest):
        raise ValueError("manifest protocol binding mismatch")
    clusters = validate_manifest(manifest)
    candidates, metrics, slots = _protocol_indexes(protocol)
    cases = _index(manifest.cases, "case_id")
    journal_report = summarize_journal(protocol, journal)
    by_slot = _index(evidence, "slot_id")
    if set(by_slot) - set(slots):
        raise ValueError("unexpected evidence slot")
    for slot in slots.values():
        case = cases.get(slot.case_id)
        if case is None or not case.evaluated or case.purpose == "synthetic_training":
            raise ValueError("slot requires declared non-training evaluated case")
    for exposure in exposures:
        if exposure.partition_digest != partition_digest(manifest, exposure.purpose):
            raise ValueError("exposure partition binding mismatch")
    statuses = []

    def auth(purpose, subject, refs):
        return _authenticate(purpose, subject, refs, protocol, authenticate, statuses)

    # Evaluate every attestation; short-circuiting would hide which evidence is missing.
    trust = [
        auth("protocol", protocol.to_dict(), protocol.evidence_refs),
        auth("provenance", manifest.to_dict(), manifest.evidence_refs),
        auth("journal", journal.to_dict(), journal.evidence_refs),
        auth(
            "exposure_history",
            dict(
                protocol_digest=record_digest(protocol), exposures=[e.to_dict() for e in exposures]
            ),
            protocol.evidence_refs,
        ),
    ]
    trusted = all(trust)
    rows, sources, severe, calibration = _join_captures(
        protocol, manifest, journal_report, by_slot, final_authorizations, auth, trusted
    )
    comparisons, verified, overstatements = _compare_captures(
        protocol, manifest, journal, exposures, clusters, rows, trusted
    )
    failures = _failure_inventory(rows)
    result = dict(
        schema_version="improvement-audit-v1",
        training_eligibility="DEVELOPMENT",
        confers_authority=False,
        deployment_eligibility="not_assessed",
        protocol=protocol.to_dict(),
        manifest=manifest.to_dict(),
        authentication=statuses,
        optimization_progress=journal_report,
        evaluated_behavior=dict(rows=rows, comparisons=comparisons, sources=sources),
        verified_improvement_evidence=verified,
        overstatement=overstatements,
        failures=failures,
        severe_events=list(severe.values()),
        calibration_aggregates=calibration,
        interval_scope="Fixed configurations; no simultaneous or post-selection guarantee.",
    )
    result["result_digest"] = content_digest(result)
    return json.loads(canonical_json(result))


def _join_captures(
    protocol, manifest, journal_report, by_slot, final_authorizations, auth, trusted
):
    candidates, metrics, slots = _protocol_indexes(protocol)
    cases = _index(manifest.cases, "case_id")
    rows, sources, severe, calibration = [], [], {}, []
    for sid, slot in sorted(slots.items()):
        case, metric = cases[slot.case_id], metrics[slot.metric_id]
        ev = by_slot.get(sid)
        output = dict(
            slot=slot.to_dict(),
            purpose=case.purpose,
            status="missing",
            value=None,
            source_value=None,
            source_metric=metric.source_metric,
            source_row=None,
        )
        if ev is None:
            if case.purpose == "final_test":
                output["status"] = "not_run"
            rows.append(output)
            continue
        source, row = _source_row(
            ev, slot, metric, candidates[slot.candidate_id], protocol.evaluator
        )
        start = journal_report["starts"].get(sid)
        allowed = case.purpose != "final_test"
        if not allowed and start:
            valid = [
                a
                for a in final_authorizations
                if a.candidate_id == slot.candidate_id
                and a.candidate_digest == record_digest(candidates[slot.candidate_id])
                and a.protocol_digest == record_digest(protocol)
                and a.partition_digest == partition_digest(manifest, "final_test")
                and a.evaluation_event_id == start["event_id"]
            ]
            allowed = any(auth("final_authorization", a.to_dict(), a.evidence_refs) for a in valid)
        if not allowed:
            output["status"] = "unauthorized"
            output["training_eligibility"] = source["training_eligibility"]
            _retain_severe(severe, source, row, slot, case.purpose, False, unauthorized=True)
            rows.append(output)
            continue
        output["source_digest"] = content_digest(source)
        accepted = auth(
            "evidence",
            dict(
                protocol_digest=record_digest(protocol), evidence=ev.to_dict(), slot=slot.to_dict()
            ),
            ev.evidence_refs,
        )
        status = row.get("status", "unresolved")
        complete_event = start is not None and sid in journal_report["finished"]
        if not trusted or not accepted:
            status = "unverified"
        elif not complete_event:
            status = "missing_attempt_completion"
        value = _numeric(row, metric) if trusted and accepted and complete_event else None
        output.update(
            status=status,
            value=value,
            source_value=row.get("value"),
            source_row=row,
            training_eligibility=source["training_eligibility"],
            seed_policy=slot.seed_policy,
        )
        sources.append(dict(slot_id=sid, source=source, authentication=accepted))
        if "calibration" in metric.source_metric and metric.calibration_digest is not None:
            calibration.append(
                dict(
                    slot_id=sid,
                    candidate_id=slot.candidate_id,
                    arm=slot.arm,
                    purpose=case.purpose,
                    authentication=trusted and accepted,
                    calibration_digest=metric.calibration_digest,
                    groups=source.get("metrics", {})
                    .get(metric.source_metric, {})
                    .get("groups", []),
                )
            )
        _retain_severe(severe, source, row, slot, case.purpose, accepted and trusted)
        rows.append(output)
    return rows, sources, severe, calibration


def _retain_severe(severe, source, row, slot, purpose, authenticated, *, unauthorized=False):
    inventory = list(source.get("severe_observations", source.get("severe_events", [])))
    if row.get("severity") not in (None, "routine"):
        inventory.append(row)
    for event in inventory:
        key = content_digest(
            dict(
                candidate=slot.candidate_id,
                arm=slot.arm,
                purpose=purpose,
                source=source["result_digest"],
                event=event,
            )
        )
        # Retain public incident provenance without releasing unauthorized test outcomes.
        if unauthorized:
            event = {
                k: v
                for k, v in event.items()
                if k
                in (
                    "probe_id",
                    "opportunity_id",
                    "pair_id",
                    "event_id",
                    "metric",
                    "severity",
                    "status",
                    "evidence_refs",
                )
            }
        severe[key] = dict(
            candidate_id=slot.candidate_id,
            arm=slot.arm,
            purpose=purpose,
            event=event,
            authentication=authenticated,
            training_eligibility=source["training_eligibility"],
            status="unauthorized" if unauthorized else event.get("status", "unresolved"),
        )
        if not unauthorized:
            severe[key]["source_digest"] = source["result_digest"]


def _compare_captures(protocol, manifest, journal, exposures, clusters, rows, trusted):
    candidates, metrics, slots = _protocol_indexes(protocol)
    cases = _index(manifest.cases, "case_id")
    comparisons, verified, estimates = [], [], {}
    by_id = {r["slot"]["slot_id"]: r for r in rows}
    for cid, candidate in sorted(candidates.items()):
        stratum = content_digest(
            dict(baseline=candidate.baseline.to_dict(), candidate=candidate.candidate.to_dict())
        )
        for purpose in PURPOSES[1:]:
            independent = independence_status(protocol, cid, purpose, exposures, journal=journal)
            for mid, metric in sorted(metrics.items()):
                selected = [
                    s
                    for s in slots.values()
                    if s.candidate_id == cid
                    and s.metric_id == mid
                    and cases[s.case_id].purpose == purpose
                ]
                groups = defaultdict(dict)
                for s in selected:
                    groups[(s.case_id, s.condition_id, s.repeat_id)][s.arm] = s
                matched = []
                expected = []
                for coordinate, arms in sorted(groups.items()):
                    pid = content_digest(coordinate)
                    expected.append(pid)
                    if set(arms) != {"baseline", "candidate"}:
                        continue
                    a, b = arms["baseline"], arms["candidate"]
                    ra, rb = by_id[a.slot_id], by_id[b.slot_id]
                    if (a.budget_digest, a.seed_policy) != (b.budget_digest, b.seed_policy):
                        continue
                    if a.seed_policy == "shared" and a.seed != b.seed:
                        continue
                    if ra["value"] is None or rb["value"] is None:
                        continue
                    if any(
                        ra["source_row"].get(k) != rb["source_row"].get(k)
                        for k in ("cohort", "severity", "unit")
                    ):
                        continue
                    matched.append(
                        PairedObservation(
                            pid,
                            a.case_id,
                            a.condition_id,
                            a.repeat_id,
                            clusters[a.case_id],
                            stratum,
                            metric,
                            ra["value"],
                            rb["value"],
                        )
                    )
                estimate = estimate_paired(
                    tuple(matched),
                    expected_pair_ids=tuple(expected),
                    policy=protocol.policy,
                    dependencies_known=trusted and manifest.dependencies_known,
                )
                eligible = (
                    trusted
                    and independent["independent"]
                    and manifest.dependencies_known
                    and bool(expected)
                    and estimate.matched_count == len(expected)
                )
                raw = {}
                for arm in ("baseline", "candidate"):
                    values = [
                        by_id[s.slot_id]["value"]
                        for s in selected
                        if s.arm == arm and by_id[s.slot_id]["value"] is not None
                    ]
                    raw[arm] = dict(
                        known_count=len(values),
                        planned=sum(s.arm == arm for s in selected),
                        mean=sum(values) / len(values) if values else None,
                    )
                report = dict(
                    candidate_id=cid,
                    purpose=purpose,
                    metric_id=mid,
                    estimate=estimate.to_dict(),
                    raw=raw,
                    independence=independent,
                    independent_evidence_eligible=eligible,
                )
                comparisons.append(report)
                estimates[(cid, purpose, mid)] = (estimate, trusted and independent["independent"])
                if eligible and purpose in ("independent_audit", "final_test", "ood_combinations"):
                    verified.append(report)
    overstatements = []
    for cid in sorted(candidates):
        for mid in sorted(metrics):
            selection, _ = estimates[(cid, "optimizer_selection", mid)]
            item = dict(candidate_id=cid, metric_id=mid)
            for label, purpose in (("audit", "independent_audit"), ("final", "final_test")):
                independent, eligible = estimates[(cid, purpose, mid)]
                item[label] = (
                    estimate_overstatement(selection, independent, policy=protocol.policy)
                    if eligible
                    else dict(
                        overstatement=None, interval=None, reason="independent_evidence_unavailable"
                    )
                )
            overstatements.append(item)
    return comparisons, verified, overstatements


def _failure_inventory(rows):
    failures = {}
    for family, (name, success) in FAILURE_METRICS.items():
        members = [r for r in rows if r["source_metric"] == name]
        failure_rows = []
        for r in members:
            value = r["source_value"]
            failure_value = (1 - value if success else value) if type(value) is bool else r["value"]
            failure_rows.append(
                dict(
                    slot=r["slot"],
                    purpose=r["purpose"],
                    status=r["status"],
                    value=failure_value if r["value"] is not None else None,
                    source_row=r["source_row"],
                )
            )
        grouped = defaultdict(list)
        for failure in failure_rows:
            s = failure["slot"]
            grouped[(s["candidate_id"], failure["purpose"], s["arm"])].append(failure)
        summaries = []
        for (cid, purpose, arm), own in sorted(grouped.items()):
            values = [r["value"] for r in own if r["value"] is not None]
            unresolved = sum(
                r["value"] is None and (r["source_row"] or {}).get("eligible") is True for r in own
            )
            observed = sum(values) / len(values) if values else None
            summaries.append(
                dict(
                    candidate_id=cid,
                    purpose=purpose,
                    arm=arm,
                    planned=len(own),
                    denominator=len(values) + unresolved,
                    known_count=len(values),
                    unresolved_eligible=unresolved,
                    missing=len(own) - len(values),
                    numerator=sum(values) if family != "calibration" else None,
                    rate=observed if not unresolved and family != "calibration" else None,
                    observed_rate=observed if family != "calibration" else None,
                    mean_brier=(observed if not unresolved and family == "calibration" else None),
                    observed_mean_brier=observed if family == "calibration" else None,
                )
            )
        failures[family] = dict(
            rows=failure_rows, groups=summaries, status="reported" if members else "missing"
        )
    return failures


def _index(items: tuple, key: str) -> dict[str, Any]:
    result = {}
    for item in items:
        name = getattr(item, key)
        if name in result:
            raise ValueError(f"duplicate {key}")
        result[name] = item
    return result


def _acyclic(parents: dict[str, tuple[str, ...]]) -> None:
    children = defaultdict(list)
    degrees = {name: len(values) for name, values in parents.items()}
    for child, values in parents.items():
        for parent in values:
            if parent not in parents:
                raise ValueError("unknown parent")
            children[parent].append(child)
    ready = deque(name for name, degree in degrees.items() if degree == 0)
    visited = 0
    while ready:
        visited += 1
        for child in children[ready.popleft()]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(parents):
        raise ValueError("cyclic ancestry")


def validate_manifest(manifest: DatasetManifest) -> dict[str, str]:
    manifest = DatasetManifest.from_dict(manifest.to_dict())
    cases = _index(manifest.cases, "case_id")
    if not cases:
        raise ValueError("manifest requires cases")
    _acyclic({name: c.parent_ids for name, c in cases.items()})
    roots = {name: name for name in cases}

    def root(name: str) -> str:
        while roots[name] != name:
            roots[name] = roots[roots[name]]
            name = roots[name]
        return name

    def join(left: str, right: str) -> None:
        a, b = sorted((root(left), root(right)))
        roots[b] = a

    owners = {}
    definitions = {content_digest(json.loads(d)) for d in manifest.ood_definitions}
    for name, case in sorted(cases.items()):
        if case.purpose == "ood_combinations" and case.combination_digest not in definitions:
            raise ValueError("OOD case requires a bound combination definition")
        for parent in case.parent_ids:
            join(name, parent)
        keys = [("family", case.family_id), ("content", case.content_digest)]
        keys += [("chain", chain) for chain in case.transformation_ids]
        keys += [("dependency", dep) for dep in case.dependency_ids]
        for key in keys:
            if key in owners:
                join(name, owners[key])
            owners[key] = name
    groups = defaultdict(set)
    result = {name: root(name) for name in sorted(cases)}
    for name, cluster in result.items():
        groups[cluster].add(cases[name].purpose)
    if any(len(purposes) != 1 for purposes in groups.values()):
        raise ValueError("cross-partition lineage, family, content or dependency overlap")
    return result


def partition_digest(manifest: DatasetManifest, purpose: str) -> str:
    """Include ancestor-only nodes, not merely observed cases."""
    return content_digest(
        [
            c.to_dict()
            for c in sorted(manifest.cases, key=lambda c: c.case_id)
            if c.purpose == purpose
        ]
    )


def _protocol_indexes(protocol: ImprovementProtocol) -> tuple[dict, dict, dict]:
    protocol.__post_init__()
    candidates = _index(protocol.candidates, "candidate_id")
    metrics = _index(protocol.metrics, "metric_id")
    slots = _index(protocol.slots, "slot_id")
    _acyclic(
        {
            n: (c.parent_candidate_id,) if c.parent_candidate_id else ()
            for n, c in candidates.items()
        }
    )
    for c in candidates.values():
        if (
            c.parent_candidate_id
            and candidates[c.parent_candidate_id].freeze_ordinal > c.freeze_ordinal
        ):
            raise ValueError("candidate ancestry contradicts freeze ordering")
    coordinates = set()
    for slot in slots.values():
        if slot.candidate_id not in candidates or slot.metric_id not in metrics:
            raise ValueError("slot references unknown candidate or metric")
        coordinate = (
            slot.candidate_id,
            slot.case_id,
            slot.metric_id,
            slot.arm,
            slot.repeat_id,
            slot.condition_id,
        )
        if coordinate in coordinates:
            raise ValueError("duplicate evaluation coordinates")
        coordinates.add(coordinate)
    return candidates, metrics, slots


def summarize_journal(protocol: ImprovementProtocol, journal: AttemptJournal) -> dict[str, Any]:
    protocol = ImprovementProtocol.from_dict(protocol.to_dict())
    journal = AttemptJournal.from_dict(journal.to_dict())
    if journal.protocol_digest != record_digest(protocol):
        raise ValueError("journal protocol binding mismatch")
    candidates, _, slots = _protocol_indexes(protocol)
    _index(journal.events, "event_id")
    proposed, started, finished, decisions = set(), {}, set(), {}
    rounds, candidate_rounds = set(), defaultdict(set)
    previous = -1
    charges = defaultdict(list)
    measured = set()
    for event in journal.events:
        cid = event.candidate_id
        if event.ordinal <= previous or cid not in candidates:
            raise ValueError("invalid event ordering or candidate")
        previous = event.ordinal
        if cid in decisions:
            raise ValueError("event after terminal decision")
        candidate_rounds[cid].add(event.round_id)
        if event.kind == "proposal":
            if cid in proposed:
                raise ValueError("duplicate proposal")
            proposed.add(cid)
        else:
            if cid not in proposed:
                raise ValueError("event before proposal")
            if event.kind == "decision":
                decisions[cid] = event.terminal_status
            else:
                slot = slots.get(event.slot_id)
                if slot is None or slot.candidate_id != cid:
                    raise ValueError("evaluation slot/candidate mismatch")
                if event.ordinal < candidates[cid].freeze_ordinal:
                    raise ValueError("evaluation before candidate freeze")
                if event.kind == "evaluation_started":
                    if slot.slot_id in started:
                        raise ValueError("duplicate evaluation start; use a distinct repeat")
                    started[slot.slot_id] = event
                    rounds.add(event.round_id)
                else:
                    start = started.get(slot.slot_id)
                    if (
                        start is None
                        or slot.slot_id in finished
                        or start.round_id != event.round_id
                    ):
                        raise ValueError("unmatched or repeated evaluation finish")
                    finished.add(slot.slot_id)
        for cost in event.costs:
            charges[(cost.unit, cost.currency, cost.basis)].append(cost.value)
        if event.slot_id and event.costs:
            measured.add(event.slot_id)
    attempted = {e.candidate_id for e in started.values()}
    cost_groups = []
    for (unit, currency, basis), values in sorted(charges.items(), key=lambda item: str(item[0])):
        known = [v for v in values if v is not None]
        cost_groups.append(
            dict(
                unit=unit,
                currency=currency,
                basis=basis,
                known_total=sum(known) if known else None,
                known_count=len(known),
                unknown_count=len(values) - len(known),
                partial=len(known) != len(values),
            )
        )
    return dict(
        candidate_count=len(attempted),
        evaluation_attempt_count=len(started),
        selection_round_count=len(rounds),
        planned_unattempted=sorted(set(candidates) - attempted),
        candidates=[
            dict(
                candidate_id=cid,
                status=decisions.get(cid, "pending"),
                attempted=cid in attempted,
                rounds=sorted(candidate_rounds[cid]),
                spec=candidates[cid].to_dict(),
            )
            for cid in sorted(candidates)
        ],
        starts={name: e.to_dict() for name, e in sorted(started.items())},
        finished=sorted(finished),
        events=[e.to_dict() for e in journal.events],
        cost=dict(groups=cost_groups, unmeasured_attempts=len(set(started) - measured)),
    )


def independence_status(
    protocol: ImprovementProtocol,
    candidate_id: str,
    purpose: str,
    exposures: tuple[ExposureRecord, ...],
    *,
    journal: AttemptJournal | None = None,
) -> dict[str, Any]:
    candidates, _, _ = _protocol_indexes(protocol)
    if candidate_id not in candidates:
        raise ValueError("unknown independence target")
    ancestors = set()
    cursor = candidate_id
    while cursor is not None:
        ancestors.add(cursor)
        cursor = candidates[cursor].parent_candidate_id
    round_members = defaultdict(set)
    if journal is not None:
        for event in journal.events:
            round_members[event.round_id].add(event.candidate_id)
    reasons = []
    for exposure in exposures:
        exposure.__post_init__()
        if set(exposure.candidate_ids) - set(candidates):
            raise ValueError("unknown exposure recipient")
        if set(exposure.round_ids) - set(round_members):
            raise ValueError("exposure round requires matching journal")
        recipients = set(exposure.candidate_ids)
        for round_id in exposure.round_ids:
            recipients.update(round_members[round_id])
        if exposure.purpose == purpose and exposure.use != "evaluation_only":
            if not recipients or recipients & ancestors:
                reasons.append(dict(ordinal=exposure.ordinal, use=exposure.use))
    return dict(independent=not reasons, reasons=reasons, authentication="not_assessed")
