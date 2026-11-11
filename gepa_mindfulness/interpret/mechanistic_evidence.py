"""Experimental host-verified mechanistic findings and protected diagnostic registration.

Independent replay checks measurements. Trusted host authentication establishes custody,
precommit order and actual reviewer independence; role labels alone never suffice. Existing
WRITE authorization is consumed for admission. Every snapshot remains DEVELOPMENT data.
"""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_enum,
    _read_evaluator,
    _read_records,
    _record,
    array,
    digest,
    enum,
    evaluator,
    references,
    text,
    texts,
)
from gepa_mindfulness.verification.diagnostic_records import restore_refs
from gepa_mindfulness.verification.runtime_governance import (
    AuthorityGrantRegistry,
    AuthorizationDecision,
    RuntimeCapability,
    TrustedClock,
    consume_authorization,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.action_bound_events import ActionRecord
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

from .causal_interventions import EXECUTOR_ID, EXECUTOR_VERSION, _make_events, execute_causal_plan
from .native_causal_records import CausalCommit, InterventionMode, run_measurement_digest
from .native_neurons import FrozenNativeModel
from .native_records import NativeBatch, NativeModelIdentity, finite


class EvidenceLevel(str, Enum):
    DISCOVERED_CORRELATION = "DISCOVERED_CORRELATION"
    PREDICTIVE_HYPOTHESIS = "PREDICTIVE_HYPOTHESIS"
    HELD_OUT_SUPPORTED = "HELD_OUT_SUPPORTED"
    INTERVENTION_SUPPORTED = "INTERVENTION_SUPPORTED"
    REPLICATED_CAUSAL_EVIDENCE = "REPLICATED_CAUSAL_EVIDENCE"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("mechanistic evidence requires enabled=True")


def _timestamp(clock: TrustedClock | None) -> str:
    now = datetime.now(timezone.utc) if clock is None else clock.now()
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("evidence clock requires an aware datetime")
    return now.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class VerificationPacket(CVARecord):
    actor_id: str
    report_jsons: tuple[str, ...]

    schema_version = "mechanistic-verification-packet-v1"
    restorers = {"report_jsons": _read_array}

    def __post_init__(self) -> None:
        text(self.actor_id, "actor_id")
        for source in array(self.report_jsons, 16, required=True):
            text(source, "report_json", 4 * 1024 * 1024)


@dataclass(frozen=True)
class MechanisticVerifierContext:
    backend: FrozenNativeModel
    evaluator: TrustedEvaluatorContract
    actor_id: str
    clock: TrustedClock
    authenticate: Callable[[VerificationPacket], bool] | None

    def __post_init__(self) -> None:
        if type(self.backend) is not FrozenNativeModel:
            raise ValueError("mechanistic verifier requires the trusted frozen backend")
        evaluator(self.evaluator)
        text(self.actor_id, "actor_id")
        if self.actor_id == self.evaluator.evaluator_id:
            raise ValueError("self-review cannot establish independent evidence")
        if not callable(getattr(self.clock, "now", None)):
            raise ValueError("host clock must implement now")
        if self.authenticate is not None and not callable(self.authenticate):
            raise ValueError("host authentication must be a callable or None")


@dataclass(frozen=True)
class MechanisticFinding(CVARecord):
    finding_id: str
    hypothesis_key: str
    level: EvidenceLevel
    source_reports_json: tuple[str, ...]
    report_digests: tuple[str, ...]
    measurement_digests: tuple[str, ...]
    replication_count: int
    host_evaluator: TrustedEvaluatorContract | None
    evidence_refs: tuple[EvidenceReference, ...]
    remaining_uncertainty: tuple[str, ...]
    created_at: str
    source_kind: str = "causal"

    schema_version = "mechanistic-finding-v1"
    restorers = {
        "level": lambda v: _read_enum(v, EvidenceLevel),
        "source_reports_json": _read_array,
        "report_digests": _read_array,
        "measurement_digests": _read_array,
        "host_evaluator": lambda v: None if v is None else _read_evaluator(v),
        "evidence_refs": restore_refs,
        "remaining_uncertainty": _read_array,
    }

    def __post_init__(self) -> None:
        if self.source_kind not in ("causal", "prediction"):
            raise ValueError("unsupported mechanistic evidence source kind")
        digest(self.finding_id)
        digest(self.hypothesis_key)
        enum(self.level, EvidenceLevel)
        sources = array(self.source_reports_json, 16, required=True)
        hashes = array(self.report_digests, 16, required=True)
        measurements = array(self.measurement_digests, 16, required=True)
        if len(sources) != len(hashes) or len(sources) != len(measurements):
            raise ValueError("finding source lengths mismatch")
        for source, expected, measurement in zip(sources, hashes, measurements):
            text(source, "report_json", 4 * 1024 * 1024)
            digest(expected)
            digest(measurement)
            try:
                payload = json.loads(source)
                validate_input_envelope(payload)
            except (RecursionError, TypeError) as error:
                raise ValueError("invalid source report JSON") from error
            if canonical_json(payload) != source or content_digest(payload) != expected:
                raise ValueError("finding source digest mismatch")
            measured_key = (
                "measurement_digest" if self.source_kind == "causal" else "source_batch_digest"
            )
            if payload.get(measured_key) != measurement:
                raise ValueError("finding measurement binding mismatch")
        if type(self.replication_count) is not int or self.replication_count != len(sources):
            raise ValueError("replication count must retain every submitted source")
        if self.host_evaluator is not None:
            evaluator(self.host_evaluator)
        references(self.evidence_refs)
        texts(self.remaining_uncertainty, required=True)
        parse_rfc3339_datetime(self.created_at, "created_at")


def _hypothesis_key(commit: CausalCommit) -> str:
    p = commit.plan
    return content_digest(
        {
            "hypothesis": p.hypothesis,
            "model": p.model_digest,
            "source": p.source.to_dict(),
            "mediator": None if p.mediator is None else p.mediator.to_dict(),
            "mode": p.mode.value,
            "metric": p.metric.to_dict(),
            "prediction": p.prediction.to_dict(),
            "replacement_values": p.to_dict()["replacement_values"],
        }
    )


def _causal_context_key(commit: CausalCommit) -> str:
    p = commit.plan
    # Future-token changes or renamed IDs are not independent causal-context replication.
    maximum = p.metric.position
    mask = p.donor.attention_mask[: maximum + 1]
    return content_digest(
        {
            "donor": [t if m else None for t, m in zip(p.donor.token_ids, mask)],
            "recipient": [t if m else None for t, m in zip(p.recipient.token_ids, mask)],
            "mask": list(mask),
        }
    )


def _check_metric_arithmetic(report: dict[str, Any], commit: CausalCommit) -> None:
    """Verifier-side arithmetic from recorded logits, independent of executor effect code."""
    conditions = report["conditions"]
    if type(conditions) is not dict or not {
        "donor",
        "recipient",
        "identity",
        "ablation",
        "intervention",
    } <= set(conditions):
        raise ValueError("missing required causal conditions")
    metric = commit.plan.metric
    for name, c in conditions.items():
        required = {"input", "logits", "score", "sites", "hook_order", "effect"}
        if name not in ("donor", "recipient"):
            required.add("reference_score")
        if type(c) is not dict or set(c) != required:
            raise ValueError("incorrect causal condition fields")
        logits = c["logits"]
        if (
            type(logits) is not list
            or len(logits) != len(commit.plan.donor.token_ids)
            or any(
                type(row) is not list
                or max(metric.target_token, metric.reference_token) >= len(row)
                for row in logits
            )
        ):
            raise ValueError("incorrect recorded logit geometry")
        for row in logits:
            for value in row:
                finite(value)
        finite(c["score"])
        finite(c["effect"])
        score = (
            logits[metric.position][metric.target_token]
            - logits[metric.position][metric.reference_token]
        )
        if c["score"] != score:
            raise ValueError("reported score disagrees with recorded logits")
        if name in ("donor", "recipient"):
            if c["effect"] != 0.0:
                raise ValueError("baseline cannot claim intervention effect")
        else:
            finite(c["reference_score"])
            if c["effect"] != score - c["reference_score"]:
                raise ValueError("reported effect disagrees with recorded contrast")

    def derive(prefix: str) -> float:
        if commit.plan.mode is InterventionMode.ABLATION:
            return conditions["donor"]["score"] - conditions[prefix]["score"]
        if commit.plan.mode is InterventionMode.RESTORATION:
            return conditions[prefix]["score"] - conditions[prefix + "_source_ablated"]["score"]
        if commit.plan.mode is InterventionMode.MEDIATION:
            restored = (
                "mediator_restored" if prefix == "intervention" else prefix + "_mediator_restored"
            )
            return conditions[prefix]["effect"] - conditions[restored]["effect"]
        return conditions[prefix]["effect"]

    try:
        primary = derive("intervention")
        control = derive("random_component") if "random_component" in conditions else None
        direction = derive("magnitude_matched_direction")
    except KeyError as error:
        raise ValueError("incomplete causal intervention/control data") from error
    if (
        report["primary_effect"] != primary
        or report["control_effect"] != control
        or report["direction_control_effect"] != direction
    ):
        raise ValueError("causal summary disagrees with controlled effect arithmetic")
    prediction = commit.plan.prediction
    signed = prediction.direction * primary
    outcome = "MET"
    if signed < prediction.minimum_effect or signed > prediction.maximum_effect:
        outcome = "FALSIFIED"
    elif (
        control is None
        or prediction.direction * (primary - control) < prediction.minimum_control_gap
    ):
        outcome = "NOT_SPECIFIC"
    if report["prediction_result"] != outcome:
        raise ValueError("causal result disagrees with preregistered prediction")


def _report_snapshot(source: object) -> dict[str, Any]:
    validate_input_envelope(source)
    if type(source) is not dict:
        raise ValueError("causal evidence requires exact report objects")
    report = json.loads(canonical_json(source))
    expected = {
        "schema_version": "native-causal-run-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "measurement_authentication": "NOT_PERFORMED",
        "evidence_status": "UNRESOLVED",
        "executor_id": EXECUTOR_ID,
        "executor_version": EXECUTOR_VERSION,
    }
    fields = set(expected) | {
        "plan_digest",
        "commit_digest",
        "commit",
        "model",
        "conditions",
        "primary_effect",
        "control_effect",
        "direction_control_effect",
        "mediation",
        "prediction_result",
        "executed_at",
        "evidence_refs",
        "control_roles",
        "limitations",
        "measurement_digest",
        "events",
    }
    if set(report) != fields:
        raise ValueError("incorrect causal report fields")
    for name, value in expected.items():
        if type(report.get(name)) is not type(value) or report[name] != value:
            raise ValueError("unsupported or relabelled causal evidence")
    commit = CausalCommit.from_dict(report.get("commit"))
    if (
        report.get("plan_digest") != commit.plan.digest
        or report.get("commit_digest") != commit.digest
    ):
        raise ValueError("causal report plan binding mismatch")
    if report.get("measurement_digest") != run_measurement_digest(report):
        raise ValueError("causal report measurement digest mismatch")
    if NativeModelIdentity.from_dict(report["model"]).digest != commit.plan.model_digest:
        raise ValueError("causal report physical model binding mismatch")
    if report["evidence_refs"] != commit.plan.to_dict()["evidence_refs"]:
        raise ValueError("causal report evidence reference mismatch")
    if parse_rfc3339_datetime(report["executed_at"], "executed_at") < parse_rfc3339_datetime(
        commit.created_at, "created_at"
    ):
        raise ValueError("causal execution predates its prediction")
    _check_metric_arithmetic(report, commit)
    events = report.get("events")
    if type(events) is not list or len(events) != 4:
        raise ValueError("causal report requires complete action-bound events")
    validate_action_bound_sequence(tuple(EventEnvelope(**event) for event in events))
    if events != _make_events(commit, report, report["executed_at"]):
        raise ValueError("action-bound events do not bind the exact prediction/action/measurement")
    return report


def verify_causal_runs(
    reports: tuple[dict[str, Any], ...],
    *,
    context: MechanisticVerifierContext | None = None,
    enabled: bool = False,
) -> MechanisticFinding:
    """Authenticate host custody, independently replay, then classify scoped predictions.

    Finding serialization is inert. Consumers must recheck sources during protected admission.
    Temporal order and actual independence rely on host authentication, not role-name strings.
    """
    _enabled(enabled)
    snapshots = tuple(_report_snapshot(r) for r in array(reports, 16, required=True))
    commits = tuple(CausalCommit.from_dict(r["commit"]) for r in snapshots)
    key = _hypothesis_key(commits[0])
    if any(_hypothesis_key(c) != key for c in commits):
        raise ValueError("replication requires matched mechanism/model/metric/prediction")
    if len({_causal_context_key(c) for c in commits}) != len(commits):
        raise ValueError("replication requires distinct causal input contexts")
    if context is not None and type(context) is not MechanisticVerifierContext:
        raise ValueError("exact host verifier context required")
    host = (
        None
        if context is None
        else MechanisticVerifierContext(
            context.backend,
            evaluator(context.evaluator),
            context.actor_id,
            context.clock,
            context.authenticate,
        )
    )
    sources = tuple(canonical_json(r) for r in snapshots)
    packet = VerificationPacket("unknown-actor" if host is None else host.actor_id, sources)
    level = EvidenceLevel.UNRESOLVED
    reasons = ["Host custody, prediction order and actual reviewer independence not established."]
    authenticated = False
    if host is not None and host.authenticate is not None:
        detached = VerificationPacket.from_dict(packet.to_dict())
        before = detached.to_dict()
        try:
            authenticated = host.authenticate(detached) is True and detached.to_dict() == before
        except Exception:
            authenticated = False
    if authenticated and host is not None:
        matches = True
        for original, commit in zip(snapshots, commits):
            try:
                replay = execute_causal_plan(host.backend, commit, clock=host.clock, enabled=True)
                if replay["measurement_digest"] != original["measurement_digest"]:
                    matches = False
            except (ValueError, RuntimeError, TypeError):
                matches = False
        if not matches:
            reasons = ["Recorded measurements do not match independent frozen-model replay."]
        else:
            outcomes = [r["prediction_result"] for r in snapshots]
            if "FALSIFIED" in outcomes:
                level = EvidenceLevel.CONTRADICTED
            elif all(o == "MET" for o in outcomes):
                level = (
                    EvidenceLevel.REPLICATED_CAUSAL_EVIDENCE
                    if len(outcomes) > 1
                    else EvidenceLevel.INTERVENTION_SUPPORTED
                )
            if level is EvidenceLevel.CONTRADICTED:
                scope_reason = "The preregistered prediction was falsified by independent replay."
            elif level is EvidenceLevel.UNRESOLVED:
                scope_reason = (
                    "Replay matched, but specificity against controls remains unresolved."
                )
            else:
                scope_reason = (
                    "Only the tested model, contexts, sites, controls and output metric "
                    "are supported."
                )
            reasons = [
                scope_reason,
                "Redundancy, interactions, off-manifold states and untested routes remain unknown.",
                "Semantic intent, ethical understanding and general alignment remain unverified.",
            ]
    refs: dict[str, EvidenceReference] = {}
    for c in commits:
        for ref in c.plan.evidence_refs:
            if ref.reference_id in refs and refs[ref.reference_id] != ref:
                raise ValueError("conflicting evidence reference kinds")
            refs[ref.reference_id] = ref
    hashes = tuple(content_digest(r) for r in snapshots)
    return MechanisticFinding(
        content_digest(
            {
                "sources": list(hashes),
                "verifier": None if host is None else host.evaluator.evaluator_id,
            }
        ),
        key,
        level,
        sources,
        hashes,
        tuple(r["measurement_digest"] for r in snapshots),
        len(snapshots),
        None if host is None else host.evaluator,
        tuple(refs.values()),
        tuple(reasons),
        _timestamp(None if host is None else host.clock),
    )


def verify_predictive_run(
    report: dict[str, Any],
    *,
    context: MechanisticVerifierContext | None = None,
    enabled: bool = False,
) -> MechanisticFinding:
    """Verify exact held-out activation predictions without claiming a causal mechanism.

    Host attestation covers locked-data isolation, prior predictor custody and the scope of
    its preregistered criterion. A met numeric tolerance alone is not semantic understanding.
    """
    _enabled(enabled)
    from .explanation_tests import ExplanationCommit, evaluate_explanation

    validate_input_envelope(report)
    if type(report) is not dict:
        raise ValueError("prediction evidence requires an exact report")
    payload = json.loads(canonical_json(report))
    commit = ExplanationCommit.from_dict(payload.get("commit"))
    source = NativeBatch.from_dict(payload.get("source_measurements"))
    reconstructed = evaluate_explanation(commit, source, enabled=True)
    if reconstructed != payload:
        raise ValueError("prediction report differs from its bound metric/source records")
    if context is not None and type(context) is not MechanisticVerifierContext:
        raise ValueError("exact host verifier context required")
    host = (
        None
        if context is None
        else MechanisticVerifierContext(
            context.backend,
            evaluator(context.evaluator),
            context.actor_id,
            context.clock,
            context.authenticate,
        )
    )
    packet = VerificationPacket(
        "unknown-actor" if host is None else host.actor_id, (canonical_json(payload),)
    )
    authenticated = False
    if host is not None and host.authenticate is not None:
        detached = VerificationPacket.from_dict(packet.to_dict())
        before = detached.to_dict()
        try:
            authenticated = host.authenticate(detached) is True and detached.to_dict() == before
        except Exception:
            authenticated = False
    level = EvidenceLevel.UNRESOLVED
    reasons = [
        "Prediction custody, locked-data isolation and authentic measurements are not established."
    ]
    if authenticated and host is not None:
        import torch

        try:
            replay = host.backend.capture(
                source.hook,
                torch.tensor(
                    [i.token_ids for i in source.inputs],
                    dtype=torch.long,
                    device=host.backend._device,
                ),
                attention_mask=torch.tensor(
                    [i.attention_mask for i in source.inputs],
                    dtype=torch.long,
                    device=host.backend._device,
                ),
                component_indices=source.component_indices,
                token_positions=source.token_positions,
                example_ids=tuple(i.example_id for i in source.inputs),
                enabled=True,
            )
            matched = (
                replay == source and evaluate_explanation(commit, replay, enabled=True) == payload
            )
        except (ValueError, RuntimeError, TypeError):
            matched = False
        if matched:
            level = (
                EvidenceLevel.HELD_OUT_SUPPORTED
                if payload["prediction_result"] == "PREREGISTERED_CRITERION_MET"
                else EvidenceLevel.CONTRADICTED
            )
            reasons = [
                (
                    "The declared numeric activation-prediction criterion was falsified."
                    if level is EvidenceLevel.CONTRADICTED
                    else "Only the declared numeric activation-prediction criterion is supported."
                ),
                "No causal, semantic, baseline-superiority or alignment claim is established.",
                "Retrospective offset explanations are not necessarily causal-prefix predictors.",
            ]
    report_hash = content_digest(payload)
    key = content_digest(
        {
            "kind": "prediction",
            "model": commit.model_digest,
            "hook": commit.hook_digest,
            "component": commit.component_index,
            "hypothesis": commit.hypothesis.to_dict(),
        }
    )
    from gepa_mindfulness.core.evidence import EvidenceSourceKind

    refs = (
        EvidenceReference(
            "mechanistic-prediction:sha256:" + report_hash, EvidenceSourceKind.EXTERNAL_RECORD
        ),
    )
    return MechanisticFinding(
        content_digest(
            {
                "source": report_hash,
                "verifier": None if host is None else host.evaluator.evaluator_id,
            }
        ),
        key,
        level,
        (canonical_json(payload),),
        (report_hash,),
        (source.digest,),
        1,
        None if host is None else host.evaluator,
        refs,
        tuple(reasons),
        _timestamp(None if host is None else host.clock),
        "prediction",
    )


@dataclass(frozen=True)
class MechanisticRegistry(CVARecord):
    """Host-persisted append-only snapshot; not an authority-bearing deserialized handle."""

    registry_id: str
    revision: int
    entries: tuple[MechanisticFinding, ...]

    schema_version = "mechanistic-registry-v1"
    restorers = {"entries": lambda v: _read_records(v, MechanisticFinding)}

    def __post_init__(self) -> None:
        text(self.registry_id, "registry_id")
        if (
            type(self.revision) is not int
            or self.revision < 0
            or self.revision != len(self.entries)
        ):
            raise ValueError("registry revision must preserve every appended finding")
        entries = tuple(_record(e, MechanisticFinding) for e in array(self.entries, 128))
        if len({e.finding_id for e in entries}) != len(entries):
            raise ValueError("duplicate registry finding IDs")
        if any(e.level is EvidenceLevel.UNRESOLVED for e in entries):
            raise ValueError("unverified unresolved finding cannot enter the accepted registry")


def _proposed_registry(
    before: MechanisticRegistry, finding: MechanisticFinding
) -> MechanisticRegistry:
    state = _record(before, MechanisticRegistry)
    result = _record(finding, MechanisticFinding)
    return MechanisticRegistry(state.registry_id, state.revision + 1, state.entries + (result,))


def mechanistic_update_scope(before: MechanisticRegistry, finding: MechanisticFinding) -> str:
    after = _proposed_registry(before, finding)
    return "mechanistic-evidence-update:sha256:" + content_digest(
        {"before": before.to_dict(), "after": after.to_dict()}
    )


def admit_mechanistic_finding(
    before: MechanisticRegistry,
    finding: MechanisticFinding,
    *,
    context: MechanisticVerifierContext,
    action: ActionRecord,
    authorization: AuthorizationDecision,
    grant_registry: AuthorityGrantRegistry,
    clock: TrustedClock,
    enabled: bool = False,
) -> MechanisticRegistry:
    """Reverify all sources, consume existing exact WRITE authority, return new snapshot.

    Host must authenticate/pin the current snapshot and serialize persistence, just as for
    existing evidence-state updates. This function creates neither credentials nor reward.
    """
    _enabled(enabled)
    proposal = _record(finding, MechanisticFinding)
    if proposal.source_kind == "causal":
        refreshed = verify_causal_runs(
            tuple(json.loads(s) for s in proposal.source_reports_json),
            context=context,
            enabled=True,
        )
    else:
        if len(proposal.source_reports_json) != 1:
            raise PermissionError("prediction admission requires one complete held-out report")
        refreshed = verify_predictive_run(
            json.loads(proposal.source_reports_json[0]), context=context, enabled=True
        )
    if (
        refreshed.level is EvidenceLevel.UNRESOLVED
        or replace(refreshed, created_at=proposal.created_at) != proposal
    ):
        raise PermissionError("finding is not independently verified by the current trusted host")
    after = _proposed_registry(before, proposal)
    scope = mechanistic_update_scope(before, proposal)
    if (
        type(action) is not ActionRecord
        or action.authorization_scope != scope
        or action.action_class != "mechanistic_evidence_registration"
        or action.reversible is not True
    ):
        raise PermissionError("publication action must bind exact before/after registry snapshots")
    if (
        type(authorization) is not AuthorizationDecision
        or authorization.capability is not RuntimeCapability.WRITE
    ):
        raise PermissionError("mechanistic admission requires issued WRITE authority")
    consume_authorization(authorization, action, grant_registry=grant_registry, clock=clock)
    return after
