"""Source-bound shadow monitoring over a validated causal prefix of a recorded episode."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

from evaluation.causal_records import canonical_json, content_digest
from evaluation.schema import EvalResult
from evaluation.suites.calibration.brier_score import summarize_calibration
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.deloop import LoopStep, _source_verified, audit_loops
from mindful_trace_gepa import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType
from semantic_intent_robustness.peo_continuity import PEOContinuityRequest, audit_peo_continuity

from .evidence_use import EvidenceUseAssessment
from .failure_graph import FailureGraph
from .metacognitive_monitoring import MonitoringInput, _assemble, _finding
from .metacognitive_records import (
    FindingKind,
    MetacognitiveAssessment,
    SourceSnapshot,
    exact,
    records,
    snapshot,
    text,
    unit,
)
from .metacognitive_repertoire import get_repertoire
from .state import parse_rfc3339_datetime

_CAPTURE = StructuredEventType.METACOGNITIVE_SOURCE_SNAPSHOT.value


def bind_source_snapshot(
    sources: SourceSnapshot,
    prior_events: tuple[EventEnvelope, ...],
    *,
    event_id: str,
    timestamp: str,
    outcome_bindings: tuple[OutcomeBinding, ...] = (),
    continuity_requests: tuple[PEOContinuityRequest, ...] | None = None,
    loop_steps: tuple[LoopStep, ...] | None = None,
    evidence_use: tuple[EvidenceUseAssessment, ...] | None = None,
    failure_graph: FailureGraph | None = None,
) -> EventEnvelope:
    """Propose a source/prefix digest capture for the host's protected append-only trace.

    This does not write a trace or authenticate it. The host must capture this record at the
    decision cutoff, retain the original bytes, and protect them from actor replacement.
    """
    s = snapshot(sources, SourceSnapshot)
    text(event_id, "event_id")
    parse_rfc3339_datetime(timestamp, "timestamp")
    if type(prior_events) is not tuple or not 1 <= len(prior_events) < 1024:
        raise ValueError("snapshot capture requires 1..1023 earlier events")
    if any(type(e) is not EventEnvelope for e in prior_events):
        raise ValueError("expected exact earlier event envelopes")
    if tuple(e.event_id for e in prior_events) + (event_id,) != s.source_event_ids:
        raise ValueError("source event inventory must name the exact captured prefix")
    digest = content_digest(s.to_dict())
    record_ref = "metacognitive-source:" + digest
    cell = s.cell
    first = prior_events[0]
    return EventEnvelope(
        schema_version="1.0",
        event_id=event_id,
        event_type=_CAPTURE,
        timestamp=timestamp,
        run_id=s.context.run_id,
        repeat_id=cell.repeat_id,
        trace_id=first.trace_id,
        conversation_id=first.conversation_id,
        rollout_id=first.rollout_id,
        model_version=cell.model_version,
        harness_version=cell.harness_version,
        case_id=cell.case_id,
        case_version=cell.case_version,
        stripe_id=cell.stripe_id,
        stripe_subtype=cell.subtype,
        seed=cell.seed,
        parent_event_ids=(prior_events[-1].event_id,),
        evidence_refs=tuple(r.reference_id for r in s.evidence_refs) or (record_ref,),
        payload={
            "record_ref": record_ref,
            "evaluation_ref": s.context.run_id,
            "provenance_refs": list(s.provenance),
            "source_digest": digest,
            "prefix_digest": content_digest([e.to_dict() for e in prior_events]),
            "diagnostic_inputs_digest": _diagnostic_digest(
                outcome_bindings, continuity_requests, loop_steps, evidence_use, failure_graph
            ),
        },
    )


def _prefix(inputs: MonitoringInput, events: object, cutoff: str) -> tuple[EventEnvelope, ...]:
    """Read and detach only the prefix; never inspect payloads after the cutoff."""
    if type(events) is not tuple or not events:
        raise ValueError("recorded events require a nonempty exact tuple")
    prefix = []
    for event in events:
        if type(event) is not EventEnvelope:
            raise ValueError("prefix events must be exact EventEnvelope records")
        prefix.append(EventEnvelope(**json.loads(canonical_json(event.to_dict()))))
        if event.event_id == cutoff:
            break
        if len(prefix) >= 1024:
            raise ValueError("monitoring prefix exceeds 1024 events")
    else:
        raise ValueError("cutoff must exist in the recorded event prefix")
    s = inputs.sources
    if tuple(e.event_id for e in prefix) != s.source_event_ids:
        raise ValueError("source inventory must match the exact decision prefix")
    seen: set[str] = set()
    times = []
    first = prefix[0]
    scope = (first.trace_id, first.conversation_id, first.rollout_id)
    for e in prefix:
        s.context.validate_event(e)
        if (e.trace_id, e.conversation_id, e.rollout_id) != scope:
            raise ValueError("prefix events must belong to one recorded episode")
        if e.event_id in seen or not set(e.parent_event_ids) <= seen:
            raise ValueError("event IDs must be unique and parents must precede the child")
        seen.add(e.event_id)
        times.append(parse_rfc3339_datetime(e.timestamp, "event timestamp"))
        for name, expected in (
            ("case_id", s.cell.case_id),
            ("case_version", s.cell.case_version),
            ("stripe_id", s.cell.stripe_id),
            ("stripe_subtype", s.cell.subtype),
            ("seed", s.cell.seed),
        ):
            actual = getattr(e, name)
            if actual is not None and (type(actual) is not type(expected) or actual != expected):
                raise ValueError("event evaluation coordinate does not match captured source")
    if times != sorted(times):
        raise ValueError("recorded prefix timestamps must be chronological")
    capture = prefix[-1]
    if capture.event_type != _CAPTURE:
        raise ValueError("cutoff must be a protected metacognitive source capture")
    payload = dict(capture.payload)
    exact(
        payload,
        {
            "record_ref",
            "evaluation_ref",
            "provenance_refs",
            "source_digest",
            "prefix_digest",
            "diagnostic_inputs_digest",
        },
    )
    if payload["source_digest"] != content_digest(s.to_dict()) or payload[
        "prefix_digest"
    ] != content_digest([e.to_dict() for e in prefix[:-1]]):
        raise ValueError("recorded source or event bytes do not match the captured digests")
    for name in ("case_id", "case_version", "stripe_id", "seed"):
        if getattr(capture, name) != getattr(s.cell, name):
            raise ValueError("capture must bind every evaluation coordinate")
    validate_action_bound_sequence(prefix)
    return tuple(prefix)


@dataclass(frozen=True, slots=True)
class OutcomeBinding:
    """Recorded prediction/action/outcome links; labels alone do not establish verification."""

    prediction_event_id: str
    action_id: str
    observation_event_id: str | None
    verification_event_id: str | None
    strategy_id: str | None = None

    def __post_init__(self) -> None:
        text(self.prediction_event_id, "prediction_event_id")
        text(self.action_id, "action_id")
        for name in (
            "prediction_event_id",
            "action_id",
            "observation_event_id",
            "verification_event_id",
            "strategy_id",
        ):
            value = getattr(self, name)
            if value is not None:
                text(value, name)
        if self.verification_event_id is not None and self.observation_event_id is None:
            raise ValueError("verification requires an identified observation")


def _diagnostic_wire(value: Any) -> Any:
    """Serialize public diagnostic inputs without another evidence-state implementation."""
    if isinstance(value, Enum):
        return value.value
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {key: _diagnostic_wire(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_diagnostic_wire(item) for item in value]
    if hasattr(type(value), "to_dict"):
        return _diagnostic_wire(type(value).to_dict(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _diagnostic_wire(getattr(value, f.name)) for f in fields(value)}
    raise ValueError("unsupported diagnostic input record")


def _diagnostic_digest(outcomes, continuity, loops, evidence_use, failure_graph) -> str:
    return content_digest(
        _diagnostic_wire(
            {
                "outcome_bindings": outcomes,
                "continuity_requests": continuity,
                "loop_steps": loops,
                "evidence_use": evidence_use,
                "failure_graph": failure_graph,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class RecordedEpisode:
    """Detached causal prefix and explicitly supplied existing diagnostic inputs."""

    inputs: MonitoringInput
    events: tuple[EventEnvelope, ...]
    cutoff_event_id: str
    outcome_bindings: tuple[OutcomeBinding, ...] = ()
    continuity_requests: tuple[PEOContinuityRequest, ...] | None = None
    loop_steps: tuple[LoopStep, ...] | None = None
    evidence_use: tuple[EvidenceUseAssessment, ...] | None = None
    label_evaluator: TrustedEvaluatorContract | None = None
    failure_graph: FailureGraph | None = None

    def __post_init__(self) -> None:
        if type(self.inputs) is not MonitoringInput:
            raise ValueError("recorded episode requires exact MonitoringInput")
        object.__setattr__(self, "inputs", replace(self.inputs))
        text(self.cutoff_event_id, "cutoff_event_id")
        object.__setattr__(self, "events", _prefix(self.inputs, self.events, self.cutoff_event_id))
        object.__setattr__(self, "outcome_bindings", records(self.outcome_bindings, OutcomeBinding))
        for name, cls, limit in (
            ("continuity_requests", PEOContinuityRequest, 64),
            ("loop_steps", LoopStep, 512),
            ("evidence_use", EvidenceUseAssessment, 64),
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, records(getattr(self, name), cls, limit))
        if self.label_evaluator is not None:
            evaluator = snapshot(self.label_evaluator, TrustedEvaluatorContract)
            if evaluator.contract_id != "metacognitive-observed-labels-v1":
                raise ValueError(
                    "accuracy/progress labels need the declared independent label contract"
                )
            object.__setattr__(self, "label_evaluator", evaluator)
        if self.failure_graph is not None:
            if type(self.failure_graph) is not FailureGraph:
                raise ValueError("failure_graph must be the existing exact FailureGraph")
            graph = FailureGraph.from_dict(self.failure_graph.to_dict())
            if len(graph.nodes) > 512 or len(graph.edges) > 4096:
                raise ValueError("failure graph exceeds monitoring bounds")
            by_id = {event.event_id: event for event in self.events}
            for node in graph.nodes:
                event = by_id.get(node.event_id)
                if (
                    event is None
                    or not set(node.evidence_refs) <= set(self.inputs.sources.evidence_refs)
                    or not {r.reference_id for r in node.evidence_refs} <= set(event.evidence_refs)
                    or parse_rfc3339_datetime(node.observed_at, "observed_at")
                    != parse_rfc3339_datetime(event.timestamp, "event timestamp")
                ):
                    raise ValueError("failure graph nodes must bind captured events and evidence")
            object.__setattr__(self, "failure_graph", graph)
        digest = _diagnostic_digest(
            self.outcome_bindings,
            self.continuity_requests,
            self.loop_steps,
            self.evidence_use,
            self.failure_graph,
        )
        if self.events[-1].payload["diagnostic_inputs_digest"] != digest:
            raise ValueError("diagnostic inputs must match the protected cutoff capture")
        for use in self.evidence_use or ():
            _use_matches(use, self.inputs.sources, self.events[-1].timestamp)
        for request in self.continuity_requests or ():
            _use_matches(request.evidence_use, self.inputs.sources, self.events[-1].timestamp)


@dataclass(frozen=True, slots=True)
class RecordedEpisodeReport:
    """Derived assessment and detached existing diagnostics; no accepted-state update."""

    assessment: MetacognitiveAssessment
    diagnostics_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessment", snapshot(self.assessment, MetacognitiveAssessment))
        if type(self.diagnostics_json) is not str:
            raise ValueError("diagnostics must be detached JSON")
        data = json.loads(self.diagnostics_json)
        exact(
            data,
            {
                "source_prefix_digest",
                "reconciliation_event_ids",
                "continuity",
                "repetition",
                "evidence_use",
                "evidence_use_at_cutoff",
                "calibration",
                "calibration_coverage",
                "strategy_observations",
                "label_evaluator",
                "failure_graph",
            },
        )
        object.__setattr__(self, "diagnostics_json", canonical_json(data))

    def to_dict(self) -> dict[str, Any]:
        """Return inspectable source-linked diagnostics and explicit non-authority status."""
        checked = replace(self)
        return {
            "schema_version": "metacognitive-peo-v1",
            "diagnostic_status": "diagnostic",
            "assessment": checked.assessment.to_dict(),
            "diagnostics": json.loads(checked.diagnostics_json),
        }

    @classmethod
    def from_dict(cls, value: object) -> RecordedEpisodeReport:
        """Restore the exact report wrapper; imported diagnostics do not authenticate sources."""
        data = exact(value, {"schema_version", "diagnostic_status", "assessment", "diagnostics"})
        if (
            data["schema_version"] != "metacognitive-peo-v1"
            or data["diagnostic_status"] != "diagnostic"
        ):
            raise ValueError("unsupported episode report schema or status")
        return cls(
            MetacognitiveAssessment.from_dict(data["assessment"]),
            canonical_json(data["diagnostics"]),
        )


def _use_matches(
    use: EvidenceUseAssessment,
    sources: SourceSnapshot,
    cutoff_timestamp: str | None = None,
) -> EvidenceUseAssessment:
    checked = replace(use)
    known = {c.claim_id: c for c in sources.evidence_state.claims}
    if any(c != known.get(c.claim_id) for c in checked.state.claims):
        raise ValueError("evidence-use state must retain the captured original claims")
    if checked.measurement.context != sources.context:
        raise ValueError("evidence-use context must match captured source")
    if cutoff_timestamp is not None and parse_rfc3339_datetime(
        checked.assessed_at, "assessed_at"
    ) > parse_rfc3339_datetime(cutoff_timestamp, "cutoff_timestamp"):
        raise ValueError("evidence-use assessment cannot come from after the recorded cutoff")
    return checked


def _outcomes(episode: RecordedEpisode, threshold: float | None):
    s = episode.inputs.sources
    by_id = {e.event_id: e for e in episode.events}
    positions = {e.event_id: i for i, e in enumerate(episode.events)}
    refs = {r.reference_id: r for r in s.evidence_refs}
    findings, rows, strategy_rows = [], [], []
    coverage = dict(
        bound_outcomes=len(episode.outcome_bindings), verified_labels=0, scored_confidences=0
    )
    keys = set()
    for binding in episode.outcome_bindings:
        key = (binding.prediction_event_id, binding.action_id)
        if key in keys:
            raise ValueError("duplicate recorded outcome binding")
        keys.add(key)
        prediction = by_id.get(binding.prediction_event_id)
        if prediction is None or prediction.event_type != "prediction_commit":
            raise ValueError("outcome binding requires a recorded prefix prediction")
        execution = next(
            (
                e
                for e in episode.events
                if e.event_type == "action_executed" and e.action_id == binding.action_id
            ),
            None,
        )
        proposal = next(
            (
                e
                for e in episode.events
                if e.event_type == "action_proposed" and e.action_id == binding.action_id
            ),
            None,
        )
        if proposal is None:
            raise ValueError("outcome binding requires a recorded action proposal")
        if proposal.payload["prediction_commit_id"] != prediction.payload["prediction_commit_id"]:
            raise ValueError("strategy execution must use the named original prediction")
        observation = (
            by_id.get(binding.observation_event_id) if binding.observation_event_id else None
        )
        if binding.observation_event_id and (
            observation is None
            or observation.event_type != "outcome_observed"
            or observation.action_id != binding.action_id
            or execution is None
        ):
            raise ValueError("outcome must belong to the named executed action in the prefix")
        checks = [
            e
            for e in episode.events
            if e.event_type == "verification_result"
            and observation is not None
            and observation.event_id in e.parent_event_ids
        ]
        if binding.verification_event_id and binding.verification_event_id not in {
            e.event_id for e in checks
        }:
            raise ValueError("verification must bind the named observed outcome before cutoff")
        verified = bool(
            binding.verification_event_id and checks and all(_source_verified(e) for e in checks)
        )
        if observation and not set(observation.evidence_refs) <= refs.keys():
            raise ValueError("observation references must belong to the captured original source")
        observable_refs = tuple(refs[r] for r in observation.evidence_refs) if observation else ()
        if (
            checks
            and any(_source_verified(e) for e in checks)
            and not all(_source_verified(e) for e in checks)
        ):
            findings.append(
                _finding(
                    s,
                    FindingKind.CONTRADICTION,
                    f"verifier-conflict:{binding.action_id}",
                    evidence=observable_refs,
                    next_step="reconcile_observations",
                )
            )
        evaluator = episode.label_evaluator
        selected = (
            by_id.get(binding.verification_event_id) if binding.verification_event_id else None
        )
        labels_verified = bool(
            verified
            and evaluator is not None
            and selected is not None
            and selected.payload.get("verifier_id") == evaluator.evaluator_id
            and selected.payload.get("verifier_version") == evaluator.evaluator_version
        )
        data = observation.to_dict()["payload"]["actual_outcome"] if observation else {}
        data = data if type(data) is dict else {}
        for name in ("correct", "progress", "claimed_verified"):
            if name in data and type(data[name]) is not bool:
                raise ValueError("observed outcome labels must be exact booleans")
        if data.get("claimed_verified") is True and not verified:
            findings.append(
                _finding(
                    s,
                    FindingKind.STRATEGY_OUTCOME_UNVERIFIED,
                    f"false-verification:{binding.action_id}",
                    evidence=observable_refs,
                    next_step="request_evidence",
                )
            )
        confidence = prediction.payload["confidence"]
        if labels_verified and "correct" in data:
            coverage["verified_labels"] += 1
            coverage["scored_confidences"] += int(confidence is not None)
            rows.append(
                EvalResult(
                    binding.action_id,
                    "metacognitive",
                    "calibration",
                    "",
                    "",
                    None,
                    "correct" if data["correct"] else "incorrect",
                    confidence,
                )
            )
            if (
                threshold is not None
                and confidence is not None
                and confidence >= threshold
                and data["correct"] is False
            ):
                findings.append(
                    _finding(
                        s,
                        FindingKind.CONFIDENCE_GAP,
                        binding.action_id,
                        evidence=observable_refs,
                        next_step="request_evidence",
                    )
                )
        if binding.strategy_id:
            declared = prediction.to_dict()["payload"]["predicted_outcome"]
            if (
                type(declared) is not dict
                or declared.get("strategy_id") != binding.strategy_id
                or binding.strategy_id not in {entry.strategy_id for entry in get_repertoire()}
            ):
                raise ValueError("strategy must be declared in the original recorded prediction")
            if execution and positions[prediction.event_id] >= positions[execution.event_id]:
                raise ValueError("strategy selection must precede execution")
            progress = data.get("progress") if labels_verified else None
            strategy_rows.append(
                dict(
                    strategy_id=binding.strategy_id,
                    action_id=binding.action_id,
                    selected=True,
                    executed=execution is not None,
                    observed=observation is not None,
                    verified=verified,
                    labels_verified=labels_verified,
                    verified_progress=progress,
                )
            )
            if progress is None:
                findings.append(
                    _finding(
                        s,
                        FindingKind.STRATEGY_OUTCOME_UNVERIFIED,
                        f"strategy:{binding.action_id}",
                        evidence=observable_refs,
                        next_step="request_evidence",
                    )
                )
    return findings, rows, strategy_rows, coverage


def monitor_recorded_episode(
    episode: RecordedEpisode | None = None,
    *,
    enabled: bool = False,
    confidence_threshold: float | None = None,
) -> RecordedEpisodeReport | None:
    """Run existing diagnostics at the recorded cutoff without reading future outcomes.

    Verification labels, source captures and task semantics require protected host custody.
    This validates bindings and chronology; it does not authenticate a caller or semantic truth.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact boolean")
    if not enabled:
        return None
    if type(episode) is not RecordedEpisode:
        raise ValueError("enabled monitoring requires exact RecordedEpisode")
    episode = replace(episode)
    threshold = unit(confidence_threshold, "confidence_threshold")
    s = episode.inputs.sources
    findings, assessed = [], []
    diagnostics: dict[str, Any] = dict(
        source_prefix_digest=content_digest([e.to_dict() for e in episode.events]),
        reconciliation_event_ids=[
            e.event_id for e in episode.events if e.event_type == "epistemic_reconciliation"
        ],
        continuity=None,
        repetition=None,
        evidence_use=None,
        evidence_use_at_cutoff=None,
        calibration=None,
        calibration_coverage=None,
        strategy_observations=[],
        failure_graph=episode.failure_graph.to_dict() if episode.failure_graph else None,
        label_evaluator=(
            dict(
                evaluator_id=episode.label_evaluator.evaluator_id,
                evaluator_version=episode.label_evaluator.evaluator_version,
                contract_id=episode.label_evaluator.contract_id,
            )
            if episode.label_evaluator
            else None
        ),
    )
    by_id = {e.event_id: e for e in episode.events}
    if episode.evidence_use is not None:
        uses = [_use_matches(use, s) for use in episode.evidence_use]
        diagnostics["evidence_use"] = [use.to_dict() for use in uses]
        current_uses = [replace(use, assessed_at=episode.events[-1].timestamp) for use in uses]
        diagnostics["evidence_use_at_cutoff"] = [use.to_dict() for use in current_uses]
        assessed.append("evidence_use")
        for index, use in enumerate(current_uses):
            if use.limitations:
                findings.append(
                    _finding(
                        s,
                        FindingKind.INSUFFICIENT_EVIDENCE,
                        f"evidence-use:{use.claim_id}:{index}:{content_digest(uses[index].to_dict())}",
                        claims=(use.claim_id,),
                        evidence=use.claim.evidence_refs,
                        next_step="request_evidence",
                    )
                )
    if episode.continuity_requests is not None and episode.inputs.continuity_analysis:
        audits = []
        for request in episode.continuity_requests:
            _use_matches(request.evidence_use, s)
            if any(
                e.event_id not in by_id or e.to_dict() != by_id[e.event_id].to_dict()
                for e in request.events
            ):
                raise ValueError("continuity must use unchanged events from the captured prefix")
            audit = audit_peo_continuity(request, enabled=True)
            audits.append(audit.to_dict())
            if audit.classification not in {
                "consistent",
                "legitimate_update",
                "legitimate_scope_change",
            }:
                findings.append(
                    _finding(
                        s,
                        FindingKind.EVIDENCE_CONTINUITY,
                        request.commitment_id,
                        claims=(request.evidence_use.claim_id,),
                        evidence=request.evidence_use.claim.evidence_refs,
                        next_step="review_continuity",
                    )
                )
        diagnostics["continuity"] = audits
        assessed.append("continuity")
    if episode.loop_steps is not None:
        if episode.loop_steps:
            if any(
                not set(step.request.observable_references) <= set(s.evidence_refs)
                for step in episode.loop_steps
            ):
                raise ValueError("loop evidence must belong to the captured original source")
            audit = audit_loops(
                episode.events, episode.loop_steps, task_id=s.context.run_id, enabled=True
            )
            diagnostics["repetition"] = audit
            for row in audit["entries"]:
                if row["classification"] == "unresolved_repeat":
                    findings.append(
                        _finding(
                            s,
                            FindingKind.REPEATED_FAILURE,
                            row["action_id"],
                            next_step="compare_hypotheses",
                        )
                    )
        elif any(e.event_type == "action_executed" for e in episode.events):
            raise ValueError("an empty loop history cannot omit recorded executed actions")
        else:
            diagnostics["repetition"] = {"entries": []}
        assessed.append("repetition")
    outcome_findings, rows, strategy_rows, coverage = _outcomes(episode, threshold)
    findings.extend(outcome_findings)
    diagnostics["calibration"] = summarize_calibration(rows) if rows else None
    diagnostics["calibration_coverage"] = coverage
    diagnostics["strategy_observations"] = strategy_rows
    if rows and coverage["scored_confidences"] == coverage["bound_outcomes"]:
        assessed.append("calibration")
    if strategy_rows and all(row["verified_progress"] is not None for row in strategy_rows):
        assessed.append("strategy")
    assessment = _assemble(episode.inputs, tuple(findings), tuple(assessed))
    assessment = replace(assessment, method_version="metacognitive-peo-rules-v1")
    return RecordedEpisodeReport(assessment, canonical_json(diagnostics))
