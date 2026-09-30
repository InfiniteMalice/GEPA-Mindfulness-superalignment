"""Reference-only composition of experimental diagnostics beside unchanged V5 records.

Attributions are concurrent observations, not a causal verdict. Hosts authenticate and bind
the referenced audit artifacts to the same episode; this module validates V5 event provenance.
No diagnostic result is an action grant, reward signal, training promotion, or repair receipt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from gepa_mindfulness.verification.formal_reasoning import (
    FormalAuditResult,
    FormalAuditStatus,
    PremiseGroundingStatus,
)
from mindful_trace_gepa.logging_schema import (
    RESEARCH_DIAGNOSTIC_EVENT_TYPES,
    EventEnvelope,
    StructuredEventType,
    make_event_envelope,
)
from semantic_intent_robustness._continuity_validation import (
    boolean,
    index_field,
    references,
    text_field,
)
from semantic_intent_robustness.evolutionary_atlas import SemanticStrategy
from semantic_intent_robustness.latent_language_transition import (
    LatentLanguageTransitionAssessment,
    TransitionStatus,
)

from .serialization_roundtrip import (
    EquivalenceResult,
    EquivalenceStatus,
    FaultStatus,
    RoundTripResult,
)
from .v5_provenance import validate_v5_record_provenance
from .v5_records import V5EvaluationRecord

_ATTRIBUTION_REFERENCES = {
    "MEANING_CHANGED": "semantic_equivalence_ref",
    "MEANING_PRESERVED_JUDGMENT_CHANGED": "semantic_equivalence_ref",
    "SERIALIZATION_FAULT": "communication_roundtrip_ref",
    "EXTRACTION_FAULT": "communication_roundtrip_ref",
    "COMMUNICATION_FAILURE_UNATTRIBUTED": "communication_roundtrip_ref",
    "INTERNAL_STATE_DRIFT": "latent_language_transition_ref",
    "LATENT_LANGUAGE_DECOUPLING": "latent_language_transition_ref",
    "STABLE_MEASURED_STATE_OUTPUT_DIVERGED": "latent_language_transition_ref",
    "INVALID_PUBLIC_INFERENCE": "formal_audit_ref",
    "PREMISES_UNSUPPORTED": "formal_audit_ref",
}


@dataclass(frozen=True, slots=True)
class ResearchAuditReferences:
    """Stable IDs for externally retained typed artifacts; no duplicate trace payloads."""

    semantic_equivalence_ref: str | None = None
    communication_roundtrip_ref: str | None = None
    formal_audit_ref: str | None = None
    latent_language_transition_ref: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None:
                text_field(value, name)


@dataclass(frozen=True, slots=True)
class ResearchAuditBundle:
    """An immutable external sidecar bound to the serialized V5 observation digest.

    Reference collections contain at most 128 unique IDs. Construction rejects
    mutable containers and unsupported findings. Typed artifact references must
    accompany their findings; the host still authenticates artifact contents.
    """

    evaluation_ref: str
    record_digest: str
    run_id: str
    trace_id: str
    model_version: str
    harness_version: str
    parent_event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    references: ResearchAuditReferences
    attributions: tuple[str, ...]
    strategy_id: str | None = None
    strategy_generation: int | None = None
    strategy_parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("evaluation_ref", "run_id", "trace_id", "model_version", "harness_version"):
            text_field(getattr(self, name), name)
        digest = self.record_digest
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("record_digest must be a lowercase SHA256 digest")
        if type(self.references) is not ResearchAuditReferences:
            raise ValueError("references must be a ResearchAuditReferences record")
        for name in ("parent_event_ids", "evidence_refs", "attributions", "strategy_parent_ids"):
            references(getattr(self, name), name)
        if not self.parent_event_ids or not self.evidence_refs or not self.attributions:
            raise ValueError("bundle requires parent events, evidence references, and attributions")
        for finding in self.attributions:
            if finding in {"UNKNOWN", "FINAL_OUTPUT_FAILURE"}:
                continue
            required = _ATTRIBUTION_REFERENCES.get(finding)
            if required is None or getattr(self.references, required) is None:
                raise ValueError("attribution must have a supported finding and matching audit ref")
        if "UNKNOWN" in self.attributions and any(
            finding in _ATTRIBUTION_REFERENCES for finding in self.attributions
        ):
            raise ValueError("UNKNOWN cannot accompany an attributed diagnostic finding")
        _validate_findings(self.attributions)
        if self.strategy_id is None:
            if self.strategy_generation is not None or self.strategy_parent_ids:
                raise ValueError("strategy metadata requires strategy_id")
        else:
            text_field(self.strategy_id, "strategy_id")
            index_field(self.strategy_generation, "strategy_generation")
            if self.strategy_id in self.strategy_parent_ids:
                raise ValueError("strategy cannot be its own parent")

    def to_event(self) -> EventEnvelope:
        """Log a deeply immutable reference bundle; raw evidence stays at its original IDs."""
        payload = asdict(self)
        payload.update(record_ref=self.evaluation_ref, provenance_refs=list(self.parent_event_ids))
        return make_event_envelope(
            StructuredEventType.RESEARCH_FAILURE_ATTRIBUTION,
            payload,
            run_id=self.run_id,
            trace_id=self.trace_id,
            model_version=self.model_version,
            harness_version=self.harness_version,
            parent_event_ids=self.parent_event_ids,
            evidence_refs=self.evidence_refs,
        )


def _validate_findings(findings: tuple[str, ...]) -> None:
    """Reject contradictory summaries of the single referenced artifact per diagnostic."""
    stage_fault = any(name in findings for name in ("SERIALIZATION_FAULT", "EXTRACTION_FAULT"))
    channel_failed = stage_fault or "COMMUNICATION_FAILURE_UNATTRIBUTED" in findings
    if "MEANING_PRESERVED_JUDGMENT_CHANGED" in findings and (
        "MEANING_CHANGED" in findings or channel_failed
    ):
        raise ValueError("preserved-meaning judgment attribution conflicts with failed semantics")
    if stage_fault and "COMMUNICATION_FAILURE_UNATTRIBUTED" in findings:
        raise ValueError("unattributed communication failure cannot have a stage attribution")
    drift = "INTERNAL_STATE_DRIFT" in findings
    if drift and "STABLE_MEASURED_STATE_OUTPUT_DIVERGED" in findings:
        raise ValueError("one transition cannot describe both stable and drifting measured state")
    if "LATENT_LANGUAGE_DECOUPLING" in findings and not drift:
        raise ValueError("latent-language decoupling requires observed internal-state drift")


def make_research_event(
    event_type: StructuredEventType,
    record_ref: str,
    *,
    run_id: str,
    trace_id: str,
    evaluation_ref: str,
    model_version: str,
    harness_version: str,
    parent_event_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    provenance_refs: tuple[str, ...],
) -> EventEnvelope:
    """Use existing envelopes with required research context and reference-only payloads."""
    if type(event_type) is not StructuredEventType or event_type.value not in (
        RESEARCH_DIAGNOSTIC_EVENT_TYPES
    ):
        raise ValueError("event_type must identify a research diagnostic")
    for name, value in (("record_ref", record_ref), ("evaluation_ref", evaluation_ref)):
        text_field(value, name)
    for name, values in (
        ("parent_event_ids", parent_event_ids),
        ("evidence_refs", evidence_refs),
        ("provenance_refs", provenance_refs),
    ):
        references(values, name)
    return make_event_envelope(
        event_type,
        {
            "record_ref": record_ref,
            "evaluation_ref": evaluation_ref,
            "provenance_refs": list(provenance_refs),
        },
        run_id=run_id,
        trace_id=trace_id,
        model_version=model_version,
        harness_version=harness_version,
        parent_event_ids=parent_event_ids,
        evidence_refs=evidence_refs,
    )


def bind_research_audits(
    record: V5EvaluationRecord,
    events: tuple[EventEnvelope, ...],
    *,
    evaluation_ref: str,
    trace_id: str,
    strategy: SemanticStrategy | None = None,
    semantic_equivalence: tuple[str, EquivalenceResult] | None = None,
    roundtrip: tuple[str, RoundTripResult] | None = None,
    formal: tuple[str, FormalAuditResult] | None = None,
    transition: tuple[str, LatentLanguageTransitionAssessment] | None = None,
    judgment_changed: bool = False,
) -> ResearchAuditBundle:
    """Bind independently supplied typed audits without altering V5 score or outcome fields.

    Host callers assert episode association for public audit artifacts and the judgment pair.
    Missing diagnostics yield UNKNOWN; observed downstream failure cannot infer its cause.
    """
    text_field(evaluation_ref, "evaluation_ref")
    text_field(trace_id, "trace_id")
    boolean(judgment_changed, "judgment_changed")
    proof = validate_v5_record_provenance(record, events)
    record = proof.record_snapshot()
    identifiers: list[str | None] = []
    for item, expected in (
        (semantic_equivalence, EquivalenceResult),
        (roundtrip, RoundTripResult),
        (formal, FormalAuditResult),
        (transition, LatentLanguageTransitionAssessment),
    ):
        if item is None:
            identifiers.append(None)
        else:
            if type(item) is not tuple or len(item) != 2 or type(item[1]) is not expected:
                raise ValueError("audit reference must bind an exact typed audit record")
            text_field(item[0], "audit reference")
            identifiers.append(item[0])
    if strategy is not None:
        review = record.assessment
        if (
            type(strategy) is not SemanticStrategy
            or review is None
            or (
                strategy.strategy_id != review.variant_id
                or strategy.transformation_lineage != review.transformation_lineage
                or strategy.semantic_intent_id != review.semantic_intent
                or strategy.target_case_id != record.case.case_id
                or strategy.target_stripe_id != record.robustness.stripe_id
                or strategy.target_subtype != record.robustness.subtype
            )
        ):
            raise ValueError("strategy must match the V5 observation and lineage")
    findings: list[str] = []
    evidence = list(record.epistemics.evidence_refs + record.outcome.observation_refs)
    channel_failed = False
    if roundtrip is not None:
        audit = roundtrip[1]
        evidence.extend(audit.evidence_refs)
        if audit.serialization_fault is FaultStatus.FAULT:
            findings.append("SERIALIZATION_FAULT")
        if audit.extraction_fault is FaultStatus.FAULT:
            findings.append("EXTRACTION_FAULT")
        channel_failed = audit.roundtrip_failure is True or bool(findings)
        if channel_failed and not findings:
            findings.append("COMMUNICATION_FAILURE_UNATTRIBUTED")
    if semantic_equivalence is not None:
        equivalence = semantic_equivalence[1]
        evidence.extend(equivalence.evidence_refs)
        if equivalence.status is EquivalenceStatus.NOT_EQUIVALENT:
            findings.append("MEANING_CHANGED")
        elif equivalence.semantics_preserved and judgment_changed and not channel_failed:
            findings.append("MEANING_PRESERVED_JUDGMENT_CHANGED")
    if transition is not None:
        latent = transition[1]
        evidence.extend(latent.provenance)
        for state in (latent.before, latent.after):
            if state is not None:
                evidence.extend(state.provenance)
        for public in (latent.output, latent.action):
            evidence.extend(public.endpoint_refs + public.provenance)
        if latent.status is TransitionStatus.LATENT_LANGUAGE_DECOUPLING:
            findings.extend(("INTERNAL_STATE_DRIFT", "LATENT_LANGUAGE_DECOUPLING"))
        elif latent.status is TransitionStatus.LANGUAGE_CHANGE_WITHOUT_MATCHED_LATENT_SIGNAL:
            findings.append("STABLE_MEASURED_STATE_OUTPUT_DIVERGED")
        elif latent.status is TransitionStatus.CO_CHANGE_OBSERVED:
            findings.append("INTERNAL_STATE_DRIFT")
    if formal is not None:
        logic = formal[1]
        evidence.append(logic.claim.public_reasoning_ref.reference_id)
        evidence.extend(
            ref.reference_id
            for ref in (
                logic.claim.evidence_refs + logic.claim.provenance_refs + logic.verifier_refs
            )
        )
        for grounding in logic.premise_grounding:
            evidence.extend(
                ref.reference_id for ref in grounding.evidence_refs + grounding.verifier_refs
            )
        if logic.status is FormalAuditStatus.INVALID:
            findings.append("INVALID_PUBLIC_INFERENCE")
        if logic.premise_grounding_status is PremiseGroundingStatus.UNSUPPORTED:
            findings.append("PREMISES_UNSUPPORTED")
    unknown = not findings
    if not record.outcome.passed:
        findings.append("FINAL_OUTPUT_FAILURE")
    if unknown:
        findings.append("UNKNOWN")
    digest = sha256(
        json.dumps(record.to_dict(), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return ResearchAuditBundle(
        evaluation_ref,
        digest,
        proof.run_id,
        trace_id,
        record.system.model_version,
        record.system.harness_version,
        proof.event_ids,
        tuple(dict.fromkeys(evidence)),
        ResearchAuditReferences(*identifiers),
        tuple(findings),
        None if strategy is None else strategy.strategy_id,
        None if strategy is None else strategy.generation,
        () if strategy is None else strategy.parent_ids,
    )
