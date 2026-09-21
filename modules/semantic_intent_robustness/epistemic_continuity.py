"""Audit public evidence retention and recall through the existing memory boundary.

SoT motivates state-conditioned support; this omission detector is a repository
hypothesis. It neither reproduces SoT nor infers motive from geometry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
)
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

from ._continuity_validation import boolean, index_field, references, text_field
from .epistemic_records import CommitmentStatus, CommitmentUpdate, EpistemicCommitment
from .internal_state_trajectory import SoTStateSnapshot
from .memory_safety import (
    MemoryRetrievalAssessment,
    MemoryRetrievalDecision,
    assess_retrieved_memory,
)
from .semantic_state_continuity import state_distance


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """Validated event prefix and proposal identity for one continuity decision."""

    events: tuple[EventEnvelope, ...]
    decision: EventEnvelope

    @property
    def digest(self) -> str:
        """Bind diagnostics to the immutable action-bound prefix, including decision content."""
        payload = [e.to_dict() for e in (*self.events, self.decision)]
        return _digest(payload)

    @classmethod
    def build(cls, events: tuple[EventEnvelope, ...], decision_event_id: str) -> EvidenceWindow:
        """Reuse action-bound validation, then exclude events after the current proposal."""
        if type(events) is not tuple or len(events) > 10_000:
            raise ValueError("events must be a tuple of at most 10000 envelopes")
        validate_action_bound_sequence(events)
        matches = [i for i, e in enumerate(events) if e.event_id == decision_event_id]
        if len(matches) != 1:
            raise ValueError("decision_event_id must identify one event")
        decision = events[matches[0]]
        if decision.event_type != "action_proposed":
            raise ValueError("continuity must be assessed at an action_proposed event")
        text_field(decision.conversation_id, "decision.conversation_id")
        index_field(decision.checkpoint_step, "decision.checkpoint_step")
        return cls(events[: matches[0]], decision)

    def bound_sources(
        self,
        refs: tuple[EvidenceReference, ...],
        event_ids: tuple[str, ...],
        *,
        verified: bool = False,
        required_finding: tuple[str, str | bool] | None = None,
    ) -> tuple[EventEnvelope, ...]:
        """Resolve declared public references to earlier same-unit source events.

        A required finding must bind the cited references in a typed verification result.
        Authenticating the verifier and interpreting the public claim remain host duties.
        """
        if not refs or not event_ids:
            return ()
        by_id = {event.event_id: event for event in self.events}
        sources = tuple(by_id[key] for key in event_ids if key in by_id)
        if len(sources) != len(event_ids):
            return ()
        available: set[EvidenceReference] = set()
        assert self.decision.checkpoint_step is not None
        for event in sources:
            if (
                event.run_id != self.decision.run_id
                or event.repeat_id != self.decision.repeat_id
                or event.conversation_id != self.decision.conversation_id
                or type(event.checkpoint_step) is not int
                or event.checkpoint_step < 0
                or event.checkpoint_step > self.decision.checkpoint_step
                or event.event_type
                not in {
                    "prediction_commit",
                    "outcome_observed",
                    "verification_result",
                    "epistemic_assessment",
                }
            ):
                return ()
            captured = _typed_evidence(
                event,
                verified=verified,
                required_finding=required_finding,
            )
            if captured is not None:
                available.update(captured)
            else:
                if required_finding is not None:
                    return ()
                if verified and (
                    event.event_type != "verification_result"
                    or event.payload.get("verified") is not True
                    or not event.verifier_refs
                ):
                    return ()
                ids = set(event.evidence_refs + event.verifier_refs)
                available.update(ref for ref in refs if ref.reference_id in ids)
        # Typed source declarations anywhere in the prefix cannot be overwritten by a
        # later untyped string reference or a caller's relabeled commitment.
        for event in self.events:
            captured = _typed_evidence(event, verified=False)
            if captured is not None:
                for ref in refs:
                    if any(r.reference_id == ref.reference_id and r != ref for r in captured):
                        return ()
        if not all(ref.is_observable and ref in available for ref in refs):
            return ()
        return sources


def _typed_evidence(
    event: EventEnvelope,
    *,
    verified: bool,
    required_finding: tuple[str, str | bool] | None = None,
) -> tuple[EvidenceReference, ...] | None:
    """Select captured sources bound to the required finding or an affirmative finding."""
    if event.event_type != "verification_result":
        return None
    result: LocalVerificationResult | RelationalVerificationResult
    level = event.payload.get("verification_level")
    if level == "local_execution":
        result = LocalVerificationResult.from_dict(event.payload["result"])
    elif level == "relational_evidence":
        result = RelationalVerificationResult.from_dict(event.payload["result"])
    else:
        return None
    if required_finding is not None:
        field, expected = required_finding
        if getattr(result, field, None) != expected:
            return ()
        return tuple(
            ref
            for binding in result.evidence_bindings
            if binding.field_name == field
            for ref in binding.evidence_refs
        )
    if not verified:
        return result.evidence_refs
    refs: list[EvidenceReference] = []
    for binding in result.evidence_bindings:
        finding = getattr(result, binding.field_name)
        affirmative = finding is True or (
            binding.field_name == "contradiction_status" and finding in {"none", "contradicted"}
        )
        if affirmative:
            refs.extend(binding.evidence_refs)
    return tuple(refs)


def _digest(payload: object) -> str:
    """Hash the canonical JSON representation of an assessed public input."""
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EpistemicContinuityAssessment:
    """Observed omission before recall, with explicit accepted updates and quarantines."""

    assessment_id: str
    decision_event_id: str
    conversation_id: str
    prior_commitment_ids: tuple[str, ...]
    currently_relevant_commitment_ids: tuple[str, ...]
    retained_ids: tuple[str, ...]
    explicitly_superseded_ids: tuple[str, ...]
    legitimately_scoped_out_ids: tuple[str, ...]
    unexplained_omission_ids: tuple[str, ...]
    quarantined_ids: tuple[str, ...]
    invalid_update_ids: tuple[str, ...]
    reactivated_ids: tuple[str, ...]
    new_evidence_refs: tuple[str, ...]
    current_state_snapshot_id: str | None
    prior_state_snapshot_ids: tuple[str, ...]
    decision_context_changed: bool
    continuity_status: str
    review_required: bool
    provenance: tuple[str, ...]
    commitment_digests: tuple[tuple[str, str], ...]
    event_window_digest: str
    current_state_digest: str | None
    prior_state_digests: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize auditable public assessment fields."""
        return asdict(self)


def assess_epistemic_continuity(
    *,
    assessment_id: str,
    commitments: tuple[EpistemicCommitment, ...],
    events: tuple[EventEnvelope, ...],
    decision_event_id: str,
    active_commitment_ids: tuple[str, ...],
    updates: tuple[CommitmentUpdate, ...] = (),
    relevant_commitment_ids: tuple[str, ...] = (),
    ignored_commitment_ids: tuple[str, ...] = (),
    current_state: SoTStateSnapshot | None = None,
    prior_states: tuple[SoTStateSnapshot, ...] = (),
    decision_context_changed: bool = False,
    provenance: tuple[str, ...],
) -> EpistemicContinuityAssessment:
    """Require explicit supported transitions before deactivating relevant evidence.

    Active/ignored IDs are public observations supplied by the harness; this function
    does not inspect private reasoning or infer effective use from mere text presence.
    Relevant IDs may add relevance but cannot silently remove prior relevance.
    """
    text_field(assessment_id, "assessment_id")
    references(provenance, "provenance")
    boolean(decision_context_changed, "decision_context_changed")
    if not provenance:
        raise ValueError("assessment provenance is required")
    window = EvidenceWindow.build(events, decision_event_id)
    by_id = _commitment_map(commitments)
    prior_digests = _prior_state_digests(prior_states)
    for name, ids in (
        ("active_commitment_ids", active_commitment_ids),
        ("relevant_commitment_ids", relevant_commitment_ids),
        ("ignored_commitment_ids", ignored_commitment_ids),
    ):
        references(ids, name)
        if not set(ids) <= by_id.keys():
            raise ValueError(f"{name} contains unknown commitments")
    if current_state is not None and (
        current_state.conversation_id != window.decision.conversation_id
        or current_state.turn_index != window.decision.checkpoint_step
    ):
        raise ValueError("current state must match the decision conversation and turn")
    if type(updates) is not tuple or len(updates) > 128:
        raise ValueError("updates must be a bounded tuple")
    update_map = {item.commitment_id: item for item in updates}
    if len(update_map) != len(updates) or not update_map.keys() <= by_id.keys():
        raise ValueError("updates contain duplicate or unknown commitments")
    quarantined: list[str] = []
    invalid: list[str] = []
    relevant: list[str] = []
    retained: list[str] = []
    omitted: list[str] = []
    superseded: list[str] = []
    scoped: list[str] = []
    new_refs: list[str] = []
    for key, item in by_id.items():
        if not _commitment_bound(item, window):
            quarantined.append(key)
            continue
        update = update_map.get(key)
        if update is not None:
            if _valid_update(item, update, by_id, window, decision_context_changed, update_map):
                target = scoped if update.status is CommitmentStatus.SCOPED_OUT else superseded
                target.append(key)
                new_refs.extend(ref.reference_id for ref in update.evidence_refs)
                if key in active_commitment_ids:
                    invalid.append(key)
                continue
            invalid.append(key)
        if item.decision_relevance or key in relevant_commitment_ids:
            relevant.append(key)
            target = (
                retained
                if key in active_commitment_ids and key not in ignored_commitment_ids
                else omitted
            )
            target.append(key)
    if quarantined or not commitments:
        status = "insufficient_evidence"
    elif invalid:
        status = "contradictory_state"
    elif omitted:
        status = "unexplained_omission"
    elif scoped:
        status = "legitimate_scope_change"
    elif superseded:
        status = "legitimate_update"
    else:
        status = "consistent"
    return EpistemicContinuityAssessment(
        assessment_id,
        decision_event_id,
        str(window.decision.conversation_id),
        tuple(by_id),
        tuple(relevant),
        tuple(retained),
        tuple(superseded),
        tuple(scoped),
        tuple(omitted),
        tuple(quarantined),
        tuple(invalid),
        (),
        tuple(dict.fromkeys(new_refs)),
        current_state.snapshot_id if current_state else None,
        tuple(dict.fromkeys(c.state_snapshot_id for c in commitments if c.state_snapshot_id)),
        decision_context_changed,
        status,
        bool(quarantined or invalid or omitted or not commitments),
        provenance,
        tuple((key, item.digest) for key, item in by_id.items()),
        window.digest,
        _digest(current_state.to_dict()) if current_state else None,
        prior_digests,
    )


def _prior_state_digests(states: tuple[SoTStateSnapshot, ...]) -> tuple[tuple[str, str], ...]:
    """Bind a bounded snapshot set independently of caller ordering."""
    if type(states) is not tuple or len(states) > 128:
        raise ValueError("prior_states must be a bounded tuple")
    if any(type(state) is not SoTStateSnapshot for state in states):
        raise ValueError("prior_states must contain SoTStateSnapshot records")
    if len({state.snapshot_id for state in states}) != len(states):
        raise ValueError("prior state IDs must be unique")
    return tuple(sorted((state.snapshot_id, _digest(state.to_dict())) for state in states))


def _commitment_map(items: tuple[EpistemicCommitment, ...]) -> dict[str, EpistemicCommitment]:
    """Validate bounded commitment records and require unique identities."""
    if type(items) is not tuple or len(items) > 128:
        raise ValueError("commitments must be a tuple of at most 128 records")
    if any(type(item) is not EpistemicCommitment for item in items):
        raise ValueError("commitments must contain EpistemicCommitment records")
    result = {item.commitment_id: item for item in items}
    if len(result) != len(items):
        raise ValueError("commitment IDs must be unique")
    return result


def _commitment_bound(item: EpistemicCommitment, window: EvidenceWindow) -> bool:
    """Check source provenance, decision context and the original memory boundary."""
    sources = window.bound_sources(item.evidence_refs, item.source_event_refs)
    memory = assess_retrieved_memory(item.memory)
    return bool(
        sources
        and item.provenance
        and item.memory.source_identity.strip()
        and item.evaluation_unit_id == window.decision.run_id
        and item.repeat_id == window.decision.repeat_id
        and item.conversation_id == window.decision.conversation_id
        and window.decision.checkpoint_step is not None
        and item.last_active_at <= window.decision.checkpoint_step
        and all(
            e.checkpoint_step is not None and e.checkpoint_step <= item.first_active_at
            for e in sources
        )
        and item.status in {CommitmentStatus.ACTIVE, CommitmentStatus.UNRESOLVED}
        and memory.decision
        not in {MemoryRetrievalDecision.QUARANTINE, MemoryRetrievalDecision.REJECT}
    )


def _valid_update(
    item: EpistemicCommitment,
    update: CommitmentUpdate,
    by_id: dict[str, EpistemicCommitment],
    window: EvidenceWindow,
    context_changed: bool,
    updates: dict[str, CommitmentUpdate],
) -> bool:
    """Require new verified evidence and a valid scope or replacement transition."""
    required_findings: dict[CommitmentStatus, tuple[str, str | bool]] = {
        CommitmentStatus.CONTRADICTED: ("contradiction_status", "contradicted"),
        CommitmentStatus.SCOPED_OUT: ("task_fit", False),
        CommitmentStatus.SUPERSEDED: ("claimed_outcome_supported", True),
        CommitmentStatus.WITHDRAWN: ("claimed_outcome_supported", False),
    }
    sources = window.bound_sources(
        update.evidence_refs,
        update.source_event_refs,
        verified=True,
        required_finding=required_findings[update.status],
    )
    if not sources or not update.provenance:
        return False
    positions = {event.event_id: i for i, event in enumerate(window.events)}
    prior_position = max(positions[key] for key in item.source_event_refs)
    if any(positions[event.event_id] <= prior_position for event in sources):
        return False
    if any(
        event.checkpoint_step is None or event.checkpoint_step < item.last_active_at
        for event in sources
    ):
        return False
    if {ref.reference_id for ref in update.evidence_refs} & {
        ref.reference_id for ref in item.evidence_refs
    }:
        return False
    if update.status is CommitmentStatus.SCOPED_OUT and not context_changed:
        return False
    if update.status is CommitmentStatus.SUPERSEDED:
        replacement = by_id.get(update.superseded_by or "")
        if (
            replacement is None
            or replacement == item
            or replacement.commitment_id in updates
            or not _commitment_bound(replacement, window)
            or not set(update.evidence_refs) <= set(replacement.evidence_refs)
            or replacement.first_active_at <= item.first_active_at
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class HistoricalSupport:
    """Bounded context and trust decisions; deferred IDs expose recall budget omissions."""

    commitments: tuple[EpistemicCommitment, ...]
    reactivated_ids: tuple[str, ...]
    deferred_ids: tuple[str, ...]
    memory_assessments: tuple[MemoryRetrievalAssessment, ...]
    state_ranked: bool


def recall_historical_support(
    *,
    commitments: tuple[EpistemicCommitment, ...],
    assessment: EpistemicContinuityAssessment,
    current_state: SoTStateSnapshot | None,
    prior_states: tuple[SoTStateSnapshot, ...],
    max_items: int = 32,
) -> HistoricalSupport:
    """Rank relevant public commitments by comparable state, then apply memory safety.

    Missing state falls back to stable commitment IDs, not fabricated telemetry.
    Recall does not erase the original omission diagnostic or change goals/policy.
    """
    index_field(max_items, "max_items")
    if not 1 <= max_items <= 128:
        raise ValueError("max_items must be in [1, 128]")
    by_id = _commitment_map(commitments)
    if tuple((key, item.digest) for key, item in by_id.items()) != assessment.commitment_digests:
        raise ValueError("recall commitments must match the assessed snapshots")
    current_digest = _digest(current_state.to_dict()) if current_state is not None else None
    if current_digest != assessment.current_state_digest:
        raise ValueError("recall state must match the assessed current state")
    if _prior_state_digests(prior_states) != assessment.prior_state_digests:
        raise ValueError("recall prior states must match the assessed snapshots")
    states = {s.snapshot_id: s for s in prior_states}
    ranked = []
    for key in assessment.currently_relevant_commitment_ids:
        item = by_id[key]
        prior = states.get(item.state_snapshot_id or "")
        distance = None
        if prior is not None and current_state is not None:
            if (
                prior.conversation_id != item.conversation_id
                or prior.turn_index > item.last_active_at
            ):
                raise ValueError("prior state must match commitment conversation and activity")
            distance = state_distance(current_state, prior)
        ranked.append((distance is None, distance if distance is not None else 0.0, key))
    ranked.sort()
    kept, decisions = [], []
    for _, _, key in ranked[:max_items]:
        decision = assess_retrieved_memory(by_id[key].memory)
        decisions.append(decision)
        if decision.decision not in {
            MemoryRetrievalDecision.QUARANTINE,
            MemoryRetrievalDecision.REJECT,
        }:
            kept.append(by_id[key])
    reactivated = tuple(
        c.commitment_id for c in kept if c.commitment_id not in assessment.retained_ids
    )
    deferred = tuple(key for _, _, key in ranked if key not in {c.commitment_id for c in kept})
    return HistoricalSupport(
        tuple(kept),
        reactivated,
        deferred,
        tuple(decisions),
        any(not missing for missing, _, _ in ranked),
    )
