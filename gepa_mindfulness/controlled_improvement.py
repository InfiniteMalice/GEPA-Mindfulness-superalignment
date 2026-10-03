"""Opt-in PEO investigation intake; diagnostics grant no execution or persistence authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from gepa_mindfulness.coevolution import CoevolutionStore, CorrectionProposal
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.epistemic_state import EpistemicStateEstimate
from gepa_mindfulness.verification.hypothesis_records import _number, _refs, _text

_HUMAN_THRESHOLDS = {
    "severity": 0.8,
    "irreversibility": 0.5,
    "systemic_effect": 0.8,
    "autonomy_impact": 0.5,
    "reward_hacking_signal": 0.5,
}
_UNCERTAINTIES = ("world_uncertainty", "model_uncertainty", "monitor_uncertainty")


@dataclass(frozen=True, slots=True, kw_only=True)
class TriageDiagnostics:
    """Host-declared consequence signals in a named protocol; None remains unknown.

    Each numeric value lies in [0, 1]. These values guide investigation only; the
    implementation does not authenticate measurements or infer deception intent.
    """

    protocol_id: str
    evidence_refs: tuple[EvidenceReference, ...]
    severity: float | None = None
    irreversibility: float | None = None
    recurrence: float | None = None
    ood_novelty: float | None = None
    systemic_effect: float | None = None
    autonomy_impact: float | None = None
    reward_hacking_signal: float | None = None

    def __post_init__(self) -> None:
        _text(self.protocol_id, "protocol_id")
        if type(self.evidence_refs) is not tuple or not 1 <= len(self.evidence_refs) <= 32:
            raise ValueError("triage evidence_refs requires 1..32 exact references")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))
        for item in fields(TriageDiagnostics)[2:]:
            value = getattr(self, item.name)
            _number(value, item.name, unit=True)
            if value is not None:
                object.__setattr__(self, item.name, float(value))


def assess_improvement(
    store: CoevolutionStore,
    correction: CorrectionProposal,
    estimate: EpistemicStateEstimate,
    triage: TriageDiagnostics,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Bind diagnostics to catalog evidence and recommend the next review route.

    Args:
        store: Exact live coevolution store with the recorded source trajectory.
        correction: One localized proposed correction; never executed by this function.
        estimate: Epistemic estimate naming the source action and prediction.
        triage: Observable consequence diagnostics under a host measurement protocol.
        enabled: Explicit opt-in; only exact True enables catalog reads.

    Returns:
        Detached DEVELOPMENT JSON with all diagnostics, reasons and review route.
        Only sandbox_review includes correction JSON for a later explicit store call.
        The result is neither a training example nor a persistence capability.

    Raises:
        ValueError: Disabled, malformed, private or mismatched input or catalog evidence.
        KeyError: The source epoch is absent from the pinned evaluation catalog.
    """
    if enabled is not True:
        raise ValueError("controlled improvement must be explicitly enabled=True")
    for value, cls in (
        (store, CoevolutionStore),
        (correction, CorrectionProposal),
        (estimate, EpistemicStateEstimate),
        (triage, TriageDiagnostics),
    ):
        if type(value) is not cls:
            raise ValueError(f"expected exact {cls.__name__}")
    proposal = CorrectionProposal.from_dict(CorrectionProposal.to_dict(correction))
    epistemic = EpistemicStateEstimate(
        **{f.name: getattr(estimate, f.name) for f in fields(EpistemicStateEstimate)}
    )
    consequences = TriageDiagnostics(
        **{f.name: getattr(triage, f.name) for f in fields(TriageDiagnostics)}
    )
    if epistemic.evidence_refs:
        # Validate typed public provenance without trusting an instance serializer.
        _refs(epistemic.evidence_refs)
    events = CoevolutionStore.correction_source_events(store, proposal)
    for event in events:
        epistemic.context.validate_event(event)
    if epistemic.action_id != proposal.source_action_id:
        raise ValueError("estimate action must match correction source action")
    executed = tuple(
        event
        for event in events
        if event.event_type == "action_executed" and event.action_id == proposal.source_action_id
    )
    if len(executed) != 1 or epistemic.prediction_commit_id != executed[0].payload.get(
        "prediction_commit_id"
    ):
        raise ValueError("estimate prediction must match the executed source action")
    localized = next(
        node
        for node in proposal.failure_graph.nodes
        if node.failure_id == proposal.localized_failure_id
    )
    for name, refs in (
        ("triage", consequences.evidence_refs),
        ("estimate", epistemic.evidence_refs),
    ):
        if (name == "triage" or refs) and not set(refs).intersection(localized.evidence_refs):
            raise ValueError(f"{name} evidence must intersect localized failure evidence")

    diagnostics = {f.name: getattr(consequences, f.name) for f in fields(TriageDiagnostics)[2:]}
    uncertainties = {name: getattr(epistemic, name) for name in _UNCERTAINTIES}
    reasons = []
    human = False
    for name, value in diagnostics.items():
        if value is None:
            reasons.append(f"missing_{name}")
        elif name in _HUMAN_THRESHOLDS and value >= _HUMAN_THRESHOLDS[name]:
            reasons.append(f"high_{name}")
            human = True
    for name, value in uncertainties.items():
        if value is None:
            reasons.append(f"missing_{name}")
        elif value >= 0.5:
            reasons.append(f"high_{name}")
    root_status = proposal.failure_graph.root_cause_status
    if root_status != "supported":
        reasons.append("root_cause_unqualified")
    if len(proposal.changed_components) != 1:
        reasons.append("multiple_components")
    route = "human_review" if human else "investigate" if reasons else "sandbox_review"
    values = tuple(diagnostics.values()) + tuple(uncertainties.values())
    priority = "elevated" if any(v is None or v >= 0.5 for v in values) else "routine"
    return {
        "schema_version": "controlled-improvement-v1",
        "training_eligibility": TrainingEligibility.DEVELOPMENT.value,
        "authority_granted": False,
        "proposal_id": proposal.proposal_id,
        "source_trajectory_id": proposal.source_trajectory_id,
        "source_epoch_id": proposal.source_epoch_id,
        "source_action_id": proposal.source_action_id,
        "estimate": EpistemicStateEstimate.to_dict(epistemic),
        "triage": {
            "protocol_id": consequences.protocol_id,
            "evidence_refs": [EvidenceReference.to_dict(ref) for ref in consequences.evidence_refs],
            **diagnostics,
        },
        "root_cause_status": root_status,
        "route": route,
        "priority": "urgent" if human else priority,
        "reasons": reasons,
        "correction": CorrectionProposal.to_dict(proposal) if route == "sandbox_review" else None,
    }
