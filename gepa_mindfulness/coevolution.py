"""Auditable, non-executing acceptance for offline model and harness candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, cast

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.learning_surfaces import (
    EvaluationAuthority,
    EvaluationEpoch,
    EvaluationEpochStore,
    ValidationReceipt,
    ValidationSplit,
    ValidationTarget,
)
from gepa_mindfulness.verification.failure_graph import FailureGraph

EnumT = TypeVar("EnumT", bound=Enum)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class CandidateComponent(str, Enum):
    """A versioned system component changed by one offline candidate."""

    MODEL = "model"
    HARNESS = "harness"


class CorrectionScope(str, Enum):
    """The amount of source behavior a correction proposes to imitate or change."""

    LOCALIZED_FAILURE = "localized_failure"
    WHOLE_TRAJECTORY = "whole_trajectory"


class MetricDirection(str, Enum):
    """The declared comparison direction for an acceptance metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    """A teacher proposal bound to one verifier-localized source failure.

    Teacher text proposes a change. It grants no execution or acceptance authority.
    """

    proposal_id: str
    source_trajectory_id: str
    source_action_id: str
    source_epoch_id: str
    failure_graph: FailureGraph
    localized_failure_id: str
    localization_verifier_refs: tuple[str, ...]
    source_evidence_refs: tuple[EvidenceReference, ...]
    teacher_correction: str
    teacher_evidence_refs: tuple[EvidenceReference, ...]
    changed_components: tuple[CandidateComponent, ...]
    scope: CorrectionScope = CorrectionScope.LOCALIZED_FAILURE
    _construction_binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "source_trajectory_id",
            "source_action_id",
            "source_epoch_id",
            "localized_failure_id",
            "teacher_correction",
        ):
            _require_token(getattr(self, name), name)
        if type(self.scope) is not CorrectionScope:
            raise ValueError("scope must be an exact CorrectionScope")
        if self.scope is not CorrectionScope.LOCALIZED_FAILURE:
            raise ValueError("correction scope must name one localized failure")
        graph = _snapshot_failure_graph(self.failure_graph)
        verifier_refs = _snapshot_tokens(
            self.localization_verifier_refs,
            "localization_verifier_refs",
        )
        source_refs = _snapshot_evidence_refs(self.source_evidence_refs, "source_evidence_refs")
        teacher_refs = _snapshot_evidence_refs(
            self.teacher_evidence_refs,
            "teacher_evidence_refs",
        )
        components = _snapshot_components(self.changed_components)
        _validate_correction_bindings(
            graph,
            self.localized_failure_id,
            self.source_action_id,
            verifier_refs,
            source_refs,
        )
        object.__setattr__(self, "failure_graph", graph)
        object.__setattr__(self, "localization_verifier_refs", verifier_refs)
        object.__setattr__(self, "source_evidence_refs", source_refs)
        object.__setattr__(self, "teacher_evidence_refs", teacher_refs)
        object.__setattr__(self, "changed_components", components)
        object.__setattr__(self, "_construction_binding", _digest(_proposal_payload(self)))

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible proposal snapshot."""

        return _proposal_payload(_snapshot_proposal(self))

    @classmethod
    def from_dict(cls, data: object) -> CorrectionProposal:
        """Restore a proposal from its exact JSON-compatible representation."""

        values = _exact_mapping(
            data,
            {
                "proposal_id",
                "source_trajectory_id",
                "source_action_id",
                "source_epoch_id",
                "failure_graph",
                "localized_failure_id",
                "localization_verifier_refs",
                "source_evidence_refs",
                "teacher_correction",
                "teacher_evidence_refs",
                "changed_components",
                "scope",
            },
            "CorrectionProposal",
        )
        return cls(
            proposal_id=cast(str, values["proposal_id"]),
            source_trajectory_id=cast(str, values["source_trajectory_id"]),
            source_action_id=cast(str, values["source_action_id"]),
            source_epoch_id=cast(str, values["source_epoch_id"]),
            failure_graph=FailureGraph.from_dict(values["failure_graph"]),
            localized_failure_id=cast(str, values["localized_failure_id"]),
            localization_verifier_refs=_restore_tokens(
                values["localization_verifier_refs"],
                "localization_verifier_refs",
            ),
            source_evidence_refs=_restore_evidence_refs(
                values["source_evidence_refs"],
                "source_evidence_refs",
            ),
            teacher_correction=cast(str, values["teacher_correction"]),
            teacher_evidence_refs=_restore_evidence_refs(
                values["teacher_evidence_refs"],
                "teacher_evidence_refs",
            ),
            changed_components=_restore_components(values["changed_components"]),
            scope=_parse_enum(values["scope"], CorrectionScope, "scope"),
        )


@dataclass(frozen=True, slots=True)
class CandidateSystem:
    """One store-bound offline candidate with explicit rollback provenance."""

    candidate_id: str
    correction: CorrectionProposal
    authority: EvaluationAuthority
    lineage_id: str
    source_epoch_id: str
    candidate_epoch_id: str
    model_version: str
    harness_version: str
    changed_components: tuple[CandidateComponent, ...]
    artifact_digest: str
    rollback_target_epoch_id: str
    _construction_binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "lineage_id",
            "source_epoch_id",
            "candidate_epoch_id",
            "model_version",
            "harness_version",
            "rollback_target_epoch_id",
        ):
            _require_token(getattr(self, name), name)
        proposal = _snapshot_proposal(self.correction)
        authority = _snapshot_authority(self.authority)
        components = _snapshot_components(self.changed_components)
        _require_sha256(self.artifact_digest, "artifact_digest")
        if self.source_epoch_id == self.candidate_epoch_id:
            raise ValueError("candidate_epoch_id must differ from source_epoch_id")
        if self.source_epoch_id != proposal.source_epoch_id:
            raise ValueError("source_epoch_id must match the correction source epoch")
        if self.rollback_target_epoch_id != self.source_epoch_id:
            raise ValueError("rollback_target_epoch_id must be the exact source epoch")
        if components != proposal.changed_components:
            raise ValueError("changed_components must match the correction proposal")
        object.__setattr__(self, "correction", proposal)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "changed_components", components)
        object.__setattr__(self, "_construction_binding", _digest(_candidate_payload(self)))

    def validation_target(self) -> ValidationTarget:
        """Return the exact candidate target used by store-issued validation receipts."""

        snapshot = _snapshot_candidate(self)
        return ValidationTarget(
            snapshot.candidate_id,
            "model-harness-candidate",
            snapshot.harness_version,
            snapshot.artifact_digest,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible candidate snapshot."""

        return _candidate_payload(_snapshot_candidate(self))

    @classmethod
    def from_dict(cls, data: object) -> CandidateSystem:
        """Restore a candidate from its exact JSON-compatible representation."""

        values = _exact_mapping(
            data,
            {
                "candidate_id",
                "correction",
                "authority",
                "lineage_id",
                "source_epoch_id",
                "candidate_epoch_id",
                "model_version",
                "harness_version",
                "changed_components",
                "artifact_digest",
                "rollback_target_epoch_id",
            },
            "CandidateSystem",
        )
        return cls(
            candidate_id=cast(str, values["candidate_id"]),
            correction=CorrectionProposal.from_dict(values["correction"]),
            authority=_authority_from_dict(values["authority"]),
            lineage_id=cast(str, values["lineage_id"]),
            source_epoch_id=cast(str, values["source_epoch_id"]),
            candidate_epoch_id=cast(str, values["candidate_epoch_id"]),
            model_version=cast(str, values["model_version"]),
            harness_version=cast(str, values["harness_version"]),
            changed_components=_restore_components(values["changed_components"]),
            artifact_digest=cast(str, values["artifact_digest"]),
            rollback_target_epoch_id=cast(str, values["rollback_target_epoch_id"]),
        )


@dataclass(frozen=True, slots=True)
class ComponentMetric:
    """One named candidate comparison with explicit direction and allowed tolerance."""

    name: str
    baseline_value: float
    candidate_value: float
    direction: MetricDirection
    tolerance: float

    def __post_init__(self) -> None:
        _require_token(self.name, "name")
        _require_finite_float(self.baseline_value, "baseline_value")
        _require_finite_float(self.candidate_value, "candidate_value")
        if type(self.direction) is not MetricDirection:
            raise ValueError("direction must be an exact MetricDirection")
        tolerance = _require_finite_float(self.tolerance, "tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")

    def is_non_worse(self) -> bool:
        """Return whether the candidate is within the declared degradation tolerance."""

        snapshot = _snapshot_metric(self)
        if snapshot.direction is MetricDirection.HIGHER_IS_BETTER:
            return snapshot.candidate_value >= snapshot.baseline_value - snapshot.tolerance
        return snapshot.candidate_value <= snapshot.baseline_value + snapshot.tolerance

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible metric."""

        snapshot = _snapshot_metric(self)
        return {
            "name": snapshot.name,
            "baseline_value": snapshot.baseline_value,
            "candidate_value": snapshot.candidate_value,
            "direction": snapshot.direction.value,
            "tolerance": snapshot.tolerance,
        }

    @classmethod
    def from_dict(cls, data: object) -> ComponentMetric:
        """Restore an exact component metric."""

        values = _exact_mapping(
            data,
            {"name", "baseline_value", "candidate_value", "direction", "tolerance"},
            "ComponentMetric",
        )
        return cls(
            cast(str, values["name"]),
            cast(float, values["baseline_value"]),
            cast(float, values["candidate_value"]),
            _parse_enum(values["direction"], MetricDirection, "direction"),
            cast(float, values["tolerance"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationBundle:
    """Held-out and protected receipts plus complete named component comparisons."""

    candidate: CandidateSystem
    held_out_receipt: ValidationReceipt
    protected_receipt: ValidationReceipt
    component_metrics: tuple[ComponentMetric, ...]
    primary_metric_name: str
    _construction_binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        candidate = _snapshot_candidate(self.candidate)
        held_out = _snapshot_receipt(self.held_out_receipt, "held_out_receipt")
        protected = _snapshot_receipt(self.protected_receipt, "protected_receipt")
        metrics = _snapshot_metrics(self.component_metrics)
        primary = _require_token(self.primary_metric_name, "primary_metric_name")
        names = tuple(metric.name for metric in metrics)
        if not metrics:
            raise ValueError("component_metrics must be a nonempty exact tuple")
        if len(set(names)) != len(names):
            raise ValueError("component_metrics names must be unique and non-conflicting")
        if primary not in names:
            raise ValueError("primary_metric_name must name one complete component metric")
        if held_out.split is not ValidationSplit.HELD_OUT:
            raise ValueError("held_out_receipt must have the held_out split")
        if protected.split is not ValidationSplit.PROTECTED:
            raise ValueError("protected_receipt must have the protected split")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "held_out_receipt", held_out)
        object.__setattr__(self, "protected_receipt", protected)
        object.__setattr__(self, "component_metrics", metrics)
        object.__setattr__(self, "_construction_binding", _digest(_bundle_payload(self)))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible validation bundle."""

        return _bundle_payload(_snapshot_bundle(self))

    @classmethod
    def from_dict(cls, data: object) -> ValidationBundle:
        """Restore a validation bundle from an exact JSON-compatible object."""

        values = _exact_mapping(
            data,
            {
                "candidate",
                "held_out_receipt",
                "protected_receipt",
                "component_metrics",
                "primary_metric_name",
            },
            "ValidationBundle",
        )
        metrics = values["component_metrics"]
        if type(metrics) is not list:
            raise ValueError("ValidationBundle component_metrics must be an array")
        return cls(
            CandidateSystem.from_dict(values["candidate"]),
            ValidationReceipt.from_dict(values["held_out_receipt"]),
            ValidationReceipt.from_dict(values["protected_receipt"]),
            tuple(ComponentMetric.from_dict(value) for value in cast(list[object], metrics)),
            cast(str, values["primary_metric_name"]),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """A deterministic audit decision; it never executes or installs the candidate."""

    decision_id: str
    accepted: bool
    candidate_id: str
    proposal_id: str
    authority: EvaluationAuthority
    lineage_id: str
    source_epoch_id: str
    candidate_epoch_id: str
    held_out_receipt_id: str
    protected_receipt_id: str
    protected_record_ids: tuple[str, ...]
    primary_metric_name: str
    component_metrics: tuple[ComponentMetric, ...]
    rollback_target_epoch_id: str
    reason: str
    execute_candidate: bool
    inputs_digest: str
    _construction_binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256(self.decision_id, "decision_id")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be an exact bool")
        for name in (
            "candidate_id",
            "proposal_id",
            "lineage_id",
            "source_epoch_id",
            "candidate_epoch_id",
            "held_out_receipt_id",
            "protected_receipt_id",
            "primary_metric_name",
            "rollback_target_epoch_id",
            "reason",
        ):
            _require_token(getattr(self, name), name)
        authority = _snapshot_authority(self.authority)
        protected_ids = _snapshot_sha256_ids(self.protected_record_ids, "protected_record_ids")
        metrics = _snapshot_metrics(self.component_metrics)
        if self.primary_metric_name not in {metric.name for metric in metrics}:
            raise ValueError("primary_metric_name must name one component metric")
        if self.rollback_target_epoch_id != self.source_epoch_id:
            raise ValueError("rollback_target_epoch_id must equal source_epoch_id")
        if self.execute_candidate is not False:
            raise ValueError("execute_candidate must remain false for an audit-only decision")
        _require_sha256(self.inputs_digest, "inputs_digest")
        expected_accepted = _primary_metric(metrics, self.primary_metric_name).is_non_worse()
        if self.accepted is not expected_accepted:
            raise ValueError("accepted must equal the declared primary metric result")
        expected_reason = _decision_reason(self.accepted)
        if self.reason != expected_reason:
            raise ValueError("reason does not match the acceptance result")
        expected_id = _decision_id(self.inputs_digest, self.accepted, self.reason)
        if self.decision_id != expected_id:
            raise ValueError("decision_id does not match the authoritative decision inputs")
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "protected_record_ids", protected_ids)
        object.__setattr__(self, "component_metrics", metrics)
        object.__setattr__(self, "_construction_binding", _digest(_decision_payload(self)))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible audit decision."""

        return _decision_payload(_snapshot_decision(self))

    @classmethod
    def from_dict(cls, data: object) -> AcceptanceDecision:
        """Restore an internally consistent audit decision."""

        values = _exact_mapping(
            data,
            {
                "decision_id",
                "accepted",
                "candidate_id",
                "proposal_id",
                "authority",
                "lineage_id",
                "source_epoch_id",
                "candidate_epoch_id",
                "held_out_receipt_id",
                "protected_receipt_id",
                "protected_record_ids",
                "primary_metric_name",
                "component_metrics",
                "rollback_target_epoch_id",
                "reason",
                "execute_candidate",
                "inputs_digest",
            },
            "AcceptanceDecision",
        )
        protected_ids = _restore_sha256_ids(
            values["protected_record_ids"],
            "protected_record_ids",
        )
        metrics = values["component_metrics"]
        if type(metrics) is not list:
            raise ValueError("AcceptanceDecision component_metrics must be an array")
        return cls(
            decision_id=cast(str, values["decision_id"]),
            accepted=_exact_bool(values["accepted"], "accepted"),
            candidate_id=cast(str, values["candidate_id"]),
            proposal_id=cast(str, values["proposal_id"]),
            authority=_authority_from_dict(values["authority"]),
            lineage_id=cast(str, values["lineage_id"]),
            source_epoch_id=cast(str, values["source_epoch_id"]),
            candidate_epoch_id=cast(str, values["candidate_epoch_id"]),
            held_out_receipt_id=cast(str, values["held_out_receipt_id"]),
            protected_receipt_id=cast(str, values["protected_receipt_id"]),
            protected_record_ids=protected_ids,
            primary_metric_name=cast(str, values["primary_metric_name"]),
            component_metrics=tuple(
                ComponentMetric.from_dict(value) for value in cast(list[object], metrics)
            ),
            rollback_target_epoch_id=cast(str, values["rollback_target_epoch_id"]),
            reason=cast(str, values["reason"]),
            execute_candidate=_exact_bool(values["execute_candidate"], "execute_candidate"),
            inputs_digest=cast(str, values["inputs_digest"]),
        )


def bind_candidate_system(
    store: EvaluationEpochStore,
    *,
    lineage_id: str,
    candidate_id: str,
    correction: CorrectionProposal,
    artifact_digest: str,
) -> CandidateSystem:
    """Bind a proposal to the store-issued source-to-candidate epoch transition."""

    checked_store = _exact_store(store)
    lineage = _require_token(lineage_id, "lineage_id")
    proposal = _snapshot_proposal(correction)
    epochs = checked_store.open(lineage).snapshot()
    if len(epochs) < 2:
        raise ValueError("candidate epoch requires a closed source and a new store-issued epoch")
    source, candidate = epochs[-2:]
    _validate_epoch_transition(source, candidate, proposal)
    return CandidateSystem(
        candidate_id=candidate_id,
        correction=proposal,
        authority=checked_store.authority(),
        lineage_id=lineage,
        source_epoch_id=source.epoch_id,
        candidate_epoch_id=candidate.epoch_id,
        model_version=candidate.model_version,
        harness_version=candidate.harness_version,
        changed_components=proposal.changed_components,
        artifact_digest=artifact_digest,
        rollback_target_epoch_id=source.epoch_id,
    )


def decide_candidate_acceptance(
    bundle: ValidationBundle,
    *,
    trusted_store: EvaluationEpochStore,
    trusted_authority: EvaluationAuthority,
    required_protected_record_ids: tuple[str, ...],
) -> AcceptanceDecision:
    """Return a deterministic decision after revalidating all authoritative inputs."""

    snapshot = _snapshot_bundle(bundle)
    store = _exact_store(trusted_store)
    authority = _snapshot_authority(trusted_authority)
    if store.authority() != authority:
        raise ValueError("trusted authority does not identify the supplied evaluation catalog")
    candidate = snapshot.candidate
    if candidate.authority != authority:
        raise ValueError("candidate belongs to a different authority or catalog")
    required_ids = _snapshot_sha256_ids(
        required_protected_record_ids,
        "required_protected_record_ids",
    )
    if required_ids != snapshot.protected_receipt.record_ids:
        raise ValueError("protected receipt does not cover the complete protected regression set")
    if set(snapshot.held_out_receipt.record_ids) & set(required_ids):
        raise ValueError("held-out and protected validation records must be disjoint")
    epochs = store.open(candidate.lineage_id).snapshot()
    source, candidate_epoch = _authoritative_candidate_epochs(epochs, candidate)
    _validate_epoch_transition(source, candidate_epoch, candidate.correction)
    if not candidate_epoch.closed:
        raise ValueError("candidate acceptance requires a closed candidate evaluation epoch")
    _validate_candidate_versions(candidate, candidate_epoch)
    held_out = store.validate_validation_receipt(snapshot.held_out_receipt)
    protected = store.validate_validation_receipt(snapshot.protected_receipt)
    _validate_receipt(held_out, candidate, ValidationSplit.HELD_OUT)
    _validate_receipt(protected, candidate, ValidationSplit.PROTECTED)
    primary = _primary_metric(snapshot.component_metrics, snapshot.primary_metric_name)
    accepted = primary.is_non_worse()
    reason = _decision_reason(accepted)
    inputs = {
        "bundle": snapshot.to_dict(),
        "trusted_authority": _authority_payload(authority),
        "required_protected_record_ids": list(required_ids),
    }
    inputs_digest = _digest(inputs)
    return AcceptanceDecision(
        decision_id=_decision_id(inputs_digest, accepted, reason),
        accepted=accepted,
        candidate_id=candidate.candidate_id,
        proposal_id=candidate.correction.proposal_id,
        authority=authority,
        lineage_id=candidate.lineage_id,
        source_epoch_id=candidate.source_epoch_id,
        candidate_epoch_id=candidate.candidate_epoch_id,
        held_out_receipt_id=held_out.receipt_id,
        protected_receipt_id=protected.receipt_id,
        protected_record_ids=required_ids,
        primary_metric_name=snapshot.primary_metric_name,
        component_metrics=snapshot.component_metrics,
        rollback_target_epoch_id=candidate.rollback_target_epoch_id,
        reason=reason,
        execute_candidate=False,
        inputs_digest=inputs_digest,
    )


def _validate_correction_bindings(
    graph: FailureGraph,
    failure_id: str,
    action_id: str,
    verifier_refs: tuple[str, ...],
    source_refs: tuple[EvidenceReference, ...],
) -> None:
    nodes = tuple(node for node in graph.nodes if node.failure_id == failure_id)
    if len(nodes) != 1:
        raise ValueError("localized_failure_id must name one node in FailureGraph localization")
    node = nodes[0]
    if node.event_id != action_id:
        raise ValueError("source_action_id must equal the localized failure event")
    localized_ids = {
        graph.localization.first_anomaly,
        graph.localization.root_cause,
        graph.localization.decisive_failure,
        graph.localization.recoverable_until,
        *graph.localization.symptoms,
    }
    if failure_id not in localized_ids:
        raise ValueError("localized_failure_id must name an asserted FailureGraph role")
    expected_refs: set[str] = set()
    for binding in graph.localization.role_evidence:
        if binding.failure_id == failure_id:
            expected_refs.update(binding.verifier_refs)
    if set(verifier_refs) != expected_refs:
        raise ValueError("localization_verifier_refs must bind the exact localized failure")
    source_keys = {(reference.reference_id, reference.source_kind) for reference in source_refs}
    node_keys = {
        (reference.reference_id, reference.source_kind) for reference in node.evidence_refs
    }
    if not source_refs or not node_keys.issubset(source_keys):
        raise ValueError("source_evidence_refs must include the localized failure evidence")


def _validate_epoch_transition(
    source: EvaluationEpoch,
    candidate: EvaluationEpoch,
    proposal: CorrectionProposal,
) -> None:
    if not source.closed:
        raise ValueError("source epoch must be closed before candidate evolution")
    if source.epoch_id != proposal.source_epoch_id:
        raise ValueError("correction source_epoch_id does not match the authoritative source")
    if source.epoch_id == candidate.epoch_id:
        raise ValueError("candidate epoch must have a new epoch identity")
    changes: list[CandidateComponent] = []
    if source.model_version != candidate.model_version:
        changes.append(CandidateComponent.MODEL)
    if source.harness_version != candidate.harness_version:
        changes.append(CandidateComponent.HARNESS)
    if tuple(changes) != proposal.changed_components:
        raise ValueError("changed_components do not match authoritative version changes")


def _authoritative_candidate_epochs(
    epochs: tuple[EvaluationEpoch, ...],
    candidate: CandidateSystem,
) -> tuple[EvaluationEpoch, EvaluationEpoch]:
    positions = tuple(
        index
        for index, epoch in enumerate(epochs)
        if epoch.epoch_id == candidate.candidate_epoch_id
    )
    if len(positions) != 1 or positions[0] == 0 or positions[0] != len(epochs) - 1:
        raise ValueError("candidate epoch is not the authoritative lineage tip")
    candidate_epoch = epochs[positions[0]]
    source = epochs[positions[0] - 1]
    if source.epoch_id != candidate.source_epoch_id:
        raise ValueError("candidate source epoch does not match authoritative lineage")
    return source, candidate_epoch


def _validate_candidate_versions(candidate: CandidateSystem, epoch: EvaluationEpoch) -> None:
    if (
        candidate.model_version != epoch.model_version
        or candidate.harness_version != epoch.harness_version
    ):
        raise ValueError("candidate versions do not match the authoritative candidate epoch")


def _validate_receipt(
    receipt: ValidationReceipt,
    candidate: CandidateSystem,
    split: ValidationSplit,
) -> None:
    target = candidate.validation_target()
    if receipt.split is not split:
        raise ValueError(f"candidate requires an exact {split.value} validation receipt")
    if receipt.lineage_id != candidate.lineage_id:
        raise ValueError("validation receipt lineage does not match candidate")
    if receipt.epoch_id != candidate.candidate_epoch_id:
        raise ValueError("validation receipt epoch does not match candidate")
    if (
        receipt.target_artifact_id,
        receipt.target_skill_id,
        receipt.target_version,
        receipt.target_digest,
    ) != (target.artifact_id, target.skill_id, target.version, target.artifact_digest):
        raise ValueError("validation receipt target does not match candidate identity")


def _proposal_payload(proposal: CorrectionProposal) -> dict[str, object]:
    return {
        "proposal_id": proposal.proposal_id,
        "source_trajectory_id": proposal.source_trajectory_id,
        "source_action_id": proposal.source_action_id,
        "source_epoch_id": proposal.source_epoch_id,
        "failure_graph": proposal.failure_graph.to_dict(),
        "localized_failure_id": proposal.localized_failure_id,
        "localization_verifier_refs": list(proposal.localization_verifier_refs),
        "source_evidence_refs": [value.to_dict() for value in proposal.source_evidence_refs],
        "teacher_correction": proposal.teacher_correction,
        "teacher_evidence_refs": [value.to_dict() for value in proposal.teacher_evidence_refs],
        "changed_components": [value.value for value in proposal.changed_components],
        "scope": proposal.scope.value,
    }


def _candidate_payload(candidate: CandidateSystem) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "correction": candidate.correction.to_dict(),
        "authority": _authority_payload(candidate.authority),
        "lineage_id": candidate.lineage_id,
        "source_epoch_id": candidate.source_epoch_id,
        "candidate_epoch_id": candidate.candidate_epoch_id,
        "model_version": candidate.model_version,
        "harness_version": candidate.harness_version,
        "changed_components": [value.value for value in candidate.changed_components],
        "artifact_digest": candidate.artifact_digest,
        "rollback_target_epoch_id": candidate.rollback_target_epoch_id,
    }


def _bundle_payload(bundle: ValidationBundle) -> dict[str, object]:
    return {
        "candidate": bundle.candidate.to_dict(),
        "held_out_receipt": bundle.held_out_receipt.to_dict(),
        "protected_receipt": bundle.protected_receipt.to_dict(),
        "component_metrics": [metric.to_dict() for metric in bundle.component_metrics],
        "primary_metric_name": bundle.primary_metric_name,
    }


def _decision_payload(decision: AcceptanceDecision) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "accepted": decision.accepted,
        "candidate_id": decision.candidate_id,
        "proposal_id": decision.proposal_id,
        "authority": _authority_payload(decision.authority),
        "lineage_id": decision.lineage_id,
        "source_epoch_id": decision.source_epoch_id,
        "candidate_epoch_id": decision.candidate_epoch_id,
        "held_out_receipt_id": decision.held_out_receipt_id,
        "protected_receipt_id": decision.protected_receipt_id,
        "protected_record_ids": list(decision.protected_record_ids),
        "primary_metric_name": decision.primary_metric_name,
        "component_metrics": [metric.to_dict() for metric in decision.component_metrics],
        "rollback_target_epoch_id": decision.rollback_target_epoch_id,
        "reason": decision.reason,
        "execute_candidate": decision.execute_candidate,
        "inputs_digest": decision.inputs_digest,
    }


def _snapshot_proposal(value: object) -> CorrectionProposal:
    if type(value) is not CorrectionProposal:
        raise ValueError("correction must be an exact CorrectionProposal")
    proposal = cast(CorrectionProposal, value)
    if _digest(_proposal_payload(proposal)) != proposal._construction_binding:
        raise ValueError("CorrectionProposal changed after construction")
    return CorrectionProposal.from_dict(_proposal_payload(proposal))


def _snapshot_candidate(value: object) -> CandidateSystem:
    if type(value) is not CandidateSystem:
        raise ValueError("candidate must be an exact CandidateSystem")
    candidate = cast(CandidateSystem, value)
    if _digest(_candidate_payload(candidate)) != candidate._construction_binding:
        raise ValueError("CandidateSystem changed after construction")
    return CandidateSystem.from_dict(_candidate_payload(candidate))


def _snapshot_bundle(value: object) -> ValidationBundle:
    if type(value) is not ValidationBundle:
        raise ValueError("bundle must be an exact ValidationBundle")
    bundle = cast(ValidationBundle, value)
    if _digest(_bundle_payload(bundle)) != bundle._construction_binding:
        raise ValueError("ValidationBundle changed after construction")
    return ValidationBundle.from_dict(_bundle_payload(bundle))


def _snapshot_decision(value: object) -> AcceptanceDecision:
    if type(value) is not AcceptanceDecision:
        raise ValueError("decision must be an exact AcceptanceDecision")
    decision = cast(AcceptanceDecision, value)
    if _digest(_decision_payload(decision)) != decision._construction_binding:
        raise ValueError("AcceptanceDecision changed after construction")
    return AcceptanceDecision.from_dict(_decision_payload(decision))


def _snapshot_failure_graph(value: object) -> FailureGraph:
    if type(value) is not FailureGraph:
        raise ValueError("failure_graph must be an exact FailureGraph")
    graph_type = cast(Any, FailureGraph)
    return cast(FailureGraph, graph_type.from_dict(cast(FailureGraph, value).to_dict()))


def _snapshot_authority(value: object) -> EvaluationAuthority:
    if type(value) is not EvaluationAuthority:
        raise ValueError("authority must be an exact EvaluationAuthority")
    authority = cast(EvaluationAuthority, value)
    authority_type = cast(Any, EvaluationAuthority)
    return cast(
        EvaluationAuthority,
        authority_type(
            authority.catalog_path,
            authority.catalog_id,
            authority.authority_domain,
        ),
    )


def _snapshot_receipt(value: object, field_name: str) -> ValidationReceipt:
    if type(value) is not ValidationReceipt:
        raise ValueError(f"{field_name} must be an exact ValidationReceipt")
    try:
        receipt_type = cast(Any, ValidationReceipt)
        return cast(
            ValidationReceipt,
            receipt_type.from_dict(cast(ValidationReceipt, value).to_dict()),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid: {exc}") from exc


def _snapshot_metric(value: object) -> ComponentMetric:
    if type(value) is not ComponentMetric:
        raise ValueError("component_metrics must contain exact ComponentMetric values")
    metric = cast(ComponentMetric, value)
    return ComponentMetric(
        metric.name,
        metric.baseline_value,
        metric.candidate_value,
        metric.direction,
        metric.tolerance,
    )


def _snapshot_metrics(values: object) -> tuple[ComponentMetric, ...]:
    if type(values) is not tuple:
        raise ValueError("component_metrics must be a nonempty exact tuple")
    return tuple(_snapshot_metric(value) for value in cast(tuple[object, ...], values))


def _snapshot_components(values: object) -> tuple[CandidateComponent, ...]:
    if type(values) is not tuple or not values:
        raise ValueError("changed_components must be a nonempty exact tuple")
    components = cast(tuple[object, ...], values)
    if any(type(value) is not CandidateComponent for value in components):
        raise ValueError("changed_components must contain exact CandidateComponent values")
    result = cast(tuple[CandidateComponent, ...], components)
    canonical = tuple(
        component
        for component in (CandidateComponent.MODEL, CandidateComponent.HARNESS)
        if component in result
    )
    if result != canonical:
        raise ValueError("changed_components must be unique and canonically ordered")
    return result


def _snapshot_evidence_refs(values: object, field_name: str) -> tuple[EvidenceReference, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{field_name} must be a nonempty exact tuple")
    references: list[EvidenceReference] = []
    for value in cast(tuple[object, ...], values):
        if type(value) is not EvidenceReference:
            raise ValueError(f"{field_name} must contain exact EvidenceReference values")
        reference = cast(EvidenceReference, value)
        if type(reference.reference_id) is not str or not reference.reference_id.strip():
            raise ValueError(f"{field_name} reference_id must be a nonblank built-in string")
        if type(reference.source_kind) is not EvidenceSourceKind:
            raise ValueError(f"{field_name} source_kind must be an exact EvidenceSourceKind")
        reference_type = cast(Any, EvidenceReference)
        references.append(
            cast(EvidenceReference, reference_type(reference.reference_id, reference.source_kind))
        )
    keys = tuple((value.reference_id, value.source_kind) for value in references)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field_name} must contain unique references")
    return tuple(references)


def _restore_evidence_refs(values: object, field_name: str) -> tuple[EvidenceReference, ...]:
    if type(values) is not list:
        raise ValueError(f"{field_name} must be an array")
    return tuple(EvidenceReference.from_dict(value) for value in cast(list[object], values))


def _snapshot_tokens(values: object, field_name: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{field_name} must be a nonempty exact tuple")
    result = tuple(_require_token(value, field_name) for value in cast(tuple[object, ...], values))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _restore_tokens(values: object, field_name: str) -> tuple[str, ...]:
    if type(values) is not list:
        raise ValueError(f"{field_name} must be an array")
    return tuple(cast(str, value) for value in cast(list[object], values))


def _snapshot_sha256_ids(values: object, field_name: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{field_name} must be a nonempty exact tuple")
    result = tuple(_require_sha256(value, field_name) for value in cast(tuple[object, ...], values))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _restore_sha256_ids(values: object, field_name: str) -> tuple[str, ...]:
    if type(values) is not list:
        raise ValueError(f"{field_name} must be an array")
    return tuple(cast(str, value) for value in cast(list[object], values))


def _restore_components(values: object) -> tuple[CandidateComponent, ...]:
    if type(values) is not list:
        raise ValueError("changed_components must be an array")
    return tuple(
        _parse_enum(value, CandidateComponent, "changed_components")
        for value in cast(list[object], values)
    )


def _primary_metric(
    metrics: tuple[ComponentMetric, ...],
    primary_metric_name: str,
) -> ComponentMetric:
    matches = tuple(metric for metric in metrics if metric.name == primary_metric_name)
    if len(matches) != 1:
        raise ValueError("primary_metric_name must resolve to exactly one component metric")
    return matches[0]


def _authority_payload(authority: EvaluationAuthority) -> dict[str, object]:
    return {
        "catalog_path": authority.catalog_path,
        "catalog_id": authority.catalog_id,
        "authority_domain": authority.authority_domain,
    }


def _authority_from_dict(value: object) -> EvaluationAuthority:
    fields = _exact_mapping(
        value,
        {"catalog_path", "catalog_id", "authority_domain"},
        "EvaluationAuthority",
    )
    return EvaluationAuthority(
        cast(str, fields["catalog_path"]),
        cast(str, fields["catalog_id"]),
        cast(str, fields["authority_domain"]),
    )


def _exact_store(value: object) -> EvaluationEpochStore:
    if type(value) is not EvaluationEpochStore:
        raise ValueError("trusted_store must be an exact EvaluationEpochStore")
    return cast(EvaluationEpochStore, value)


def _exact_mapping(
    value: object,
    expected: set[str],
    record_name: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{record_name} must be an exact object")
    if set(cast(dict[object, object], value)) != expected:
        raise ValueError(f"{record_name} requires exactly {sorted(expected)!r}")
    return cast(Mapping[str, object], value)


def _parse_enum(value: object, enum_type: type[EnumT], field_name: str) -> EnumT:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a built-in string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} has unknown value {value!r}") from exc


def _require_token(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank canonical built-in string")
    return value


def _require_finite_float(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite exact float")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an exact lowercase SHA-256 digest")
    return value


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be an exact bool")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _decision_id(inputs_digest: str, accepted: bool, reason: str) -> str:
    return _digest({"inputs_digest": inputs_digest, "accepted": accepted, "reason": reason})


def _decision_reason(accepted: bool) -> str:
    if accepted:
        return "candidate passed held-out and protected acceptance gates"
    return "candidate primary metric is worse than the declared tolerance"


__all__ = [
    "AcceptanceDecision",
    "CandidateComponent",
    "CandidateSystem",
    "ComponentMetric",
    "CorrectionProposal",
    "CorrectionScope",
    "MetricDirection",
    "ValidationBundle",
    "bind_candidate_system",
    "decide_candidate_acceptance",
]
