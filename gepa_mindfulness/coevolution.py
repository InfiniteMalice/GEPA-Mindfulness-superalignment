"""Durable authority for controlled offline model and harness coevolution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, cast
from uuid import uuid4

from evaluation.v5_records import V5EvaluationRecord
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.learning_surfaces import (
    EvaluationAuthority,
    EvaluationEpoch,
    EvaluationEpochStore,
    ValidationReceipt,
    ValidationSplit,
    ValidationTarget,
    evaluation_record_id,
)
from gepa_mindfulness.verification.failure_graph import FailureGraph
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

EnumT = TypeVar("EnumT", bound=Enum)
_METRIC_NAMES = ("correctness", "calibration", "abstention", "epistemic_process", "total")


class CandidateComponent(str, Enum):
    """A versioned component changed by one offline candidate."""

    MODEL = "model"
    HARNESS = "harness"


class CorrectionScope(str, Enum):
    """The source scope of a teacher correction proposal."""

    LOCALIZED_FAILURE = "localized_failure"
    WHOLE_TRAJECTORY = "whole_trajectory"


class MetricDirection(str, Enum):
    """The declared preferred direction for a component metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricAggregation(str, Enum):
    """The authority-approved aggregation over matched V5 records."""

    ARITHMETIC_MEAN = "arithmetic_mean"


@dataclass(frozen=True, slots=True)
class CoevolutionAuthority:
    """Durable coevolution catalog identity and pinned evaluation authority."""

    catalog_path: str
    catalog_id: str
    authority_domain: str
    evaluation_authority: EvaluationAuthority
    lineage_id: str

    def __post_init__(self) -> None:
        _require_token(self.catalog_path, "catalog_path")
        if _canonical_path(self.catalog_path) != self.catalog_path:
            raise ValueError("catalog_path must be canonical")
        for name in ("catalog_id", "authority_domain", "lineage_id"):
            _require_token(getattr(self, name), name)
        object.__setattr__(
            self, "evaluation_authority", _snapshot_eval_authority(self.evaluation_authority)
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible authority identity."""

        return {
            "catalog_path": self.catalog_path,
            "catalog_id": self.catalog_id,
            "authority_domain": self.authority_domain,
            "evaluation_authority": _eval_authority_payload(self.evaluation_authority),
            "lineage_id": self.lineage_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> CoevolutionAuthority:
        fields = _mapping(
            value,
            {
                "catalog_path",
                "catalog_id",
                "authority_domain",
                "evaluation_authority",
                "lineage_id",
            },
            "CoevolutionAuthority",
        )
        return cls(
            cast(str, fields["catalog_path"]),
            cast(str, fields["catalog_id"]),
            cast(str, fields["authority_domain"]),
            _eval_authority_from_dict(fields["evaluation_authority"]),
            cast(str, fields["lineage_id"]),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryBinding:
    """Store-issued identity for one canonical action-bound source trajectory."""

    trajectory_id: str
    trajectory_digest: str
    source_epoch_id: str
    event_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    source_evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "source_epoch_id"):
            _require_token(getattr(self, name), name)
        _require_sha256(self.trajectory_digest, "trajectory_digest")
        object.__setattr__(self, "event_ids", _tokens(self.event_ids, "event_ids"))
        object.__setattr__(self, "action_ids", _tokens(self.action_ids, "action_ids"))
        object.__setattr__(
            self,
            "source_evidence_refs",
            _evidence(self.source_evidence_refs, "source_evidence_refs"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible binding."""

        return {
            "trajectory_id": self.trajectory_id,
            "trajectory_digest": self.trajectory_digest,
            "source_epoch_id": self.source_epoch_id,
            "event_ids": list(self.event_ids),
            "action_ids": list(self.action_ids),
            "source_evidence_refs": [item.to_dict() for item in self.source_evidence_refs],
        }

    @classmethod
    def from_dict(cls, value: object) -> TrajectoryBinding:
        fields = _mapping(
            value,
            {
                "trajectory_id",
                "trajectory_digest",
                "source_epoch_id",
                "event_ids",
                "action_ids",
                "source_evidence_refs",
            },
            "TrajectoryBinding",
        )
        return cls(
            cast(str, fields["trajectory_id"]),
            cast(str, fields["trajectory_digest"]),
            cast(str, fields["source_epoch_id"]),
            _restore_tokens(fields["event_ids"], "event_ids"),
            _restore_tokens(fields["action_ids"], "action_ids"),
            _restore_evidence(fields["source_evidence_refs"], "source_evidence_refs"),
        )


@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    """A non-authoritative teacher proposal for one registered localized failure."""

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
    trajectory_digest: str
    scope: CorrectionScope = CorrectionScope.LOCALIZED_FAILURE
    _binding: str = field(init=False, repr=False, compare=False)

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
        _require_sha256(self.trajectory_digest, "trajectory_digest")
        if type(self.scope) is not CorrectionScope:
            raise ValueError("scope must be an exact CorrectionScope")
        if self.scope is not CorrectionScope.LOCALIZED_FAILURE:
            raise ValueError("correction scope must name one localized failure")
        graph = _failure_graph(self.failure_graph)
        verifier_refs = _tokens(self.localization_verifier_refs, "localization_verifier_refs")
        source_refs = _evidence(self.source_evidence_refs, "source_evidence_refs")
        teacher_refs = _evidence(self.teacher_evidence_refs, "teacher_evidence_refs")
        components = _components(self.changed_components)
        _validate_localization(
            graph,
            self.localized_failure_id,
            verifier_refs,
            source_refs,
        )
        object.__setattr__(self, "failure_graph", graph)
        object.__setattr__(self, "localization_verifier_refs", verifier_refs)
        object.__setattr__(self, "source_evidence_refs", source_refs)
        object.__setattr__(self, "teacher_evidence_refs", teacher_refs)
        object.__setattr__(self, "changed_components", components)
        object.__setattr__(self, "_binding", _digest(_proposal_payload(self)))

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated proposal snapshot."""

        _check_binding(self, _proposal_payload(self), self._binding, "CorrectionProposal")
        return _proposal_payload(self)

    @classmethod
    def from_dict(cls, value: object) -> CorrectionProposal:
        fields = _mapping(value, _PROPOSAL_FIELDS, "CorrectionProposal")
        return cls(
            proposal_id=cast(str, fields["proposal_id"]),
            source_trajectory_id=cast(str, fields["source_trajectory_id"]),
            source_action_id=cast(str, fields["source_action_id"]),
            source_epoch_id=cast(str, fields["source_epoch_id"]),
            failure_graph=FailureGraph.from_dict(fields["failure_graph"]),
            localized_failure_id=cast(str, fields["localized_failure_id"]),
            localization_verifier_refs=_restore_tokens(
                fields["localization_verifier_refs"], "localization_verifier_refs"
            ),
            source_evidence_refs=_restore_evidence(
                fields["source_evidence_refs"], "source_evidence_refs"
            ),
            teacher_correction=cast(str, fields["teacher_correction"]),
            teacher_evidence_refs=_restore_evidence(
                fields["teacher_evidence_refs"], "teacher_evidence_refs"
            ),
            changed_components=_restore_components(fields["changed_components"]),
            trajectory_digest=cast(str, fields["trajectory_digest"]),
            scope=_enum(fields["scope"], CorrectionScope, "scope"),
        )


_PROPOSAL_FIELDS = {
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
    "trajectory_digest",
    "scope",
}


@dataclass(frozen=True, slots=True)
class CandidateSystem:
    """Canonical candidate target registered by one coevolution authority."""

    candidate_id: str
    correction: CorrectionProposal
    authority: CoevolutionAuthority
    source_epoch_id: str
    candidate_epoch_id: str
    epoch_revision: int
    model_version: str
    harness_version: str
    changed_components: tuple[CandidateComponent, ...]
    artifact_digest: str
    rollback_target_epoch_id: str
    _binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "source_epoch_id",
            "candidate_epoch_id",
            "model_version",
            "harness_version",
            "rollback_target_epoch_id",
        ):
            _require_token(getattr(self, name), name)
        if type(self.epoch_revision) is not int or self.epoch_revision < 0:
            raise ValueError("epoch_revision must be a nonnegative exact integer")
        correction = CorrectionProposal.from_dict(self.correction.to_dict())
        authority = CoevolutionAuthority.from_dict(self.authority.to_dict())
        components = _components(self.changed_components)
        _require_sha256(self.artifact_digest, "artifact_digest")
        if self.source_epoch_id != correction.source_epoch_id:
            raise ValueError("source_epoch_id must match correction")
        if components != correction.changed_components:
            raise ValueError("changed_components must match correction")
        if self.rollback_target_epoch_id != self.source_epoch_id:
            raise ValueError("rollback target must be the source epoch")
        object.__setattr__(self, "correction", correction)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "changed_components", components)
        object.__setattr__(self, "_binding", _digest(_candidate_payload(self)))

    @property
    def lineage_id(self) -> str:
        """Return the pinned evaluation lineage."""

        return self.authority.lineage_id

    def validation_target(self) -> ValidationTarget:
        """Return the registered evaluation target identity."""

        self.to_dict()
        return ValidationTarget(
            self.candidate_id,
            "model-harness-candidate",
            self.harness_version,
            self.artifact_digest,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated candidate snapshot."""

        _check_binding(self, _candidate_payload(self), self._binding, "CandidateSystem")
        return _candidate_payload(self)

    @classmethod
    def from_dict(cls, value: object) -> CandidateSystem:
        fields = _mapping(value, _CANDIDATE_FIELDS, "CandidateSystem")
        return cls(
            cast(str, fields["candidate_id"]),
            CorrectionProposal.from_dict(fields["correction"]),
            CoevolutionAuthority.from_dict(fields["authority"]),
            cast(str, fields["source_epoch_id"]),
            cast(str, fields["candidate_epoch_id"]),
            cast(int, fields["epoch_revision"]),
            cast(str, fields["model_version"]),
            cast(str, fields["harness_version"]),
            _restore_components(fields["changed_components"]),
            cast(str, fields["artifact_digest"]),
            cast(str, fields["rollback_target_epoch_id"]),
        )


_CANDIDATE_FIELDS = {
    "candidate_id",
    "correction",
    "authority",
    "source_epoch_id",
    "candidate_epoch_id",
    "epoch_revision",
    "model_version",
    "harness_version",
    "changed_components",
    "artifact_digest",
    "rollback_target_epoch_id",
}


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One authority-pinned V5 score comparison rule."""

    name: str
    direction: MetricDirection
    tolerance: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _METRIC_NAMES:
            raise ValueError("name must be a derivable V5 score component")
        if type(self.direction) is not MetricDirection:
            raise ValueError("direction must be an exact MetricDirection")
        _finite(self.tolerance, "tolerance")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: object) -> MetricSpec:
        fields = _mapping(value, {"name", "direction", "tolerance"}, "MetricSpec")
        return cls(
            cast(str, fields["name"]),
            _enum(fields["direction"], MetricDirection, "direction"),
            cast(float, fields["tolerance"]),
        )


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """Complete schema and aggregation policy for V5 component comparisons."""

    policy_id: str
    specs: tuple[MetricSpec, ...]
    primary_metric_name: str
    aggregation: MetricAggregation

    def __post_init__(self) -> None:
        _require_token(self.policy_id, "policy_id")
        if type(self.specs) is not tuple:
            raise ValueError("specs must be an exact tuple")
        specs = tuple(_metric_spec(item) for item in self.specs)
        if tuple(item.name for item in specs) != _METRIC_NAMES:
            raise ValueError("specs must contain every V5 score key exactly once in schema order")
        if (
            type(self.primary_metric_name) is not str
            or self.primary_metric_name not in _METRIC_NAMES
        ):
            raise ValueError("primary_metric_name must name one required metric")
        if type(self.aggregation) is not MetricAggregation:
            raise ValueError("aggregation must be an exact MetricAggregation")
        object.__setattr__(self, "specs", specs)

    @property
    def policy_digest(self) -> str:
        """Return the complete content digest for the policy."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "specs": [item.to_dict() for item in self.specs],
            "primary_metric_name": self.primary_metric_name,
            "aggregation": self.aggregation.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> MetricPolicy:
        fields = _mapping(
            value,
            {"policy_id", "specs", "primary_metric_name", "aggregation"},
            "MetricPolicy",
        )
        return cls(
            cast(str, fields["policy_id"]),
            tuple(MetricSpec.from_dict(item) for item in _list(fields["specs"], "specs")),
            cast(str, fields["primary_metric_name"]),
            _enum(fields["aggregation"], MetricAggregation, "aggregation"),
        )


@dataclass(frozen=True, slots=True)
class ComponentMetric:
    """One record-derived metric comparison with mutation-resistant construction binding."""

    name: str
    baseline_value: float
    candidate_value: float
    direction: MetricDirection
    tolerance: float
    _binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _METRIC_NAMES:
            raise ValueError("name must be a derivable V5 score component")
        _finite(self.baseline_value, "baseline_value")
        _finite(self.candidate_value, "candidate_value")
        if type(self.direction) is not MetricDirection:
            raise ValueError("direction must be an exact MetricDirection")
        _finite(self.tolerance, "tolerance")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        object.__setattr__(self, "_binding", _digest(_component_payload(self)))

    def is_non_worse(self) -> bool:
        """Compare only after validating the original component construction."""

        self.to_dict()
        if self.direction is MetricDirection.HIGHER_IS_BETTER:
            return self.candidate_value >= self.baseline_value - self.tolerance
        return self.candidate_value <= self.baseline_value + self.tolerance

    def to_dict(self) -> dict[str, object]:
        payload = _component_payload(self)
        _check_binding(self, payload, self._binding, "ComponentMetric")
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ComponentMetric:
        fields = _mapping(
            value,
            {"name", "baseline_value", "candidate_value", "direction", "tolerance"},
            "ComponentMetric",
        )
        return cls(
            cast(str, fields["name"]),
            cast(float, fields["baseline_value"]),
            cast(float, fields["candidate_value"]),
            _enum(fields["direction"], MetricDirection, "direction"),
            cast(float, fields["tolerance"]),
        )


@dataclass(frozen=True, slots=True)
class MetricComparisonReceipt:
    """Store-issued comparison derived from matched canonical V5 record sets."""

    receipt_id: str
    catalog_id: str
    authority_domain: str
    candidate_id: str
    source_epoch_id: str
    candidate_epoch_id: str
    policy_id: str
    policy_digest: str
    primary_metric_name: str
    aggregation: MetricAggregation
    source_receipt_id: str
    candidate_receipt_id: str
    source_record_ids: tuple[str, ...]
    candidate_record_ids: tuple[str, ...]
    logical_cell_digests: tuple[str, ...]
    component_metrics: tuple[ComponentMetric, ...]

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "catalog_id",
            "authority_domain",
            "candidate_id",
            "source_epoch_id",
            "candidate_epoch_id",
            "policy_id",
            "primary_metric_name",
            "source_receipt_id",
            "candidate_receipt_id",
        ):
            _require_token(getattr(self, name), name)
        _require_sha256(self.policy_digest, "policy_digest")
        if type(self.aggregation) is not MetricAggregation:
            raise ValueError("aggregation must be an exact MetricAggregation")
        object.__setattr__(
            self, "source_record_ids", _sha_ids(self.source_record_ids, "source_record_ids")
        )
        object.__setattr__(
            self,
            "candidate_record_ids",
            _sha_ids(self.candidate_record_ids, "candidate_record_ids"),
        )
        object.__setattr__(
            self,
            "logical_cell_digests",
            _sha_ids(self.logical_cell_digests, "logical_cell_digests"),
        )
        metrics = _metrics(self.component_metrics)
        if tuple(item.name for item in metrics) != _METRIC_NAMES:
            raise ValueError("component_metrics must preserve every V5 score component")
        object.__setattr__(self, "component_metrics", metrics)

    def to_dict(self) -> dict[str, object]:
        return _metric_receipt_payload(self)

    @classmethod
    def from_dict(cls, value: object) -> MetricComparisonReceipt:
        fields = _mapping(value, _METRIC_RECEIPT_FIELDS, "MetricComparisonReceipt")
        return cls(
            cast(str, fields["receipt_id"]),
            cast(str, fields["catalog_id"]),
            cast(str, fields["authority_domain"]),
            cast(str, fields["candidate_id"]),
            cast(str, fields["source_epoch_id"]),
            cast(str, fields["candidate_epoch_id"]),
            cast(str, fields["policy_id"]),
            cast(str, fields["policy_digest"]),
            cast(str, fields["primary_metric_name"]),
            _enum(fields["aggregation"], MetricAggregation, "aggregation"),
            cast(str, fields["source_receipt_id"]),
            cast(str, fields["candidate_receipt_id"]),
            _restore_tokens(fields["source_record_ids"], "source_record_ids"),
            _restore_tokens(fields["candidate_record_ids"], "candidate_record_ids"),
            _restore_tokens(fields["logical_cell_digests"], "logical_cell_digests"),
            tuple(
                ComponentMetric.from_dict(item)
                for item in _list(fields["component_metrics"], "component_metrics")
            ),
        )


_METRIC_RECEIPT_FIELDS = {
    "receipt_id",
    "catalog_id",
    "authority_domain",
    "candidate_id",
    "source_epoch_id",
    "candidate_epoch_id",
    "policy_id",
    "policy_digest",
    "primary_metric_name",
    "aggregation",
    "source_receipt_id",
    "candidate_receipt_id",
    "source_record_ids",
    "candidate_record_ids",
    "logical_cell_digests",
    "component_metrics",
}


@dataclass(frozen=True, slots=True)
class ProtectedSuiteManifest:
    """Authority-pinned protected suite derived from a source protected receipt."""

    suite_id: str
    suite_digest: str
    source_epoch_id: str
    source_receipt_id: str
    source_record_ids: tuple[str, ...]
    logical_cell_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("suite_id", "source_epoch_id", "source_receipt_id"):
            _require_token(getattr(self, name), name)
        _require_sha256(self.suite_digest, "suite_digest")
        object.__setattr__(
            self, "source_record_ids", _sha_ids(self.source_record_ids, "source_record_ids")
        )
        object.__setattr__(
            self,
            "logical_cell_digests",
            _sha_ids(self.logical_cell_digests, "logical_cell_digests"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "suite_digest": self.suite_digest,
            "source_epoch_id": self.source_epoch_id,
            "source_receipt_id": self.source_receipt_id,
            "source_record_ids": list(self.source_record_ids),
            "logical_cell_digests": list(self.logical_cell_digests),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProtectedSuiteManifest:
        fields = _mapping(
            value,
            {
                "suite_id",
                "suite_digest",
                "source_epoch_id",
                "source_receipt_id",
                "source_record_ids",
                "logical_cell_digests",
            },
            "ProtectedSuiteManifest",
        )
        return cls(
            cast(str, fields["suite_id"]),
            cast(str, fields["suite_digest"]),
            cast(str, fields["source_epoch_id"]),
            cast(str, fields["source_receipt_id"]),
            _restore_tokens(fields["source_record_ids"], "source_record_ids"),
            _restore_tokens(fields["logical_cell_digests"], "logical_cell_digests"),
        )


@dataclass(frozen=True, slots=True)
class ValidationBundle:
    """Candidate evidence resolved entirely through one coevolution authority."""

    candidate: CandidateSystem
    held_out_receipt: ValidationReceipt
    protected_receipt: ValidationReceipt
    metric_receipt: MetricComparisonReceipt
    protected_suite_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", CandidateSystem.from_dict(self.candidate.to_dict()))
        object.__setattr__(
            self, "held_out_receipt", _validation_receipt(self.held_out_receipt, "held_out_receipt")
        )
        object.__setattr__(
            self,
            "protected_receipt",
            _validation_receipt(self.protected_receipt, "protected_receipt"),
        )
        object.__setattr__(
            self, "metric_receipt", MetricComparisonReceipt.from_dict(self.metric_receipt.to_dict())
        )
        _require_token(self.protected_suite_id, "protected_suite_id")

    @property
    def component_metrics(self) -> tuple[ComponentMetric, ...]:
        """Expose the complete record-derived comparisons without scalar collapse."""

        receipt = MetricComparisonReceipt.from_dict(self.metric_receipt.to_dict())
        return receipt.component_metrics

    @property
    def primary_metric_name(self) -> str:
        """Return the policy primary metric after authority validation at decision time."""

        return next(
            item.name for item in self.metric_receipt.component_metrics if item.name == "total"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "held_out_receipt": self.held_out_receipt.to_dict(),
            "protected_receipt": self.protected_receipt.to_dict(),
            "metric_receipt": self.metric_receipt.to_dict(),
            "protected_suite_id": self.protected_suite_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationBundle:
        fields = _mapping(
            value,
            {
                "candidate",
                "held_out_receipt",
                "protected_receipt",
                "metric_receipt",
                "protected_suite_id",
            },
            "ValidationBundle",
        )
        return cls(
            CandidateSystem.from_dict(fields["candidate"]),
            ValidationReceipt.from_dict(fields["held_out_receipt"]),
            ValidationReceipt.from_dict(fields["protected_receipt"]),
            MetricComparisonReceipt.from_dict(fields["metric_receipt"]),
            cast(str, fields["protected_suite_id"]),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """Store-issued, audit-only decision; deserialized copies are untrusted."""

    decision_id: str
    catalog_id: str
    authority_domain: str
    revision: int
    accepted: bool
    candidate: CandidateSystem
    proposal_id: str
    trajectory_id: str
    trajectory_digest: str
    held_out_receipt_id: str
    held_out_record_ids: tuple[str, ...]
    held_out_cell_digests: tuple[str, ...]
    protected_receipt_id: str
    protected_record_ids: tuple[str, ...]
    protected_cell_digests: tuple[str, ...]
    rollback_target_epoch_id: str
    protected_suite_id: str
    protected_suite_digest: str
    metric_receipt: MetricComparisonReceipt
    reason: str
    execute_candidate: bool
    decision_digest: str
    _authoritative: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "decision_id",
            "catalog_id",
            "authority_domain",
            "proposal_id",
            "trajectory_id",
            "held_out_receipt_id",
            "protected_receipt_id",
            "protected_suite_id",
            "rollback_target_epoch_id",
            "reason",
        ):
            _require_token(getattr(self, name), name)
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be a positive exact integer")
        if type(self.accepted) is not bool or self.execute_candidate is not False:
            raise ValueError("decision booleans must be exact and execute_candidate must be false")
        object.__setattr__(self, "candidate", CandidateSystem.from_dict(self.candidate.to_dict()))
        _require_sha256(self.trajectory_digest, "trajectory_digest")
        object.__setattr__(
            self, "held_out_record_ids", _sha_ids(self.held_out_record_ids, "held_out_record_ids")
        )
        object.__setattr__(
            self,
            "held_out_cell_digests",
            _sha_ids(self.held_out_cell_digests, "held_out_cell_digests"),
        )
        object.__setattr__(
            self,
            "protected_record_ids",
            _sha_ids(self.protected_record_ids, "protected_record_ids"),
        )
        object.__setattr__(
            self,
            "protected_cell_digests",
            _sha_ids(self.protected_cell_digests, "protected_cell_digests"),
        )
        _require_sha256(self.protected_suite_digest, "protected_suite_digest")
        object.__setattr__(
            self, "metric_receipt", MetricComparisonReceipt.from_dict(self.metric_receipt.to_dict())
        )
        if self.rollback_target_epoch_id != self.candidate.source_epoch_id:
            raise ValueError("rollback target must equal source epoch")
        _require_sha256(self.decision_digest, "decision_digest")
        payload = _decision_payload(self, include_digest=False)
        if self.decision_digest != _digest(payload):
            raise ValueError("decision_digest does not match complete decision provenance")

    @property
    def is_authoritative(self) -> bool:
        """Return whether a live authority validated this in-memory copy."""

        return self._authoritative

    @property
    def component_metrics(self) -> tuple[ComponentMetric, ...]:
        self.to_dict()
        return self.metric_receipt.component_metrics

    @property
    def primary_metric_name(self) -> str:
        self.to_dict()
        return _primary_metric_name(self.metric_receipt)

    def to_dict(self) -> dict[str, object]:
        return _decision_payload(self, include_digest=True)

    @classmethod
    def from_dict(cls, value: object) -> AcceptanceDecision:
        fields = _mapping(value, _DECISION_FIELDS, "AcceptanceDecision")
        return cls(
            decision_id=cast(str, fields["decision_id"]),
            catalog_id=cast(str, fields["catalog_id"]),
            authority_domain=cast(str, fields["authority_domain"]),
            revision=cast(int, fields["revision"]),
            accepted=cast(bool, fields["accepted"]),
            candidate=CandidateSystem.from_dict(fields["candidate"]),
            proposal_id=cast(str, fields["proposal_id"]),
            trajectory_id=cast(str, fields["trajectory_id"]),
            trajectory_digest=cast(str, fields["trajectory_digest"]),
            held_out_receipt_id=cast(str, fields["held_out_receipt_id"]),
            held_out_record_ids=_restore_tokens(
                fields["held_out_record_ids"], "held_out_record_ids"
            ),
            held_out_cell_digests=_restore_tokens(
                fields["held_out_cell_digests"], "held_out_cell_digests"
            ),
            protected_receipt_id=cast(str, fields["protected_receipt_id"]),
            protected_record_ids=_restore_tokens(
                fields["protected_record_ids"], "protected_record_ids"
            ),
            protected_cell_digests=_restore_tokens(
                fields["protected_cell_digests"], "protected_cell_digests"
            ),
            rollback_target_epoch_id=cast(str, fields["rollback_target_epoch_id"]),
            protected_suite_id=cast(str, fields["protected_suite_id"]),
            protected_suite_digest=cast(str, fields["protected_suite_digest"]),
            metric_receipt=MetricComparisonReceipt.from_dict(fields["metric_receipt"]),
            reason=cast(str, fields["reason"]),
            execute_candidate=cast(bool, fields["execute_candidate"]),
            decision_digest=cast(str, fields["decision_digest"]),
        )


_DECISION_FIELDS = {
    "decision_id",
    "catalog_id",
    "authority_domain",
    "revision",
    "accepted",
    "candidate",
    "proposal_id",
    "trajectory_id",
    "trajectory_digest",
    "held_out_receipt_id",
    "held_out_record_ids",
    "held_out_cell_digests",
    "protected_receipt_id",
    "protected_record_ids",
    "protected_cell_digests",
    "rollback_target_epoch_id",
    "protected_suite_id",
    "protected_suite_digest",
    "metric_receipt",
    "reason",
    "execute_candidate",
    "decision_digest",
}


class CoevolutionStore:
    """SQLite authority pinned to one evaluation catalog, domain, and lineage."""

    __slots__ = ("_database_path", "_domain", "_evaluation_store", "_lineage_id")
    _database_path: str
    _domain: str
    _evaluation_store: EvaluationEpochStore
    _lineage_id: str

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        authority_domain: str,
        evaluation_store: EvaluationEpochStore,
        *,
        lineage_id: str,
    ) -> None:
        path = _canonical_path(database_path)
        domain = _require_token(authority_domain, "authority_domain")
        lineage = _require_token(lineage_id, "lineage_id")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError("coevolution store parent directory must already exist")
        if type(evaluation_store) is not EvaluationEpochStore:
            raise ValueError("evaluation_store must be an exact EvaluationEpochStore")
        evaluation_store.open(lineage)
        object.__setattr__(self, "_database_path", path)
        object.__setattr__(self, "_domain", domain)
        object.__setattr__(self, "_evaluation_store", evaluation_store)
        object.__setattr__(self, "_lineage_id", lineage)
        with _connect(path) as connection:
            _initialize_store(connection)
            _pin_metadata(connection, domain, evaluation_store.authority(), lineage)

    def authority(self) -> CoevolutionAuthority:
        """Return the pinned durable authority identity."""

        with _connect(self._database_path) as connection:
            row = _metadata(connection)
            _validate_store_metadata(
                row,
                self._domain,
                self._evaluation_store.authority(),
                self._lineage_id,
            )
        return CoevolutionAuthority(
            self._database_path,
            _require_token(row[0], "catalog_id"),
            self._domain,
            self._evaluation_store.authority(),
            self._lineage_id,
        )

    def evaluation_epochs(self) -> tuple[EvaluationEpoch, ...]:
        """Return a detached view of the pinned evaluation lineage."""

        self.authority()
        return self._evaluation_store.open(self._lineage_id).snapshot()

    def register_trajectory(
        self,
        *,
        trajectory_id: str,
        source_epoch_id: str,
        events: tuple[EventEnvelope, ...],
        source_evidence_refs: tuple[EvidenceReference, ...],
    ) -> TrajectoryBinding:
        """Atomically register one canonical action-bound trajectory snapshot."""

        identifier = _require_token(trajectory_id, "trajectory_id")
        epoch_id = _require_token(source_epoch_id, "source_epoch_id")
        canonical_events = _events(events)
        validate_action_bound_sequence(canonical_events)
        references = _evidence(source_evidence_refs, "source_evidence_refs")
        event_evidence = tuple(
            reference for event in canonical_events for reference in event.evidence_refs
        )
        if tuple(item.reference_id for item in references) != event_evidence:
            raise ValueError("source_evidence_refs must exactly match trajectory event evidence")
        _revision, epoch, _records = self._evaluation_store.resolve_epoch(
            self._lineage_id, epoch_id
        )
        if not epoch.closed:
            raise ValueError("trajectory source epoch must be closed")
        action_ids = tuple(
            dict.fromkeys(
                event.action_id for event in canonical_events if event.action_id is not None
            )
        )
        event_payloads = [event.to_dict() for event in canonical_events]
        digest = _digest(
            {
                "trajectory_id": identifier,
                "source_epoch_id": epoch_id,
                "events": event_payloads,
                "source_evidence_refs": [item.to_dict() for item in references],
            }
        )
        binding = TrajectoryBinding(
            identifier,
            digest,
            epoch_id,
            tuple(event.event_id for event in canonical_events),
            action_ids,
            references,
        )
        payload = json.dumps(
            {"binding": binding.to_dict(), "events": event_payloads},
            separators=(",", ":"),
            sort_keys=True,
        )
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            try:
                connection.execute(
                    "INSERT INTO trajectories VALUES (?, ?, ?, ?)",
                    (self._domain, identifier, digest, payload),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError("trajectory identity or digest is already registered") from exc
        return TrajectoryBinding.from_dict(binding.to_dict())

    def register_candidate(
        self,
        *,
        candidate_id: str,
        correction: CorrectionProposal,
        artifact_digest: str,
    ) -> CandidateSystem:
        """Atomically bind one candidate target to the current store-issued epoch."""

        identifier = _require_token(candidate_id, "candidate_id")
        proposal = CorrectionProposal.from_dict(correction.to_dict())
        _require_sha256(artifact_digest, "artifact_digest")
        epochs = self.evaluation_epochs()
        if len(epochs) < 2:
            raise ValueError("candidate requires a closed source and new candidate epoch")
        source, candidate_epoch = epochs[-2:]
        _validate_transition(source, candidate_epoch, proposal)
        if candidate_epoch.closed or candidate_epoch.record_ids:
            raise ValueError("candidate target must be registered in a new empty open epoch")
        trajectory, events = self._read_trajectory(proposal.source_trajectory_id)
        _validate_proposal_trajectory(proposal, trajectory, events)
        revision, canonical_epoch, _records = self._evaluation_store.resolve_epoch(
            self._lineage_id, candidate_epoch.epoch_id
        )
        candidate = CandidateSystem(
            identifier,
            proposal,
            self.authority(),
            source.epoch_id,
            canonical_epoch.epoch_id,
            revision,
            canonical_epoch.model_version,
            canonical_epoch.harness_version,
            proposal.changed_components,
            artifact_digest,
            source.epoch_id,
        )
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            try:
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                    (
                        self._domain,
                        identifier,
                        canonical_epoch.epoch_id,
                        artifact_digest,
                        json.dumps(candidate.to_dict(), separators=(",", ":"), sort_keys=True),
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError("candidate epoch or target is already registered") from exc
        return CandidateSystem.from_dict(candidate.to_dict())

    def register_candidate_snapshot(self, candidate: CandidateSystem) -> CandidateSystem:
        """Reject aliases instead of accepting caller-authored candidate snapshots."""

        snapshot = CandidateSystem.from_dict(candidate.to_dict())
        try:
            canonical = self.read_candidate(snapshot.candidate_id)
        except KeyError as exc:
            raise ValueError("candidate target alias is not registered") from exc
        if canonical != snapshot:
            raise ValueError("candidate target alias does not match the registered target")
        return canonical

    def read_candidate(self, candidate_id: str) -> CandidateSystem:
        """Read and revalidate one canonical registered candidate."""

        identifier = _require_token(candidate_id, "candidate_id")
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload FROM candidates WHERE authority_domain = ? AND candidate_id = ?",
                (self._domain, identifier),
            ).fetchone()
        if row is None:
            raise KeyError(identifier)
        candidate = CandidateSystem.from_dict(json.loads(row[0]))
        self._validate_candidate_live(candidate)
        return candidate

    def issue_source_validation_receipt(
        self,
        *,
        epoch_id: str,
        split: ValidationSplit,
        records: tuple[V5EvaluationRecord, ...],
    ) -> ValidationReceipt:
        """Issue a receipt using a store-derived baseline target, never a caller label."""

        _revision, epoch, _canonical = self._evaluation_store.resolve_epoch(
            self._lineage_id, epoch_id
        )
        target = _source_validation_target(epoch)
        return self._evaluation_store.issue_validation_receipt(
            self._evaluation_store.open(self._lineage_id),
            epoch_id=epoch.epoch_id,
            target=target,
            split=split,
            records=records,
        )

    def issue_candidate_validation_receipt(
        self,
        *,
        candidate_id: str,
        split: ValidationSplit,
        records: tuple[V5EvaluationRecord, ...],
    ) -> ValidationReceipt:
        """Issue a receipt using the registered canonical candidate target."""

        candidate = self.read_candidate(candidate_id)
        return self._evaluation_store.issue_validation_receipt(
            self._evaluation_store.open(self._lineage_id),
            epoch_id=candidate.candidate_epoch_id,
            target=candidate.validation_target(),
            split=split,
            records=records,
        )

    def register_protected_suite(
        self,
        *,
        suite_id: str,
        source_receipt: ValidationReceipt,
    ) -> ProtectedSuiteManifest:
        """Pin exact source protected logical cells as the required regression suite."""

        identifier = _require_token(suite_id, "suite_id")
        receipt = self._evaluation_store.validate_validation_receipt(source_receipt)
        if receipt.split is not ValidationSplit.PROTECTED:
            raise ValueError("protected suite source receipt must have protected split")
        self._validate_source_receipt(receipt)
        records = self._records_for_receipt(receipt)
        cells = tuple(sorted(_logical_cell(record) for record in records))
        content = {
            "suite_id": identifier,
            "source_epoch_id": receipt.epoch_id,
            "source_receipt_id": receipt.receipt_id,
            "source_record_ids": list(receipt.record_ids),
            "logical_cell_digests": list(cells),
        }
        manifest = ProtectedSuiteManifest(
            identifier,
            _digest(content),
            receipt.epoch_id,
            receipt.receipt_id,
            receipt.record_ids,
            cells,
        )
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            try:
                connection.execute(
                    "INSERT INTO protected_suites VALUES (?, ?, ?, ?)",
                    (self._domain, identifier, manifest.suite_digest, _json(manifest.to_dict())),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError(
                    "protected suite identity or digest is already registered"
                ) from exc
        return ProtectedSuiteManifest.from_dict(manifest.to_dict())

    def register_metric_policy(self, policy: MetricPolicy) -> MetricPolicy:
        """Atomically pin one complete V5 metric schema and aggregation policy."""

        snapshot = MetricPolicy.from_dict(policy.to_dict())
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            try:
                connection.execute(
                    "INSERT INTO metric_policies VALUES (?, ?, ?, ?)",
                    (
                        self._domain,
                        snapshot.policy_id,
                        snapshot.policy_digest,
                        _json(snapshot.to_dict()),
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError("metric policy identity or digest is already registered") from exc
        return MetricPolicy.from_dict(snapshot.to_dict())

    def issue_metric_comparison(
        self,
        *,
        candidate_id: str,
        policy_id: str,
        source_receipt: ValidationReceipt,
        candidate_receipt: ValidationReceipt,
    ) -> MetricComparisonReceipt:
        """Derive and persist comparisons from matched canonical held-out records."""

        candidate = self.read_candidate(candidate_id)
        policy = self._read_policy(policy_id)
        source = self._evaluation_store.validate_validation_receipt(source_receipt)
        evaluated = self._evaluation_store.validate_validation_receipt(candidate_receipt)
        if (
            source.split is not ValidationSplit.HELD_OUT
            or evaluated.split is not ValidationSplit.HELD_OUT
        ):
            raise ValueError("metric comparisons require source and candidate held-out receipts")
        if source.epoch_id != candidate.source_epoch_id:
            raise ValueError("metric baseline receipt must belong to candidate source epoch")
        self._validate_source_receipt(source)
        self._validate_candidate_receipt(evaluated, candidate, ValidationSplit.HELD_OUT)
        source_records = self._records_for_receipt(source)
        candidate_records = self._records_for_receipt(evaluated)
        source_by_cell = {_logical_cell(record): record for record in source_records}
        candidate_by_cell = {_logical_cell(record): record for record in candidate_records}
        if len(source_by_cell) != len(source_records) or len(candidate_by_cell) != len(
            candidate_records
        ):
            raise ValueError("metric receipts must not contain duplicate logical cells")
        source_cells = tuple(sorted(source_by_cell))
        candidate_cells = tuple(sorted(candidate_by_cell))
        if source_cells != candidate_cells:
            raise ValueError("metric source and candidate records require exact matched coverage")
        matched_source = tuple(source_by_cell[cell] for cell in source_cells)
        matched_candidate = tuple(candidate_by_cell[cell] for cell in source_cells)
        comparisons = tuple(
            ComponentMetric(
                spec.name,
                _aggregate(matched_source, spec.name, policy.aggregation),
                _aggregate(matched_candidate, spec.name, policy.aggregation),
                spec.direction,
                spec.tolerance,
            )
            for spec in policy.specs
        )
        receipt = MetricComparisonReceipt(
            str(uuid4()),
            self.authority().catalog_id,
            self._domain,
            candidate.candidate_id,
            candidate.source_epoch_id,
            candidate.candidate_epoch_id,
            policy.policy_id,
            policy.policy_digest,
            policy.primary_metric_name,
            policy.aggregation,
            source.receipt_id,
            evaluated.receipt_id,
            source.record_ids,
            evaluated.record_ids,
            source_cells,
            comparisons,
        )
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            connection.execute(
                "INSERT INTO metric_receipts VALUES (?, ?, ?, ?)",
                (
                    self._domain,
                    receipt.receipt_id,
                    candidate.candidate_id,
                    _json(receipt.to_dict()),
                ),
            )
            connection.commit()
        return MetricComparisonReceipt.from_dict(receipt.to_dict())

    def decide(self, bundle: ValidationBundle) -> AcceptanceDecision:
        """Persist one audit-only decision after reloading every canonical dependency."""

        snapshot = ValidationBundle.from_dict(bundle.to_dict())
        candidate = self.read_candidate(snapshot.candidate.candidate_id)
        if snapshot.candidate != candidate:
            raise ValueError("bundle candidate differs from canonical registered target")
        held = self._evaluation_store.validate_validation_receipt(snapshot.held_out_receipt)
        protected = self._evaluation_store.validate_validation_receipt(snapshot.protected_receipt)
        self._validate_candidate_receipt(held, candidate, ValidationSplit.HELD_OUT)
        self._validate_candidate_receipt(protected, candidate, ValidationSplit.PROTECTED)
        metric_receipt = self._validate_metric_receipt(snapshot.metric_receipt, candidate, held)
        manifest = self._read_manifest(snapshot.protected_suite_id)
        held_cells = tuple(
            sorted(_logical_cell(record) for record in self._records_for_receipt(held))
        )
        protected_records = self._records_for_receipt(protected)
        protected_cells = tuple(sorted(_logical_cell(record) for record in protected_records))
        if protected_cells != manifest.logical_cell_digests:
            raise ValueError("candidate protected receipt does not exactly cover pinned manifest")
        if manifest.source_epoch_id != candidate.source_epoch_id:
            raise ValueError("protected manifest must be pinned to candidate source epoch")
        primary = next(
            item
            for item in metric_receipt.component_metrics
            if item.name == _primary_metric_name(metric_receipt)
        )
        accepted = primary.is_non_worse()
        reason = (
            "candidate passed held-out and protected acceptance gates"
            if accepted
            else "candidate primary metric is worse than the declared tolerance"
        )
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            revision = _next_revision(connection, self._domain)
            values: dict[str, Any] = {
                "decision_id": str(uuid4()),
                "catalog_id": self.authority().catalog_id,
                "authority_domain": self._domain,
                "revision": revision,
                "accepted": accepted,
                "candidate": candidate,
                "proposal_id": candidate.correction.proposal_id,
                "trajectory_id": candidate.correction.source_trajectory_id,
                "trajectory_digest": candidate.correction.trajectory_digest,
                "held_out_receipt_id": held.receipt_id,
                "held_out_record_ids": held.record_ids,
                "held_out_cell_digests": held_cells,
                "protected_receipt_id": protected.receipt_id,
                "protected_record_ids": protected.record_ids,
                "protected_cell_digests": protected_cells,
                "rollback_target_epoch_id": candidate.source_epoch_id,
                "protected_suite_id": manifest.suite_id,
                "protected_suite_digest": manifest.suite_digest,
                "metric_receipt": metric_receipt,
                "reason": reason,
                "execute_candidate": False,
            }
            digest = _digest(_decision_values_payload(values))
            decision = AcceptanceDecision(**values, decision_digest=digest)
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, 0)",
                (
                    self._domain,
                    decision.decision_id,
                    revision,
                    candidate.candidate_id,
                    _json(decision.to_dict()),
                ),
            )
            connection.commit()
        return _mark_authoritative(decision)

    def validate_decision(self, decision: AcceptanceDecision) -> AcceptanceDecision:
        """Reload the canonical decision and all dependencies before granting authority."""

        snapshot = AcceptanceDecision.from_dict(decision.to_dict())
        canonical, _consumed = self._read_decision_row(snapshot.decision_id)
        if canonical != snapshot:
            raise ValueError("decision differs from canonical catalog record")
        self._revalidate_decision_dependencies(canonical)
        return _mark_authoritative(canonical)

    def read_decision(self, decision_id: str) -> AcceptanceDecision:
        """Read and revalidate one canonical decision."""

        decision, _consumed = self._read_decision_row(decision_id)
        self._revalidate_decision_dependencies(decision)
        return _mark_authoritative(decision)

    def consume_decision(self, decision: AcceptanceDecision) -> AcceptanceDecision:
        """Mark a validated decision consumed once without executing the candidate."""

        if type(decision) is not AcceptanceDecision or not decision.is_authoritative:
            raise ValueError("decision must be validated and authoritative before consumption")
        snapshot = AcceptanceDecision.from_dict(decision.to_dict())
        self._revalidate_decision_dependencies(snapshot)
        with _connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload, consumed FROM decisions "
                "WHERE authority_domain = ? AND decision_id = ?",
                (self._domain, snapshot.decision_id),
            ).fetchone()
            if row is None:
                raise ValueError("decision is absent from canonical catalog")
            canonical = AcceptanceDecision.from_dict(json.loads(row[0]))
            if canonical != snapshot:
                raise ValueError("decision differs from canonical catalog record")
            if row[1] != 0:
                raise ValueError("decision was already consumed")
            connection.execute(
                "UPDATE decisions SET consumed = 1 "
                "WHERE authority_domain = ? AND decision_id = ? AND consumed = 0",
                (self._domain, snapshot.decision_id),
            )
            connection.commit()
        self._revalidate_decision_dependencies(canonical)
        return _mark_authoritative(canonical)

    def _check_connection(self, connection: sqlite3.Connection) -> None:
        _validate_store_metadata(
            _metadata(connection),
            self._domain,
            self._evaluation_store.authority(),
            self._lineage_id,
        )

    def _read_trajectory(
        self, trajectory_id: str
    ) -> tuple[TrajectoryBinding, tuple[EventEnvelope, ...]]:
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload FROM trajectories "
                "WHERE authority_domain = ? AND trajectory_id = ?",
                (self._domain, trajectory_id),
            ).fetchone()
        if row is None:
            raise ValueError("correction references an unknown trajectory binding")
        payload = json.loads(row[0])
        binding = TrajectoryBinding.from_dict(payload["binding"])
        events = tuple(_event_from_dict(item) for item in payload["events"])
        validate_action_bound_sequence(events)
        rebuilt = _trajectory_digest(binding, events)
        if rebuilt != binding.trajectory_digest:
            raise ValueError("trajectory binding digest does not match canonical events")
        return binding, events

    def _read_policy(self, policy_id: str) -> MetricPolicy:
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload FROM metric_policies "
                "WHERE authority_domain = ? AND policy_id = ?",
                (self._domain, policy_id),
            ).fetchone()
        if row is None:
            raise ValueError("metric policy is not registered")
        return MetricPolicy.from_dict(json.loads(row[0]))

    def _read_manifest(self, suite_id: str) -> ProtectedSuiteManifest:
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload FROM protected_suites "
                "WHERE authority_domain = ? AND suite_id = ?",
                (self._domain, suite_id),
            ).fetchone()
        if row is None:
            raise ValueError("protected suite is not registered")
        manifest = ProtectedSuiteManifest.from_dict(json.loads(row[0]))
        content = manifest.to_dict()
        content.pop("suite_digest")
        if manifest.suite_digest != _digest(content):
            raise ValueError("protected suite digest is invalid")
        return manifest

    def _records_for_receipt(self, receipt: ValidationReceipt) -> tuple[V5EvaluationRecord, ...]:
        _revision, _epoch, records = self._evaluation_store.resolve_epoch(
            self._lineage_id, receipt.epoch_id
        )
        by_id = {evaluation_record_id(record): record for record in records}
        try:
            return tuple(by_id[item] for item in receipt.record_ids)
        except KeyError as exc:
            raise ValueError("receipt record is absent from canonical epoch") from exc

    def _validate_candidate_live(self, candidate: CandidateSystem) -> None:
        if candidate.authority != self.authority():
            raise ValueError("candidate belongs to a different coevolution catalog")
        revision, epoch, _records = self._evaluation_store.resolve_epoch(
            self._lineage_id, candidate.candidate_epoch_id
        )
        if revision < candidate.epoch_revision:
            raise ValueError("candidate epoch revision is invalid")
        if (epoch.model_version, epoch.harness_version) != (
            candidate.model_version,
            candidate.harness_version,
        ):
            raise ValueError("candidate versions differ from authoritative epoch")

    def _validate_candidate_receipt(
        self,
        receipt: ValidationReceipt,
        candidate: CandidateSystem,
        split: ValidationSplit,
    ) -> None:
        target = candidate.validation_target()
        if receipt.split is not split:
            raise ValueError(f"candidate receipt must have exact {split.value} split")
        if (
            receipt.lineage_id != self._lineage_id
            or receipt.epoch_id != candidate.candidate_epoch_id
        ):
            raise ValueError("candidate receipt lineage or epoch is invalid")
        if (
            receipt.target_artifact_id,
            receipt.target_skill_id,
            receipt.target_version,
            receipt.target_digest,
        ) != (target.artifact_id, target.skill_id, target.version, target.artifact_digest):
            raise ValueError("candidate receipt target differs from registered candidate")

    def _validate_source_receipt(self, receipt: ValidationReceipt) -> None:
        if receipt.lineage_id != self._lineage_id:
            raise ValueError("source receipt belongs to a different lineage")
        _revision, epoch, _records = self._evaluation_store.resolve_epoch(
            self._lineage_id, receipt.epoch_id
        )
        target = _source_validation_target(epoch)
        if (
            receipt.target_artifact_id,
            receipt.target_skill_id,
            receipt.target_version,
            receipt.target_digest,
        ) != (target.artifact_id, target.skill_id, target.version, target.artifact_digest):
            raise ValueError("source receipt target is not the authority-derived baseline")

    def _validate_metric_receipt(
        self,
        receipt: MetricComparisonReceipt,
        candidate: CandidateSystem,
        held: ValidationReceipt,
    ) -> MetricComparisonReceipt:
        snapshot = MetricComparisonReceipt.from_dict(receipt.to_dict())
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload FROM metric_receipts "
                "WHERE authority_domain = ? AND receipt_id = ?",
                (self._domain, snapshot.receipt_id),
            ).fetchone()
        if row is None or MetricComparisonReceipt.from_dict(json.loads(row[0])) != snapshot:
            raise ValueError("metric receipt differs from canonical catalog record")
        if snapshot.candidate_id != candidate.candidate_id:
            raise ValueError("metric receipt candidate is invalid")
        if snapshot.candidate_receipt_id != held.receipt_id:
            raise ValueError("metric receipt does not bind the supplied held-out receipt")
        policy = self._read_policy(snapshot.policy_id)
        declared = tuple((item.name, item.direction, item.tolerance) for item in policy.specs)
        compared = tuple(
            (item.name, item.direction, item.tolerance) for item in snapshot.component_metrics
        )
        if (
            snapshot.policy_digest != policy.policy_digest
            or snapshot.primary_metric_name != policy.primary_metric_name
            or snapshot.aggregation is not policy.aggregation
            or compared != declared
        ):
            raise ValueError("metric receipt policy snapshot is invalid")
        return snapshot

    def _read_decision_row(self, decision_id: str) -> tuple[AcceptanceDecision, bool]:
        identifier = _require_token(decision_id, "decision_id")
        with _connect(self._database_path) as connection:
            self._check_connection(connection)
            row = connection.execute(
                "SELECT payload, consumed FROM decisions "
                "WHERE authority_domain = ? AND decision_id = ?",
                (self._domain, identifier),
            ).fetchone()
        if row is None:
            raise ValueError("decision is absent from canonical catalog")
        return AcceptanceDecision.from_dict(json.loads(row[0])), bool(row[1])

    def _revalidate_decision_dependencies(self, decision: AcceptanceDecision) -> None:
        if (decision.catalog_id, decision.authority_domain) != (
            self.authority().catalog_id,
            self._domain,
        ):
            raise ValueError("decision belongs to a different catalog or authority")
        candidate = self.read_candidate(decision.candidate.candidate_id)
        if candidate != decision.candidate:
            raise ValueError("decision candidate differs from canonical target")
        binding, events = self._read_trajectory(decision.trajectory_id)
        if (
            binding.trajectory_digest != decision.trajectory_digest
            or candidate.correction.source_trajectory_id != binding.trajectory_id
            or candidate.correction.trajectory_digest != binding.trajectory_digest
        ):
            raise ValueError("decision trajectory provenance is no longer canonical")
        _validate_proposal_trajectory(candidate.correction, binding, events)
        held = self._receipt_by_id(decision.held_out_receipt_id)
        protected = self._receipt_by_id(decision.protected_receipt_id)
        self._validate_candidate_receipt(held, candidate, ValidationSplit.HELD_OUT)
        self._validate_candidate_receipt(protected, candidate, ValidationSplit.PROTECTED)
        if held.record_ids != decision.held_out_record_ids:
            raise ValueError("decision held-out record provenance is invalid")
        held_cells = tuple(
            sorted(_logical_cell(record) for record in self._records_for_receipt(held))
        )
        if held_cells != decision.held_out_cell_digests:
            raise ValueError("decision held-out cell provenance is invalid")
        if protected.record_ids != decision.protected_record_ids:
            raise ValueError("decision protected record provenance is invalid")
        metric = self._validate_metric_receipt(decision.metric_receipt, candidate, held)
        manifest = self._read_manifest(decision.protected_suite_id)
        protected_cells = tuple(
            sorted(_logical_cell(record) for record in self._records_for_receipt(protected))
        )
        primary = next(
            item for item in metric.component_metrics if item.name == metric.primary_metric_name
        )
        expected_accepted = primary.is_non_worse()
        expected_reason = (
            "candidate passed held-out and protected acceptance gates"
            if expected_accepted
            else "candidate primary metric is worse than the declared tolerance"
        )
        if (
            metric != decision.metric_receipt
            or manifest.suite_digest != decision.protected_suite_digest
            or manifest.source_epoch_id != candidate.source_epoch_id
            or protected_cells != manifest.logical_cell_digests
            or protected_cells != decision.protected_cell_digests
            or decision.rollback_target_epoch_id != candidate.source_epoch_id
            or decision.accepted is not expected_accepted
            or decision.reason != expected_reason
        ):
            raise ValueError("decision metric or protected provenance is no longer canonical")

    def _receipt_by_id(self, receipt_id: str) -> ValidationReceipt:
        with sqlite3.connect(self._evaluation_store.authority().catalog_path) as connection:
            row = connection.execute(
                "SELECT payload FROM validation_receipts "
                "WHERE authority_domain = ? AND receipt_id = ?",
                (self._evaluation_store.authority().authority_domain, receipt_id),
            ).fetchone()
        if row is None:
            raise ValueError("evaluation receipt is absent")
        return self._evaluation_store.validate_validation_receipt(
            ValidationReceipt.from_dict(json.loads(row[0]))
        )


def bind_candidate_system(
    store: CoevolutionStore,
    *,
    candidate_id: str,
    correction: CorrectionProposal,
    artifact_digest: str,
) -> CandidateSystem:
    """Register a candidate through the durable coevolution authority."""

    if type(store) is not CoevolutionStore:
        raise ValueError("store must be an exact CoevolutionStore")
    return store.register_candidate(
        candidate_id=candidate_id,
        correction=correction,
        artifact_digest=artifact_digest,
    )


def decide_candidate_acceptance(
    bundle: ValidationBundle,
    *,
    authority_store: CoevolutionStore,
) -> AcceptanceDecision:
    """Issue an authoritative audit-only decision; never execute a candidate."""

    if type(authority_store) is not CoevolutionStore:
        raise ValueError("authority_store must be an exact CoevolutionStore")
    return authority_store.decide(bundle)


def _initialize_store(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            catalog_id TEXT NOT NULL,
            authority_domain TEXT NOT NULL,
            evaluation_catalog_path TEXT NOT NULL,
            evaluation_catalog_id TEXT NOT NULL,
            evaluation_authority_domain TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            revision INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            authority_domain TEXT NOT NULL,
            trajectory_id TEXT NOT NULL,
            trajectory_digest TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, trajectory_id),
            UNIQUE (authority_domain, trajectory_digest)
        );
        CREATE TABLE IF NOT EXISTS candidates (
            authority_domain TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_epoch_id TEXT NOT NULL,
            artifact_digest TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, candidate_id),
            UNIQUE (authority_domain, candidate_epoch_id),
            UNIQUE (authority_domain, artifact_digest)
        );
        CREATE TABLE IF NOT EXISTS protected_suites (
            authority_domain TEXT NOT NULL,
            suite_id TEXT NOT NULL,
            suite_digest TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, suite_id),
            UNIQUE (authority_domain, suite_digest)
        );
        CREATE TABLE IF NOT EXISTS metric_policies (
            authority_domain TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            policy_digest TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, policy_id),
            UNIQUE (authority_domain, policy_digest)
        );
        CREATE TABLE IF NOT EXISTS metric_receipts (
            authority_domain TEXT NOT NULL,
            receipt_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, receipt_id)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            authority_domain TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            consumed INTEGER NOT NULL,
            PRIMARY KEY (authority_domain, decision_id),
            UNIQUE (authority_domain, revision)
        );
        """)


def _pin_metadata(
    connection: sqlite3.Connection,
    domain: str,
    evaluation: EvaluationAuthority,
    lineage_id: str,
) -> None:
    row = connection.execute(
        "SELECT catalog_id, authority_domain, evaluation_catalog_path, "
        "evaluation_catalog_id, evaluation_authority_domain, lineage_id, revision "
        "FROM metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO metadata VALUES (1, ?, ?, ?, ?, ?, ?, 0)",
            (
                str(uuid4()),
                domain,
                evaluation.catalog_path,
                evaluation.catalog_id,
                evaluation.authority_domain,
                lineage_id,
            ),
        )
        connection.commit()
        return
    _validate_store_metadata(row, domain, evaluation, lineage_id)


def _metadata(connection: sqlite3.Connection) -> tuple[object, ...]:
    row = connection.execute(
        "SELECT catalog_id, authority_domain, evaluation_catalog_path, "
        "evaluation_catalog_id, evaluation_authority_domain, lineage_id, revision "
        "FROM metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise ValueError("coevolution catalog metadata is missing")
    return cast(tuple[object, ...], row)


def _validate_store_metadata(
    row: tuple[object, ...],
    domain: str,
    evaluation: EvaluationAuthority,
    lineage_id: str,
) -> None:
    if row[1:6] != (
        domain,
        evaluation.catalog_path,
        evaluation.catalog_id,
        evaluation.authority_domain,
        lineage_id,
    ):
        raise ValueError("coevolution store pin differs from evaluation catalog/domain/lineage")
    _require_token(row[0], "catalog_id")
    if type(row[6]) is not int or row[6] < 0:
        raise ValueError("coevolution catalog revision is invalid")


def _next_revision(connection: sqlite3.Connection, domain: str) -> int:
    row = _metadata(connection)
    current = cast(int, row[6])
    updated = connection.execute(
        "UPDATE metadata SET revision = ? WHERE singleton = 1 AND authority_domain = ? "
        "AND revision = ?",
        (current + 1, domain, current),
    )
    if updated.rowcount != 1:
        raise RuntimeError("coevolution decision revision changed concurrently")
    return current + 1


def _connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _validate_transition(
    source: EvaluationEpoch,
    candidate: EvaluationEpoch,
    proposal: CorrectionProposal,
) -> None:
    if not source.closed:
        raise ValueError("source epoch must be closed before candidate evolution")
    if source.epoch_id != proposal.source_epoch_id:
        raise ValueError("correction source epoch differs from authoritative source")
    changes = tuple(
        component
        for component, changed in (
            (CandidateComponent.MODEL, source.model_version != candidate.model_version),
            (CandidateComponent.HARNESS, source.harness_version != candidate.harness_version),
        )
        if changed
    )
    if changes != proposal.changed_components:
        raise ValueError("changed_components differ from authoritative version transition")


def _validate_proposal_trajectory(
    proposal: CorrectionProposal,
    trajectory: TrajectoryBinding,
    events: tuple[EventEnvelope, ...],
) -> None:
    if proposal.trajectory_digest != trajectory.trajectory_digest:
        raise ValueError("correction trajectory digest differs from registered binding")
    if proposal.source_epoch_id != trajectory.source_epoch_id:
        raise ValueError("correction source epoch differs from trajectory binding")
    if proposal.source_action_id not in trajectory.action_ids:
        raise ValueError("correction source action is absent from trajectory binding")
    if proposal.source_evidence_refs != trajectory.source_evidence_refs:
        raise ValueError("correction source evidence must exactly equal trajectory binding")
    node = next(
        item
        for item in proposal.failure_graph.nodes
        if item.failure_id == proposal.localized_failure_id
    )
    matches = tuple(event for event in events if event.event_id == node.event_id)
    if len(matches) != 1 or matches[0].action_id != proposal.source_action_id:
        raise ValueError("localized failure event/action linkage differs from trajectory")
    if tuple(item.reference_id for item in node.evidence_refs) != matches[0].evidence_refs:
        raise ValueError("localized failure evidence differs from trajectory event evidence")
    verifier_refs = tuple(
        reference
        for event in events
        if node.event_id in event.parent_event_ids
        for reference in event.verifier_refs
    )
    if verifier_refs != proposal.localization_verifier_refs:
        raise ValueError("localization verifier evidence differs from trajectory verification")


def _validate_localization(
    graph: FailureGraph,
    failure_id: str,
    verifier_refs: tuple[str, ...],
    source_refs: tuple[EvidenceReference, ...],
) -> None:
    nodes = tuple(item for item in graph.nodes if item.failure_id == failure_id)
    if len(nodes) != 1:
        raise ValueError("localized_failure_id must name one failure graph node")
    localized = {
        graph.localization.first_anomaly,
        graph.localization.root_cause,
        graph.localization.decisive_failure,
        graph.localization.recoverable_until,
        *graph.localization.symptoms,
    }
    if failure_id not in localized:
        raise ValueError("localized_failure_id must name an asserted localization role")
    expected = {
        reference
        for binding in graph.localization.role_evidence
        if binding.failure_id == failure_id
        for reference in binding.verifier_refs
    }
    if set(verifier_refs) != expected:
        raise ValueError("localization_verifier_refs must equal complete role evidence")
    source_ids = {item.reference_id for item in source_refs}
    if not {item.reference_id for item in nodes[0].evidence_refs}.issubset(source_ids):
        raise ValueError("source_evidence_refs must include localized failure evidence")


def _events(values: object) -> tuple[EventEnvelope, ...]:
    if type(values) is not tuple or not values:
        raise ValueError("events must be a nonempty exact tuple")
    events: list[EventEnvelope] = []
    for value in cast(tuple[object, ...], values):
        if type(value) is not EventEnvelope:
            raise ValueError("events must contain exact EventEnvelope values")
        events.append(_event_from_dict(cast(EventEnvelope, value).to_dict()))
    return tuple(events)


def _event_from_dict(value: object) -> EventEnvelope:
    if type(value) is not dict:
        raise ValueError("stored event must be an object")
    fields = dict(cast(dict[str, Any], value))
    for name in ("parent_event_ids", "evidence_refs", "verifier_refs"):
        if name in fields:
            fields[name] = tuple(fields[name])
    return EventEnvelope(**fields)


def _trajectory_digest(
    binding: TrajectoryBinding,
    events: tuple[EventEnvelope, ...],
) -> str:
    return _digest(
        {
            "trajectory_id": binding.trajectory_id,
            "source_epoch_id": binding.source_epoch_id,
            "events": [item.to_dict() for item in events],
            "source_evidence_refs": [item.to_dict() for item in binding.source_evidence_refs],
        }
    )


def _logical_cell(record: V5EvaluationRecord) -> str:
    return _digest(
        {
            "case_id": record.case.case_id,
            "case_version": record.case.case_version,
            "stripe_id": record.robustness.stripe_id,
            "subtype": record.robustness.subtype,
            "repeat_id": record.system.repeat_id,
            "seed": record.system.seed,
        }
    )


def _source_validation_target(epoch: EvaluationEpoch) -> ValidationTarget:
    return ValidationTarget(
        f"baseline:{epoch.epoch_id}",
        "model-harness-baseline",
        epoch.harness_version,
        _digest(epoch.to_dict()),
    )


def _aggregate(
    records: tuple[V5EvaluationRecord, ...],
    name: str,
    aggregation: MetricAggregation,
) -> float:
    if aggregation is not MetricAggregation.ARITHMETIC_MEAN:
        raise ValueError("unsupported metric aggregation")
    values = tuple(float(getattr(record.scores, name)) for record in records)
    return sum(values) / len(values)


def _primary_metric_name(receipt: MetricComparisonReceipt) -> str:
    return receipt.primary_metric_name


def _mark_authoritative(decision: AcceptanceDecision) -> AcceptanceDecision:
    result = AcceptanceDecision.from_dict(decision.to_dict())
    object.__setattr__(result, "_authoritative", True)
    return result


def _decision_values_payload(values: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            value.to_dict()
            if isinstance(value, (CandidateSystem, MetricComparisonReceipt))
            else list(value) if type(value) is tuple else value
        )
        for key, value in values.items()
    }


def _decision_payload(
    decision: AcceptanceDecision,
    *,
    include_digest: bool,
) -> dict[str, object]:
    result = {
        "decision_id": decision.decision_id,
        "catalog_id": decision.catalog_id,
        "authority_domain": decision.authority_domain,
        "revision": decision.revision,
        "accepted": decision.accepted,
        "candidate": decision.candidate.to_dict(),
        "proposal_id": decision.proposal_id,
        "trajectory_id": decision.trajectory_id,
        "trajectory_digest": decision.trajectory_digest,
        "held_out_receipt_id": decision.held_out_receipt_id,
        "held_out_record_ids": list(decision.held_out_record_ids),
        "held_out_cell_digests": list(decision.held_out_cell_digests),
        "protected_receipt_id": decision.protected_receipt_id,
        "protected_record_ids": list(decision.protected_record_ids),
        "protected_cell_digests": list(decision.protected_cell_digests),
        "rollback_target_epoch_id": decision.rollback_target_epoch_id,
        "protected_suite_id": decision.protected_suite_id,
        "protected_suite_digest": decision.protected_suite_digest,
        "metric_receipt": decision.metric_receipt.to_dict(),
        "reason": decision.reason,
        "execute_candidate": decision.execute_candidate,
    }
    if include_digest:
        result["decision_digest"] = decision.decision_digest
    return result


def _proposal_payload(value: CorrectionProposal) -> dict[str, object]:
    return {
        "proposal_id": value.proposal_id,
        "source_trajectory_id": value.source_trajectory_id,
        "source_action_id": value.source_action_id,
        "source_epoch_id": value.source_epoch_id,
        "failure_graph": value.failure_graph.to_dict(),
        "localized_failure_id": value.localized_failure_id,
        "localization_verifier_refs": list(value.localization_verifier_refs),
        "source_evidence_refs": [item.to_dict() for item in value.source_evidence_refs],
        "teacher_correction": value.teacher_correction,
        "teacher_evidence_refs": [item.to_dict() for item in value.teacher_evidence_refs],
        "changed_components": [item.value for item in value.changed_components],
        "trajectory_digest": value.trajectory_digest,
        "scope": value.scope.value,
    }


def _candidate_payload(value: CandidateSystem) -> dict[str, object]:
    return {
        "candidate_id": value.candidate_id,
        "correction": value.correction.to_dict(),
        "authority": value.authority.to_dict(),
        "source_epoch_id": value.source_epoch_id,
        "candidate_epoch_id": value.candidate_epoch_id,
        "epoch_revision": value.epoch_revision,
        "model_version": value.model_version,
        "harness_version": value.harness_version,
        "changed_components": [item.value for item in value.changed_components],
        "artifact_digest": value.artifact_digest,
        "rollback_target_epoch_id": value.rollback_target_epoch_id,
    }


def _component_payload(value: ComponentMetric) -> dict[str, object]:
    return {
        "name": value.name,
        "baseline_value": value.baseline_value,
        "candidate_value": value.candidate_value,
        "direction": value.direction.value,
        "tolerance": value.tolerance,
    }


def _metric_receipt_payload(value: MetricComparisonReceipt) -> dict[str, object]:
    return {
        "receipt_id": value.receipt_id,
        "catalog_id": value.catalog_id,
        "authority_domain": value.authority_domain,
        "candidate_id": value.candidate_id,
        "source_epoch_id": value.source_epoch_id,
        "candidate_epoch_id": value.candidate_epoch_id,
        "policy_id": value.policy_id,
        "policy_digest": value.policy_digest,
        "primary_metric_name": value.primary_metric_name,
        "aggregation": value.aggregation.value,
        "source_receipt_id": value.source_receipt_id,
        "candidate_receipt_id": value.candidate_receipt_id,
        "source_record_ids": list(value.source_record_ids),
        "candidate_record_ids": list(value.candidate_record_ids),
        "logical_cell_digests": list(value.logical_cell_digests),
        "component_metrics": [item.to_dict() for item in value.component_metrics],
    }


def _validation_receipt(value: object, name: str) -> ValidationReceipt:
    if type(value) is not ValidationReceipt:
        raise ValueError(f"{name} must be an exact ValidationReceipt")
    return ValidationReceipt.from_dict(cast(ValidationReceipt, value).to_dict())


def _failure_graph(value: object) -> FailureGraph:
    if type(value) is not FailureGraph:
        raise ValueError("failure_graph must be an exact FailureGraph")
    return FailureGraph.from_dict(cast(FailureGraph, value).to_dict())


def _metric_spec(value: object) -> MetricSpec:
    if type(value) is not MetricSpec:
        raise ValueError("specs must contain exact MetricSpec values")
    return MetricSpec.from_dict(cast(MetricSpec, value).to_dict())


def _metrics(value: object) -> tuple[ComponentMetric, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("component_metrics must be a nonempty exact tuple")
    return tuple(
        (
            ComponentMetric.from_dict(cast(ComponentMetric, item).to_dict())
            if type(item) is ComponentMetric
            else _raise_metric()
        )
        for item in cast(tuple[object, ...], value)
    )


def _raise_metric() -> ComponentMetric:
    raise ValueError("component_metrics must contain exact ComponentMetric values")


def _components(value: object) -> tuple[CandidateComponent, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("changed_components must be a nonempty exact tuple")
    result = cast(tuple[CandidateComponent, ...], value)
    if any(type(item) is not CandidateComponent for item in result):
        raise ValueError("changed_components must contain exact CandidateComponent values")
    expected = tuple(item for item in CandidateComponent if item in result)
    if result != expected:
        raise ValueError("changed_components must be unique and canonically ordered")
    return result


def _restore_components(value: object) -> tuple[CandidateComponent, ...]:
    return tuple(
        _enum(item, CandidateComponent, "changed_components")
        for item in _list(value, "changed_components")
    )


def _evidence(value: object, name: str) -> tuple[EvidenceReference, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty exact tuple")
    result: list[EvidenceReference] = []
    for item in cast(tuple[object, ...], value):
        if type(item) is not EvidenceReference:
            raise ValueError(f"{name} must contain exact EvidenceReference values")
        reference = cast(EvidenceReference, item)
        if type(reference.reference_id) is not str or not reference.reference_id.strip():
            raise ValueError(f"{name} reference_id must be a nonblank built-in string")
        if type(reference.source_kind) is not EvidenceSourceKind:
            raise ValueError(f"{name} source_kind must be exact")
        result.append(EvidenceReference(reference.reference_id, reference.source_kind))
    keys = tuple((item.reference_id, item.source_kind) for item in result)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must contain unique references")
    return tuple(result)


def _restore_evidence(value: object, name: str) -> tuple[EvidenceReference, ...]:
    return tuple(EvidenceReference.from_dict(item) for item in _list(value, name))


def _tokens(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty exact tuple")
    result = tuple(_require_token(item, name) for item in cast(tuple[object, ...], value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _sha_ids(value: object, name: str) -> tuple[str, ...]:
    result = _tokens(value, name)
    for item in result:
        _require_sha256(item, name)
    return result


def _restore_tokens(value: object, name: str) -> tuple[str, ...]:
    return tuple(cast(str, item) for item in _list(value, name))


def _list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be an array")
    return cast(list[object], value)


def _mapping(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"{name} requires exactly {sorted(fields)!r}")
    return cast(Mapping[str, object], value)


def _enum(value: object, enum_type: type[EnumT], name: str) -> EnumT:
    if type(value) is not str:
        raise ValueError(f"{name} must be a built-in string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{name} has unknown value {value!r}") from exc


def _require_token(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonblank canonical built-in string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be exactly sha256:<64 lowercase hex>")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    return value


def _canonical_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))


def _eval_authority_payload(value: EvaluationAuthority) -> dict[str, object]:
    return {
        "catalog_path": value.catalog_path,
        "catalog_id": value.catalog_id,
        "authority_domain": value.authority_domain,
    }


def _snapshot_eval_authority(value: object) -> EvaluationAuthority:
    if type(value) is not EvaluationAuthority:
        raise ValueError("evaluation_authority must be exact")
    return _eval_authority_from_dict(_eval_authority_payload(cast(EvaluationAuthority, value)))


def _eval_authority_from_dict(value: object) -> EvaluationAuthority:
    fields = _mapping(
        value,
        {"catalog_path", "catalog_id", "authority_domain"},
        "EvaluationAuthority",
    )
    return EvaluationAuthority(
        cast(str, fields["catalog_path"]),
        cast(str, fields["catalog_id"]),
        cast(str, fields["authority_domain"]),
    )


def _check_binding(value: object, payload: object, binding: str, name: str) -> None:
    del value
    if _digest(payload) != binding:
        raise ValueError(f"{name} changed after construction")


def _digest(value: object) -> str:
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


__all__ = [
    "AcceptanceDecision",
    "CandidateComponent",
    "CandidateSystem",
    "CoevolutionAuthority",
    "CoevolutionStore",
    "ComponentMetric",
    "CorrectionProposal",
    "CorrectionScope",
    "MetricAggregation",
    "MetricComparisonReceipt",
    "MetricDirection",
    "MetricPolicy",
    "MetricSpec",
    "ProtectedSuiteManifest",
    "TrajectoryBinding",
    "ValidationBundle",
    "bind_candidate_system",
    "decide_candidate_acceptance",
]
