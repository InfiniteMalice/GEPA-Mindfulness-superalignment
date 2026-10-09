"""Opt-in artifact retrieval with current access checks and an allowlisted producer view."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

# Third-party
# Local
from semantic_intent_robustness.memory_safety import (
    MemoryRetrievalDecision,
    assess_retrieved_memory,
)

from .artifact_records import (
    ADMISSIONS,
    ArtifactDiagnosticRecord,
    ArtifactKey,
    ArtifactLocation,
    ArtifactRecord,
    ArtifactSnapshot,
    DerivedInterpretation,
    SourceFragment,
    _lineage,
    _mapping,
    _tuple,
    artifact_digest,
    artifact_key,
    json_value,
    unique,
)
from .artifact_topology import EvidenceTopology, assess_support_routes, validate_topology
from .debate_records import _digest, _record
from .diagnostic_records import _text, choice, strings
from .evidence_use import EvidenceUsePolicy
from .state import parse_rfc3339_datetime

CHANNELS = ("candidate_evidence", "unverified", "historical", "untrusted", "withheld")
_ITEM_FIELDS = {
    "item_id",
    "ancestor_source_ids",
    "channel",
    "reasons",
    "original_observed_at",
    "inherited_training_eligibility",
}
_ACCESS_FIELDS = {"artifact_key", "artifact_digest", "decision", "reason"}
_VIEW_FIELDS = {"handle", "text", "location", "observed_at", "channel"}


def _keys(value: object) -> tuple[ArtifactKey, ...]:
    result = tuple(artifact_key(k) for k in _tuple(value))
    unique(list(result), "artifact keys")
    if len(result) > 64:
        raise ValueError("too many artifact keys")
    return result


def _policy(data: object) -> EvidenceUsePolicy:
    return EvidenceUsePolicy(**_mapping(data, EvidenceUsePolicy))


@dataclass(frozen=True)
class ArtifactQuery(ArtifactDiagnosticRecord):
    """Host scope, explicit query time and source-use policy, separate from expected outcomes."""

    request_id: str
    public_query: str
    principal_id: str
    scope_id: str
    entity_ids: tuple[str, ...]
    artifact_keys: tuple[ArtifactKey, ...]
    assessed_at: str
    policy_version: str
    policy: EvidenceUsePolicy
    purpose: str
    excluded_artifacts: tuple[ArtifactKey, ...]
    schema_version = "artifact-query-v1"
    restorers = {"artifact_keys": _keys, "excluded_artifacts": _keys, "policy": _policy}

    def __post_init__(self) -> None:
        for name in ("request_id", "public_query", "principal_id", "scope_id", "policy_version"):
            _text(getattr(self, name), name)
        object.__setattr__(
            self, "entity_ids", strings(self.entity_ids, "entity_ids", required=True)
        )
        for name in ("artifact_keys", "excluded_artifacts"):
            object.__setattr__(self, name, _keys(getattr(self, name)))
        parse_rfc3339_datetime(self.assessed_at, "assessed_at")
        if type(self.policy) is not EvidenceUsePolicy:
            raise ValueError("exact EvidenceUsePolicy required")
        object.__setattr__(self, "policy", _policy(json_value(self.policy)))
        choice(self.purpose, "purpose", ("current", "historical"))


@dataclass(frozen=True)
class ArtifactAccessRequest:
    """A host ACL descriptor containing identities and a digest, never source text."""

    request_id: str
    principal_id: str
    scope_id: str
    artifact_key: ArtifactKey
    artifact_digest: str
    policy_version: str
    assessed_at: str

    def __post_init__(self) -> None:
        for name in ("request_id", "principal_id", "scope_id", "policy_version"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "artifact_key", artifact_key(self.artifact_key))
        _digest(self.artifact_digest)
        parse_rfc3339_datetime(self.assessed_at, "assessed_at")

    def to_dict(self) -> dict[str, Any]:
        """Return only the declared access descriptor, without diagnostic envelope fields."""
        self.__post_init__()
        return dict(
            request_id=self.request_id,
            principal_id=self.principal_id,
            scope_id=self.scope_id,
            artifact_key=list(self.artifact_key),
            artifact_digest=self.artifact_digest,
            policy_version=self.policy_version,
            assessed_at=self.assessed_at,
        )


def _rows(value: object, names: set[str], limit: int) -> tuple[dict[str, Any], ...]:
    values = _tuple(value)
    if len(values) > limit or any(type(v) is not dict or set(v) != names for v in values):
        raise ValueError("incorrect row fields or bounds")
    return tuple(json.loads(json.dumps(json_value(v), allow_nan=False)) for v in values)


@dataclass(frozen=True)
class ArtifactRetrieval(ArtifactDiagnosticRecord):
    """Host audit and separate producer context; saved rows are never access credentials."""

    snapshot_digest: str
    topology_digest: str
    query: ArtifactQuery
    mode: str
    selected_item_ids: tuple[str, ...]
    item_rows: tuple[dict[str, Any], ...]
    access_rows: tuple[dict[str, Any], ...]
    producer_view: dict[str, Any]
    schema_version = "artifact-retrieval-v1"
    restorers = {
        "query": ArtifactQuery.from_dict,
        "item_rows": lambda v: _tuple(v),
        "access_rows": lambda v: _tuple(v),
    }

    def __post_init__(self) -> None:
        _digest(self.snapshot_digest)
        _digest(self.topology_digest)
        object.__setattr__(self, "query", _record(self.query, ArtifactQuery))
        choice(self.mode, "mode", ("artifact_index", "artifact_topology"))
        object.__setattr__(self, "selected_item_ids", strings(self.selected_item_ids, "selected"))
        items = _rows(self.item_rows, _ITEM_FIELDS, 256)
        access = _rows(self.access_rows, _ACCESS_FIELDS, 64)
        for row in items:
            _text(row["item_id"], "item ID")
            strings(row["ancestor_source_ids"], "source IDs", required=True)
            strings(row["reasons"], "reasons")
            choice(row["channel"], "channel", CHANNELS)
            choice(
                row["inherited_training_eligibility"],
                "admission",
                tuple(a.value for a in ADMISSIONS),
            )
            for time in strings(row["original_observed_at"], "times", required=True):
                parse_rfc3339_datetime(time, "original_observed_at")
        unique([r["item_id"] for r in items], "item rows")
        for row in access:
            artifact_key(row["artifact_key"])
            _digest(row["artifact_digest"])
            choice(row["decision"], "decision", ("allowed", "denied", "unresolved", "excluded"))
            _text(row["reason"], "reason")
        unique([tuple(r["artifact_key"]) for r in access], "access rows")
        if type(self.producer_view) is not dict or set(self.producer_view) != {
            "public_query",
            "items",
        }:
            raise ValueError("producer view requires exact public fields")
        if self.producer_view["public_query"] != self.query.public_query:
            raise ValueError("producer query mismatch")
        view_items = _rows(self.producer_view["items"], _VIEW_FIELDS, 256)
        emitted = tuple(r["item_id"] for r in items if r["channel"] != "withheld")
        if self.selected_item_ids != emitted or len(view_items) != len(emitted):
            raise ValueError("projection must cover exactly selected items in order")
        channels = [r["channel"] for r in items if r["channel"] != "withheld"]
        for i, row in enumerate(view_items):
            if row["handle"] != f"item-{i}" or row["channel"] != channels[i]:
                raise ValueError("producer handle/channel mismatch")
            _text(row["text"], "producer text")
            for time in strings(row["observed_at"], "observed_at", required=True):
                parse_rfc3339_datetime(time, "observed_at")
            if row["location"] is not None:
                ArtifactLocation(**_mapping(row["location"], ArtifactLocation))
        object.__setattr__(self, "item_rows", items)
        object.__setattr__(self, "access_rows", access)
        object.__setattr__(
            self,
            "producer_view",
            dict(public_query=self.query.public_query, items=list(view_items)),
        )


def _access(
    artifact: ArtifactRecord,
    query: ArtifactQuery,
    authorize: Callable[[ArtifactAccessRequest], bool] | None,
) -> tuple[str, str]:
    if artifact.availability != "available":
        return "denied", f"artifact_{artifact.availability}"
    if authorize is None:
        return "unresolved", "no_authorizer"
    descriptor = ArtifactAccessRequest(
        query.request_id,
        query.principal_id,
        query.scope_id,
        (artifact.artifact_id, artifact.version),
        artifact.observation.digest,
        query.policy_version,
        query.assessed_at,
    )
    detached = replace(descriptor)
    original = detached.to_dict()
    try:
        accepted = authorize(detached)
        if detached.to_dict() != original:
            return "unresolved", "authorizer_mutated_descriptor"
        if accepted is True:
            return "allowed", "host_authorized"
        return (
            ("denied", "host_denied")
            if accepted is False
            else ("unresolved", "nonboolean_authorizer")
        )
    except Exception as error:
        return "unresolved", f"callback_error:{type(error).__name__}"


def _quality_reasons(source: SourceFragment, query: ArtifactQuery) -> list[str]:
    q, p = source.quality, query.policy
    reasons = []
    if q.source_reliability is None or q.source_reliability < p.min_source_reliability:
        reasons.append("source_reliability_unknown_or_below_policy")
    if q.compression_distortion is None or q.compression_distortion > p.max_compression_distortion:
        reasons.append("compression_distortion_unknown_or_above_policy")
    if q.integrity != "intact":
        reasons.append(f"integrity_{q.integrity}")
    if q.authority == "unknown":
        reasons.append("authority_unknown")
    return reasons


def _dependency_items(
    item_id: str, items: dict[str, SourceFragment | DerivedInterpretation]
) -> tuple[str, ...]:
    found: set[str] = set()
    pending = [item_id]
    while pending:
        key = pending.pop()
        if key in found:
            continue
        found.add(key)
        item = items[key]
        if isinstance(item, DerivedInterpretation):
            pending.extend(parent for parent, _ in item.inputs)
    return tuple(sorted(found))


def _classify(
    item: SourceFragment | DerivedInterpretation,
    ancestors: tuple[SourceFragment, ...],
    dependencies: tuple[SourceFragment | DerivedInterpretation, ...],
    query: ArtifactQuery,
    access: dict[ArtifactKey, tuple[str, str]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not set(item.entity_ids) <= set(query.entity_ids):
        reasons.append("entity_mismatch")
    for source in ancestors:
        decision, reason = access[source.artifact_key]
        if decision != "allowed":
            reasons.append(reason)
        reasons.extend(_quality_reasons(source, query))
    historical, unverified, untrusted = False, False, False
    now = parse_rfc3339_datetime(query.assessed_at, "assessed_at")
    for dependency in dependencies:
        boundary = assess_retrieved_memory(dependency.memory)
        if boundary.decision in {
            MemoryRetrievalDecision.REJECT,
            MemoryRetrievalDecision.QUARANTINE,
        }:
            reasons.extend(boundary.reasons)
        if boundary.decision is MemoryRetrievalDecision.TREAT_AS_UNTRUSTED_CONTEXT:
            untrusted = True
        status = dependency.claim.status
        if status == "unavailable":
            reasons.append("evidence_unavailable")
        historical |= status in {"stale", "superseded", "contradicted"}
        unverified |= status == "unverified"
    for source in ancestors:
        age = (
            now - parse_rfc3339_datetime(source.quality.recorded_at, "recorded_at")
        ).total_seconds()
        historical |= age > query.policy.max_age_seconds
    if historical and query.purpose == "current":
        reasons.append("historical_evidence_not_current")
    if reasons:
        return "withheld", sorted(set(reasons))
    if untrusted:
        return "untrusted", ["bounded_untrusted_context"]
    if historical:
        return "historical", ["historical_not_current_support"]
    if unverified or isinstance(item, DerivedInterpretation):
        return "unverified", ["unverified_interpretation_or_claim"]
    return "candidate_evidence", ["declared_current_evidence_not_certified"]


def retrieve_artifact_evidence(
    snapshot: ArtifactSnapshot,
    topology: EvidenceTopology,
    query: ArtifactQuery,
    *,
    mode: str = "artifact_index",
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    enabled: bool = False,
) -> ArtifactRetrieval:
    """Recheck access and validity before emitting any source or derived content."""
    if enabled is not True:
        raise ValueError("artifact retrieval requires enabled=True")
    snapshot = _record(snapshot, ArtifactSnapshot)
    topology = _record(topology, EvidenceTopology)
    query = _record(query, ArtifactQuery)
    choice(mode, "mode", ("artifact_index", "artifact_topology"))
    validate_topology(snapshot, topology)
    if authorize is not None and not callable(authorize):
        raise ValueError("authorize must be callable")
    artifacts = {(a.artifact_id, a.version): a for a in snapshot.artifacts}
    if not set(query.entity_ids) <= set(snapshot.entity_ids):
        raise ValueError("query has unknown entities")
    if not (set(query.artifact_keys) | set(query.excluded_artifacts)) <= set(artifacts):
        raise ValueError("query has unknown artifact versions")
    now = parse_rfc3339_datetime(query.assessed_at, "assessed_at")
    if any(
        now < parse_rfc3339_datetime(a.observation.observed_at, "observed_at")
        for a in snapshot.artifacts
    ):
        raise ValueError("source observation is in the future")
    if any(
        now < parse_rfc3339_datetime(item.created_at, "created_at")
        for item in snapshot.interpretations
    ):
        raise ValueError("derived interpretation is in the future")
    sources = {s.item_id: s for s in snapshot.sources}
    items: dict[str, SourceFragment | DerivedInterpretation] = {
        s.item_id: s for s in snapshot.sources + snapshot.interpretations
    }
    ancestry = _lineage(snapshot)
    relevant = {
        sources[source_id].artifact_key
        for item in items.values()
        if set(item.entity_ids) <= set(query.entity_ids)
        for source_id in ancestry[item.item_id]
    }
    access: dict[ArtifactKey, tuple[str, str]] = {}
    for key, artifact in sorted(artifacts.items()):
        if key in query.excluded_artifacts:
            access[key] = ("excluded", "planned_removal")
        elif key not in relevant or (query.artifact_keys and key not in query.artifact_keys):
            access[key] = ("excluded", "not_selected")
        else:
            access[key] = _access(artifact, query, authorize)
    order = sorted(
        items, key=lambda key: (min(sources[s].artifact_key for s in ancestry[key]), key)
    )
    rows: list[dict[str, Any]] = []
    for item_id in order:
        item = items[item_id]
        ancestors = tuple(sources[s] for s in ancestry[item_id])
        dependencies = tuple(items[k] for k in _dependency_items(item_id, items))
        channel, reasons = _classify(item, ancestors, dependencies, query, access)
        admissions = [artifacts[s.artifact_key].source_training_eligibility for s in ancestors]
        rows.append(
            dict(
                item_id=item_id,
                ancestor_source_ids=list(ancestry[item_id]),
                channel=channel,
                reasons=reasons,
                original_observed_at=sorted({s.quality.recorded_at for s in ancestors}),
                inherited_training_eligibility=max(admissions, key=ADMISSIONS.index).value,
            )
        )
    if mode == "artifact_topology":
        available = tuple(r["item_id"] for r in rows if r["channel"] == "candidate_evidence")
        structure = assess_support_routes(
            snapshot, topology, available_item_ids=available, enabled=True
        )
        available_routes = set(structure["available_route_ids"])
        participating = {
            key
            for route in topology.routes
            if route.route_id in available_routes
            for key in route.item_ids
        }
        for row in rows:
            if row["channel"] == "candidate_evidence" and row["item_id"] not in participating:
                row.update(channel="withheld", reasons=["no_available_support_route"])
    selected = tuple(row["item_id"] for row in rows if row["channel"] != "withheld")
    view_items: list[dict[str, Any]] = []
    for row in rows:
        if row["channel"] == "withheld":
            continue
        item = items[row["item_id"]]
        view_items.append(
            dict(
                handle=f"item-{len(view_items)}",
                text=item.quotation if isinstance(item, SourceFragment) else item.claim.proposition,
                location=(
                    dict(kind=item.location.kind, value=item.location.value)
                    if isinstance(item, SourceFragment)
                    else None
                ),
                observed_at=list(row["original_observed_at"]),
                channel=row["channel"],
            )
        )
    access_rows = tuple(
        dict(
            artifact_key=list(key),
            artifact_digest=artifacts[key].observation.digest,
            decision=decision,
            reason=reason,
        )
        for key, (decision, reason) in access.items()
    )
    return ArtifactRetrieval(
        artifact_digest(snapshot),
        artifact_digest(topology),
        query,
        mode,
        selected,
        tuple(rows),
        access_rows,
        dict(public_query=query.public_query, items=view_items),
    )
