"""Experimental source-bound statistics and finite evidence reconstruction; no promotion."""

# Standard library
from __future__ import annotations

import gzip
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter_ns
from typing import Any, cast

# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import MetricSpec, ResamplingPolicy
from evaluation.improvement_statistics import PairedObservation, estimate_paired
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_evaluator,
    _read_records,
    _record,
    array,
    digest,
    evaluator,
    references,
    text,
)
from gepa_mindfulness.verification.diagnostic_records import restore_refs
from gepa_mindfulness.verification.research_archive import (
    MAX_SOURCE_BYTES,
    ResearchArchive,
    ResearchCatalog,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from synthetic_data.cva_fixtures import SecurityFixture, qualify_security_fixture


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("phase2 reporting requires enabled=True")


def _legacy(value: Any, kind: type) -> Any:
    if type(value) is not kind:
        raise ValueError("incorrect legacy reporting record")
    payload = cast(Any, value).to_dict()
    validate_input_envelope(payload)
    return kind.from_dict(payload)


@dataclass(frozen=True)
class SourceSelector(CVARecord):
    artifact_id: str
    version: str
    source_digest: str
    pointer: str
    kind: str
    codec: str = "json"

    schema_version = "research-source-selector-v1"

    def __post_init__(self) -> None:
        text(self.artifact_id, "artifact_id")
        text(self.version, "version")
        digest(self.source_digest)
        if (
            type(self.pointer) is not str
            or len(self.pointer) > 2048
            or (self.pointer and not self.pointer.startswith("/"))
        ):
            raise ValueError("selector requires a bounded JSON pointer")
        if len(self.pointer.split("/")) > 17:
            raise ValueError("source pointer exceeds depth budget")
        for part in self.pointer.split("/"):
            rest = part.replace("~0", "").replace("~1", "")
            if "~" in rest:
                raise ValueError("unsupported pointer escape")
        if self.kind not in ("number", "boolean", "protected", "document") or self.codec not in (
            "json",
            "gzip_json",
        ):
            raise ValueError("unsupported source kind/codec")


@dataclass(frozen=True)
class ModelExperimentManifest(CVARecord):
    experiment_version: str
    code_revision: str
    baseline_model_digest: str
    candidate_model_digest: str
    model_version: str
    harness_version: str
    dataset_digest: str
    seeds: tuple[int, ...]
    condition: str
    created_at: str

    schema_version = "research-model-experiment-manifest-v1"
    restorers = {"seeds": _read_array}

    def __post_init__(self) -> None:
        for name in ("experiment_version", "model_version", "harness_version"):
            text(getattr(self, name), name)
        if (
            type(self.code_revision) is not str
            or len(self.code_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.code_revision)
        ):
            raise ValueError("code revision requires exact Git SHA")
        for d in (self.baseline_model_digest, self.candidate_model_digest, self.dataset_digest):
            digest(d)
        values = array(self.seeds, 32, required=True)
        if any(type(v) is not int or not 0 <= v < 2**32 for v in values) or len(
            set(values)
        ) != len(values):
            raise ValueError("manifest seeds must be unique bounded integers")
        if self.condition not in (
            "baseline",
            "behaviorally_trained",
            "diagnostic_instrumented",
            "intervention_informed",
        ):
            raise ValueError("unknown research condition")
        if (
            self.condition == "diagnostic_instrumented"
            and self.baseline_model_digest != self.candidate_model_digest
        ):
            raise ValueError("diagnostic condition cannot silently change model state")
        parse_rfc3339_datetime(self.created_at, "created_at")


@dataclass(frozen=True)
class PairedSource(CVARecord):
    pair_id: str
    case_id: str
    condition_id: str
    repeat_id: int
    cluster_id: str
    baseline: SourceSelector
    candidate: SourceSelector

    schema_version = "research-paired-source-v1"
    restorers = {"baseline": SourceSelector.from_dict, "candidate": SourceSelector.from_dict}

    def __post_init__(self) -> None:
        for name in ("pair_id", "case_id", "condition_id", "cluster_id"):
            text(getattr(self, name), name)
        if type(self.repeat_id) is not int or not 0 <= self.repeat_id < 1000000:
            raise ValueError("invalid repeat")
        for selector in (self.baseline, self.candidate):
            s = _record(selector, SourceSelector)
            if s.kind not in ("number", "boolean"):
                raise ValueError("paired metric source must be scalar")


@dataclass(frozen=True)
class ProtectedRegression(CVARecord):
    candidate_model_digest: str | None
    suite_digest: str
    passed: bool
    kind: str
    checker: TrustedEvaluatorContract
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "research-protected-regression-v1"
    restorers = {"checker": _read_evaluator, "evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        digest(self.suite_digest)
        evaluator(self.checker)
        references(self.evidence_refs)
        if type(self.passed) is not bool or self.kind not in ("model", "software"):
            raise ValueError("invalid protected regression record")
        if self.kind == "model":
            digest(self.candidate_model_digest)
        elif self.candidate_model_digest is not None:
            raise ValueError("software test record cannot claim model regression identity")


@dataclass(frozen=True)
class ResearchComparisonProtocol(CVARecord):
    protocol_id: str
    manifest: ModelExperimentManifest
    metric: MetricSpec
    pairs: tuple[PairedSource, ...]
    policy: ResamplingPolicy
    dependencies_known: bool
    protected_source: SourceSelector | None = None

    schema_version = "research-comparison-protocol-v1"
    restorers = {
        "manifest": ModelExperimentManifest.from_dict,
        "metric": MetricSpec.from_dict,
        "pairs": lambda v: _read_records(v, PairedSource),
        "policy": ResamplingPolicy.from_dict,
        "protected_source": lambda v: None if v is None else SourceSelector.from_dict(v),
    }

    def __post_init__(self) -> None:
        text(self.protocol_id, "protocol_id")
        _record(self.manifest, ModelExperimentManifest)
        metric = _legacy(self.metric, MetricSpec)
        digest(metric.rubric_digest)
        if metric.transform == "one_minus" and metric.unit != "rate":
            raise ValueError("one-minus requires a declared unit-rate metric")
        policy = _legacy(self.policy, ResamplingPolicy)
        if (
            type(policy.seed) is not int
            or not 0 <= policy.seed < 2**32
            or type(policy.resamples) is not int
            or not 1 <= policy.resamples <= 100000
            or type(policy.minimum_clusters) is not int
            or not 2 <= policy.minimum_clusters <= 128
            or type(policy.confidence_level) is not float
            or not 0 < policy.confidence_level < 1
        ):
            raise ValueError("unsupported confidence policy budget")
        if type(self.dependencies_known) is not bool:
            raise ValueError("dependency status requires bool")
        pairs = tuple(_record(p, PairedSource) for p in array(self.pairs, 128, required=True))
        if len({p.pair_id for p in pairs}) != len(pairs):
            raise ValueError("duplicate pair IDs")
        coordinates = set()
        sources = set()
        clusters: dict[str, str] = {}
        for p in pairs:
            coordinate = (p.case_id, p.condition_id, p.repeat_id)
            if coordinate in coordinates:
                raise ValueError("duplicate paired observation coordinate")
            coordinates.add(coordinate)
            if p.case_id in clusters and clusters[p.case_id] != p.cluster_id:
                raise ValueError("same case cannot split across dependency clusters")
            clusters[p.case_id] = p.cluster_id
            for s in (p.baseline, p.candidate):
                key = (s.source_digest, s.pointer, s.codec)
                if key in sources:
                    raise ValueError("reused or aliased source observation")
                sources.add(key)
        if (
            self.protected_source is not None
            and _record(self.protected_source, SourceSelector).kind != "protected"
        ):
            raise ValueError("protected source must retain typed model/software distinction")


@dataclass(frozen=True)
class ResearchVerificationPacket(CVARecord):
    protocol: ResearchComparisonProtocol
    catalog: ResearchCatalog
    observations: tuple[PairedObservation, ...]
    protected: ProtectedRegression | None

    schema_version = "research-verification-packet-v1"
    restorers = {
        "protocol": ResearchComparisonProtocol.from_dict,
        "catalog": ResearchCatalog.from_dict,
        "observations": lambda v: tuple(PairedObservation.from_dict(x) for x in _read_array(v)),
        "protected": lambda v: None if v is None else ProtectedRegression.from_dict(v),
    }

    def __post_init__(self) -> None:
        _record(self.protocol, ResearchComparisonProtocol)
        _record(self.catalog, ResearchCatalog)
        for row in array(self.observations, 128):
            _legacy(row, PairedObservation)
        if self.protected is not None:
            _record(self.protected, ProtectedRegression)


@dataclass(frozen=True)
class ResearchHostContext:
    evaluator: TrustedEvaluatorContract
    actor_id: str
    authenticate: Callable[[ResearchVerificationPacket], bool] | None

    def __post_init__(self) -> None:
        evaluator(self.evaluator)
        text(self.actor_id, "actor_id")
        if self.actor_id == self.evaluator.evaluator_id:
            raise ValueError("self-review is not independent checking")
        if self.authenticate is not None and not callable(self.authenticate):
            raise ValueError("host callable required")


def _load_json(payload: bytes, codec: str) -> Any:
    if codec == "gzip_json":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                data = stream.read(MAX_SOURCE_BYTES + 1)
        except (OSError, EOFError) as error:
            raise ValueError("invalid compressed source") from error
    else:
        data = payload
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("decoded source exceeds archive byte budget")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON source keys")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid bounded JSON source") from error
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > 32:
            raise ValueError("source JSON exceeds 32 levels")
        if type(node) is dict:
            stack.extend((v, depth + 1) for v in node.values())
        elif type(node) is list:
            stack.extend((v, depth + 1) for v in node)
        elif type(node) is float and not math.isfinite(node):
            raise ValueError("nonfinite source value")
    return value


def _pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        try:
            if type(value) is dict:
                value = value[key]
            elif type(value) is list and (
                key == "0" or (key.isascii() and key.isdigit() and not key.startswith("0"))
            ):
                value = value[int(key)]
            else:
                raise KeyError(key)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("source pointer is missing or incompatible") from error
    return value


def _source_document(
    archive: ResearchArchive, catalog: ResearchCatalog, selector: SourceSelector, pin: str
) -> Any | None:
    if not any(
        (a.artifact_id, a.version) == (selector.artifact_id, selector.version)
        for a in catalog.snapshot.artifacts
    ):
        return None
    record = next(
        a
        for a in catalog.snapshot.artifacts
        if (a.artifact_id, a.version) == (selector.artifact_id, selector.version)
    )
    if record.observation.digest != selector.source_digest:
        raise ValueError("source selector digest differs from catalog")
    payload = archive.read(
        catalog, selector.artifact_id, selector.version, trusted_catalog_digest=pin, enabled=True
    )
    return _load_json(payload, selector.codec)


def report_research_comparison(
    protocol: ResearchComparisonProtocol,
    archive: ResearchArchive,
    catalog: ResearchCatalog,
    *,
    trusted_catalog_digest: str,
    context: ResearchHostContext | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    _enabled(enabled)
    p = _record(protocol, ResearchComparisonProtocol)
    c = _record(catalog, ResearchCatalog)
    if type(archive) is not ResearchArchive or trusted_catalog_digest != c.digest:
        raise PermissionError("reporting requires exact host archive/catalog pin")
    rows = []
    missing = []
    for pair in p.pairs:
        values = []
        for s in (pair.baseline, pair.candidate):
            root = _source_document(archive, c, s, trusted_catalog_digest)
            if root is None:
                break
            if type(root) is not dict or root.get("schema_version") != p.metric.source_schema:
                raise ValueError("source schema differs from frozen metric rubric")
            value = _pointer(root, s.pointer)
            if s.kind == "boolean":
                if type(value) is not bool:
                    raise ValueError("boolean source value required")
            elif type(value) not in (float, int) or (type(value) is int and abs(value) > 2**53 - 1):
                raise ValueError("finite JSON-safe numeric source value required")
            number = float(value)
            if not math.isfinite(number) or (p.metric.unit == "rate" and not 0 <= number <= 1):
                raise ValueError("source value violates finite metric/rate contract")
            values.append(1.0 - number if p.metric.transform == "one_minus" else number)
        if len(values) != 2:
            missing.append(pair.pair_id)
            continue
        rows.append(
            PairedObservation(
                pair.pair_id,
                pair.case_id,
                pair.condition_id,
                pair.repeat_id,
                pair.cluster_id,
                p.digest,
                p.metric,
                values[0],
                values[1],
            )
        )
    protected = None
    if p.protected_source is not None:
        document = _source_document(archive, c, p.protected_source, trusted_catalog_digest)
        if document is not None:
            protected = ProtectedRegression.from_dict(
                _pointer(document, p.protected_source.pointer)
            )
            if (
                protected.kind == "model"
                and protected.candidate_model_digest != p.manifest.candidate_model_digest
            ):
                raise ValueError("protected regression belongs to another candidate model")
    if context is not None and type(context) is not ResearchHostContext:
        raise ValueError("exact host context required")
    if context is not None:
        context = ResearchHostContext(
            evaluator(context.evaluator), context.actor_id, context.authenticate
        )
    packet = ResearchVerificationPacket(p, c, tuple(rows), protected)
    authenticated = False
    if context is not None and context.authenticate is not None:
        detached = ResearchVerificationPacket.from_dict(packet.to_dict())
        original = detached.to_dict()
        try:
            authenticated = (
                context.authenticate(detached) is True and detached.to_dict() == original
            )
        except Exception:
            authenticated = False
    expected = tuple(v.pair_id for v in p.pairs)
    estimate = estimate_paired(
        tuple(rows) if authenticated else (),
        expected_pair_ids=expected,
        policy=p.policy,
        dependencies_known=p.dependencies_known and authenticated,
    )
    signed_interval = None
    if estimate.interval is not None:
        sign = -1 if p.metric.direction == "lower" else 1
        signed_interval = sorted(sign * v for v in estimate.interval)
    status = "UNVERIFIED" if not authenticated else "NOT_RUN"
    if authenticated and protected is not None:
        status = (
            "SOFTWARE_ONLY"
            if protected.kind == "software"
            else "PASSED" if protected.passed else "FAILED"
        )
    if not authenticated:
        interpretation = "UNVERIFIED_SOURCES"
    elif status == "FAILED":
        interpretation = "WITHHELD_PROTECTED_REGRESSION"
    elif status != "PASSED":
        interpretation = "WITHHELD_MODEL_REGRESSION_UNRUN"
    elif signed_interval is None:
        interpretation = "WITHHELD_UNCERTAINTY"
    elif signed_interval[0] <= 0:
        interpretation = "NO_SUPPORTED_METRIC_IMPROVEMENT"
    else:
        interpretation = "OBSERVED_METRIC_IMPROVEMENT_WITH_PROTECTED_CHECK"
    result = {
        "schema_version": "research-comparison-report-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "protocol": p.to_dict(),
        "catalog_digest": c.digest,
        "source_authentication": "HOST_CHECKED" if authenticated else "UNRESOLVED",
        "observations": [v.to_dict() for v in rows],
        "missing_pair_ids": missing,
        "estimate": estimate.to_dict(),
        "improvement_interval": signed_interval,
        "protected_regression": {
            "status": status,
            "record": None if protected is None else protected.to_dict(),
        },
        "interpretation": interpretation,
        "alignment_improvement": "NOT_ESTABLISHED",
        "promotion": "NOT_AUTHORIZED",
        "independence_note": (
            "Host attests custody, preregistration, rubric and actual independence."
        ),
        "limitations": [
            "Conditional paired metric evidence is not general alignment improvement.",
            "Existing protected coevolution stores alone decide candidate admission.",
            "Intervals require declared independent clusters and complete authenticated pairs.",
        ],
    }
    validate_input_envelope(result)
    result["result_digest"] = content_digest(result)
    return result


def finite_reconstruction_trial(
    archive: ResearchArchive,
    catalog: ResearchCatalog,
    selector: SourceSelector,
    *,
    trusted_catalog_digest: str,
    enabled: bool = False,
) -> dict[str, Any]:
    _enabled(enabled)
    s = _record(selector, SourceSelector)
    c = _record(catalog, ResearchCatalog)
    if s.kind != "document" or s.pointer:
        raise ValueError("resource trial requires an entire finite source document")
    root = _source_document(archive, c, s, trusted_catalog_digest)
    if (
        type(root) is not dict
        or set(root) != {"schema_version", "fixture", "qualification"}
        or root["schema_version"] != "finite-cva-source-v1"
    ):
        raise ValueError("unsupported finite reconstruction source")
    fixture = SecurityFixture.from_dict(root["fixture"])
    before = archive.read(
        c, s.artifact_id, s.version, trusted_catalog_digest=trusted_catalog_digest, enabled=True
    )
    started = perf_counter_ns()
    full = qualify_security_fixture(fixture, enabled=True)
    full_ns = perf_counter_ns() - started
    if canonical_json(full) != canonical_json(root["qualification"]):
        raise ValueError("cached finite qualification differs from independently checked source")
    full_state = {"source": s.to_dict(), "checked_result": full}
    compact_state = {
        "source": s.to_dict(),
        "fixture_digest": fixture.digest,
        "checker": "gepa-cva-finite-v1",
    }
    started = perf_counter_ns()
    restored = _source_document(archive, c, s, trusted_catalog_digest)
    if type(restored) is not dict:
        raise ValueError("finite source unavailable during reconstruction")
    compact = qualify_security_fixture(SecurityFixture.from_dict(restored["fixture"]), enabled=True)
    compact_ns = perf_counter_ns() - started
    after = archive.read(
        c, s.artifact_id, s.version, trusted_catalog_digest=trusted_catalog_digest, enabled=True
    )
    result = {
        "schema_version": "finite-verification-reconstruction-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "source": s.to_dict(),
        "fixture_digest": fixture.digest,
        "source_bytes_retained": len(before),
        "full": {
            "retained_state_bytes": len(canonical_json(full_state).encode()),
            "verification_wall_time_ns": full_ns,
        },
        "compact": {
            "retained_state_bytes": len(canonical_json(compact_state).encode()),
            "reconstruction_wall_time_ns": compact_ns,
        },
        "verification_fidelity": canonical_json(full) == canonical_json(compact),
        "source_unchanged": before == after and sha256(after).hexdigest() == s.source_digest,
        "independent_checker_calls": 2,
        "checked_result_digest": content_digest(full),
        "measurement_scope": "serialized checkpoint state; original archive retained",
        "mathematical_theorem_claim": False,
        "peak_memory_measured": False,
        "limitations": [
            "Only exact bundled finite fixtures and their assumptions are checked.",
            "Original source bytes and audit metadata remain in the immutable archive.",
            "Checkpoint bytes do not establish peak memory savings or a transformer theorem.",
        ],
    }
    validate_input_envelope(result)
    result["result_digest"] = content_digest(result)
    return result
