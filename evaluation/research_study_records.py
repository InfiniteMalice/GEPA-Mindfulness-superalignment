"""Versioned declarations for research overlays, never canonical cases or authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

# Third-party
# Local
from evaluation.improvement_records import (
    DatasetManifest,
    DiagnosticEvidence,
    EvaluationSlot,
    ImprovementRecord,
    MetricSpec,
    ResamplingPolicy,
    SystemConfig,
    _encode,
)
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

STUDY_SOURCE_SCHEMA = "research-study-observation-v1"
STUDIES = MappingProxyType(
    {
        "A": (
            "baseline_finetuning",
            "actionopd",
            "locd",
            "locd_actionopd",
            "locd_sst",
            "locd_actionopd_sst",
        ),
        "B": (
            "outcome_grpo_dapo",
            "knowgpo",
            "prm4rl",
            "vpo",
            "knowgpo_prm4rl",
            "vpo_knowgpo",
            "vpo_knowgpo_prm4rl",
        ),
        "C": (
            "peo_baseline",
            "action_ledger",
            "deloop",
            "deloop_peo_memory",
            "deloop_locd",
            "deloop_peo_memory_locd",
        ),
        "D": ("bounded_routing", "generative_routing", "jev_decisions", "lev_native_depth"),
        "E": (
            "fixed_data_harness",
            "autosft",
            "rsiflow",
            "harness_evolution",
            "autosft_rsiflow",
            "autosft_rsiflow_harness_evolution",
        ),
        "F": ("structured_retrieval", "librarian_workspace", "latent_memory", "hybrid_memory"),
    }
)


def _refs(values: tuple[EvidenceReference, ...], *, required: bool = True) -> None:
    if not (1 if required else 0) <= len(values) <= 32:
        raise ValueError("bounded public references required")
    if len({v.reference_id for v in values}) != len(values) or any(
        len(v.reference_id) > 256 for v in values
    ):
        raise ValueError("duplicate or oversized reference identity")


@dataclass(frozen=True, kw_only=True)
class _StudyRecord(ImprovementRecord):
    training_eligibility: str = "DEVELOPMENT"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.training_eligibility != "DEVELOPMENT"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("study declarations confer no authority or optimizer eligibility")
        validate_input_envelope(_encode(self))


@dataclass(frozen=True)
class StudyRuntime(_StudyRecord):
    """External runtime declarations do not establish qualification or permission to run."""

    condition_id: str
    availability: str
    reason: str
    system: SystemConfig | None = None
    runtime_version: str | None = None
    runtime_config_digest: str | None = None
    parent_checkpoint_digest: str | None = None
    source_refs: tuple[EvidenceReference, ...] = ()
    schema_version: str = "research-study-runtime-v1"

    def _validate(self) -> None:
        if self.schema_version != "research-study-runtime-v1":
            raise ValueError("unsupported runtime declaration")
        if self.availability not in ("configured", "unsupported", "not_requested"):
            raise ValueError("invalid runtime availability")
        if len(self.condition_id) > 128 or len(self.reason) > 2048:
            raise ValueError("runtime declaration text exceeds limit")
        values = (
            self.system,
            self.runtime_version,
            self.runtime_config_digest,
            self.parent_checkpoint_digest,
        )
        if self.availability == "configured":
            if any(v is None for v in values):
                raise ValueError("configured runtime requires complete version/config/checkpoint")
            if len(self.runtime_version) > 256:
                raise ValueError("runtime version exceeds limit")
            if any(
                len(getattr(self.system, f)) > 256 for f in ("model_version", "harness_version")
            ):
                raise ValueError("system version exceeds limit")
            _refs(self.source_refs)
        elif any(v is not None for v in values) or self.source_refs:
            raise ValueError("unrun runtime must not fabricate system/version provenance")


@dataclass(frozen=True)
class StudyConfiguration(_StudyRecord):
    study_id: str
    runtimes: tuple[StudyRuntime, ...]
    manifest: DatasetManifest
    metrics: tuple[MetricSpec, ...]
    evaluator: TrustedEvaluatorContract
    evidence_refs: tuple[EvidenceReference, ...]
    seeds: tuple[int, ...]
    repeats: int
    inference_budget_digest: str
    training_budget_digest: str
    parent_checkpoint_digest: str
    training_data_digest: str
    protected_manifest_digest: str
    source_revision: str
    policy: ResamplingPolicy
    schema_version: str = "research-study-configuration-v1"

    def _validate(self) -> None:
        if self.schema_version != "research-study-configuration-v1" or self.study_id not in STUDIES:
            raise ValueError("unsupported study configuration")
        if tuple(r.condition_id for r in self.runtimes) != STUDIES[self.study_id]:
            raise ValueError("complete ordered study roster required, including unrun arms")
        active = [r for r in self.runtimes if r.availability == "configured"]
        if active and self.runtimes[0].availability != "configured":
            raise ValueError("configured comparisons require the declared study baseline")
        if any(r.parent_checkpoint_digest != self.parent_checkpoint_digest for r in active):
            raise ValueError("runtime parent checkpoint does not match common initialization")
        if not 2 <= len(self.seeds) <= 16 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("requires 2..16 unique declared seeds")
        if any(v < 0 for v in self.seeds) or not 1 <= self.repeats <= 8:
            raise ValueError("invalid seed or repeat count")
        if not 1 <= len(self.manifest.cases) <= 128 or any(
            len(c.case_id) > 256 for c in self.manifest.cases
        ):
            raise ValueError("bounded task manifest required")
        if not 1 <= len(self.metrics) <= 16 or len({m.metric_id for m in self.metrics}) != len(
            self.metrics
        ):
            raise ValueError("bounded unique component metrics required")
        if any(m.source_schema != STUDY_SOURCE_SCHEMA for m in self.metrics):
            raise ValueError("research captures require the explicit study source schema")
        if any(
            len(m.metric_id) > 128 or len(m.source_metric) > 128 or len(m.unit) > 64
            for m in self.metrics
        ):
            raise ValueError("metric identity exceeds limit")
        if len(self.source_revision) > 256 or not 1 <= self.policy.resamples <= 10000:
            raise ValueError("revision or resampling budget exceeds limit")
        if any(
            type(v) is not str or len(v) > 256
            for v in (
                self.evaluator.evaluator_id,
                self.evaluator.evaluator_version,
                self.evaluator.contract_id,
            )
        ):
            raise ValueError("invalid evaluator identity")
        _refs(self.evidence_refs)


@dataclass(frozen=True, kw_only=True)
class _PrivateStudyRecord(ImprovementRecord):
    training_eligibility: str = "HIDDEN_EVAL"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.training_eligibility != "HIDDEN_EVAL"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("private study captures cannot grant optimizer or runtime authority")
        validate_input_envelope(_encode(self))


@dataclass(frozen=True)
class StudyMetricCapture(_PrivateStudyRecord):
    """One declared observable component; status never substitutes for authentication."""

    slot_id: str
    value: float | bool | None
    unit: str
    status: str
    cohort: str
    severity: str
    evidence_refs: tuple[EvidenceReference, ...]
    outcome: bool | None = None
    reason: str | None = None
    eligible: bool = True
    schema_version: str = "research-study-metric-capture-v1"

    def _validate(self) -> None:
        if self.schema_version != "research-study-metric-capture-v1":
            raise ValueError("unsupported metric capture")
        if self.status not in ("observed", "unresolved", "failed", "not_run"):
            raise ValueError("invalid capture status")
        if (self.status != "observed" or not self.eligible) and self.value is not None:
            raise ValueError("unresolved or unrun captures must not invent numeric outcomes")
        if self.severity not in ("routine", "consequential", "catastrophic"):
            raise ValueError("unsupported consequence stratum")
        if len(self.slot_id) > 128 or len(self.unit) > 64 or len(self.cohort) > 256:
            raise ValueError("capture identity exceeds limit")
        if self.reason is not None and len(self.reason) > 1024:
            raise ValueError("capture reason exceeds limit")
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class StudyCaptureEnvelope(_PrivateStudyRecord):
    """Retain outer privacy when handing a legacy DiagnosticEvidence record to the host."""

    study_plan_digest: str
    evidence: DiagnosticEvidence
    schema_version: str = "research-study-capture-envelope-v1"

    def _validate(self) -> None:
        if self.schema_version != "research-study-capture-envelope-v1":
            raise ValueError("unsupported study envelope")


def validate_research_capture_source(
    source: dict,
    row: dict,
    slot: EvaluationSlot,
    evidence_refs: tuple[EvidenceReference, ...],
    protocol_digest: str,
) -> None:
    """Validate the new source schema before the existing host authentication boundary."""
    if (
        type(protocol_digest) is not str
        or len(protocol_digest) != 64
        or any(c not in "0123456789abcdef" for c in protocol_digest)
    ):
        raise ValueError("research source requires an exact protocol digest context")
    keys = {
        "schema_version",
        "training_eligibility",
        "confers_authority",
        "study_plan_digest",
        "protocol_digest",
        "slot_id",
        "case_id",
        "budget_digest",
        "system",
        "evaluator",
        "rows",
        "result_digest",
    }
    if set(source) != keys or (
        source["schema_version"] != STUDY_SOURCE_SCHEMA
        or source["training_eligibility"] != "HIDDEN_EVAL"
        or source["confers_authority"] is not False
        or source["protocol_digest"] != protocol_digest
        or source["slot_id"] != slot.slot_id
        or source["case_id"] != slot.case_id
        or source["budget_digest"] != slot.budget_digest
        or len(source["rows"]) != 1
    ):
        raise ValueError("research source protocol/slot/case/budget/privacy mismatch")
    digest = source["study_plan_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid research plan digest")
    system = source["system"]
    if (
        type(system) is not dict
        or set(system) != {"model_version", "harness_version", "repeat_id", "seed"}
        or any(type(system[k]) is not int for k in ("repeat_id", "seed"))
        or any(
            type(system[k]) is not str or not system[k].strip() or len(system[k]) > 256
            for k in ("model_version", "harness_version")
        )
    ):
        raise ValueError("invalid research system binding")
    evaluator = source["evaluator"]
    if (
        type(evaluator) is not dict
        or set(evaluator) != {"evaluator_id", "evaluator_version", "contract_id"}
        or any(type(v) is not str or not v.strip() or len(v) > 256 for v in evaluator.values())
    ):
        raise ValueError("invalid research evaluator binding")
    if set(row) != {
        "opportunity_id",
        "metric",
        "unit",
        "value",
        "outcome",
        "status",
        "eligible",
        "cohort",
        "severity",
        "reason",
        "evidence_refs",
    }:
        raise ValueError("research row requires exact capture fields")
    capture = StudyMetricCapture.from_dict(
        dict(
            training_eligibility="HIDDEN_EVAL",
            optimizer_input=False,
            confers_authority=False,
            schema_version="research-study-metric-capture-v1",
            slot_id=row["opportunity_id"],
            value=row["value"],
            unit=row["unit"],
            status=row["status"],
            cohort=row["cohort"],
            severity=row["severity"],
            evidence_refs=row["evidence_refs"],
            outcome=row["outcome"],
            reason=row["reason"],
            eligible=row["eligible"],
        )
    )
    if capture.slot_id != slot.slot_id or capture.evidence_refs != evidence_refs:
        raise ValueError("research capture identity or evidence references changed")
