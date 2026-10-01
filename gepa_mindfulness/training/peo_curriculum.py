"""Opt-in PEO dataset scheduling with persistent anchors and existing admission gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.participatory_agency.training.curriculum import (
    TEMPORAL_FEATURES,
    CurriculumPhase,
    PEOStage,
)
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

from .eligibility import require_training_eligible
from .seeds import validate_seed
from .trajectory import RolloutRequest

DIMENSIONS = frozenset("UAHPDTMCRISJE")


def _text(value: object, name: str) -> None:
    """Reject absent or coerced identifiers at the public planning boundary."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _strings(value: object, name: str) -> None:
    """Require unique immutable labels so a repeated ID cannot gain sampling weight."""
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")


def _records(values: object, kind: type) -> None:
    """Keep nested records immutable and prevent duck-typed admission bypasses."""
    if not isinstance(values, tuple) or not values or any(type(v) is not kind for v in values):
        raise ValueError(f"expected a nonempty tuple of {kind.__name__}")


def _request_dict(request: RolloutRequest) -> dict[str, Any]:
    """Detach complete request data for an auditable plan identity."""
    return dict(
        prompt=request.prompt,
        case_id=request.case_id,
        num_samples=request.num_samples,
        sampling_parameters=thaw_json_mapping(request.sampling_parameters),
        metadata=thaw_json_mapping(request.metadata),
        policy_version=request.policy_version,
        seed=request.seed,
    )


class CurriculumBucket(str, Enum):
    """Persistent anchors plus three curator-defined sources of variation."""

    ANCHOR = "anchor"
    WEAKNESS = "weakness"
    FRONTIER = "frontier"
    OOD = "ood"


@dataclass(frozen=True)
class CurriculumUnit:
    """A reviewed data group kept contiguous when a curriculum round is materialized."""

    unit_id: str
    stage: PEOStage
    bucket: CurriculumBucket
    requests: tuple[RolloutRequest, ...]
    dimensions: Mapping[str, float] = field(default_factory=dict)
    temporal_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot request metadata without reclassifying the source or altering prompts."""
        _text(self.unit_id, "unit_id")
        if not isinstance(self.stage, PEOStage) or not isinstance(self.bucket, CurriculumBucket):
            raise ValueError("stage and bucket must use their canonical enums")
        if self.bucket is CurriculumBucket.ANCHOR and self.stage is not PEOStage.CAUSAL:
            raise ValueError("persistent anchors must belong to the foundational causal stage")
        _records(self.requests, RolloutRequest)
        snapshots = []
        for request in self.requests:
            _text(request.prompt, "prompt")
            for name in ("case_id", "policy_version"):
                if getattr(request, name) is not None:
                    _text(getattr(request, name), name)
            if type(request.num_samples) is not int or request.num_samples <= 0:
                raise ValueError("num_samples must be a positive integer")
            validate_seed(request.seed, sample_count=request.num_samples)
            metadata = freeze_json_mapping(request.metadata, field_name="metadata")
            if "peo_curriculum" in metadata:
                raise ValueError("source metadata already contains reserved peo_curriculum data")
            snapshots.append(
                replace(
                    request,
                    metadata=metadata,
                    sampling_parameters=freeze_json_mapping(
                        request.sampling_parameters, field_name="sampling_parameters"
                    ),
                )
            )
        object.__setattr__(self, "requests", tuple(snapshots))
        if not isinstance(self.dimensions, Mapping) or set(self.dimensions) - DIMENSIONS:
            raise ValueError("dimensions must use U/A/H/P/D/T/M/C/R/I/S/J/E")
        for value in self.dimensions.values():
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("dimension annotations must be finite numbers in [0, 1]")
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        _strings(self.temporal_features, "temporal_features")
        if set(self.temporal_features) - set(TEMPORAL_FEATURES):
            raise ValueError("unknown temporal feature")

    def to_dict(self) -> dict[str, Any]:
        """Export detached annotations and original source-bearing requests."""
        return dict(
            unit_id=self.unit_id,
            stage=self.stage.value,
            bucket=self.bucket.value,
            requests=[_request_dict(r) for r in self.requests],
            dimensions=dict(self.dimensions),
            temporal_features=list(self.temporal_features),
        )


@dataclass(frozen=True)
class CurriculumMixture:
    """Exact unit quotas; at least one quarter of every round remains anchors."""

    anchor: int = 4
    weakness: int = 2
    frontier: int = 1
    ood: int = 1

    def __post_init__(self) -> None:
        """Fail closed instead of silently normalizing away a protected pool."""
        values = self.to_dict().values()
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("each mixture quota must be a positive integer")
        if 4 * self.anchor < sum(values):
            raise ValueError("anchor quota must cover at least one quarter of selected units")

    def to_dict(self) -> dict[str, int]:
        """Expose stable bucket names for manifests and sampling."""
        return {bucket.value: getattr(self, bucket.value) for bucket in CurriculumBucket}


@dataclass(frozen=True)
class AnchorResult:
    """Host-observed retention outcome; an external reference is not self-authenticating."""

    unit_id: str
    passed: bool
    evidence: EvidenceReference

    def __post_init__(self) -> None:
        """Exclude private reasoning and non-boolean judgments from retention evidence."""
        _text(self.unit_id, "anchor unit_id")
        if type(self.passed) is not bool:
            raise ValueError("anchor passed must be a boolean")
        if type(self.evidence) is not EvidenceReference or (
            self.evidence.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD
        ):
            raise ValueError("anchor outcomes require external-record evidence")

    def to_dict(self) -> dict[str, Any]:
        """Retain the source reference with its declared outcome."""
        return dict(unit_id=self.unit_id, passed=self.passed, evidence=self.evidence.to_dict())


@dataclass(frozen=True)
class AnchorEvaluation:
    """A host evaluation of one exact plan, with the model/harness actually evaluated."""

    plan_digest: str
    system: EvaluatedSystemVersion
    results: tuple[AnchorResult, ...]

    def __post_init__(self) -> None:
        """Require exact plan identity and unambiguous anchor coverage."""
        if (
            not isinstance(self.plan_digest, str)
            or len(self.plan_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.plan_digest)
        ):
            raise ValueError("plan_digest must be a lowercase SHA-256 digest")
        if type(self.system) is not EvaluatedSystemVersion:
            raise ValueError("anchor evaluation requires an evaluated system version")
        _records(self.results, AnchorResult)
        if len({r.unit_id for r in self.results}) != len(self.results):
            raise ValueError("anchor results cannot duplicate unit IDs")

    def to_dict(self) -> dict[str, Any]:
        """Export evaluator identity and each retained observation."""
        return dict(
            plan_digest=self.plan_digest,
            system=dict(
                model_version=self.system.model_version, harness_version=self.system.harness_version
            ),
            results=[r.to_dict() for r in sorted(self.results, key=lambda r: r.unit_id)],
        )


@dataclass(frozen=True, kw_only=True)
class PEOCurriculumDataset:
    """An immutable opt-in DatasetProvider, suitable for RLTrainingEngine.dataset_factory."""

    units: tuple[CurriculumUnit, ...]
    phase: CurriculumPhase
    stage: PEOStage = PEOStage.CAUSAL
    mixture: CurriculumMixture = CurriculumMixture()
    seed: int = 0
    round_id: int = 0
    weakness_ids: tuple[str, ...] = ()
    last_evaluation: AnchorEvaluation | None = None
    enabled: bool = False

    def __post_init__(self) -> None:
        """Validate all pools before sampling and keep the full anchor bank immutable."""
        if self.enabled is not True:
            raise ValueError("PEO curriculum requires enabled=True")
        _records(self.units, CurriculumUnit)
        if len({u.unit_id for u in self.units}) != len(self.units):
            raise ValueError("curriculum unit IDs must be unique")
        object.__setattr__(self, "units", tuple(sorted(self.units, key=lambda u: u.unit_id)))
        if type(self.phase) is not CurriculumPhase or not isinstance(self.stage, PEOStage):
            raise ValueError("phase and stage must use validated curriculum records")
        if type(self.mixture) is not CurriculumMixture:
            raise ValueError("mixture must be CurriculumMixture")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        validate_seed(self.seed)
        if type(self.round_id) is not int or not 0 <= self.round_id <= 2**53 - 1:
            raise ValueError("round_id must be a nonnegative JSON-safe integer")
        if self.last_evaluation is not None and type(self.last_evaluation) is not AnchorEvaluation:
            raise ValueError("last_evaluation must be AnchorEvaluation")
        _strings(self.weakness_ids, "weakness_ids")
        active = self._active_units()
        for bucket in CurriculumBucket:
            if not any(u.bucket is bucket for u in active):
                raise ValueError(f"current stage has no {bucket.value} pool")
        eligible_targets = {u.unit_id for u in active if u.bucket is CurriculumBucket.WEAKNESS}
        if set(self.weakness_ids) - eligible_targets:
            raise ValueError("weakness targets must name current-stage weakness units")

    def _active_units(self) -> tuple[CurriculumUnit, ...]:
        """Retain foundational anchors across every data-stage transition."""
        return tuple(
            u for u in self.units if u.bucket is CurriculumBucket.ANCHOR or u.stage is self.stage
        )

    @property
    def digest(self) -> str:
        """Bind evaluation evidence to this exact source-bearing schedule and round."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return an evaluator-side manifest; source metadata is not an actor prompt."""
        return dict(
            schema_version="peo-curriculum-v1",
            units=[u.to_dict() for u in self.units],
            phase=dict(
                name=self.phase.name,
                description=self.phase.description,
                active_heads=list(self.phase.active_heads),
                loss_weights=dict(self.phase.loss_weights),
            ),
            stage=self.stage.value,
            mixture=self.mixture.to_dict(),
            seed=self.seed,
            round_id=self.round_id,
            weakness_ids=sorted(self.weakness_ids),
            last_evaluation=(
                None if self.last_evaluation is None else self.last_evaluation.to_dict()
            ),
        )

    def materialize(self, mode: str) -> tuple[RolloutRequest, ...]:
        """Return one deterministic round; train/resume require explicit existing admission."""
        if mode not in {"train", "resume", "collect", "evaluate"}:
            raise ValueError("unknown dataset materialization mode")
        active = self._active_units()
        if mode in {"train", "resume"}:
            # Even inactive sources contribute to the plan digest carried into training.
            for unit in self.units:
                for request in unit.requests:
                    if request.metadata.get("training_eligibility") != "TRAIN":
                        raise ValueError("curriculum training_eligibility requires explicit TRAIN")
                    require_training_eligible(request.metadata)
        selected: list[CurriculumUnit] = []
        for bucket, count in self.mixture.to_dict().items():
            pool = [
                u
                for u in active
                if u.bucket.value == bucket
                and (
                    bucket != "weakness" or not self.weakness_ids or u.unit_id in self.weakness_ids
                )
            ]
            random.Random(f"{self.seed}:{self.stage.value}:{bucket}").shuffle(pool)
            selected.extend(
                pool[(self.round_id * count + offset) % len(pool)] for offset in range(count)
            )
        random.Random(f"{self.seed}:{self.round_id}:order").shuffle(selected)
        result = []
        digest = self.digest
        for position, unit in enumerate(selected):
            report = dict(
                plan_digest=digest,
                phase=self.phase.name,
                stage=self.stage.value,
                bucket=unit.bucket.value,
                unit_id=unit.unit_id,
                unit_position=position,
                dimensions=dict(unit.dimensions),
                temporal_features=list(unit.temporal_features),
                seed=self.seed,
                round_id=self.round_id,
                anchor_evaluation=(
                    None if self.last_evaluation is None else self.last_evaluation.to_dict()
                ),
            )
            for index, request in enumerate(unit.requests):
                metadata = thaw_json_mapping(request.metadata)
                metadata["peo_curriculum"] = json.loads(
                    json.dumps(dict(report, request_index=index))
                )
                result.append(
                    replace(
                        request,
                        metadata=metadata,
                        sampling_parameters=thaw_json_mapping(request.sampling_parameters),
                    )
                )
        return tuple(result)

    def adapt(
        self,
        evaluation: AnchorEvaluation,
        *,
        stage: PEOStage | None = None,
        mixture: CurriculumMixture | None = None,
        weakness_ids: tuple[str, ...] | None = None,
    ) -> PEOCurriculumDataset:
        """Create a new round only after current, complete, passing anchor observations."""
        if type(evaluation) is not AnchorEvaluation or evaluation.plan_digest != self.digest:
            raise ValueError("adaptation requires evaluation of the current plan digest")
        anchors = {u.unit_id for u in self.units if u.bucket is CurriculumBucket.ANCHOR}
        if {r.unit_id for r in evaluation.results} != anchors or not all(
            r.passed for r in evaluation.results
        ):
            raise ValueError("adaptation requires passing observations for every anchor")
        target_stage = self.stage if stage is None else stage
        if not isinstance(target_stage, PEOStage):
            raise ValueError("target stage must be PEOStage")
        stages = list(PEOStage)
        if stages.index(target_stage) - stages.index(self.stage) not in (0, 1):
            raise ValueError("adaptation may stay at the current stage or advance one stage")
        targets = self.weakness_ids if weakness_ids is None else weakness_ids
        if target_stage is not self.stage and weakness_ids is None:
            targets = ()
        return replace(
            self,
            stage=target_stage,
            mixture=self.mixture if mixture is None else mixture,
            weakness_ids=targets,
            round_id=self.round_id + 1,
            last_evaluation=evaluation,
        )
