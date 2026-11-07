"""Experimental prediction-bound intervention records; serialization grants no authority."""

# Standard library
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_enum,
    _record,
    array,
    digest,
    enum,
    references,
    text,
)
from gepa_mindfulness.verification.diagnostic_records import restore_refs
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

from .native_records import HookKind, HookSpec, NativeInput, finite, index, indices


class InterventionMode(str, Enum):
    ACTIVATION_PATCH = "activation_patch"
    ACTIVATION_REPLACE = "activation_replace"
    ABLATION = "ablation"
    COMPONENT_SWAP = "component_swap"
    RESTORATION = "targeted_restoration"
    PATH_PATCH = "path_patch"
    MEDIATION = "causal_mediation"


def _float_rows(value: object) -> tuple[tuple[float, ...], ...]:
    rows = array(value, 128, required=True)
    for row in rows:
        for v in array(row, 1_000_000, required=True):
            finite(v)
    return rows


def _read_rows(value: object) -> tuple[tuple[float, ...], ...]:
    return tuple(_read_array(row) for row in _read_array(value))


@dataclass(frozen=True)
class InterventionSite(CVARecord):
    hook: HookSpec
    component_indices: tuple[int, ...]
    token_positions: tuple[int, ...]

    schema_version = "native-intervention-site-v1"
    restorers = {
        "hook": HookSpec.from_dict,
        "component_indices": _read_array,
        "token_positions": _read_array,
    }

    def __post_init__(self) -> None:
        hook = _record(self.hook, HookSpec)
        components = indices(self.component_indices, 64)
        positions = indices(self.token_positions, 128)
        width = hook.heads if hook.kind is HookKind.ATTENTION_HEAD else hook.width
        if max(components) >= width or max(positions) >= 128:
            raise ValueError("intervention site exceeds declared hook geometry")
        if len(self.columns) > 64:
            raise ValueError("expanded intervention exceeds the 64-column budget")

    @property
    def columns(self) -> tuple[int, ...]:
        if self.hook.kind is not HookKind.ATTENTION_HEAD:
            return self.component_indices
        return tuple(
            c * self.hook.head_dim + d
            for c in self.component_indices
            for d in range(self.hook.head_dim)
        )


@dataclass(frozen=True)
class LogitContrast(CVARecord):
    position: int
    target_token: int
    reference_token: int

    schema_version = "native-logit-contrast-v1"

    def __post_init__(self) -> None:
        index(self.position, maximum=128)
        index(self.target_token)
        index(self.reference_token)
        if self.target_token == self.reference_token:
            raise ValueError("contrast requires distinct tokens")


@dataclass(frozen=True)
class EffectPrediction(CVARecord):
    direction: int
    minimum_effect: float
    maximum_effect: float
    minimum_control_gap: float

    schema_version = "native-effect-prediction-v1"

    def __post_init__(self) -> None:
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("effect direction requires exact -1 or +1")
        for value in (self.minimum_effect, self.maximum_effect, self.minimum_control_gap):
            finite(value)
        if not 0 < self.minimum_effect <= self.maximum_effect or self.minimum_control_gap <= 0:
            raise ValueError("prediction requires positive effect bounds and control gap")


@dataclass(frozen=True)
class CausalPlan(CVARecord):
    experiment_id: str
    hypothesis: str
    model_digest: str
    donor: NativeInput
    recipient: NativeInput
    source: InterventionSite
    mediator: InterventionSite | None
    mode: InterventionMode
    metric: LogitContrast
    prediction: EffectPrediction
    evidence_refs: tuple[EvidenceReference, ...]
    seed: int
    replacement_values: tuple[tuple[float, ...], ...] | None = None
    max_distribution_values: int = 250_000

    schema_version = "native-causal-plan-v1"
    restorers = {
        "donor": NativeInput.from_dict,
        "recipient": NativeInput.from_dict,
        "source": InterventionSite.from_dict,
        "mediator": lambda v: None if v is None else InterventionSite.from_dict(v),
        "mode": lambda v: _read_enum(v, InterventionMode),
        "metric": LogitContrast.from_dict,
        "prediction": EffectPrediction.from_dict,
        "evidence_refs": restore_refs,
        "replacement_values": lambda v: None if v is None else _read_rows(v),
    }

    def __post_init__(self) -> None:
        text(self.experiment_id, "experiment_id", 160)
        text(self.hypothesis, "hypothesis", 4096)
        digest(self.model_digest)
        donor, recipient = _record(self.donor, NativeInput), _record(self.recipient, NativeInput)
        source = _record(self.source, InterventionSite)
        if (
            donor.example_id == recipient.example_id
            or donor.token_ids == recipient.token_ids
            or len(donor.token_ids) != len(recipient.token_ids)
            or donor.attention_mask != recipient.attention_mask
        ):
            raise ValueError(
                "causal pair requires distinct contexts and matched token/mask geometry"
            )
        enum(self.mode, InterventionMode)
        metric = _record(self.metric, LogitContrast)
        _record(self.prediction, EffectPrediction)
        references(self.evidence_refs)
        index(self.seed, maximum=2**32)
        if (
            type(self.max_distribution_values) is not int
            or not 1 <= self.max_distribution_values <= 1_000_000
        ):
            raise ValueError("invalid distribution evidence budget")
        needs_mediator = self.mode in (
            InterventionMode.PATH_PATCH,
            InterventionMode.MEDIATION,
            InterventionMode.RESTORATION,
        )
        if needs_mediator != (self.mediator is not None):
            raise ValueError("this mode requires exactly its declared mediator")
        sites: tuple[InterventionSite, ...] = (source,)
        if self.mediator is not None:
            mediator = _record(self.mediator, InterventionSite)
            if (source.hook.module_path, source.hook.location) == (
                mediator.hook.module_path,
                mediator.hook.location,
            ):
                raise ValueError("mediator must use a distinct hook")
            if mediator.hook.layer < source.hook.layer:
                raise ValueError("mediator cannot precede source layer")
            sites += (mediator,)
        for position in tuple(p for s in sites for p in s.token_positions) + (metric.position,):
            if position >= len(donor.token_ids) or not donor.attention_mask[position]:
                raise ValueError("site/metric lacks actual unmasked context coverage")
        if (self.mode is InterventionMode.ACTIVATION_REPLACE) != (
            self.replacement_values is not None
        ):
            raise ValueError("literal replacement values belong only to replacement mode")
        if self.replacement_values is not None:
            rows = _float_rows(self.replacement_values)
            if len(rows) != len(source.token_positions) or any(
                len(r) != len(source.columns) for r in rows
            ):
                raise ValueError("replacement values do not match intervention geometry")


def _snapshot_json(value: str, kind: type) -> Any:
    text(value, "action-bound snapshot", 8192)
    try:
        data = json.loads(value)
        if type(data) is not dict:
            raise ValueError("snapshot must be an object")
        if kind is PredictionCommit:
            if type(data.get("evidence_refs")) is not list:
                raise ValueError("prediction references must be an array")
            data["evidence_refs"] = tuple(data["evidence_refs"])
        result = kind(**data)
        if canonical_json(result.to_dict()) != value:
            raise ValueError("snapshot requires exact canonical fields")
        return result
    except (TypeError, RecursionError, OverflowError) as error:
        raise ValueError("invalid action-bound snapshot") from error


@dataclass(frozen=True)
class CausalCommit(CVARecord):
    plan: CausalPlan
    prediction_json: str
    action_json: str
    created_at: str

    schema_version = "native-causal-commit-v1"
    restorers = {"plan": CausalPlan.from_dict}

    def __post_init__(self) -> None:
        plan = _record(self.plan, CausalPlan)
        prediction = self.prediction_record
        action = self.action_record
        if prediction.to_dict()["predicted_outcome"] != {"causal_plan_digest": plan.digest}:
            raise ValueError("prediction does not bind the exact causal plan")
        if set(prediction.evidence_refs) != {r.reference_id for r in plan.evidence_refs}:
            raise ValueError("prediction must bind the plan evidence references")
        if (
            action.prediction_commit_id != prediction.prediction_commit_id
            or action.authorization_scope != "mechanistic-plan:" + plan.digest
            or action.action_class != "mechanistic_intervention"
            or action.reversible is not True
        ):
            raise ValueError("action does not bind the exact prediction/plan scope")
        parse_rfc3339_datetime(self.created_at, "created_at")

    @property
    def prediction_record(self) -> PredictionCommit:
        return _snapshot_json(self.prediction_json, PredictionCommit)

    @property
    def action_record(self) -> ActionRecord:
        return _snapshot_json(self.action_json, ActionRecord)


def run_measurement_digest(report: dict[str, Any]) -> str:
    """Bind all numeric conditions/controls; timestamps and event copies are separate."""
    keys = (
        "plan_digest",
        "commit",
        "commit_digest",
        "model",
        "conditions",
        "primary_effect",
        "control_effect",
        "direction_control_effect",
        "evidence_refs",
        "control_roles",
        "limitations",
        "mediation",
        "prediction_result",
        "executor_id",
        "executor_version",
    )
    return content_digest({key: report[key] for key in keys})
