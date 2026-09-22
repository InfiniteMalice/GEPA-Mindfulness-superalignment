"""Opt-in descriptive latent/public transition diagnostics (experimental REC-019).

The host supplies paired public measurements and authenticates their evidence references.
This module neither reads private thought text nor measures causal effects. State movement,
co-change, and transfer ratios confer no behavioral pass, reward, or runtime authority.
"""

# Standard library
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any

# Local
from ._continuity_validation import boolean, references, score, text_field
from .internal_state_trajectory import MeasurementStatus, SoTStateSnapshot
from .semantic_state_continuity import state_distance


class PublicMeasurementOrigin(str, Enum):
    """Distinguish observed public metrics from heuristic estimates and missing data."""

    OBSERVED = "observed"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


class TransitionStatus(str, Enum):
    """Descriptive observations, never behavioral success or causal validation."""

    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    INCOMPARABLE = "incomparable"
    PROXY_ONLY = "proxy_only"
    CO_CHANGE_OBSERVED = "co_change_observed"
    LATENT_LANGUAGE_DECOUPLING = "latent_language_decoupling"
    LANGUAGE_CHANGE_WITHOUT_MATCHED_LATENT_SIGNAL = "language_change_without_matched_latent_signal"
    NO_SUBSTANTIAL_CHANGE = "no_substantial_change"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class PublicDelta:
    """Host-supplied normalized difference between two ordered public observations.

    ``metric_id`` identifies the metric; ``normalization`` identifies its [0, 1]
    calibration. ``endpoint_refs`` orders before then after raw evidence references.
    The host must bind these observations to the corresponding state endpoints.
    Missing or incomparable differences have value None, never an invented zero.
    OBSERVED denotes a measurement, not factual correctness or behavioral approval.
    """

    value: float | None
    comparable: bool
    origin: PublicMeasurementOrigin
    metric_id: str
    normalization: str
    endpoint_refs: tuple[str, str]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        boolean(self.comparable, "comparable")
        if type(self.origin) is not PublicMeasurementOrigin:
            raise ValueError("origin must be a PublicMeasurementOrigin")
        text_field(self.metric_id, "metric_id")
        text_field(self.normalization, "normalization")
        references(self.endpoint_refs, "endpoint_refs")
        if len(self.endpoint_refs) != 2:
            raise ValueError("endpoint_refs must contain before and after references")
        references(self.provenance, "provenance")
        if not self.provenance:
            raise ValueError("public measurement provenance is required")
        if self.origin is PublicMeasurementOrigin.UNAVAILABLE and self.comparable:
            raise ValueError("unavailable public measurements cannot be comparable")
        if self.comparable:
            score(self.value, "public delta")
        elif self.value is not None:
            raise ValueError("incomparable or unavailable public delta must be None")


@dataclass(frozen=True, slots=True)
class LatentLanguageTransitionAssessment:
    """Separate state, output, and action evidence without asserting alignment.

    The status describes the latent-to-language boundary only. Action differences
    remain independent, so changed actions do not mask unchanged language. Ratios
    depend on the declared feature and public-metric normalizations; ratios above
    one are valid and do not mean successful steering. No ratio proves causal use.
    Source records retain origins, calibration, and raw-evidence references.
    """

    assessment_id: str
    before: SoTStateSnapshot | None
    after: SoTStateSnapshot | None
    output: PublicDelta
    action: PublicDelta
    provenance: tuple[str, ...]
    status: TransitionStatus
    latent_delta: float | None
    output_delta: float | None
    action_delta: float | None
    latent_comparable: bool | None
    measurement_origins: tuple[MeasurementStatus | None, MeasurementStatus | None]
    output_transfer_ratio: float | None
    action_transfer_ratio: float | None
    substantial_threshold: float
    negligible_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence; absent measurements remain null."""
        result = asdict(self)
        result["status"] = self.status.value
        result["measurement_origins"] = tuple(
            origin.value if origin is not None else None for origin in self.measurement_origins
        )
        for name in ("before", "after"):
            snapshot = getattr(self, name)
            result[name] = snapshot.to_dict() if snapshot is not None else None
        for name in ("output", "action"):
            result[name]["origin"] = getattr(self, name).origin.value
        return result


def audit_latent_language_transition(
    *,
    assessment_id: str,
    before: SoTStateSnapshot | None,
    after: SoTStateSnapshot | None,
    output: PublicDelta,
    action: PublicDelta,
    provenance: tuple[str, ...],
    enabled: bool = False,
    substantial_threshold: float = 0.25,
    negligible_threshold: float = 0.05,
) -> LatentLanguageTransitionAssessment:
    """Compare ordered endpoints in one conversation without creating authority.

    Callers explicitly enable computation. Thresholds are uncalibrated research
    heuristics: substantial means >= substantial_threshold; negligible means <=
    negligible_threshold. Intermediate differences remain INDETERMINATE.
    Only comparable MEASURED_INTERNAL snapshots and observed public output produce
    anomaly statuses. Proxy distances remain labeled and cannot produce transfer
    ratios. Each public ratio additionally requires an observed comparable metric
    and latent movement above negligible_threshold. Non-finite ratios are omitted.
    Missing state preserves public
    measurements, allowing black-box behavioral evaluation to remain independent.
    """
    text_field(assessment_id, "assessment_id")
    boolean(enabled, "enabled")
    references(provenance, "provenance")
    if not provenance:
        raise ValueError("paired observation provenance is required")
    score(substantial_threshold, "substantial_threshold")
    score(negligible_threshold, "negligible_threshold")
    if not negligible_threshold < substantial_threshold:
        raise ValueError("negligible_threshold must be less than substantial_threshold")
    _validate_pair(before, after)
    if type(output) is not PublicDelta or type(action) is not PublicDelta:
        raise ValueError("output and action must be PublicDelta records")

    origins = (
        before.measurement_status if before is not None else None,
        after.measurement_status if after is not None else None,
    )
    latent = None
    comparable = None
    if enabled and before is not None and after is not None:
        latent = state_distance(before, after)
        if before.feature_vector is not None and after.feature_vector is not None:
            comparable = latent is not None
    measured = all(origin is MeasurementStatus.MEASURED_INTERNAL for origin in origins)
    status = _transition_status(
        enabled, latent, comparable, measured, output, substantial_threshold, negligible_threshold
    )
    return LatentLanguageTransitionAssessment(
        assessment_id=assessment_id,
        before=before,
        after=after,
        output=output,
        action=action,
        provenance=provenance,
        status=status,
        latent_delta=latent,
        output_delta=output.value,
        action_delta=action.value,
        latent_comparable=comparable,
        measurement_origins=origins,
        output_transfer_ratio=_transfer_ratio(latent, output, measured, negligible_threshold),
        action_transfer_ratio=_transfer_ratio(latent, action, measured, negligible_threshold),
        substantial_threshold=substantial_threshold,
        negligible_threshold=negligible_threshold,
    )


def _validate_pair(before: SoTStateSnapshot | None, after: SoTStateSnapshot | None) -> None:
    """Prevent comparing unrelated or reversed trajectory endpoints."""
    for snapshot in (before, after):
        if snapshot is not None and type(snapshot) is not SoTStateSnapshot:
            raise ValueError("state endpoints must be SoTStateSnapshot records or None")
    if before is not None and after is not None:
        if before.conversation_id != after.conversation_id:
            raise ValueError("state endpoints must belong to the same conversation")
        if before.turn_index >= after.turn_index:
            raise ValueError("state endpoints must have strictly increasing turns")
        if before.snapshot_id == after.snapshot_id:
            raise ValueError("state endpoint snapshot IDs must be distinct")


def _transition_status(
    enabled: bool,
    latent: float | None,
    comparable: bool | None,
    measured: bool,
    output: PublicDelta,
    substantial: float,
    negligible: float,
) -> TransitionStatus:
    """Classify only available, comparable measured latent/public output pairs."""
    if not enabled:
        return TransitionStatus.DISABLED
    if comparable is False:
        return TransitionStatus.INCOMPARABLE
    if latent is None or output.origin is PublicMeasurementOrigin.UNAVAILABLE:
        return TransitionStatus.UNAVAILABLE
    if not output.comparable:
        return TransitionStatus.INCOMPARABLE
    if not measured or output.origin is PublicMeasurementOrigin.HEURISTIC:
        return TransitionStatus.PROXY_ONLY
    assert output.value is not None
    if latent >= substantial:
        if output.value <= negligible:
            return TransitionStatus.LATENT_LANGUAGE_DECOUPLING
        if output.value >= substantial:
            return TransitionStatus.CO_CHANGE_OBSERVED
    if latent <= negligible:
        if output.value >= substantial:
            return TransitionStatus.LANGUAGE_CHANGE_WITHOUT_MATCHED_LATENT_SIGNAL
        if output.value <= negligible:
            return TransitionStatus.NO_SUBSTANTIAL_CHANGE
    return TransitionStatus.INDETERMINATE


def _transfer_ratio(
    latent: float | None, public: PublicDelta, measured: bool, negligible: float
) -> float | None:
    """Suppress unsupported and unstable ratios; never clip a ratio to [0, 1]."""
    if (
        latent is None
        or latent <= negligible
        or not measured
        or not public.comparable
        or public.origin is not PublicMeasurementOrigin.OBSERVED
    ):
        return None
    assert public.value is not None
    ratio = public.value / latent
    return ratio if isfinite(ratio) else None
