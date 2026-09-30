"""Explicit scalar fusion diagnostics with conservative unknown-correlation defaults."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf, nextafter
from typing import Any

from .epistemic_state import (
    Availability,
    CorrelationTreatment,
    DiagonalState,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    _array,
    _enum,
    _nonnegative,
    _number,
    _snapshot,
    _strings,
    _text,
)

FUSION_VERSION = "scalar-source-fusion-v1"
MAX_SOURCES = 16


@dataclass(frozen=True, slots=True)
class ScalarFusionResult:
    """Detached computation output; source labels and covariance declarations are not authority."""

    estimate: EpistemicStateEstimate
    measurements: tuple[EpistemicMeasurement, ...]
    correlation_treatment: CorrelationTreatment
    weight_inputs: tuple[float, ...]
    weights: tuple[float, ...]
    mean_weights: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    covariance_provenance: tuple[str, ...]
    peer_exposed: bool
    uncertainty_scale: float

    def to_dict(self) -> dict[str, Any]:
        """Export inputs and output for audit/replay; this is not an event or trust certificate."""
        return dict(
            schema_version=FUSION_VERSION,
            estimate=self.estimate.to_dict(),
            measurements=[m.to_dict() for m in self.measurements],
            correlation_treatment=self.correlation_treatment.value,
            weight_inputs=list(self.weight_inputs),
            weights=list(self.weights),
            mean_weights=list(self.mean_weights),
            covariance=None if self.covariance is None else [list(r) for r in self.covariance],
            covariance_provenance=list(self.covariance_provenance),
            peer_exposed=self.peer_exposed,
            uncertainty_scale=self.uncertainty_scale,
        )


def fuse_scalar_measurements(
    measurements: tuple[EpistemicMeasurement, ...],
    *,
    estimate_id: str,
    mode: CorrelationTreatment = CorrelationTreatment.COVARIANCE_INTERSECTION,
    weights: tuple[float, ...] | None = None,
    covariance: tuple[tuple[float, ...], ...] | None = None,
    covariance_provenance: tuple[str, ...] = (),
    peer_exposed: bool = False,
    uncertainty_scale: float = 1.0,
) -> ScalarFusionResult:
    """Fuse a common scalar target without silently assuming independent source errors.

    Known covariance computes the variance of the declared convex mean, not an optimized mean.
    Unknown covariance defaults to CI. Every variance bound assumes valid input error bounds and
    a common unbiased target; calibration and actual peer exposure remain host responsibilities.
    """
    items = _measurements(measurements)
    _text(estimate_id, "estimate_id")
    _enum(mode, CorrelationTreatment, "mode")
    if type(peer_exposed) is not bool:
        raise ValueError("peer_exposed must be a built-in bool")
    scale = _number(uncertainty_scale, "uncertainty_scale")
    if scale <= 0:
        raise ValueError("uncertainty_scale must be positive")
    refs = _strings(covariance_provenance, "covariance_provenance")
    raw_weights = tuple(
        _nonnegative(w, "weight")
        for w in _array((1.0,) * len(items) if weights is None else weights, "weights")
    )
    if len(raw_weights) != len(items) or not any(raw_weights):
        raise ValueError("weights must match measurements and have positive total")
    total = sum(map(Fraction, raw_weights), Fraction())
    alpha = tuple(Fraction(w) / total for w in raw_weights)
    normalized = tuple(_finite_fraction(w, "weight") for w in alpha)
    if any(w > 0 and reported == 0 for w, reported in zip(alpha, normalized)):
        raise ValueError("normalized weight underflow")
    if mode is not CorrelationTreatment.KNOWN_COVARIANCE and (covariance is not None or refs):
        raise ValueError("covariance declarations require KNOWN_COVARIANCE mode")

    matrix = None
    state = None
    uncertainty = None
    coefficients: tuple[Fraction, ...] = ()
    if mode is not CorrelationTreatment.UNRESOLVED_CORRELATION:
        if any(m.variance is None or m.variance <= 0 for m in items):
            raise ValueError("numeric fusion requires positive available marginal variances")
        variances = tuple(Fraction(m.variance) for m in items if m.variance is not None)
        coefficients = alpha
        if mode is CorrelationTreatment.KNOWN_COVARIANCE:
            if covariance is None or not refs:
                raise ValueError("known covariance requires a full matrix and provenance")
            matrix = _covariance(covariance, items, peer_exposed)
            variance = sum(
                (
                    alpha[i] * alpha[j] * Fraction(matrix[i][j])
                    for i in range(len(items))
                    for j in range(len(items))
                ),
                Fraction(),
            )
        elif mode is CorrelationTreatment.COVARIANCE_INTERSECTION:
            precision = sum((w / v for w, v in zip(alpha, variances)), Fraction())
            variance = 1 / precision
            coefficients = tuple(w / v / precision for w, v in zip(alpha, variances))
        else:
            # Every convex combination's error variance is bounded by the largest marginal bound.
            variance = max(variances)
        mean = sum(
            (w * Fraction(m.value) for w, m in zip(coefficients, items) if m.value is not None),
            Fraction(),
        )
        numeric_variance = _finite_fraction(variance, "variance")
        if variance > 0:
            if numeric_variance == 0:
                raise ValueError("positive fused variance underflow")
            # Preserve the bound when converting exact arithmetic back to binary floats.
            if Fraction(numeric_variance) < variance:
                numeric_variance = _number(nextafter(numeric_variance, inf), "variance")
        state = DiagonalState(
            items[0].representation_id,
            (items[0].target_dimension,),
            (_finite_fraction(mean, "mean"),),
            (numeric_variance,),
        )
        uncertainty = float(
            Fraction(numeric_variance) / (Fraction(numeric_variance) + Fraction(scale))
        )
    evidence = tuple(dict.fromkeys(ref for m in items for ref in m.evidence_refs))
    provenance = tuple(
        dict.fromkeys((FUSION_VERSION, *refs, *(p for m in items for p in m.provenance)))
    )
    estimate = EpistemicStateEstimate(
        estimate_id=estimate_id,
        context=items[0].context,
        estimator_version=FUSION_VERSION,
        world_uncertainty=uncertainty,
        model_uncertainty=None,
        monitor_uncertainty=None,
        state=state,
        status=Availability.AVAILABLE if state is not None else Availability.UNAVAILABLE,
        evidence_refs=evidence,
        provenance=provenance,
    )
    mean_weights = tuple(_finite_fraction(w, "mean weight") for w in coefficients)
    if any(w > 0 and reported == 0 for w, reported in zip(coefficients, mean_weights)):
        raise ValueError("mean weight underflow")
    return ScalarFusionResult(
        estimate,
        items,
        mode,
        raw_weights,
        normalized,
        mean_weights,
        matrix,
        refs,
        peer_exposed,
        scale,
    )


def _measurements(values: object) -> tuple[EpistemicMeasurement, ...]:
    raw = _array(values, "measurements")
    if not 1 <= len(raw) <= MAX_SOURCES:
        raise ValueError(f"fusion requires 1..{MAX_SOURCES} measurements")
    items = tuple(_snapshot(m, EpistemicMeasurement) for m in raw)
    if len({m.measurement_id for m in items}) != len(items):
        raise ValueError("measurement identities must be unique")
    first = items[0]
    for m in items:
        if m.status is not Availability.AVAILABLE or (
            m.context,
            m.representation_id,
            m.target_dimension,
        ) != (first.context, first.representation_id, first.target_dimension):
            raise ValueError("measurements must be available and share context and scalar target")
    return items


def _covariance(
    values: object,
    items: tuple[EpistemicMeasurement, ...],
    peer_exposed: bool,
) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(
        tuple(_number(v, "covariance") for v in _array(row, "covariance row"))
        for row in _array(values, "covariance")
    )
    n = len(items)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("covariance shape must match measurement order")
    for i in range(n):
        if matrix[i][i] != items[i].variance:
            raise ValueError("covariance diagonal must equal measurement variance")
        for j in range(i):
            if matrix[i][j] != matrix[j][i]:
                raise ValueError("covariance must be symmetric")
            shared = bool(
                {r.reference_id for r in items[i].evidence_refs}
                & {r.reference_id for r in items[j].evidence_refs}
            )
            group = items[i].correlation_group
            if matrix[i][j] == 0 and (
                peer_exposed
                or shared
                or (group is not None and group == items[j].correlation_group)
            ):
                raise ValueError("zero cross-covariance contradicts reported dependence")
    # Exact LDL decomposition handles singular PSD matrices without accepting negative pivots.
    residual = [[Fraction(v) for v in row] for row in matrix]
    for k in range(n):
        pivot = residual[k][k]
        if pivot < 0 or (pivot == 0 and any(residual[i][k] != 0 for i in range(k + 1, n))):
            raise ValueError("covariance must be positive semidefinite")
        if pivot:
            for i in range(k + 1, n):
                for j in range(i, n):
                    residual[j][i] -= residual[i][k] * residual[j][k] / pivot
                    residual[i][j] = residual[j][i]
    return matrix


def _finite_fraction(value: Fraction, name: str) -> float:
    try:
        return _number(float(value), name)
    except OverflowError as exc:
        raise ValueError(f"{name} must remain finite") from exc
