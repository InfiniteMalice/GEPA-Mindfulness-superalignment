"""Descriptive estimators for evaluator-annotated MDAC opportunity rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite, sqrt
from typing import Any

from .diagnostic_records import strings
from .epistemic_state import _number, _text

Row = Mapping[str, Any]
Verify = Callable[[Row], bool]


def label(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or "/" in value:
        raise ValueError(f"{name} requires a nonblank label without '/'")
    return value


def flag(row: Row, name: str) -> bool | None:
    value = row.get(name)
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} requires a boolean or null")
    return value


def number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def labels(values: object, name: str) -> tuple[str, ...]:
    return strings(values, name)


def invert(value: bool | None) -> bool | None:
    return None if value is None else not value


@dataclass(frozen=True)
class Annotation:
    opportunity_id: str
    evidence_refs: tuple[str, ...]
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class MetricResult:
    mechanism_id: str
    name: str
    numerator: float | None
    denominator: int
    missing: int
    raw: float | None
    normalized: float | None
    normalization: str
    interval: tuple[float, float] | None
    uncertainty: str
    unavailable_reason: str | None
    annotations: tuple[Annotation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = "mdac-metric-v1"
        result["annotations"] = [
            dict(
                opportunity_id=a.opportunity_id,
                evidence_refs=list(a.evidence_refs),
                alternatives=list(a.alternatives),
            )
            for a in self.annotations
        ]
        return result


def _wilson(rate: float, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    center = rate + z * z / (2 * n)
    radius = z * sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    scale = 1 + z * z / n
    return (
        0.0 if rate == 0 else max(0.0, (center - radius) / scale),
        1.0 if rate == 1 else min(1.0, (center + radius) / scale),
    )


def _normalize(raw: float, rule: str) -> float | None:
    if rule == "descriptive":
        return None
    if rule == "higher":
        quality = raw
    elif rule == "lower":
        quality = 1 - raw
    elif rule == "signed_gain":
        quality = (raw + 1) / 2
    elif rule == "standardized_error":
        quality = 1 / (1 + raw)
    elif rule == "reward_divergence":
        quality = 1 - max(0.0, raw) / 2
    else:
        raise ValueError(f"unknown normalization rule {rule}")
    if not -1e-12 <= quality <= 1 + 1e-12:
        raise ValueError("result outside declared normalization range")
    return max(0.0, min(1.0, quality))


@dataclass
class _Samples:
    rule: str
    binary: bool
    values: list[float] = field(default_factory=list)
    missing: int = 0
    annotations: list[Annotation] = field(default_factory=list)


class Measurements:
    def __init__(self, mechanism: str, independent: bool):
        self.mechanism = mechanism
        self.independent = independent
        self.samples: dict[str, _Samples] = {}
        self.derived: dict[str, MetricResult] = {}

    def add(
        self,
        name: str,
        value: float | None,
        *,
        rule: str = "higher",
        binary: bool = True,
        eligible: bool | None = True,
        annotation: Annotation | None = None,
    ) -> None:
        item = self.samples.setdefault(name, _Samples(rule, binary))
        if (item.rule, item.binary) != (rule, binary):
            raise ValueError("inconsistent metric definition")
        if eligible is False:
            return
        if eligible is None or value is None:
            item.missing += 1
        else:
            if not isfinite(value) or (binary and value not in (0, 1)):
                raise ValueError(f"invalid measurement for {name}")
            item.values.append(float(value))
        if annotation is not None:
            item.annotations.append(annotation)

    def aggregate(self, name: str, raw: float | None, count: int, missing: int, rule: str) -> None:
        # These nonlinear estimators expose the effective sum and sample count. Their
        # underlying additive components are also emitted as named results.
        self.derived[name] = MetricResult(
            self.mechanism,
            name,
            None if raw is None else raw * count,
            count,
            missing,
            raw,
            None if raw is None else _normalize(raw, rule),
            rule,
            None,
            "nonlinear estimator; no interval specified",
            "insufficient component observations" if raw is None else None,
        )

    def finish(self) -> dict[str, MetricResult]:
        result = dict(self.derived)
        for name, sample in self.samples.items():
            n = len(sample.values)
            numerator = fsum(sample.values) if n else None
            raw = numerator / n if numerator is not None else None
            interval = None
            uncertainty = "independent Bernoulli trials not asserted"
            if not sample.binary:
                uncertainty = "no sampling model specified for continuous observations"
            elif self.independent and raw is not None:
                interval = _wilson(raw, n)
                uncertainty = "95% Wilson interval conditional on measured independent trials"
            if raw is None:
                uncertainty = "no measured opportunities"
            result[name] = MetricResult(
                self.mechanism,
                name,
                numerator,
                n,
                sample.missing,
                raw,
                None if raw is None else _normalize(raw, sample.rule),
                sample.rule,
                interval,
                uncertainty,
                None if n else "no measured opportunities",
                tuple(sample.annotations),
            )
        return result


def unique_rows(rows: Sequence[Row]) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence of opportunity mappings")
    if any(not isinstance(r, Mapping) or "id" not in r for r in rows):
        raise ValueError("each opportunity row requires an id")
    ids = [_text(r["id"], "id") for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate opportunity ID")
