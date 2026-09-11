"""Deterministic planning for V5 evaluation cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from mindful_trace_gepa._json_values import require_serialization_safe_integer

from .cases.registry import FRAMEWORK_VERSION, load_case_manifest, load_stripe_registry
from .v5_records import (
    CaseIdentity,
    OutcomeRecord,
    RobustnessIdentity,
    SystemIdentity,
    V5EvaluationRecord,
)

_MAX_UNSIGNED_32_BIT_INTEGER = 4_294_967_295


@dataclass(frozen=True, slots=True)
class RepeatMetrics:
    """Outcome metrics computed from one nonempty set of boolean repeats."""

    k: int
    pass_at_k: float
    mean_at_k: float
    pass_power_k: float
    consistency_gap_at_k: float


@dataclass(frozen=True, slots=True)
class V5RepeatGroupKey:
    """Full V5 identity shared by records in one repeat summary."""

    case_id: int
    case_version: str
    case_key: str
    case_title: str
    stripe_id: str
    subtype: str | None
    model_version: str
    harness_version: str


@dataclass(frozen=True, slots=True)
class V5RepeatGroupSummary:
    """One immutable V5 repeat-group key and its outcome-only metrics."""

    key: V5RepeatGroupKey
    metrics: RepeatMetrics


def summarize_repeats(passed: Sequence[bool]) -> RepeatMetrics:
    """Summarize one nonempty sequence of exact boolean repeat outcomes."""

    if not isinstance(passed, Sequence) or isinstance(passed, (str, bytes, bytearray)):
        raise ValueError("passed must be a non-string sequence of built-in bools")
    if not passed:
        raise ValueError("passed must not be empty")
    if any(type(value) is not bool for value in passed):
        raise ValueError("passed must contain only built-in bools")

    k = len(passed)
    successes = sum(passed)
    pass_at_k = float(any(passed))
    mean_at_k = successes / k
    pass_power_k = float(all(passed))
    return RepeatMetrics(
        k=k,
        pass_at_k=pass_at_k,
        mean_at_k=mean_at_k,
        pass_power_k=pass_power_k,
        consistency_gap_at_k=mean_at_k - pass_power_k,
    )


def summarize_v5_record_groups(
    records: Sequence[V5EvaluationRecord],
    *,
    allow_partial: bool = False,
) -> tuple[V5RepeatGroupSummary, ...]:
    """Summarize V5 records by first-seen full identity and contiguous repeat IDs.

    By default every returned group must have the same repeat count. Set ``allow_partial`` to
    summarize complete, contiguous groups with different counts independently.
    """

    validated_records = tuple(
        _revalidate_record(record) for record in _require_record_sequence(records)
    )
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be a built-in bool")

    grouped_records: dict[V5RepeatGroupKey, list[V5EvaluationRecord]] = {}
    for record in validated_records:
        key = _record_group_key(record)
        grouped_records.setdefault(key, []).append(record)

    summaries = tuple(
        V5RepeatGroupSummary(key=key, metrics=_summarize_record_group(group))
        for key, group in grouped_records.items()
    )
    repeat_counts = {summary.metrics.k for summary in summaries}
    if not allow_partial and len(repeat_counts) != 1:
        raise ValueError("V5 record groups must have equal repeat counts unless allow_partial=True")
    return summaries


def _require_record_sequence(records: object) -> Sequence[V5EvaluationRecord]:
    """Return a nonempty sequence containing only exact V5 evaluation records."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("records must be a non-string sequence of V5EvaluationRecord instances")
    if not records:
        raise ValueError("records must not be empty")
    if any(type(record) is not V5EvaluationRecord for record in records):
        raise ValueError("records must contain only exact V5EvaluationRecord instances")
    return records


def _revalidate_record(record: V5EvaluationRecord) -> V5EvaluationRecord:
    """Hydrate a fresh record so grouping never trusts mutable-bypass nested fields."""

    return V5EvaluationRecord.from_dict(record.to_dict())


def _record_group_key(record: V5EvaluationRecord) -> V5RepeatGroupKey:
    """Return one explicit group key after defending the nested identity boundary."""

    if type(record.case) is not CaseIdentity:
        raise ValueError("record case must be an exact CaseIdentity")
    if type(record.robustness) is not RobustnessIdentity:
        raise ValueError("record robustness must be an exact RobustnessIdentity")
    if type(record.system) is not SystemIdentity:
        raise ValueError("record system must be an exact SystemIdentity")
    if type(record.outcome) is not OutcomeRecord:
        raise ValueError("record outcome must be an exact OutcomeRecord")
    return V5RepeatGroupKey(
        case_id=record.case.case_id,
        case_version=record.case.case_version,
        case_key=record.case.case_key,
        case_title=record.case.case_title,
        stripe_id=record.robustness.stripe_id,
        subtype=record.robustness.subtype,
        model_version=record.system.model_version,
        harness_version=record.system.harness_version,
    )


def _summarize_record_group(records: Sequence[V5EvaluationRecord]) -> RepeatMetrics:
    """Reject malformed repeat identities before reducing only verified outcomes."""

    repeat_ids: list[int] = []
    seeds: set[int] = set()
    passed: list[bool] = []
    for record in records:
        repeat_id = record.system.repeat_id
        if type(repeat_id) is not int or repeat_id < 0:
            raise ValueError("repeat_id must be a nonnegative built-in integer")
        if repeat_id in repeat_ids:
            raise ValueError("duplicate repeat_id within V5 record group")
        repeat_ids.append(repeat_id)

        seed = record.system.seed
        if type(seed) is not int:
            raise ValueError("seed must be a built-in integer")
        if seed in seeds:
            raise ValueError("duplicate seed within V5 record group")
        seeds.add(seed)
        passed.append(record.outcome.passed)

    if set(repeat_ids) != set(range(len(repeat_ids))):
        raise ValueError("repeat IDs within a V5 record group must be contiguous from zero")
    return summarize_repeats(passed)


@dataclass(frozen=True, slots=True)
class V5EvaluationCell:
    """One registry-validated, reproducibly seeded V5 evaluation execution."""

    case_id: int
    case_version: str
    stripe_id: str
    subtype: str | None
    repeat_id: int
    seed: int
    model_version: str
    harness_version: str

    def __post_init__(self) -> None:
        """Validate direct construction at the same boundary as planner output."""

        _require_canonical_case_id(self.case_id)
        _require_exact_version(self.case_version, "case_version", FRAMEWORK_VERSION)
        _require_canonical_stripe_id(self.stripe_id)
        if self.subtype is not None:
            raise ValueError("subtype must be None in the V5 base cell planner")
        _require_nonnegative_integer(self.repeat_id, "repeat_id")
        _require_unsigned_32_bit_integer(self.seed, "seed")
        _require_version(self.model_version, "model_version")
        _require_version(self.harness_version, "harness_version")


def plan_v5_cells(
    *,
    case_ids: Sequence[int] | None = None,
    stripe_ids: Sequence[str] | None = None,
    repeats: int = 5,
    base_seed: int = 0,
    model_version: str,
    harness_version: str,
) -> tuple[V5EvaluationCell, ...]:
    """Return a stable Cartesian grid of registry-validated V5 evaluation cells."""

    selected_cases = _select_case_ids(case_ids)
    selected_stripes = _select_stripe_ids(stripe_ids)
    repeat_count = _require_positive_integer(repeats, "repeats")
    normalized_base_seed = require_serialization_safe_integer("base_seed", base_seed)
    _require_version(model_version, "model_version")
    _require_version(harness_version, "harness_version")

    cells: list[V5EvaluationCell] = []
    seeds: set[int] = set()
    for case_id in selected_cases:
        for stripe_id in selected_stripes:
            for repeat_id in range(repeat_count):
                seed = _derive_seed(
                    base_seed=normalized_base_seed,
                    case_id=case_id,
                    stripe_id=stripe_id,
                    repeat_id=repeat_id,
                    model_version=model_version,
                    harness_version=harness_version,
                )
                if seed in seeds:
                    raise ValueError("seed collision while planning V5 evaluation cells")
                seeds.add(seed)
                cells.append(
                    V5EvaluationCell(
                        case_id=case_id,
                        case_version=FRAMEWORK_VERSION,
                        stripe_id=stripe_id,
                        subtype=None,
                        repeat_id=repeat_id,
                        seed=seed,
                        model_version=model_version,
                        harness_version=harness_version,
                    )
                )
    return tuple(cells)


def _select_case_ids(case_ids: Sequence[int] | None) -> tuple[int, ...]:
    """Validate requested cases while retaining explicit caller order."""

    if case_ids is None:
        return tuple(case.id for case in load_case_manifest().cases)
    requested = _require_sequence(case_ids, "case_ids")
    parsed = tuple(_require_canonical_case_id(case_id) for case_id in requested)
    _require_unique(parsed, "case_ids")
    return parsed


def _select_stripe_ids(stripe_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Validate requested stripes while retaining explicit caller order."""

    if stripe_ids is None:
        return tuple(stripe.id for stripe in load_stripe_registry().stripes)
    requested = _require_sequence(stripe_ids, "stripe_ids")
    parsed = tuple(_require_canonical_stripe_id(stripe_id) for stripe_id in requested)
    _require_unique(parsed, "stripe_ids")
    return parsed


def _derive_seed(
    *,
    base_seed: int,
    case_id: int,
    stripe_id: str,
    repeat_id: int,
    model_version: str,
    harness_version: str,
) -> int:
    """Hash one compact JSON tuple into an unsigned 32-bit deterministic seed."""

    encoded = json.dumps(
        [base_seed, case_id, stripe_id, repeat_id, model_version, harness_version],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], byteorder="big")


@cache
def _canonical_case_ids() -> frozenset[int]:
    """Load the PR-1 case registry once for repeated direct-cell validation."""

    return frozenset(case.id for case in load_case_manifest().cases)


@cache
def _canonical_stripe_ids() -> frozenset[str]:
    """Load the PR-1 stripe registry once for repeated direct-cell validation."""

    return frozenset(stripe.id for stripe in load_stripe_registry().stripes)


def _require_canonical_case_id(value: object) -> int:
    """Return a built-in canonical case identifier."""

    case_id = require_serialization_safe_integer("case_id", value)
    if case_id not in _canonical_case_ids():
        raise ValueError(f"case_id must identify a canonical V5 case; received {case_id!r}")
    return case_id


def _require_canonical_stripe_id(value: object) -> str:
    """Return a canonical stripe identifier."""

    if type(value) is not str or value not in _canonical_stripe_ids():
        raise ValueError(f"stripe_id must identify a canonical V5 stripe; received {value!r}")
    return value


def _require_exact_version(value: object, field_name: str, expected: str) -> str:
    """Return the required literal case-framework version."""

    if type(value) is not str or value != expected:
        raise ValueError(f"{field_name} must be {expected!r}; received {value!r}")
    return value


def _require_version(value: object, field_name: str) -> str:
    """Return a nonblank, boundary-trimmed system version string."""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank string without boundary whitespace")
    return value


def _require_nonnegative_integer(value: object, field_name: str) -> int:
    """Return one serialization-safe nonnegative integer."""

    integer = require_serialization_safe_integer(field_name, value)
    if integer < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return integer


def _require_positive_integer(value: object, field_name: str) -> int:
    """Return one serialization-safe positive integer."""

    integer = require_serialization_safe_integer(field_name, value)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _require_unsigned_32_bit_integer(value: object, field_name: str) -> int:
    """Return one seed-range integer without allowing wraparound."""

    integer = require_serialization_safe_integer(field_name, value)
    if not 0 <= integer <= _MAX_UNSIGNED_32_BIT_INTEGER:
        raise ValueError(f"{field_name} must be an unsigned 32-bit integer")
    return integer


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    """Reject scalar strings and unordered containers at planner selection boundaries."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a non-string sequence")
    return value


def _require_unique(values: tuple[object, ...], field_name: str) -> None:
    """Reject duplicate explicit selection requests before cell enumeration."""

    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate values")
