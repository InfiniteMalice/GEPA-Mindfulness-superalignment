"""Deterministic planning for V5 evaluation cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import NoReturn

from mindful_trace_gepa._json_values import require_serialization_safe_integer
from mindful_trace_gepa.logging_schema import EventEnvelope

from .cases.registry import FRAMEWORK_VERSION, load_case_manifest, load_stripe_registry
from .v5_provenance import validate_v5_record_provenance
from .v5_records import V5EvaluationRecord

_MAX_UNSIGNED_32_BIT_INTEGER = 4_294_967_295
MAX_V5_PLANNED_CELLS = 10_000


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
    """Planned and observed coverage plus outcome-only metrics for one group."""

    key: V5RepeatGroupKey
    expected_count: int
    observed_count: int
    metrics: RepeatMetrics | None


def summarize_repeats(passed: Sequence[bool]) -> RepeatMetrics:
    """Summarize one nonempty sequence of exact boolean repeat outcomes."""

    if not isinstance(passed, Sequence) or isinstance(passed, (str, bytes, bytearray)):
        raise ValueError("passed must be a non-string sequence of built-in bools")
    values = tuple(passed)
    if not values:
        raise ValueError("passed must not be empty")
    if any(type(value) is not bool for value in values):
        raise ValueError("passed must contain only built-in bools")

    k = len(values)
    successes = sum(values)
    pass_at_k = float(successes > 0)
    mean_at_k = successes / k
    pass_power_k = float(successes == k)
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
    planned_cells: Sequence[V5EvaluationCell],
    event_sequences: Mapping[V5EvaluationCell, Sequence[EventEnvelope]],
    allow_partial: bool = False,
) -> tuple[V5RepeatGroupSummary, ...]:
    """Summarize records only within an explicit plan and verified per-cell event boundary."""

    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be a built-in bool")
    validated_records = _snapshot_records(records, allow_empty=allow_partial)
    plan = _snapshot_planned_cells(planned_cells)
    sequences = _snapshot_event_mapping(event_sequences)

    plan_groups, plan_by_coordinate = _index_plan(plan)
    record_by_cell = _index_records(validated_records, plan, plan_by_coordinate)
    observed_cells = set(record_by_cell)
    sequence_cells = set(sequences)
    missing_sequences = observed_cells - sequence_cells
    extra_sequences = sequence_cells - observed_cells
    if missing_sequences:
        raise ValueError(f"missing event_sequences for {len(missing_sequences)} observed cells")
    if extra_sequences:
        raise ValueError(f"extra event_sequences for {len(extra_sequences)} unobserved cells")

    if not allow_partial and observed_cells != set(plan):
        raise ValueError(
            "strict plan coverage requires every planned cell exactly once; "
            f"expected {len(plan)}, observed {len(observed_cells)}"
        )
    if allow_partial:
        _require_prefix_coverage(plan_groups, observed_cells)

    verified_records = {
        cell: validate_v5_record_provenance(record, sequences[cell]).record_snapshot()
        for cell, record in record_by_cell.items()
    }
    summaries: list[V5RepeatGroupSummary] = []
    for key, group_cells in plan_groups.items():
        observed = tuple(cell for cell in group_cells if cell in verified_records)
        outcomes = tuple(verified_records[cell].outcome.passed for cell in observed)
        summaries.append(
            V5RepeatGroupSummary(
                key=key,
                expected_count=len(group_cells),
                observed_count=len(observed),
                metrics=summarize_repeats(outcomes) if outcomes else None,
            )
        )
    return tuple(summaries)


def _snapshot_records(
    records: object,
    *,
    allow_empty: bool,
) -> tuple[V5EvaluationRecord, ...]:
    """Iterate records once, require exact types, and revalidate detached snapshots."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise ValueError("records must be a non-string sequence of V5EvaluationRecord instances")
    supplied = tuple(records)
    if not supplied and not allow_empty:
        raise ValueError("records must not be empty")
    if any(type(record) is not V5EvaluationRecord for record in supplied):
        raise ValueError("records must contain only exact V5EvaluationRecord instances")
    return tuple(_revalidate_record(record) for record in supplied)


def _revalidate_record(record: V5EvaluationRecord) -> V5EvaluationRecord:
    """Hydrate a fresh record so grouping never trusts mutable-bypass nested fields."""

    return V5EvaluationRecord.from_dict(record.to_dict())


def _snapshot_planned_cells(cells: object) -> tuple[V5EvaluationCell, ...]:
    """Iterate a plan once and detach every exact cell from caller-owned objects."""

    if isinstance(cells, (str, bytes, bytearray)) or not isinstance(cells, Sequence):
        raise ValueError("planned_cells must be a non-string sequence of V5EvaluationCell values")
    supplied = tuple(cells)
    if not supplied:
        raise ValueError("planned_cells must not be empty")
    snapshots: list[V5EvaluationCell] = []
    for cell in supplied:
        if type(cell) is not V5EvaluationCell:
            raise ValueError("planned_cells must contain only exact V5EvaluationCell values")
        snapshots.append(
            V5EvaluationCell(
                case_id=cell.case_id,
                case_version=cell.case_version,
                stripe_id=cell.stripe_id,
                subtype=cell.subtype,
                repeat_id=cell.repeat_id,
                seed=cell.seed,
                model_version=cell.model_version,
                harness_version=cell.harness_version,
            )
        )
    return tuple(snapshots)


def _snapshot_event_mapping(
    event_sequences: object,
) -> dict[V5EvaluationCell, tuple[EventEnvelope, ...]]:
    """Snapshot every mapping entry and each per-cell sequence exactly once."""

    if not isinstance(event_sequences, Mapping):
        raise ValueError("event_sequences must map V5EvaluationCell values to event sequences")
    entries = tuple(event_sequences.items())
    snapshots: dict[V5EvaluationCell, tuple[EventEnvelope, ...]] = {}
    for cell, events in entries:
        if type(cell) is not V5EvaluationCell:
            raise ValueError("event_sequences keys must be exact V5EvaluationCell values")
        cell_snapshot = _snapshot_planned_cells((cell,))[0]
        if cell_snapshot in snapshots:
            raise ValueError("event_sequences must not contain duplicate cell keys")
        if isinstance(events, (str, bytes, bytearray)) or not isinstance(events, Sequence):
            raise ValueError("each event_sequences value must be a non-string event sequence")
        snapshots[cell_snapshot] = tuple(events)
    return snapshots


def _index_plan(
    plan: tuple[V5EvaluationCell, ...],
) -> tuple[
    dict[V5RepeatGroupKey, list[V5EvaluationCell]],
    dict[tuple[object, ...], V5EvaluationCell],
]:
    """Validate global plan uniqueness and retain deterministic group order."""

    groups: dict[V5RepeatGroupKey, list[V5EvaluationCell]] = {}
    by_coordinate: dict[tuple[object, ...], V5EvaluationCell] = {}
    seen_cells: set[V5EvaluationCell] = set()
    seen_seeds: set[int] = set()
    repeat_ids_by_group: dict[V5RepeatGroupKey, set[int]] = {}
    for cell in plan:
        if cell in seen_cells:
            raise ValueError("duplicate planned cell")
        seen_cells.add(cell)
        key = _cell_group_key(cell)
        seen_repeat_ids = repeat_ids_by_group.setdefault(key, set())
        if cell.repeat_id in seen_repeat_ids:
            raise ValueError("duplicate repeat_id within a planned V5 record group")
        seen_repeat_ids.add(cell.repeat_id)
        coordinate = _cell_coordinate(cell)
        by_coordinate[coordinate] = cell
        if cell.seed in seen_seeds:
            raise ValueError("duplicate seed in planned_cells")
        seen_seeds.add(cell.seed)
        groups.setdefault(key, []).append(cell)
    for group_cells in groups.values():
        repeat_ids = tuple(cell.repeat_id for cell in group_cells)
        if repeat_ids != tuple(range(len(group_cells))):
            raise ValueError("repeat IDs within a planned V5 group must be contiguous from zero")
    return groups, by_coordinate


def _index_records(
    records: tuple[V5EvaluationRecord, ...],
    plan: tuple[V5EvaluationCell, ...],
    plan_by_coordinate: Mapping[tuple[object, ...], V5EvaluationCell],
) -> dict[V5EvaluationCell, V5EvaluationRecord]:
    """Bind each unique record to its exact planned cell, including seed and versions."""

    plan_cells = set(plan)
    record_by_cell: dict[V5EvaluationCell, V5EvaluationRecord] = {}
    for record in records:
        cell = _record_cell(record)
        planned = plan_by_coordinate.get(_cell_coordinate(cell))
        if planned is None:
            _raise_record_plan_identity_error(cell, plan)
        if planned.seed != cell.seed:
            raise ValueError(
                f"record seed {cell.seed!r} does not match planned seed {planned.seed!r}"
            )
        if cell not in plan_cells:  # pragma: no cover - coordinate and seed establish equality
            raise ValueError("record cell is not present in planned_cells")
        if cell in record_by_cell:
            raise ValueError("duplicate record cell")
        record_by_cell[cell] = record
    return record_by_cell


def _raise_record_plan_identity_error(
    cell: V5EvaluationCell,
    plan: tuple[V5EvaluationCell, ...],
) -> NoReturn:
    """Report version drift separately before classifying other identities as extras."""

    same_base = tuple(item for item in plan if _cell_base(item) == _cell_base(cell))
    if same_base:
        planned = same_base[0]
        if cell.model_version != planned.model_version:
            raise ValueError("record model_version does not match planned_cells")
        if cell.harness_version != planned.harness_version:
            raise ValueError("record harness_version does not match planned_cells")
    raise ValueError("record cell is not present in planned_cells")


def _require_prefix_coverage(
    plan_groups: Mapping[V5RepeatGroupKey, list[V5EvaluationCell]],
    observed: set[V5EvaluationCell],
) -> None:
    """Require every partial group to contain only its first planned repeat cells."""

    for group_cells in plan_groups.values():
        observed_group = {cell for cell in group_cells if cell in observed}
        expected_prefix = set(group_cells[: len(observed_group)])
        if observed_group != expected_prefix:
            raise ValueError("partial coverage must be a prefix of each planned repeat group")


def _record_cell(record: V5EvaluationRecord) -> V5EvaluationCell:
    """Return the exact planned-cell identity carried by a validated record."""

    return V5EvaluationCell(
        case_id=record.case.case_id,
        case_version=record.case.case_version,
        stripe_id=record.robustness.stripe_id,
        subtype=record.robustness.subtype,
        repeat_id=record.system.repeat_id,
        seed=record.system.seed,
        model_version=record.system.model_version,
        harness_version=record.system.harness_version,
    )


def _cell_group_key(cell: V5EvaluationCell) -> V5RepeatGroupKey:
    """Resolve the canonical case labels for one planned repeat-group identity."""

    case_key, case_title = _canonical_case_map()[cell.case_id]
    return V5RepeatGroupKey(
        case_id=cell.case_id,
        case_version=cell.case_version,
        case_key=case_key,
        case_title=case_title,
        stripe_id=cell.stripe_id,
        subtype=cell.subtype,
        model_version=cell.model_version,
        harness_version=cell.harness_version,
    )


def _cell_coordinate(cell: V5EvaluationCell) -> tuple[object, ...]:
    """Return planned identity excluding the seed value that the plan assigns."""

    return (
        cell.case_id,
        cell.case_version,
        cell.stripe_id,
        cell.subtype,
        cell.repeat_id,
        cell.model_version,
        cell.harness_version,
    )


def _cell_base(cell: V5EvaluationCell) -> tuple[object, ...]:
    """Return identity used only to distinguish version drift from an extra record."""

    return (
        cell.case_id,
        cell.case_version,
        cell.stripe_id,
        cell.subtype,
        cell.repeat_id,
    )


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
        _require_registered_subtype(self.stripe_id, self.subtype)
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
    cell_count = len(selected_cases) * len(selected_stripes) * repeat_count
    if cell_count > MAX_V5_PLANNED_CELLS:
        raise ValueError(
            f"V5 plans are limited to {MAX_V5_PLANNED_CELLS:,} cells; " f"received {cell_count:,}"
        )

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
        return tuple(_canonical_case_map())
    requested = _require_sequence(case_ids, "case_ids")
    parsed = tuple(_require_canonical_case_id(case_id) for case_id in requested)
    _require_unique(parsed, "case_ids")
    return parsed


def _select_stripe_ids(stripe_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Validate requested stripes while retaining explicit caller order."""

    if stripe_ids is None:
        return tuple(_canonical_stripe_map())
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
def _canonical_case_map() -> Mapping[int, tuple[str, str]]:
    """Load one immutable ordered PR-1 case identity map."""

    return MappingProxyType(
        {case.id: (case.key, case.title) for case in load_case_manifest().cases}
    )


@cache
def _canonical_stripe_map() -> Mapping[str, tuple[str, ...]]:
    """Load one immutable ordered PR-1 stripe/subtype identity map."""

    identities = {
        stripe.id: tuple(stripe.allowed_subtypes) for stripe in load_stripe_registry().stripes
    }
    return MappingProxyType(identities)


def _require_canonical_case_id(value: object) -> int:
    """Return a built-in canonical case identifier."""

    case_id = require_serialization_safe_integer("case_id", value)
    if case_id not in _canonical_case_map():
        raise ValueError(f"case_id must identify a canonical V5 case; received {case_id!r}")
    return case_id


def _require_canonical_stripe_id(value: object) -> str:
    """Return a canonical stripe identifier."""

    if type(value) is not str or value not in _canonical_stripe_map():
        raise ValueError(f"stripe_id must identify a canonical V5 stripe; received {value!r}")
    return value


def _require_registered_subtype(stripe_id: str, value: object) -> str | None:
    """Return None or one exact subtype allowed for the selected stripe."""

    if value is None:
        return None
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError("subtype must be an exact nonblank built-in string or None")
    if value not in _canonical_stripe_map()[stripe_id]:
        raise ValueError(f"subtype {value!r} is not registered for stripe {stripe_id!r}")
    return value


def _require_exact_version(value: object, field_name: str, expected: str) -> str:
    """Return the required literal case-framework version."""

    if type(value) is not str or value != expected:
        raise ValueError(f"{field_name} must be {expected!r}; received {value!r}")
    return value


def _require_version(value: object, field_name: str) -> str:
    """Return a nonblank, boundary-trimmed system version string."""

    if type(value) is not str or not value.strip() or value != value.strip():
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
