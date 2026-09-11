"""Deterministic planning for V5 evaluation cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from mindful_trace_gepa._json_values import require_serialization_safe_integer

from .cases.registry import FRAMEWORK_VERSION, load_case_manifest, load_stripe_registry

_MAX_UNSIGNED_32_BIT_INTEGER = 4_294_967_295


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
