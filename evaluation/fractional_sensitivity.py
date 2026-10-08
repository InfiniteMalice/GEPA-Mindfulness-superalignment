"""Certified pointwise fractional block sensitivity for complete 1..4-bit Boolean tables.

Packing and covering follow Li et al. (2026), arXiv:2610.02557, Definitions 3.3–3.4.
Equal feasible primal/dual objectives prove optimality; no language-model guarantee follows.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from itertools import combinations
from typing import Any

from .causal_records import content_digest


def _domain(table: tuple[bool, ...], point: int) -> int:
    if type(table) is not tuple or len(table) not in (2, 4, 8, 16):
        raise ValueError("table requires a complete 1..4-input Boolean tuple")
    if any(type(v) is not bool for v in table):
        raise ValueError("truth table entries must be exact booleans")
    if type(point) is not int or not 0 <= point < len(table):
        raise ValueError("point must be an integer input bitmask in the table")
    return len(table).bit_length() - 1


def _fraction(value: Any) -> Fraction:
    if type(value) is not list or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("fraction requires an integer numerator/denominator pair")
    if value[1] <= 0:
        raise ValueError("fraction denominator must be positive")
    result = Fraction(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError("fraction must be reduced")
    return result


@dataclass(frozen=True)
class FractionalSensitivity:
    """Exact pointwise optimum and independently checkable packing/covering witnesses."""

    input_count: int
    point: int
    table_digest: str
    blocks: tuple[int, ...]
    primal: tuple[Fraction, ...]
    dual: tuple[Fraction, ...]
    value: Fraction

    def __post_init__(self) -> None:
        if type(self.input_count) is not int or not 1 <= self.input_count <= 4:
            raise ValueError("input_count must be 1..4")
        _domain((False,) * (1 << self.input_count), self.point)
        if (
            type(self.table_digest) is not str
            or len(self.table_digest) != 64
            or set(self.table_digest) - set("0123456789abcdef")
        ):
            raise ValueError("table_digest must be lowercase SHA-256")
        if type(self.blocks) is not tuple or any(
            type(b) is not int or not 0 < b < 1 << self.input_count for b in self.blocks
        ):
            raise ValueError("blocks must be nonempty coordinate bitmasks")
        if tuple(sorted(set(self.blocks))) != self.blocks:
            raise ValueError("blocks must be sorted and unique")
        if type(self.primal) is not tuple or type(self.dual) is not tuple:
            raise ValueError("certificate weights must be tuples")
        if len(self.primal) != len(self.blocks) or len(self.dual) != self.input_count:
            raise ValueError("certificate dimensions do not match")
        if any(type(x) is not Fraction for x in (*self.primal, *self.dual, self.value)):
            raise ValueError("certificate values must be exact Fractions")

    def to_dict(self) -> dict[str, Any]:
        """Encode exact rational pairs; this certificate makes no admission/authority claim."""
        self.__post_init__()

        def pair(x: Fraction) -> list[int]:
            return [x.numerator, x.denominator]

        return dict(
            schema_version="fractional-sensitivity-v1",
            input_count=self.input_count,
            point=self.point,
            table_digest=self.table_digest,
            blocks=list(self.blocks),
            primal=[pair(x) for x in self.primal],
            dual=[pair(x) for x in self.dual],
            value=pair(self.value),
        )

    @classmethod
    def from_dict(cls, value: object) -> FractionalSensitivity:
        """Reject coerced, noncanonical or authority-shaped certificate data."""
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)} | {
            "schema_version"
        }:
            raise ValueError("invalid certificate fields")
        data: dict[str, Any] = dict(value)
        if data.pop("schema_version") != "fractional-sensitivity-v1":
            raise ValueError("invalid certificate version")
        for name in ("blocks", "primal", "dual"):
            if type(data[name]) is not list:
                raise ValueError("certificate arrays must be lists")
        data["blocks"] = tuple(data["blocks"])
        for name in ("primal", "dual"):
            data[name] = tuple(_fraction(x) for x in data[name])
        data["value"] = _fraction(data["value"])
        return cls(**data)


def _solve(matrix: list[list[int]], rhs: list[int]) -> tuple[Fraction, ...] | None:
    """Solve a square system exactly; singular bases are not LP vertices."""
    n = len(rhs)
    rows = [[Fraction(x) for x in row] + [Fraction(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if rows[i][col]), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for i in range(n):
            if i != col:
                scale = rows[i][col]
                rows[i] = [v - scale * p for v, p in zip(rows[i], rows[col])]
    return tuple(row[-1] for row in rows)


def fractional_block_sensitivity(table: tuple[bool, ...], point: int) -> FractionalSensitivity:
    """Enumerate at most 3876 bases and return an exact primal/dual feasible optimum."""
    n = _domain(table, point)
    blocks = tuple(b for b in range(1, len(table)) if table[point] != table[point ^ b])
    m = len(blocks)
    columns = [tuple(int(bool(b & (1 << i))) for i in range(n)) for b in blocks]
    columns += [tuple(int(i == j) for i in range(n)) for j in range(n)]
    for basis in combinations(range(m + n), n):
        matrix = [[columns[j][i] for j in basis] for i in range(n)]
        weights = _solve(matrix, [1] * n)
        if weights is None or any(w < 0 for w in weights):
            continue
        dual = _solve([list(col) for col in zip(*matrix)], [int(j < m) for j in basis])
        if dual is None or any(y < 0 for y in dual):
            continue
        if any(sum(dual[i] * col[i] for i in range(n)) < 1 for col in columns[:m]):
            continue
        primal = [Fraction(0)] * m
        for j, w in zip(basis, weights):
            if j < m:
                primal[j] = w
        return FractionalSensitivity(
            n, point, content_digest(table), blocks, tuple(primal), dual, sum(primal, Fraction(0))
        )
    raise ArithmeticError("no feasible primal/dual certificate found")


def verify_fbs_certificate(table: tuple[bool, ...], certificate: FractionalSensitivity) -> bool:
    """Check constraints and equal objectives without running the optimizer."""
    if type(certificate) is not FractionalSensitivity:
        raise ValueError("expected FractionalSensitivity")
    certificate.__post_init__()
    n = _domain(table, certificate.point)
    c = certificate
    blocks = tuple(b for b in range(1, len(table)) if table[c.point] != table[c.point ^ b])
    if n != c.input_count or content_digest(table) != c.table_digest or blocks != c.blocks:
        return False
    if any(x < 0 for x in (*c.primal, *c.dual)):
        return False
    if any(sum(w for b, w in zip(blocks, c.primal) if b & (1 << i)) > 1 for i in range(n)):
        return False
    if any(sum(c.dual[i] for i in range(n) if b & (1 << i)) < 1 for b in blocks):
        return False
    return sum(c.primal) == sum(c.dual) == c.value
