"""Exact values, malformed domains and independently checked LP certificates."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest

from evaluation.fractional_sensitivity import (
    FractionalSensitivity,
    fractional_block_sensitivity,
    verify_fbs_certificate,
)


@pytest.mark.parametrize(
    "table,point,want",
    [
        ((False,) * 16, 5, 0),
        ((False, True), 0, 1),
        (tuple(bool(bin(i).count("1") % 2) for i in range(16)), 3, 4),
        ((False,) * 15 + (True,), 0, 1),
        ((False,) * 15 + (True,), 15, 4),
        ((False, False, False, True, False, True, True, True), 0, Fraction(3, 2)),
    ],
)
def test_known_exact_values(table, point, want):
    result = fractional_block_sensitivity(table, point)
    assert type(result.value) is Fraction
    assert result.value == want
    assert verify_fbs_certificate(table, result)
    assert FractionalSensitivity.from_dict(result.to_dict()) == result


def test_all_two_bit_tables_have_valid_certificates():
    for table in product((False, True), repeat=4):
        for point in range(4):
            result = fractional_block_sensitivity(table, point)
            assert verify_fbs_certificate(table, result)
            assert result == fractional_block_sensitivity(table, point)


@pytest.mark.parametrize(
    "table,point",
    [
        ((False,), 0),
        ((False,) * 3, 0),
        ((False,) * 32, 0),
        ((0, 1), 0),
        ((False, True), True),
        ((False, True), -1),
        ((False, True), 2),
        ((False, True), 0.0),
        ([False, True], 0),
    ],
)
def test_malformed_domains_rejected(table, point):
    with pytest.raises(ValueError):
        fractional_block_sensitivity(table, point)


def test_tampered_certificates_rejected_without_solving(monkeypatch):
    table = (False, False, False, True, False, True, True, True)
    result = fractional_block_sensitivity(table, 0)
    monkeypatch.setattr(
        "evaluation.fractional_sensitivity.fractional_block_sensitivity",
        lambda *args: pytest.fail("verifier called solver"),
    )
    assert verify_fbs_certificate(table, result)
    for forged in (
        replace(result, primal=tuple(Fraction(2) for _ in result.primal)),
        replace(result, dual=(Fraction(0),) * 3),
        replace(result, value=Fraction(7)),
        replace(result, table_digest="0" * 64),
    ):
        assert not verify_fbs_certificate(table, forged)
    assert not verify_fbs_certificate(tuple(not x for x in table), result)


@pytest.mark.parametrize("bad", [[3, 0], [True, 2], [6, 4], [3.0, 2], [-3, -2]])
def test_fraction_serialization_is_exact(bad):
    raw = fractional_block_sensitivity((False, True), 0).to_dict()
    raw["value"] = bad
    with pytest.raises(ValueError):
        FractionalSensitivity.from_dict(raw)


def test_degenerate_four_bit_function_and_extra_fields():
    table = tuple(bool(i & 1) for i in range(16))
    result = fractional_block_sensitivity(table, 0)
    assert result.value == 1
    assert verify_fbs_certificate(table, result)
    raw = result.to_dict()
    raw["reward"] = 1
    with pytest.raises(ValueError):
        FractionalSensitivity.from_dict(raw)
