"""Contract tests for deterministic V5 evaluation-cell planning."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluation import V5EvaluationCell, plan_v5_cells


class _IntegerSubclass(int):
    """Deliberately non-built-in integer for public-boundary tests."""


def test_selected_grid_preserves_order_and_has_hand_written_cell_identities() -> None:
    """A planner reordering either selected dimension or repeats breaks this contract."""

    cells = plan_v5_cells(
        case_ids=(14, 2),
        stripe_ids=("TOOL_ERROR", "NONE"),
        repeats=3,
        base_seed=23,
        model_version="mindful-model-2026-09-10",
        harness_version="v5-harness-1.0.0",
    )

    assert [
        (
            cell.case_id,
            cell.case_version,
            cell.stripe_id,
            cell.subtype,
            cell.repeat_id,
            cell.seed,
            cell.model_version,
            cell.harness_version,
        )
        for cell in cells
    ] == [
        (
            14,
            "17case-v5",
            "TOOL_ERROR",
            None,
            0,
            1985107622,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            14,
            "17case-v5",
            "TOOL_ERROR",
            None,
            1,
            3366342688,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            14,
            "17case-v5",
            "TOOL_ERROR",
            None,
            2,
            1317237387,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            14,
            "17case-v5",
            "NONE",
            None,
            0,
            2640671407,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            14,
            "17case-v5",
            "NONE",
            None,
            1,
            4164973019,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            14,
            "17case-v5",
            "NONE",
            None,
            2,
            2630719089,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "TOOL_ERROR",
            None,
            0,
            3927924709,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "TOOL_ERROR",
            None,
            1,
            4254388189,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "TOOL_ERROR",
            None,
            2,
            4188725124,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "NONE",
            None,
            0,
            4053472628,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "NONE",
            None,
            1,
            1770673556,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
        (
            2,
            "17case-v5",
            "NONE",
            None,
            2,
            3786074924,
            "mindful-model-2026-09-10",
            "v5-harness-1.0.0",
        ),
    ]


def test_default_plan_has_all_canonical_cells_in_manifest_order() -> None:
    """Dropping a registry member or changing dimension order breaks the default battery."""

    cells = plan_v5_cells(
        model_version="mindful-model-2026-09-10",
        harness_version="v5-harness-1.0.0",
    )

    assert len(cells) == 935
    assert [cell.repeat_id for cell in cells[:5]] == [0, 1, 2, 3, 4]
    assert (cells[0].case_id, cells[0].stripe_id, cells[0].repeat_id) == (1, "NONE", 0)
    assert (cells[-1].case_id, cells[-1].stripe_id, cells[-1].repeat_id) == (
        17,
        "TIME_BUDGET_PRESSURE",
        4,
    )


def test_seeds_are_deterministic_unsigned_and_unique() -> None:
    """A seed derivation that wraps or omits cell identity breaks repeat independence."""

    kwargs: dict[str, Any] = {
        "case_ids": (1, 2),
        "stripe_ids": ("NONE", "TOOL_ERROR"),
        "repeats": 3,
        "base_seed": -23,
        "model_version": "mindful-model-2026-09-10",
        "harness_version": "v5-harness-1.0.0",
    }

    first = plan_v5_cells(**kwargs)
    second = plan_v5_cells(**kwargs)

    assert first == second
    assert len({cell.seed for cell in first}) == 12
    assert all(0 <= cell.seed <= 4_294_967_295 for cell in first)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_ids": (1, 1)},
        {"case_ids": (0,)},
        {"case_ids": (18,)},
        {"case_ids": (True,)},
        {"case_ids": (_IntegerSubclass(1),)},
        {"case_ids": "1"},
        {"case_ids": {1}},
        {"stripe_ids": ("NONE", "NONE")},
        {"stripe_ids": ("UNKNOWN",)},
        {"stripe_ids": "NONE"},
        {"repeats": 0},
        {"repeats": -1},
        {"repeats": True},
        {"repeats": _IntegerSubclass(1)},
        {"base_seed": True},
        {"base_seed": _IntegerSubclass(0)},
        {"base_seed": 10**400},
        {"model_version": ""},
        {"model_version": " \t "},
        {"harness_version": ""},
        {"harness_version": " \t "},
    ],
)
def test_planner_rejects_invalid_requests_at_the_public_boundary(kwargs: dict[str, object]) -> None:
    """Unchecked request types or values could make planned work non-reproducible."""

    values: dict[str, object] = {
        "model_version": "mindful-model-2026-09-10",
        "harness_version": "v5-harness-1.0.0",
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        plan_v5_cells(**cast(Any, values))


@pytest.mark.parametrize(
    "changes",
    [
        {"case_id": 0},
        {"case_id": True},
        {"case_version": "wrong-version"},
        {"stripe_id": "UNKNOWN"},
        {"subtype": "unexpected"},
        {"repeat_id": -1},
        {"repeat_id": True},
        {"seed": -1},
        {"seed": True},
        {"seed": 4_294_967_296},
        {"model_version": ""},
        {"harness_version": " \t "},
    ],
)
def test_direct_cell_construction_validates_every_identity_boundary(
    changes: dict[str, object],
) -> None:
    """Direct construction must not bypass the planner's registry and scalar checks."""

    values: dict[str, object] = {
        "case_id": 14,
        "case_version": "17case-v5",
        "stripe_id": "TOOL_ERROR",
        "subtype": None,
        "repeat_id": 2,
        "seed": 4_242,
        "model_version": "mindful-model-2026-09-10",
        "harness_version": "v5-harness-1.0.0",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        V5EvaluationCell(**cast(Any, values))
