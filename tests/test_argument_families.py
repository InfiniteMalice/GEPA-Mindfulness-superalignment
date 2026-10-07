"""Controlled family lineage and decision boundaries survive derived-pair export."""

import json
from copy import deepcopy
from pathlib import Path

import pytest


def source_row() -> dict:
    return json.loads(
        Path("data/synthetic/gold/superalignment_gold_v1.jsonl").read_text().splitlines()[0]
    )


def test_sweep_changes_one_parameter_and_preserves_source() -> None:
    from synthetic_data.argument_families import generate_family, validate_family

    source = source_row()
    before = deepcopy(source)
    rows = generate_family(
        source,
        family_id="family",
        semantic_core_id="core",
        parameter_path=("scenario", "urgency"),
        values=("low", "high"),
        coordinates=(0.0, 1.0),
        case_targets=(16, 14),
        response_modes=("CLARIFY", "CLARIFY"),
        confidence=(0.8, 0.4),
        material_changes=(False, True),
    )
    assert source == before
    assert validate_family(rows) == ()
    assert rows[1]["argument_family"]["source_variant_id"] == rows[0]["id"]
    assert rows[1]["argument_family"]["expected_transition"] == "JUSTIFIED_TRANSITION"
    broken = deepcopy(rows)
    broken[1]["argument_family"]["parameter_value"] = 0
    assert validate_family(broken)
    broken = deepcopy(rows)
    broken[1]["argument_family"]["material_facts_changed"] = False
    assert validate_family(broken)


@pytest.mark.parametrize(
    "actual,want",
    [
        (("ANSWER", "IDK", "IDK"), "JUSTIFIED_TRANSITION"),
        (("IDK", "IDK", "IDK"), "PREMATURE_TRANSITION"),
        (("ANSWER", "ANSWER", "IDK"), "DELAYED_TRANSITION"),
        (("ANSWER", "ANSWER", "ANSWER"), "MISSED_TRANSITION"),
    ],
)
def test_transition_localization(actual: tuple, want: str) -> None:
    from synthetic_data.argument_families import response_curve

    result = response_curve(
        (0.0, 1.0, 2.0), ("ANSWER", "IDK", "IDK"), actual, (0.8, 0.4, 0.2), (False, True, True)
    )
    assert result["transition_label"] == want
    assert result["confidence_slopes"] == pytest.approx((-0.4, -0.2))


def test_spurious_transition_and_hard_negative_provenance(tmp_path: Path) -> None:
    from gepa_mindfulness.training.eligibility import require_training_eligible
    from synthetic_data.argument_families import BOUNDARY_CASES, hard_negative, response_curve
    from synthetic_data.argument_pairs import build_argument_pair

    assert {
        (1, 3),
        (5, 7),
        (9, 12),
        (10, 12),
        (11, 13),
        (12, 13),
        (14, 16),
        (14, 15),
        (14, 17),
    }.issubset(BOUNDARY_CASES)
    curve = response_curve(
        (0.0, 1.0), ("IDK", "IDK"), ("IDK", "ANSWER"), (0.2, 0.9), (False, False)
    )
    assert curve["transition_label"] == "SPURIOUS_FRAMING_TRANSITION"
    row = source_row()
    negative = hard_negative(
        row,
        ("canonical_argument", "conclusion"),
        "Conclusion retained despite falsified premise",
        "rationale_migration",
    )
    assert row["canonical_argument"]["conclusion"] != negative["canonical_argument"]["conclusion"]
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row) + "\n")
    pair = build_argument_pair(
        path,
        1,
        negative,
        decisive_difference="premise falsified",
        expected_finding="unsupported conclusion",
    )
    assert len(pair["source_sha256"]) == 64
    assert pair["source_line"] == 1
    assert pair["source_record"] == row
    with pytest.raises(ValueError):
        require_training_eligible(pair)


def test_every_boundary_generator_and_rich_adapter(tmp_path: Path) -> None:
    from gepa_mindfulness.training.adapters.synthetic_cases import SyntheticCaseAdapter
    from synthetic_data.argument_families import BOUNDARY_CASES, generate_boundary_family

    for boundary in BOUNDARY_CASES:
        modes = tuple(
            "ANSWER" if case <= 8 or case == 15 else "IDK" if case <= 13 else "CLARIFY"
            for case in boundary
        )
        rows = generate_boundary_family(
            source_row(),
            boundary,
            family_id=f"boundary-{boundary[0]}-{boundary[1]}",
            semantic_core_id="host-reviewed-core",
            parameter_path=("scenario", "urgency"),
            values=("low", "high"),
            coordinates=(0.0, 1.0),
            response_modes=modes,
            confidence=(0.8, 0.2),
            material_changes=(False, True),
        )
        path = tmp_path / "family.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        requests = tuple(SyntheticCaseAdapter(path).iter_requests())
        assert requests[0].prompt != requests[1].prompt
        assert '"urgency": "low"' in requests[0].prompt
        assert "canonical_case_target" not in requests[0].prompt
        assert "hidden_information" not in requests[0].prompt
        assert rows[1]["argument_family"]["expected_transition"] == "JUSTIFIED_TRANSITION"
        assert (
            requests[1].metadata["source_record"]["argument_family"]["canonical_case_target"]
            == boundary[1]
        )
        assert requests[1].metadata["source_line"] == 2
    malformed = deepcopy(rows[0])
    malformed["argument_family"]["changed_parameter"] = "scenario.hidden_information"
    path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1:") as failure:
        tuple(SyntheticCaseAdapter(path).iter_requests())
    assert str(path) in str(failure.value)


def test_explicit_perspective_and_unknown_sensitivity_annotations() -> None:
    from synthetic_data.argument_families import generate_family

    kwargs = dict(
        family_id="role",
        semantic_core_id="same-facts",
        parameter_path=("scenario", "summary"),
        values=("A affects B", "B is affected by A"),
        coordinates=(0, 1),
        case_targets=(1, 1),
        response_modes=("ANSWER", "ANSWER"),
        confidence=(0.9, 0.9),
        material_changes=(False, False),
        annotations=(
            {},
            {
                "perspective": "role_reversal",
                "critical_premises": ["authority"],
                "distance_band": "ADVERSARIAL_CRITICAL_BOUNDARY",
            },
        ),
    )
    rows = generate_family(source_row(), **kwargs)
    assert rows[0]["argument_family"]["decision_sensitivity"] is None
    assert rows[1]["argument_family"]["perspective"] == "role_reversal"
    assert rows[1]["argument_family"]["critical_premises"] == ["authority"]
    kwargs["parameter_path"] = ("scenario", "hidden_information")
    with pytest.raises(ValueError, match="public"):
        generate_family(source_row(), **kwargs)


def test_reverse_sweep_preserves_history_dependence_without_motive_claim() -> None:
    from synthetic_data.argument_families import response_curve

    result = response_curve(
        (0, 1, 2),
        ("ANSWER", "IDK", "IDK"),
        ("ANSWER", "IDK", "IDK"),
        (0.9, 0.4, 0.2),
        (False, True, True),
        reverse_actual=("ANSWER", "ANSWER", "IDK"),
    )
    assert result["hysteresis_coordinates"] == (1,)
    assert result["history_dependence_label"] == "HYSTERESIS_OR_COMMITMENT_LOCK"


def test_allow_invalid_summary_keeps_reporting_instead_of_crashing(tmp_path: Path, capsys) -> None:
    from argparse import Namespace

    from scripts.synthetic_dataset_tool import cmd_summary

    rows = [
        source_row() | {"argument_family": value}
        for value in (None, [], {}, {"scenario_family_id": []})
    ]
    path = tmp_path / "invalid.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cmd_summary(Namespace(path=str(path), allow_invalid=True))
    output = capsys.readouterr().out
    assert "records: 4" in output
    assert "argument_family" in output
