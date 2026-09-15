"""Failure diversity, lineage and independently verified repair regressions."""

from dataclasses import replace

import pytest
from test_v5_provenance import _record, _verified_sequence

from evaluation.failure_atlas import FailureAtlas
from evaluation.v5_records import AssessmentRecord, V5EvaluationRecord
from mindful_trace_gepa.logging_schema import EventEnvelope


def failed(
    variant: str = "original", intent: str = "preserve authorized scope"
) -> V5EvaluationRecord:
    return replace(
        _record(passed=False, epistemic_process=0.0),
        assessment=AssessmentRecord(
            variant_id=variant,
            transformation_lineage=("source:1", variant),
            semantic_intent=intent,
            failure_family="ambiguity",
            observed_behavior="guessed",
            expected_behavior="clarify",
            evaluator_attribution="MODEL_DEFICIENCY",
        ),
    )


def failure_events() -> tuple[EventEnvelope, ...]:
    return _verified_sequence(outcome_passed=False, epistemic_assessment="unverified")


def repaired_record() -> V5EvaluationRecord:
    return replace(
        _record(),
        assessment=replace(
            failed().assessment,
            evaluator_attribution=None,
            task_success=True,
            epistemic_success=True,
            alignment_success=True,
            evaluation_success=True,
            evidence_sufficiency=True,
            provenance_complete=True,
            verification_rung="DETERMINISTIC",
            regression_status="PASSED",
        ),
    )


def regression_events() -> tuple[EventEnvelope, ...]:
    return tuple(replace(event, run_id="run:verified-regression") for event in _verified_sequence())


def test_equivalent_failures_group_without_erasing_observations() -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    atlas = atlas.observe(
        "failure:2", failed("paraphrase"), failure_events(), "2026-09-15T12:01:00Z"
    )
    assert len(atlas.entries) == 2
    assert atlas.entries[1].status == "PERSISTENT"
    assert atlas.entries[1].equivalent_failures == ("failure:1",)
    assert atlas.entries[0].record.assessment.variant_id == "original"
    assert atlas.entries[1].record.assessment.variant_id == "paraphrase"
    assert atlas.report()["14"]["TOOL_ERROR"]["NONE"]["ambiguity"]["PERSISTENT"] == 1
    assert FailureAtlas.from_dict(atlas.to_dict()) == atlas


def test_failed_repair_stays_failed_then_verified_repair_becomes_regression() -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    with pytest.raises(ValueError, match="passing"):
        atlas.repair(
            "failure:1", "repair:1", failed(), failure_events(), ("test:1",), "2026-09-15T12:01:00Z"
        )
    repaired = atlas.repair(
        "failure:1",
        "repair:1",
        repaired_record(),
        regression_events(),
        ("test:1",),
        "2026-09-15T12:01:00Z",
    )
    assert repaired.entries[0].status == "REPAIRED"
    assert repaired.entries[0].run_id != repaired.entries[0].regression_run_id
    assert FailureAtlas.from_dict(repaired.to_dict()) == repaired
    assert repaired.entries[0].record.outcome.passed is False
    assert repaired.entries[0].regression_record.outcome.passed is True
    assert repaired.entries[0].regression_tests == ("test:1",)
    with pytest.raises(ValueError, match="already repaired"):
        repaired.repair(
            "failure:1",
            "repair:replacement",
            repaired_record(),
            regression_events(),
            ("test:replacement",),
            "2026-09-15T12:02:00Z",
        )
    regressed = repaired.observe("failure:2", failed(), failure_events(), "2026-09-15T12:02:00Z")
    assert regressed.entries[-1].status == "REGRESSION"
    assert regressed.entries[-1].parent_failure == "failure:1"


def test_missing_regression_or_foreign_coordinate_cannot_close_failure() -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    with pytest.raises(ValueError, match="regression"):
        atlas.repair(
            "failure:1", "repair:1", _record(), _verified_sequence(), (), "2026-09-15T12:01:00Z"
        )
    with pytest.raises(ValueError, match="duplicate"):
        atlas.observe("failure:1", failed(), failure_events(), "2026-09-15T12:02:00Z")


def test_family_balancing_preserves_rare_intents_and_hidden_isolation() -> None:
    atlas = FailureAtlas()
    for index, intent in enumerate(["common", "common", "common", "rare"]):
        atlas = atlas.observe(
            f"failure:{index}",
            failed(str(index), intent),
            failure_events(),
            f"2026-09-15T12:00:0{index}Z",
        )
    assert {entry.record.assessment.semantic_intent for entry in atlas.repair_candidates()} == {
        "common",
        "rare",
    }
    hidden = replace(
        failed(), assessment=replace(failed().assessment, training_eligibility="HIDDEN_EVAL")
    )
    atlas = atlas.observe("hidden", hidden, failure_events(), "2026-09-15T12:01:00Z")
    assert "hidden" not in {entry.failure_id for entry in atlas.repair_candidates()}


def test_hidden_family_blocks_direct_public_alias_repair() -> None:
    atlas = FailureAtlas().observe("public", failed(), failure_events(), "2026-09-15T12:00:00Z")
    hidden = replace(
        failed(), assessment=replace(failed().assessment, training_eligibility="HIDDEN_EVAL")
    )
    atlas = atlas.observe("hidden", hidden, failure_events(), "2026-09-15T12:01:00Z")
    assert atlas.repair_candidates() == ()
    with pytest.raises(ValueError, match="hidden"):
        atlas.repair(
            "public",
            "repair:1",
            repaired_record(),
            regression_events(),
            ("test:1",),
            "2026-09-15T12:02:00Z",
        )


@pytest.mark.parametrize("entries", ["", {}, None])
def test_atlas_requires_json_array(entries: object) -> None:
    with pytest.raises(ValueError, match="array"):
        FailureAtlas.from_dict({"entries": entries})


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_id", "repair:unearned"),
        ("regression_tests", ["test:unearned"]),
        ("regression_run_id", "run:unearned"),
    ],
)
def test_unrepaired_entry_cannot_claim_repair_metadata(field: str, value: object) -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    payload = atlas.to_dict()
    payload["entries"][0][field] = value
    with pytest.raises(ValueError, match="non-repaired"):
        FailureAtlas.from_dict(payload)


def test_repair_preserves_target_intent() -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    foreign = replace(
        repaired_record(),
        assessment=replace(repaired_record().assessment, semantic_intent="another objective"),
    )
    with pytest.raises(ValueError, match="semantic intent"):
        atlas.repair(
            "failure:1",
            "repair:1",
            foreign,
            regression_events(),
            ("test:1",),
            "2026-09-15T12:01:00Z",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"answer_correctness": 0.0},
        {"evaluation_success": False},
        {"evaluator_disagreement": True},
        {"regression_status": "FAILED"},
    ],
)
def test_disputed_or_failed_regression_cannot_close_failure(change: dict[str, object]) -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    bad = replace(repaired_record(), assessment=replace(repaired_record().assessment, **change))
    with pytest.raises(ValueError):
        atlas.repair(
            "failure:1", "repair:1", bad, regression_events(), ("test:1",), "2026-09-15T12:01:00Z"
        )


def test_repair_requires_a_distinct_run_and_valid_time_and_tests() -> None:
    atlas = FailureAtlas().observe("failure:1", failed(), failure_events(), "2026-09-15T12:00:00Z")
    with pytest.raises(ValueError, match="distinct run"):
        atlas.repair(
            "failure:1",
            "repair:1",
            repaired_record(),
            _verified_sequence(),
            ("test:1",),
            "2026-09-15T12:01:00Z",
        )
    with pytest.raises(ValueError, match="last_seen"):
        atlas.repair(
            "failure:1",
            "repair:1",
            repaired_record(),
            regression_events(),
            ("test:1",),
            "2026-09-15T11:00:00Z",
        )
    with pytest.raises(ValueError, match="array"):
        atlas.repair(
            "failure:1",
            "repair:1",
            repaired_record(),
            regression_events(),
            "test:1",
            "2026-09-15T12:01:00Z",
        )
