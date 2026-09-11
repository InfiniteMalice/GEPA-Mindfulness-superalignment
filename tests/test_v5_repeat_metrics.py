"""Contract tests for V5 repeat-aware evaluation metrics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from evaluation import (
    BehaviorRecord,
    CaseIdentity,
    DiagnosticRecord,
    EpistemicRecord,
    OutcomeRecord,
    RobustnessIdentity,
    ScoreRecord,
    SystemIdentity,
    V5EvaluationRecord,
    v5_records,
)
from evaluation.cases.registry import RobustnessStripeRegistry
from evaluation.v5_runner import (
    RepeatMetrics,
    V5RepeatGroupKey,
    V5RepeatGroupSummary,
    summarize_repeats,
    summarize_v5_record_groups,
)


class _SubclassV5EvaluationRecord(V5EvaluationRecord):
    """Deliberately non-canonical record used at the aggregation boundary."""


@pytest.mark.parametrize(
    ("passed", "expected_k", "expected_pass", "expected_mean", "expected_power", "expected_gap"),
    [
        ([True, True, True], 3, 1.0, 1.0, 1.0, 0.0),
        ([True, False, False], 3, 1.0, 1 / 3, 0.0, 1 / 3),
        ([False, False], 2, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_summarize_repeats_preserves_distinct_hand_computed_metrics(
    passed: list[bool],
    expected_k: int,
    expected_pass: float,
    expected_mean: float,
    expected_power: float,
    expected_gap: float,
) -> None:
    """A reducer bug in any boolean reduction must change a literal expectation."""

    result = summarize_repeats(passed)

    assert result == RepeatMetrics(
        k=expected_k,
        pass_at_k=expected_pass,
        mean_at_k=expected_mean,
        pass_power_k=expected_power,
        consistency_gap_at_k=expected_gap,
    )


def test_summarize_repeats_rejects_empty_input() -> None:
    """An empty repetition set has no meaningful repeat metrics."""

    with pytest.raises(ValueError, match="passed must not be empty"):
        summarize_repeats([])


@pytest.mark.parametrize("passed", [[True, 1], [False, "false"], "true"])
def test_summarize_repeats_rejects_non_boolean_sequence_members(passed: object) -> None:
    """Non-boolean truthiness must not be counted as a passing repeat."""

    with pytest.raises(ValueError, match="built-in bools"):
        summarize_repeats(passed)  # type: ignore[arg-type]


def test_summarize_record_groups_keeps_first_seen_full_identities_separate() -> None:
    """A grouping-key omission must merge a literal group that remains distinct here."""

    records = (
        _record(repeat_id=0, seed=101, passed=True),
        _record(repeat_id=0, seed=201, passed=False, stripe_id="PARAPHRASE"),
        _record(repeat_id=1, seed=102, passed=False),
        _record(repeat_id=1, seed=202, passed=False, stripe_id="PARAPHRASE"),
        _record(repeat_id=2, seed=103, passed=False),
        _record(repeat_id=2, seed=203, passed=False, stripe_id="PARAPHRASE"),
    )

    summaries = summarize_v5_record_groups(records)

    assert summaries == (
        V5RepeatGroupSummary(
            key=V5RepeatGroupKey(
                case_id=14,
                case_version="17case-v5",
                case_key="correct_high_stakes_clarifying_abstention",
                case_title="Correct high-stakes clarifying abstention",
                stripe_id="TOOL_ERROR",
                subtype=None,
                model_version="mindful-model-2026-09-10",
                harness_version="v5-harness-1.0.0",
            ),
            metrics=RepeatMetrics(
                k=3,
                pass_at_k=1.0,
                mean_at_k=1 / 3,
                pass_power_k=0.0,
                consistency_gap_at_k=1 / 3,
            ),
        ),
        V5RepeatGroupSummary(
            key=V5RepeatGroupKey(
                case_id=14,
                case_version="17case-v5",
                case_key="correct_high_stakes_clarifying_abstention",
                case_title="Correct high-stakes clarifying abstention",
                stripe_id="PARAPHRASE",
                subtype=None,
                model_version="mindful-model-2026-09-10",
                harness_version="v5-harness-1.0.0",
            ),
            metrics=RepeatMetrics(
                k=3,
                pass_at_k=0.0,
                mean_at_k=0.0,
                pass_power_k=0.0,
                consistency_gap_at_k=0.0,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("record_values", "message"),
    [
        (
            (
                (0, 101, True),
                (0, 102, False),
            ),
            "duplicate repeat_id",
        ),
        (
            (
                (0, 101, True),
                (2, 102, False),
            ),
            "contiguous",
        ),
        (
            (
                (0, 101, True),
                (1, 101, False),
            ),
            "duplicate seed",
        ),
    ],
)
def test_summarize_record_groups_rejects_invalid_group_repeat_identity(
    record_values: tuple[tuple[int, int, bool], ...], message: str
) -> None:
    """Duplicate or missing repeat identities must not silently distort a group summary."""

    records = tuple(
        _record(repeat_id=repeat_id, seed=seed, passed=passed)
        for repeat_id, seed, passed in record_values
    )

    with pytest.raises(ValueError, match=message):
        summarize_v5_record_groups(records)


def test_summarize_record_groups_requires_equal_repeat_counts_unless_partial() -> None:
    """Default aggregate comparisons require equally sized repeat samples."""

    records = (
        _record(repeat_id=0, seed=101, passed=True),
        _record(repeat_id=1, seed=102, passed=False),
        _record(repeat_id=0, seed=201, passed=False, stripe_id="PARAPHRASE"),
    )

    with pytest.raises(ValueError, match="equal repeat counts"):
        summarize_v5_record_groups(records)

    partial_summaries = summarize_v5_record_groups(records, allow_partial=True)

    assert tuple(summary.metrics.k for summary in partial_summaries) == (2, 1)


def test_summarize_record_groups_keeps_model_and_harness_versions_separate() -> None:
    """A system-version key omission must merge the distinct literal groups in this test."""

    records = (
        _record(repeat_id=0, seed=101, passed=True, model_version="model-a"),
        _record(repeat_id=0, seed=201, passed=False, harness_version="harness-b"),
    )

    summaries = summarize_v5_record_groups(records)

    assert tuple(summary.key.model_version for summary in summaries) == (
        "model-a",
        "mindful-model-2026-09-10",
    )
    assert tuple(summary.key.harness_version for summary in summaries) == (
        "v5-harness-1.0.0",
        "harness-b",
    )
    assert tuple(summary.metrics.pass_at_k for summary in summaries) == (1.0, 0.0)


@pytest.mark.parametrize(
    ("section_name", "field_name", "invalid_value", "message"),
    [
        ("case", "case_id", 99, "case_id"),
        ("case", "case_version", "wrong-version", "case_version"),
        ("case", "case_key", "wrong-key", "case_key"),
        ("case", "case_title", "wrong title", "case_title"),
        ("robustness", "stripe_id", "UNKNOWN", "stripe_id"),
        ("robustness", "subtype", "unknown-subtype", "subtype"),
        ("system", "model_version", "", "model_version"),
        ("system", "harness_version", " ", "harness_version"),
        ("system", "seed", 9_007_199_254_740_992, "seed"),
    ],
)
def test_summarize_record_groups_revalidates_corrupted_group_identities(
    section_name: str,
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    """A frozen-record corruption must fail canonical record validation before grouping."""

    record = _record(repeat_id=0, seed=101, passed=True)
    section = getattr(record, section_name)
    object.__setattr__(section, field_name, invalid_value)

    with pytest.raises(ValueError, match=message):
        summarize_v5_record_groups((record,))


def test_summarize_record_groups_keeps_a_second_canonical_case_separate() -> None:
    """Case identity must prevent distinct canonical cases from being conflated."""

    summaries = summarize_v5_record_groups(
        (
            _record(repeat_id=0, seed=101, passed=True, case_id=14),
            _record(repeat_id=0, seed=201, passed=False, case_id=15),
        )
    )

    assert tuple(summary.key.case_id for summary in summaries) == (14, 15)
    assert tuple(summary.metrics.pass_at_k for summary in summaries) == (1.0, 0.0)


def test_summarize_record_groups_keeps_valid_stripe_subtypes_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subtype-key omission must conflate the two valid subtype groups in this test."""

    _allow_tool_subtypes(monkeypatch)
    summaries = summarize_v5_record_groups(
        (
            _record(repeat_id=0, seed=101, passed=True, subtype="retry"),
            _record(repeat_id=0, seed=201, passed=False, subtype="fallback"),
        )
    )

    assert tuple(summary.key.subtype for summary in summaries) == ("retry", "fallback")
    assert tuple(summary.metrics.pass_at_k for summary in summaries) == (1.0, 0.0)


def test_summarize_record_groups_does_not_mutate_valid_input() -> None:
    """Canonical revalidation must use a new record rather than alter the caller's record."""

    record = _record(repeat_id=0, seed=101, passed=True)
    before = record.to_dict()

    summarize_v5_record_groups((record,))

    assert record.to_dict() == before


def test_summarize_record_groups_accepts_a_valid_record_seed_outside_planner_range() -> None:
    """Aggregation accepts the V5 record contract without imposing planner-only seed bounds."""

    summaries = summarize_v5_record_groups((_record(repeat_id=0, seed=4_294_967_296, passed=True),))

    assert summaries[0].metrics == RepeatMetrics(
        k=1,
        pass_at_k=1.0,
        mean_at_k=1.0,
        pass_power_k=1.0,
        consistency_gap_at_k=0.0,
    )


def test_summarize_record_groups_rejects_empty_noncanonical_and_negative_records() -> None:
    """Malformed aggregate inputs must fail before they can affect a repeat summary."""

    negative_repeat = _record(repeat_id=0, seed=101, passed=True)
    object.__setattr__(negative_repeat.system, "repeat_id", -1)
    base = _record(repeat_id=0, seed=201, passed=True)
    noncanonical = _SubclassV5EvaluationRecord(
        case=base.case,
        robustness=base.robustness,
        system=base.system,
        epistemics=base.epistemics,
        behavior=base.behavior,
        outcome=base.outcome,
        scores=base.scores,
        diagnostics=base.diagnostics,
    )

    with pytest.raises(ValueError, match="records must not be empty"):
        summarize_v5_record_groups(())
    with pytest.raises(ValueError, match="exact V5EvaluationRecord"):
        summarize_v5_record_groups((noncanonical,))
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_v5_record_groups((negative_repeat,))


def test_repeat_metrics_are_exported_from_the_evaluation_package() -> None:
    """Evaluation callers must be able to reach the public repeat-metric interface."""

    from evaluation import (  # pylint: disable=import-outside-toplevel
        RepeatMetrics as ExportedRepeatMetrics,
    )
    from evaluation import (  # pylint: disable=import-outside-toplevel
        V5RepeatGroupKey as ExportedV5RepeatGroupKey,
    )
    from evaluation import (  # pylint: disable=import-outside-toplevel
        V5RepeatGroupSummary as ExportedV5RepeatGroupSummary,
    )
    from evaluation import (  # pylint: disable=import-outside-toplevel
        summarize_repeats as exported_summarize_repeats,
    )
    from evaluation import (  # pylint: disable=import-outside-toplevel
        summarize_v5_record_groups as exported_summarize_v5_record_groups,
    )

    assert ExportedRepeatMetrics is RepeatMetrics
    assert ExportedV5RepeatGroupKey is V5RepeatGroupKey
    assert ExportedV5RepeatGroupSummary is V5RepeatGroupSummary
    assert exported_summarize_repeats is summarize_repeats
    assert exported_summarize_v5_record_groups is summarize_v5_record_groups


def test_summarize_record_groups_uses_only_outcome_passed() -> None:
    """Changes to scores and diagnostics must not alter outcome repeat metrics."""

    original = _record(repeat_id=0, seed=101, passed=True)
    mutated = replace(
        original,
        scores=ScoreRecord(
            correctness=0.0,
            calibration=0.0,
            abstention=0.0,
            epistemic_process=0.0,
            total=0.0,
        ),
        diagnostics=DiagnosticRecord(
            trace_summary="A different diagnostic trace.",
            deception_signal=9.0,
            mechanistic_signal=-2.0,
        ),
    )

    original_summary = summarize_v5_record_groups((original,))
    mutated_summary = summarize_v5_record_groups((mutated,))

    assert mutated_summary == original_summary


def _record(
    *,
    repeat_id: int,
    seed: int,
    passed: bool,
    case_id: int = 14,
    stripe_id: str = "TOOL_ERROR",
    subtype: str | None = None,
    model_version: str = "mindful-model-2026-09-10",
    harness_version: str = "v5-harness-1.0.0",
) -> V5EvaluationRecord:
    """Return a literal valid V5 record with one independently controlled outcome."""

    case_key, case_title = _CANONICAL_CASES[case_id]

    return V5EvaluationRecord(
        case=CaseIdentity(
            case_id=case_id,
            case_version="17case-v5",
            case_key=case_key,
            case_title=case_title,
        ),
        robustness=RobustnessIdentity(stripe_id=stripe_id, subtype=subtype),
        system=SystemIdentity(
            repeat_id=repeat_id,
            seed=seed,
            model_version=model_version,
            harness_version=harness_version,
        ),
        epistemics=EpistemicRecord(
            prediction_ref="event:prediction-14",
            evidence_refs=("evidence:request-14",),
            verifier_refs=("verifier:contract-v1",),
            confidence=0.82,
        ),
        behavior=BehaviorRecord(
            action_refs=("event:action-proposed-14",),
            abstained=True,
            requested_clarification=True,
        ),
        outcome=OutcomeRecord(
            observation_refs=("event:outcome-observed-14",),
            verifier_refs=("event:verification-result-14",),
            passed=passed,
        ),
        scores=ScoreRecord(
            correctness=1.0,
            calibration=0.82,
            abstention=1.0,
            epistemic_process=0.75,
            total=0.8925,
        ),
        diagnostics=DiagnosticRecord(
            trace_summary="The model asked a targeted question after a tool failure.",
            deception_signal=0.13,
            mechanistic_signal=0.44,
        ),
    )


_CANONICAL_CASES = {
    14: (
        "correct_high_stakes_clarifying_abstention",
        "Correct high-stakes clarifying abstention",
    ),
    15: (
        "over_eager_ambiguous_compliance",
        "Over-eager ambiguous/high-stakes compliance",
    ),
}


def _allow_tool_subtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow two test-only TOOL_ERROR subtypes through the real record validator."""

    registry = v5_records.load_stripe_registry()
    stripes = tuple(
        (
            replace(stripe, allowed_subtypes=("retry", "fallback"))
            if stripe.id == "TOOL_ERROR"
            else stripe
        )
        for stripe in registry.stripes
    )
    subtype_registry = RobustnessStripeRegistry(
        registry_version=registry.registry_version,
        stripes=stripes,
    )
    monkeypatch.setattr(v5_records, "load_stripe_registry", lambda: subtype_registry)
