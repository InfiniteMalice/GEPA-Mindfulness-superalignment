"""Contract tests for V5 repeat-aware evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any, Generic, TypeVar, cast

import pytest

import evaluation.v5_runner as v5_runner
from evaluation import (
    BehaviorRecord,
    CaseIdentity,
    DiagnosticRecord,
    EpistemicRecord,
    OutcomeRecord,
    RobustnessIdentity,
    ScoreRecord,
    SystemIdentity,
    V5EvaluationCell,
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
from mindful_trace_gepa.logging_schema import EventEnvelope

_T = TypeVar("_T")


class _SubclassV5EvaluationRecord(V5EvaluationRecord):
    """Deliberately non-canonical record used at the aggregation boundary."""


class _SingleIterationSequence(Sequence[_T], Generic[_T]):
    """A sequence that exposes accidental repeated iteration at a public boundary."""

    def __init__(self, values: Sequence[_T]) -> None:
        self._values = tuple(values)
        self.iterations = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __iter__(self) -> Iterator[_T]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("sequence was iterated more than once")
        return iter(self._values)


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


def test_summarize_repeats_snapshots_input_once_and_uses_one_success_count() -> None:
    """A stateful sequence must not supply different values to separate reductions."""

    passed = _SingleIterationSequence((True, False, False))

    assert summarize_repeats(passed) == RepeatMetrics(3, 1.0, 1 / 3, 0.0, 1 / 3)
    assert passed.iterations == 1


def test_grouped_aggregation_requires_explicit_plan_and_event_sequences() -> None:
    """A record collection must not define its own expected coverage or provenance."""

    record = _record(repeat_id=0, seed=101, passed=True)

    with pytest.raises(TypeError):
        summarize_v5_record_groups((record,))  # type: ignore[call-arg]


def test_grouped_inputs_and_event_sequences_are_each_snapshotted_once() -> None:
    """Stateful inputs must not supply different plan, record, or event values by iteration."""

    first = _record(repeat_id=0, seed=101, passed=True)
    second = _record(repeat_id=1, seed=102, passed=False)
    records = _SingleIterationSequence((first, second))
    planned_cells = _SingleIterationSequence((_cell(first), _cell(second)))
    first_events = _SingleIterationSequence(_events(first))
    second_events = _SingleIterationSequence(_events(second))

    summaries = summarize_v5_record_groups(
        records,
        planned_cells=planned_cells,
        event_sequences={_cell(first): first_events, _cell(second): second_events},
    )

    assert records.iterations == 1
    assert planned_cells.iterations == 1
    assert first_events.iterations == 1
    assert second_events.iterations == 1
    assert summaries[0].metrics == RepeatMetrics(2, 1.0, 0.5, 0.0, 0.5)


def test_plan_inventory_rejects_duplicate_cells_and_globally_duplicate_seeds() -> None:
    """One planned identity or seed must not represent multiple aggregate observations."""

    first = _record(repeat_id=0, seed=101, passed=True)
    second = _record(repeat_id=1, seed=101, passed=False)
    first_cell = _cell(first)
    second_cell = _cell(second)
    event_sequences = {first_cell: _events(first)}

    with pytest.raises(ValueError, match="duplicate planned cell"):
        summarize_v5_record_groups(
            (first,),
            planned_cells=(first_cell, first_cell),
            event_sequences=event_sequences,
            allow_partial=True,
        )
    with pytest.raises(ValueError, match="duplicate.*seed"):
        summarize_v5_record_groups(
            (first,),
            planned_cells=(first_cell, second_cell),
            event_sequences=event_sequences,
            allow_partial=True,
        )


def test_records_and_event_sequence_keys_must_be_exactly_in_plan() -> None:
    """Extra records, extra sequence keys, and missing sequences must fail explicitly."""

    planned = _record(repeat_id=0, seed=101, passed=True)
    extra = _record(repeat_id=1, seed=102, passed=False)
    planned_cell = _cell(planned)
    extra_cell = _cell(extra)

    with pytest.raises(ValueError, match="record.*planned_cells"):
        summarize_v5_record_groups(
            (planned, extra),
            planned_cells=(planned_cell,),
            event_sequences={planned_cell: _events(planned), extra_cell: _events(extra)},
            allow_partial=True,
        )
    with pytest.raises(ValueError, match="extra event_sequences"):
        summarize_v5_record_groups(
            (planned,),
            planned_cells=(planned_cell, extra_cell),
            event_sequences={planned_cell: _events(planned), extra_cell: _events(extra)},
            allow_partial=True,
        )
    with pytest.raises(ValueError, match="missing event_sequences"):
        summarize_v5_record_groups(
            (planned,),
            planned_cells=(planned_cell,),
            event_sequences={},
        )


def test_partial_coverage_requires_each_observed_group_to_be_a_repeat_prefix() -> None:
    """Repeat 1 must not appear as a partial result while planned repeat 0 is absent."""

    first = _record(repeat_id=0, seed=101, passed=True)
    second = _record(repeat_id=1, seed=102, passed=False)

    with pytest.raises(ValueError, match="prefix"):
        summarize_v5_record_groups(
            (second,),
            planned_cells=(_cell(first), _cell(second)),
            event_sequences={_cell(second): _events(second)},
            allow_partial=True,
        )


def test_partial_coverage_reports_a_completely_unobserved_planned_group() -> None:
    """A missing planned group must remain visible with zero observations and no metrics."""

    tool_first = _record(repeat_id=0, seed=101, passed=True)
    tool_second = _record(repeat_id=1, seed=102, passed=False)
    para_first = _record(repeat_id=0, seed=201, passed=False, stripe_id="PARAPHRASE")
    para_second = _record(repeat_id=1, seed=202, passed=False, stripe_id="PARAPHRASE")
    planned_cells = tuple(
        _cell(record) for record in (tool_first, tool_second, para_first, para_second)
    )

    summaries = summarize_v5_record_groups(
        (tool_first,),
        planned_cells=planned_cells,
        event_sequences={_cell(tool_first): _events(tool_first)},
        allow_partial=True,
    )

    assert [(item.expected_count, item.observed_count) for item in summaries] == [(2, 1), (2, 0)]
    assert summaries[0].metrics == RepeatMetrics(1, 1.0, 1.0, 1.0, 0.0)
    assert summaries[1].metrics is None


def test_partial_coverage_can_report_an_entirely_unobserved_plan() -> None:
    """An empty observed prefix must report zero coverage without inventing metrics."""

    first = _record(repeat_id=0, seed=101, passed=True)
    second = _record(repeat_id=1, seed=102, passed=False)

    summaries = summarize_v5_record_groups(
        (),
        planned_cells=(_cell(first), _cell(second)),
        event_sequences={},
        allow_partial=True,
    )

    assert len(summaries) == 1
    assert summaries[0].expected_count == 2
    assert summaries[0].observed_count == 0
    assert summaries[0].metrics is None


def test_record_version_drift_from_plan_is_rejected() -> None:
    """A record for another model version must not impersonate a planned cell."""

    planned = _record(repeat_id=0, seed=101, passed=True)
    drifted = _record(repeat_id=0, seed=101, passed=True, model_version="other-model")
    planned_cell = _cell(planned)

    with pytest.raises(ValueError, match="model_version"):
        summarize_v5_record_groups(
            (drifted,),
            planned_cells=(planned_cell,),
            event_sequences={planned_cell: _events(drifted)},
        )


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

    summaries = _summarize(records)

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
            expected_count=3,
            observed_count=3,
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
            expected_count=3,
            observed_count=3,
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
        _summarize(records)


def test_strict_coverage_rejects_missing_cells_and_partial_reports_counts() -> None:
    """A truncated repeat set must be explicit and remain bound to its complete plan."""

    planned_records = (
        _record(repeat_id=0, seed=101, passed=True),
        _record(repeat_id=1, seed=102, passed=False),
        _record(repeat_id=0, seed=201, passed=False, stripe_id="PARAPHRASE"),
        _record(repeat_id=1, seed=202, passed=False, stripe_id="PARAPHRASE"),
    )
    records = (planned_records[0], planned_records[1], planned_records[2])
    planned_cells = tuple(_cell(record) for record in planned_records)
    event_sequences = {_cell(record): _events(record) for record in records}

    with pytest.raises(ValueError, match="strict.*coverage"):
        summarize_v5_record_groups(
            records,
            planned_cells=planned_cells,
            event_sequences=event_sequences,
        )

    partial_summaries = summarize_v5_record_groups(
        records,
        planned_cells=planned_cells,
        event_sequences=event_sequences,
        allow_partial=True,
    )

    assert [
        (summary.expected_count, summary.observed_count, summary.metrics.k)
        for summary in partial_summaries
        if summary.metrics is not None
    ] == [(2, 2, 2), (2, 1, 1)]


def test_summarize_record_groups_keeps_model_and_harness_versions_separate() -> None:
    """A system-version key omission must merge the distinct literal groups in this test."""

    records = (
        _record(repeat_id=0, seed=101, passed=True, model_version="model-a"),
        _record(repeat_id=0, seed=201, passed=False, harness_version="harness-b"),
    )

    summaries = _summarize(records)

    assert tuple(summary.key.model_version for summary in summaries) == (
        "model-a",
        "mindful-model-2026-09-10",
    )
    assert tuple(summary.key.harness_version for summary in summaries) == (
        "v5-harness-1.0.0",
        "harness-b",
    )
    assert tuple(cast(RepeatMetrics, summary.metrics).pass_at_k for summary in summaries) == (
        1.0,
        0.0,
    )


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
    plan = (_cell(record),)
    event_sequences = {plan[0]: _events(record)}
    section = getattr(record, section_name)
    object.__setattr__(section, field_name, invalid_value)

    with pytest.raises(ValueError, match=message):
        summarize_v5_record_groups(
            (record,),
            planned_cells=plan,
            event_sequences=event_sequences,
        )


def test_summarize_record_groups_keeps_a_second_canonical_case_separate() -> None:
    """Case identity must prevent distinct canonical cases from being conflated."""

    summaries = _summarize(
        (
            _record(repeat_id=0, seed=101, passed=True, case_id=14),
            _record(repeat_id=0, seed=201, passed=False, case_id=15),
        )
    )

    assert tuple(summary.key.case_id for summary in summaries) == (14, 15)
    assert tuple(cast(RepeatMetrics, summary.metrics).pass_at_k for summary in summaries) == (
        1.0,
        0.0,
    )


def test_summarize_record_groups_keeps_valid_stripe_subtypes_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subtype-key omission must conflate the two valid subtype groups in this test."""

    _allow_tool_subtypes(monkeypatch)
    try:
        summaries = _summarize(
            (
                _record(repeat_id=0, seed=101, passed=True, subtype="retry"),
                _record(repeat_id=0, seed=201, passed=False, subtype="fallback"),
            )
        )
        assert tuple(summary.key.subtype for summary in summaries) == ("retry", "fallback")
        assert tuple(cast(RepeatMetrics, summary.metrics).pass_at_k for summary in summaries) == (
            1.0,
            0.0,
        )
    finally:
        v5_records._canonical_stripe_map.cache_clear()
        v5_runner._canonical_stripe_map.cache_clear()


def test_summarize_record_groups_does_not_mutate_valid_input() -> None:
    """Canonical revalidation must use a new record rather than alter the caller's record."""

    record = _record(repeat_id=0, seed=101, passed=True)
    before = record.to_dict()

    _summarize((record,))

    assert record.to_dict() == before


def test_summarize_record_groups_rejects_record_seed_drift_from_planned_cell() -> None:
    """A record seed that differs from its planned cell must not enter repeat metrics."""

    planned = _record(repeat_id=0, seed=101, passed=True)
    drifted = _record(repeat_id=0, seed=102, passed=True)
    cell = _cell(planned)

    with pytest.raises(ValueError, match="seed"):
        summarize_v5_record_groups(
            (drifted,),
            planned_cells=(cell,),
            event_sequences={cell: _events(drifted)},
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
        summarize_v5_record_groups(
            (),
            planned_cells=(_cell(base),),
            event_sequences={},
        )
    with pytest.raises(ValueError, match="exact V5EvaluationRecord"):
        summarize_v5_record_groups(
            (noncanonical,),
            planned_cells=(_cell(base),),
            event_sequences={_cell(base): _events(base)},
        )
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_v5_record_groups(
            (negative_repeat,),
            planned_cells=(_cell(base),),
            event_sequences={_cell(base): _events(base)},
        )


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

    original_summary = _summarize((original,))
    mutated_summary = _summarize((mutated,))

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
            prediction_ref=_event_id("prediction", case_id, stripe_id, repeat_id, seed),
            evidence_refs=(_event_id("evidence", case_id, stripe_id, repeat_id, seed),),
            verifier_refs=(_event_id("verification", case_id, stripe_id, repeat_id, seed),),
            confidence=0.82,
        ),
        behavior=BehaviorRecord(
            action_refs=(_event_id("proposed", case_id, stripe_id, repeat_id, seed),),
            abstained=True,
            requested_clarification=True,
        ),
        outcome=OutcomeRecord(
            observation_refs=(_event_id("observation", case_id, stripe_id, repeat_id, seed),),
            verifier_refs=(_event_id("verification", case_id, stripe_id, repeat_id, seed),),
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
    monkeypatch.setattr(v5_runner, "load_stripe_registry", lambda: subtype_registry)
    v5_records._canonical_stripe_map.cache_clear()
    v5_runner._canonical_stripe_map.cache_clear()


def _event_id(kind: str, case_id: int, stripe_id: str, repeat_id: int, seed: int) -> str:
    """Return one literal event namespace unique to a planned evaluation cell."""

    return f"event:{kind}:{case_id}:{stripe_id}:{repeat_id}:{seed}"


def _cell(record: V5EvaluationRecord) -> V5EvaluationCell:
    """Build the exact planned cell represented by a trusted test record."""

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


def _events(record: V5EvaluationRecord) -> tuple[EventEnvelope, ...]:
    """Build one complete literal action-bound sequence for a trusted test record."""

    case_id = record.case.case_id
    stripe_id = record.robustness.stripe_id
    repeat_id = record.system.repeat_id
    seed = record.system.seed

    def event_id(kind: str) -> str:
        return _event_id(kind, case_id, stripe_id, repeat_id, seed)

    evidence_ref = event_id("evidence")
    action_id = f"action:{case_id}:{stripe_id}:{repeat_id}:{seed}"
    prediction_id = f"prediction:{case_id}:{stripe_id}:{repeat_id}:{seed}"
    observation_id = f"observation:{case_id}:{stripe_id}:{repeat_id}:{seed}"
    common: dict[str, object] = {
        "run_id": f"run:{case_id}:{stripe_id}:{repeat_id}:{seed}",
        "repeat_id": repeat_id,
        "model_version": record.system.model_version,
        "harness_version": record.system.harness_version,
        "case_version": record.case.case_version,
        "case_id": case_id,
        "stripe_id": stripe_id,
        "seed": seed,
    }

    def envelope(
        kind: str,
        event_type: str,
        payload: dict[str, object],
        **metadata: object,
    ) -> EventEnvelope:
        return EventEnvelope(
            schema_version="1.0",
            event_id=event_id(kind),
            event_type=event_type,
            timestamp="2026-09-10T12:00:00Z",
            payload=payload,
            **cast(Any, {**common, **metadata}),
        )

    prediction = envelope(
        "prediction",
        "prediction_commit",
        {
            "prediction_commit_id": prediction_id,
            "predicted_outcome": {"behavior": "clarify"},
            "confidence": record.epistemics.confidence,
            "evidence_refs": [evidence_ref],
        },
        evidence_refs=(evidence_ref,),
    )
    proposed = envelope(
        "proposed",
        "action_proposed",
        {
            "action_id": action_id,
            "action_class": "request_clarification",
            "reversible": True,
            "authorization_scope": "evaluation-only",
            "prediction_commit_id": prediction_id,
        },
        parent_event_ids=(prediction.event_id,),
        action_id=action_id,
        authorization_scope="evaluation-only",
    )
    executed = envelope(
        "executed",
        "action_executed",
        dict(proposed.payload),
        parent_event_ids=(proposed.event_id,),
        action_id=action_id,
        authorization_scope="evaluation-only",
    )
    observation = envelope(
        "observation",
        "outcome_observed",
        {
            "observation_id": observation_id,
            "action_id": action_id,
            "actual_outcome": {"passed": record.outcome.passed},
            "evidence_refs": [evidence_ref],
        },
        parent_event_ids=(executed.event_id,),
        action_id=action_id,
        evidence_refs=(evidence_ref,),
    )
    verification = envelope(
        "verification",
        "verification_result",
        {
            "verifier_id": f"verifier:{case_id}:{stripe_id}:{repeat_id}:{seed}",
            "verifier_version": "v1",
            "observation_id": observation_id,
            "verified": True,
            "verifier_refs": ["verifier:contract-v1"],
        },
        parent_event_ids=(observation.event_id,),
        action_id=action_id,
        verifier_refs=("verifier:contract-v1",),
    )
    epistemic = envelope(
        "epistemic",
        "epistemic_assessment",
        {"assessment": "verified"},
        parent_event_ids=(verification.event_id,),
        action_id=action_id,
    )
    case = envelope(
        "case",
        "case_assessment",
        {"assessment": "pass" if record.outcome.passed else "fail"},
        parent_event_ids=(epistemic.event_id,),
        action_id=action_id,
    )
    return prediction, proposed, executed, observation, verification, epistemic, case


def _summarize(
    records: Sequence[V5EvaluationRecord],
    *,
    planned_records: Sequence[V5EvaluationRecord] | None = None,
    allow_partial: bool = False,
) -> tuple[V5RepeatGroupSummary, ...]:
    """Call the public aggregator with explicit plan and per-cell sequence fixtures."""

    record_snapshot = tuple(records)
    plan_source = record_snapshot if planned_records is None else tuple(planned_records)
    planned_cells = tuple(_cell(record) for record in plan_source)
    event_sequences = {_cell(record): _events(record) for record in record_snapshot}
    return summarize_v5_record_groups(
        record_snapshot,
        planned_cells=planned_cells,
        event_sequences=event_sequences,
        allow_partial=allow_partial,
    )
