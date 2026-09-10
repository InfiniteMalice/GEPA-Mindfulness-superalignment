# PR-3 V5 Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`. Derive
> metric expectations by hand rather than with production helpers.

**Goal:** Add typed V5 evaluation records, deterministic case-by-stripe-by-repeat planning, and
repeat-aware metrics.

**Architecture:** Nested frozen records separate case, robustness, system, epistemic, behavior,
outcome, scores, and diagnostics. A pure planner enumerates cells; a pure aggregator computes
metrics without conflating correctness and consistency.

**Tech Stack:** Python dataclasses, itertools, JSON, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Define the V5 evaluation record

**Files:**

- Create: `evaluation/v5_records.py`
- Modify: `evaluation/__init__.py`
- Test: `tests/test_v5_evaluation_record.py`

**Public record:**

```python
@dataclass(frozen=True)
class V5EvaluationRecord:
    case: CaseIdentity
    robustness: RobustnessIdentity
    system: SystemIdentity
    epistemics: EpistemicRecord
    behavior: BehaviorRecord
    outcome: OutcomeRecord
    scores: ScoreRecord
    diagnostics: DiagnosticRecord

    def to_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V5EvaluationRecord": ...
```

1. Write a complete literal fixture using Case 14, `TOOL_ERROR`, repeat 2, a fixed seed, explicit
   model/harness versions, evidence links, action links, verifier links, decomposed scores, and
   diagnostic-only signals.
2. Write failing tests for round trip, deterministic JSON, invalid case version, unknown case ID,
   unknown stripe, negative repeat, missing model/harness version, unbounded confidence/scores, and
   diagnostics omitted from `optimizer_scores()`.
3. Run the tests and verify module-not-found failure.
4. Implement focused frozen dataclasses. Use `17case-v5` as a required literal value rather than a
   silent default when hydrating external data.
5. Validate case and stripe identity through the PR-1 registries.
6. Implement `optimizer_scores()` to return only the documented score section; never return fields
   from diagnostics.
7. Run `python -m pytest tests/test_v5_evaluation_record.py -q`.

## Task 2: Enumerate V5 evaluation cells

**Files:**

- Create: `evaluation/v5_runner.py`
- Test: `tests/test_v5_cell_planner.py`

**Interface:**

```python
@dataclass(frozen=True)
class V5EvaluationCell:
    case_id: int
    case_version: str
    stripe_id: str
    subtype: str | None
    repeat_id: int
    seed: int
    model_version: str
    harness_version: str


def plan_v5_cells(
    *,
    case_ids: Sequence[int] | None = None,
    stripe_ids: Sequence[str] | None = None,
    repeats: int = 5,
    base_seed: int = 0,
    model_version: str,
    harness_version: str,
) -> tuple[V5EvaluationCell, ...]: ...
```

1. Write failing tests that expect `17 * 11 * 5 == 935` default cells, stable ordering, repeat IDs
   `0..4`, and unique deterministic seeds.
2. Write failing tests for a selected two-case, two-stripe, three-repeat grid and hand-check its 12
   exact tuple identities.
3. Write failing tests for zero/negative repeats, duplicate requested IDs, unknown cases/stripes,
   blank versions, and integer overflow avoidance in seed derivation.
4. Run tests and verify module-not-found failure.
5. Implement registry-validated Cartesian enumeration. Derive a 32-bit seed from a stable SHA-256
   digest of base seed, case, stripe, repeat, model version, and harness version.
6. Run `python -m pytest tests/test_v5_cell_planner.py -q`.

## Task 3: Compute repeat-aware metrics

**Files:**

- Modify: `evaluation/v5_runner.py`
- Test: `tests/test_v5_repeat_metrics.py`

**Interface:**

```python
@dataclass(frozen=True)
class RepeatMetrics:
    k: int
    pass_at_k: float
    mean_at_k: float
    pass_power_k: float
    consistency_gap_at_k: float


def summarize_repeats(passed: Sequence[bool]) -> RepeatMetrics: ...
```

1. Add table-driven failing tests with literal expectations:

```text
[True, True, True]   -> Pass@3=1, Mean@3=1,   Pass^3=1, Gap=0
[True, False, False] -> Pass@3=1, Mean@3=1/3, Pass^3=0, Gap=1/3
[False, False]       -> Pass@2=0, Mean@2=0,   Pass^2=0, Gap=0
```

2. Add a failing empty-input test requiring a clear `ValueError`.
3. Run tests and verify the helper is absent.
4. Implement direct boolean reductions and arithmetic. Preserve `pass_at_k`, `mean_at_k`, and
   `pass_power_k` as distinct fields.
5. Add grouped aggregation by case, stripe, model version, and harness version. Reject mixed repeat
   counts within a group unless the caller opts into partial summaries.
6. Run repeat metrics tests.

## Task 4: Add CLI-compatible serialization without changing the existing battery

**Files:**

- Create: `evaluation/run_v5_framework.py`
- Test: `tests/test_v5_runner_cli.py`
- Modify: `docs/ALIGNMENT_EVAL_BATTERY.md`

1. Write failing CLI tests for `--dry-run`, `--repeats`, `--case`, `--stripe`, `--base-seed`,
   `--model-version`, `--harness-version`, and JSONL output.
2. Require model and harness versions. Make `--dry-run` emit planned cells without model execution.
3. Keep `run_alignment_battery.py` unchanged except for a documentation link; do not merge its
   benchmark result schema into V5 records.
4. Implement the CLI and run dry-run tests.
5. Run one smoke command that produces 935 planned records and verify the exact line count.

## Task 5: Verify and commit PR-3

1. Run all `tests/test_v5_*` evaluator tests and alignment battery regressions.
2. Run serialization and deterministic-repeat tests twice and compare hashes.
3. Run Black, Ruff, and mypy on `evaluation` and modified tests.
4. Run `git diff --check`, inspect the diff, and commit with message
   `feat: add V5 repeat-aware evaluator`.
