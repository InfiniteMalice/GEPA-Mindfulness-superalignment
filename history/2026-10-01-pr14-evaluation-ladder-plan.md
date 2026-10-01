# Evaluation Ladder Implementation Plan

> Use superpowers:executing-plans inline, with one fresh whole-branch reviewer.

**Goal:** Report seven competencies independently and retain rare severe failures.
**Architecture:** One offline typed aggregation module over host-declared probes and
observable captures; reuse calibration and identity contracts. No runner or reward changes.
**Tech Stack:** Python 3.10+, standard library, existing repository modules, pytest.
**Spec:** history/2026-10-01-pr14-evaluation-ladder-spec.md

## Global Constraints

Exactly 17 canonical cases. Opt-in and fail closed. Non-TRAIN reports. No new dependencies,
private reasoning, numeric uncertainty rewards, runtime authority, or effectiveness claims.
Python lines <=100; Google public API docs; tests have docstrings. Planning stays in history.

## Review Focus

1. Host denominator omissions and severe probes missing observations must remain visible.
2. Mixed units, extreme finite scalars and censored latency must not imply favorable means.
3. Mutable nested identity/evidence records and subclass validation overrides must fail closed.
4. A high behavioral score must never become a stage promotion or mechanism claim.
5. Installed-wheel behavior and research metadata must match source checkout behavior.

### Task 1: Implement independent ladder reports

**Files:** Create evaluation/ladder.py, tests/test_evaluation_ladder.py,
docs/evaluation_ladder.md, docs/adr/0015-evaluation-ladder.md. Update four research registry
documents, evaluation/recommendations.py, tests/test_research_traceability.py, pyproject.toml.
**Interfaces:** Consumes EvidenceReference, TrustedEvaluatorContract, SystemIdentity,
TrainingEligibility, existing Brier/ECE. Produces the API defined by the spec.

1. Write contract tests for all acceptance cases, including strict and mutated records.
2. Run `../venv/Scripts/python.exe -m pytest tests/test_evaluation_ladder.py -q`.
   Expected: FAIL because the ladder API is absent.
3. Implement typed records, a fixed metric catalog, validation, independent summaries,
   severe inventory, and deterministic digests. Write exact guide definitions and source links.
4. Run focused tests with research traceability and recommendation documentation consistency.
   Expected: PASS. Measure new module coverage >=80%.
5. Run full suite, lint, format, scoped mypy, syntax/line checks, wheel/sdist and installed smoke.
   Expected: PASS; preserve 17 cases and register 65 sources.
6. Commit feature; task-done repeats focused tests. Generate review package against
   0dfdd59d44d533984d62c442ac44336ff9cb6700. Dispatch one gpt-6-astra/high fresh reviewer.
7. Fix Important/Critical findings in one RED-to-GREEN pass, repeat necessary checks and
   full suite if code fixes occur, record rulings and deferred minors. Push branch and draft PR.

## Execution evidence

Pre-flight: no shared interfaces between tasks; one cohesive reporting deliverable.

Initial RED: importing the absent evaluation.ladder API failed as expected. GREEN: 68
contract tests passed. Edge-case investigation then reproduced OverflowError for three
maximum finite residuals; `test_largest_finite_measurements_have_a_finite_accurate_mean`
failed before replacing division-plus-fsum with statistics.mean and passed afterwards.
Final focused feature run: 69 passed, 99% module coverage. Research and documentation
consistency run before the extra numeric test: 122 passed. Registry proof-language check
was corrected by replacing a negated privacy-guarantee phrase with privacy protection.

Full suite before the mean edge-case fix: 4,442 passed, 18 skipped, 16 warnings in 230.34s.
A fresh full suite is running for the final numeric fix; its result will be recorded before
shipping. Ruff, Black (620 files), scoped mypy (10 modules plus logging schema), Python
3.10 syntax and 100-column checks passed. Wheel and sdist built; the first installed-wheel
smoke verified exact module/guide/four registry documents, executable example, 17 cases,
65 references and no Torch import. Rebuilt wheel smoke follows the final numeric fix.

Documentation precision review: no unresolved BLOCK. Metric opportunity definitions,
failure polarity, censored/missing counts, observed-only rates, units and host obligations
are explicit. Contract tests cover API requirements; the guide gives the manual protocol
review for semantic judging, provenance and sampling assumptions.
