# Metacognitive experiment baseline

## Source and environment

Audit date: 2026-11-20. Source revision:
`c363bd332435cd84180d6c488e30a7ad938db195`.
Use the existing project Python 3.12 environment (`../.venv312/bin/` from the repository root).
No new runtime behavior, model invocation, model download or reward change is included in PR-0.

Superpowers design, planning, review and verification workflows are in use. The required
`repo-quality-gate` skill was unavailable; the maintainer explicitly approved the documented
fallback in the design before implementation. The fallback retains task specifications,
implementation notes, tests, lint/format/types, package checks, remote CI and actual result summaries.
No `bd` command was invoked and `.beads/*` was not modified.

## Existing contract baseline

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ../.venv312/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_epistemic_state.py tests/test_hypothesis_state.py \
  tests/test_epistemic_reconciliation.py tests/test_evidence_use.py \
  tests/test_peo_continuity.py tests/test_failure_graph.py \
  tests/test_experimental_overlay_registry.py tests/test_experimental_overlay_flags.py \
  tests/test_17_case_clarifying_abstention.py tests/test_alignment_calibration_metrics.py
```

Result: **405 passed in 1.79 seconds**.

Ruff passed. Black checked 866 files without changes. Both existing CI mypy commands passed:
the nine configured source targets and the isolated logging-schema target.

The complete existing offline suite passed: **7,339 tests, 11 skipped, and 162 subtests** in
589.12 seconds, with 18 warnings. Reproduce with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ../.venv312/bin/python -m pytest -q -p no:cacheprovider --maxfail=1 --disable-warnings
```

The baseline collection ran before the new contract tests were added and includes no new
metacognitive tests. Skips and warnings retain the existing suite behavior; they are not new
metacognitive validation evidence.

## Structural V5 plan

Call `evaluation.v5_runner.plan_v5_cells(case_ids=None, stripe_ids=None, repeats=2,
base_seed=0, model_version="baseline-no-model", harness_version="c363bd33")`.
This produced 374 cells: 17 cases × 11 stripes × two repeats. All fields in
`ExperimentalOverlayConfig()` were false. Serializing `[dataclasses.asdict(cell) for cell in cells]`
with `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)` and hashing the
UTF-8 bytes produced SHA-256
`1fafd40bc6de7c5fa1e48a7b31107a185c0b4fc9e0b89b3a1efa1183ad1597c4`.

The V5 runner currently plans identities; it does not execute a model. This is structural
coverage evidence, not behavioral or alignment performance.

## Existing synthetic response baseline

```python
from evaluation.run_alignment_battery import build_parser, run_battery

args = build_parser().parse_args([
    "--suite", "helm_calibration",
    "--responses-path", "tests/fixtures/alignment_battery/calibration_responses_toy.jsonl",
])
results, summary = run_battery(args)
print(summary)
```

The two matching existing fixture responses yielded one correct and one incorrect answer.
Accuracy was 0.5, Brier score 0.32500000000000007, and ECE 0.45. There were no abstentions;
abstention appropriateness was unavailable. These are existing synthetic-response scoring
results. They do not measure a newly executed model or a metacognitive intervention.

Model experiments and comparisons of alignment improvement: **NOT_RUN**.
