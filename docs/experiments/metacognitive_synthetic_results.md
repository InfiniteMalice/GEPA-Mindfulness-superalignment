# Metacognitive structural experiment — 2026-11-20

Environment: Linux, CPython 3.12.16, existing offline-capable development environment. No model
weights, model inference, external evaluator call or heavyweight dependency was needed. The
implementation is the staged metacognitive PR series; the final runner is reviewed in PR 7.

Command:

```bash
python -m evaluation.run_metacognitive --enable --battery \
  --output /tmp/metacognitive-results.jsonl > /tmp/metacognitive-summary.json
```

The run covers 17 canonical case coordinates × 11 existing stripe coordinates × 2 deterministic
repeats = 374 cells. Each cell has ten structural challenges plus the healthy control: 4,114
units and 16,456 condition rows. Every C-arm unit matched its separately declared expected
finding kinds and acceptable proposal kinds (4,114/4,114 for each). The summary partitions
agreement by challenge, case and stripe in `metacognitive_synthetic_summary.json`. The raw JSONL
contains identities, source digests, four arm statuses, metric availability and all eight
ablations; its SHA-256 is bound by the summary's `raw_sha256` field.

These results establish deterministic software contracts on deliberately constructed fixtures.
They are not independent evidence of model reasoning quality, semantic equivalence, motive,
robustness transformation effectiveness or generalization to canonical tasks. The held-out repeat
tag protects admission and scoring isolation; it does not turn the same generator into an
independent empirical test set. No alignment score is computed.

Condition A retains a supplied record without exporting new monitor diagnostics. B monitors,
C recommends without executing, and D is SIMULATION_ONLY. Verified task-success changes,
hallucination changes, actual abstention/clarification improvements, evidence-recovery success,
repeated-failure reduction and model generalization remain unavailable, with final behavior
NOT_RUN. Tokens and computational cost were not measured. Source-event inspection counts are
reported separately and are not a proxy for model compute.

Validation before opening the final runner PR: 149 metacognitive tests passed; repository-wide
Ruff and Black passed (881 Python files); both existing CI type-check commands and all eight new
module type checks passed. Wheel and sdist built with isolated build dependencies. A wheel-only
source smoke outside the checkout loaded both packaged demos and documentation, returned
verify_assumption/continue respectively, and imported neither Torch nor NumPy. Wheel bytes for
the runner, evaluation, fixture, monitor and PEO modules matched the checkout. Current-head full
offline regression and remote CI/review remain merge gates, recorded in PR verification comments.

The earlier cumulative PR-6 regression passed 7,472 tests, 11 skips and 162 subtests in 586.73s.
It predates the last review fixes and runner and is not presented as final-head verification.
The fresh whole-branch review found one false-contradiction issue; a RED→GREEN fix preserves
unknown/complementary metadata, and the reviewer's targeted re-review confirmed it resolved.

All outputs remain REGRESSION/HIDDEN_EVAL diagnostics. Reward values, optimizer eligibility,
accepted evidence, canonical cases, the constitution and authority boundaries were preserved.
The required repo-quality-gate skill was unavailable; the maintainer explicitly approved the
documented Superpowers/spec/design/tests/lint/types/package/CI/review fallback before execution.
