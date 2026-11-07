# Phase 2 PR 3: offline graceful-failure benchmark

Maturity: **Experimental** finite software; artifacts **Diagnostic**. Model-behavior efficacy is
**Unqualified**. This stage implements benchmark fixtures and scoring, not training or a claim
that any model has improved.

## Stage and file inventory

Base: local PR 2 commit `89167c2`. No backward-incompatible migration is required. All APIs are
additive and require explicit opt-in. Ordinary inference, canonical cases, stripe registry,
verified-process reward, accepted evidence admission and runtime grants remain unchanged.

New source: `synthetic_data/cva_fixtures.py`, `evaluation/cva_benchmark.py`.
New tests: `tests/test_cva_benchmark.py`.
New guide: `docs/graceful_failure_benchmark.md`.
New disabled configuration example: `configs/cva_benchmark.example.json` (informative API settings,
not an automatically loaded runtime configuration).
Modified: `docs/README.md`, `pyproject.toml`, `tests/test_controlled_evolution_package.py` for guide
discovery and wheel-resource verification. Validation artifacts accompany this report.

The [API and protocol guide](../graceful_failure_benchmark.md) describes exact records,
host/simulator trust boundaries, metric opportunities, limitations and the qualification CLI.
Nine fixtures cover the six requested situations plus evidence/authorization updates and bypass
pressure. Public prompts omit evaluator gold labels. Full and technical finite proof scopes stay
separate; failed search, unknown facts and denied authorization never self-certify impossibility.
The overlay joins existing V5 cells through `CausalVariant`, preserving model/harness versions,
seeds and all coordinates. Empty captures retain missing/null metrics and all 17 case groups.

## Test-first and fresh review evidence

Initial collection failed on the absent benchmark module. Subsequent targeted red/green checks
exposed and corrected inappropriate-abstention scoring, unauthorized alternative omission,
empty-trace reporting credit, unsupported wire-array coercion, and package resource omission.
The fixture qualification CLI retains raw certificates/checks and reports model experiments as
`NOT_RUN`. Missing/truthy/throwing/mutating authentication, forged/stale/over-budget observations,
unplanned/duplicate captures, bad proof bindings and prompt mutation fail closed.

A fresh-context reviewer ran 72 focused tests and identified one important defect: known-useless
authorized actions counted as useful alternatives, suppressing premature-abandonment detection.
The reproducer failed before the fix. Matched regression cases now distinguish route zero's
known uselessness from route one's verified completion or unresolved information gain. The final
focused suite passes **75 tests** (31 benchmark plus 44 existing CVA tests), in 15.44 seconds.
No remaining critical/important findings or deferred minor findings were reported. The review is
an independent code inspection, not experimental evidence of improved model alignment.

Reviewer topics explicitly outside this stage were resolved as follows: model training effects
and natural-language sincerity remain unmeasured; host V5 label selection and identity assurance
remain explicit trusted-host obligations; hostile arbitrary Python is outside the library threat
model; full-suite and packaging validation are performed separately below. These scope decisions
would become material limitations if these results were used as real-model or real-world claims;
the guide and artifact labels prohibit that inference.

## Quality gate

The maintainer-approved fallback for unavailable `repo-quality-gate` applies: stage specification,
existing design/ADR, test-first implementation, formatting/lint/type checks, fresh review, and this
recorded gate summary. Beads automation was not invoked.

Repository-wide Ruff passes. Black checks all 831 Python files. Both existing CI mypy invocations
pass, as does explicit checking of the two new modules. The wheel-resource test passes after
the new guide was added to package data. Fresh ordinary and optional CVA imports load neither
torch nor transformers. The complete protected regression run passed: **7,143 passed, 11 skipped,
18 warnings, 162 subtests passed**, 429.21 seconds, exit 0. The same 11 optional hardware/backend
skips as the frozen baseline remain unqualified. Full stdout and compressed per-test JUnit XML
are retained alongside a source/artifact hash manifest.

## Reproducibility and research status

The deterministic qualification artifact preserves all original fixture definitions, public views,
proposed witnesses, exclusion certificates and independently checked candidate and scoped
feasibility results. It records source hashes, Python version, seed zero and a declared synthetic
epoch. All nine authored fixture expectations match the independent finite checker; wh004 stays
unresolved. wh007's bounded search still fails to find a candidate without proving infeasibility.

Experiment A's finite oracle/measurement software is exercised. Experiments A/B on actual frozen
language models, comparative training, cross-seed confidence intervals and Experiments C–H
remain **NOT_RUN** here. The tests measure software correctness only. No diagnostic feature,
structured score or toy oracle becomes an optimizer-facing reward or grants execution authority.
Original red/green and environment logs remain in the task workspace; final regression and
qualification artifacts are retained with this report.

## Proposed PR description

Add nine offline white-hat feasibility fixtures and a default-off diagnostic benchmark over
unchanged V5 cells. Separate verified completion, scoped contradiction, unauthorized solutions
and evidence-limited failed search using existing CVA witnesses/exhaustion checks, with
host-authenticated structured captures and null unmeasured rates.

Preserve original records and public/evaluator separation; deny every action outside a closed
simulator without external execution. Ship a qualification CLI, typed API guide, adversarial
boundary tests and packaged documentation. No model training, inference-default changes,
optimizer signal or empirical alignment-improvement claim is introduced.
