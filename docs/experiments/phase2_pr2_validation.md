# Phase 2 PR 2: constraint and certificate contracts

Maturity: **Experimental** finite verification software; all exported records are **Diagnostic**.
Software checks establish neither improved model alignment nor real-world authorization/truth.

## Implemented boundary

The [typed API guide](../constraint_verified_alignment.md) documents the optional CVA records,
candidate verification, scoped feasibility verification, witness/exhaustion construction, bounded
search, evidence-limited noncompletion and existing action/prediction bindings.

`ConstraintSet` retains G/T/P/E conditions separately. Exact finite candidate success requires
the complete conjunction. Candidate rejection cannot become a task-infeasibility verdict.
Technical G/T witnesses are distinct from authorized G/T/P/E completion; complete finite exclusion
coverage can establish only its exact formalized scope. Missing/unavailable/unsupported evidence
remains explicit. Failed search, including exhausted search, does not self-certify impossibility.

The trusted host authenticates complete detached inputs, applicable authority, evidence and actual
reviewer independence. The deterministic checker separately recomputes predicates and coverage.
All records preserve versions/digests, original references, checker/host identity, assumptions,
uncertainty, advisory next actions and aware timestamps. They confer no accepted-store admission,
runtime grants, optimizer inputs or constitutional overrides. Current V5/authority/reward/evolution
APIs and serialized formats are unchanged.

## Test-first checks and independent review

Initial CVA tests failed because the new APIs were absent. The current focused CVA suite has
44 passing cases, covering witnesses, all four relation operators, technical/full scope, failed
search, incomplete/duplicate proof coverage, stale/foreign certificate bindings, private/conflicting
evidence, mutation, unavailable/throwing/truthy callbacks, clocks, capacity bounds and record
round trips. Related verification/runtime/governed-evidence/reward/constitutional checks pass:
280 tests. A focused coverage run reports 93% statement coverage across the three new modules;
coverage is a software diagnostic, not a research-validity score.

Fresh review found no remaining critical/important issues after the full-scope action binding
was corrected. A technical witness retains its technical interpretation; a full-validity witness
bound to another action scope fails closed. The reviewer additionally checked 28 signed
two-variable arithmetic/exhaustion cases against separately written finite predicates. These are
authored mathematical software checks, not model-behavior measurements.

The review's remaining uncertainty-composition issue was reproduced by a failing test and fixed:
valid host judgments matching generated uncertainty sentences now produce an unresolved report
with every distinct judgment retained. Other failing/passing boundary tests established rejection
of conflicting evidence before callbacks and required certificate digests on conclusive records.
Valid maximum-size uncertainty/evidence/predicate inputs now fit complete output reports.

Wheel acceptance initially failed because the new guide was not bundled. Adding its package-data
entry made the installed-wheel test pass; the test also requires all three CVA implementation
modules. Ruff and targeted mypy pass, Black checks six relevant Python files, and local Markdown
links resolve. A fresh CVA import loads neither PyTorch nor transformers.

An earlier complete regression iteration passed 7,110 tests, with 11 skips and 162 subtests.
The final complete run after review fixes passed **7,112 tests, 11 skipped, 18 warnings and
162 subtests** in 413.54 seconds, exit code 0. The [complete console log](phase2_pr2_pytest.log)
preserves this result and the unchanged optional/hardware skip reasons. Original run logs, focused
red/green/coverage results and complete JUnit evidence remain in the task's `phase2-pr2` artifact
directory. No native CUDA, Vulkan, Mojo, external model or empirical alignment qualification is
inferred from these software checks.
The approved quality-gate fallback supplies specification/design, red/green tests, lint/format/type
checks, independent review and this recorded gate; the named skill remains unavailable.

## File inventory and migration

New: `verification/cva_records.py`, `verification/cva.py`, `verification/cva_finite.py` under
`gepa_mindfulness`; `tests/test_cva.py`, `tests/test_cva_records.py`; the CVA guide and this report.
Modified: documentation index, package-data list and existing wheel-resource test. No ordinary
runner or root package initializer imports/activates CVA. New records begin at explicit v1 schemas;
legacy records keep their existing wire formats and original digests.

Only the closed finite integer fragment is implemented. External SAT/SMT proof formats,
statistical/approximation certification, external security testing, universal ethical formalization,
training, deployment and empirical alignment gains remain unsupported by these contracts. The
next stage supplies offline white-hat fixtures and orthogonal V5 benchmark captures. Preserve all
original artifacts and pass complete diagnostic envelopes to existing eligibility checks.
