# Phase 2 PR 1: constitutional graceful failure

Maturity: **Supported** document-consistency checks; constitutional guidance is normative.
Behavioral or alignment effectiveness is **experimentally unqualified**.

## Change and exact scope

The [exact constitutional diff](phase2_constitution.diff) adds 69 lines in five insertion regions.
The requested Section 17 subsection appears immediately before its existing positive scoring
signals. Six positive examples, six negative behavioral examples, five metacognitive questions
and the no-new-reward-authority caveat extend that section. Section 21 adds the requested sentence
immediately after “When blocked, respect constraints.” All original headings/prose remain intact.

The [preserved specification](../../history/2026-11-04-phase2-required-text.md) supplies the
authoritative addition. Tests remove exactly the new material and compare the remaining bytes
with the frozen baseline SHA-256. This verifies preservation of all 21 numbered sections, the
three co-equal imperatives, four wisdom values and every other original line, including pre-existing
encoding artifacts outside this additive request.

## Verification and review

`tests/test_graceful_failure_constitution.py` has four explicit acceptance/preservation checks.
Before insertion all four failed for the expected missing guidance. After insertion the new
checks and existing constitution/schema, principle, documentation and V5 consistency tests pass:
29 passed. Ruff and Black pass for the added test. `git diff --check` passes.

A fresh reviewer independently verified the text against the committed specification and
performed a byte-level insertion-only diff against PR 0's head. Fourteen independently rerun
focused tests passed; no critical, important or minor findings remained. Review establishes
document/software consistency, not improved model behavior. The complete regression run passed:
**7,068 passed, 11 skipped, 18 warnings, 162 subtests passed** in 413.83 seconds, exit code 0.
The [complete console log](phase2_pr1_pytest.log) preserves the run and skip reasons. The unchanged
hardware/optional-backend skips remain limitations, not qualification. The four additional passing
tests account for the increase over PR 0's baseline. The full JUnit source remains in the task's
`phase2-pr1/full-pytest.xml` artifact directory. Ruff, Black and a targeted mypy check pass.

The required named quality-gate skill is unavailable; the maintainer-approved documented fallback
uses specification, design, implementation, failing/passing tests, formatting/lint, fresh review
and this gate summary. No beads operations or reward/runtime changes are introduced.

## Migration and limitations

No Python runtime API, canonical case, stripe, optimizer component, checkpoint or configuration
format changes. Consumers reading the Constitution receive its extended prose. The normative
addition does not itself train a model, establish infeasibility, authorize an action or grant
optimizer credit. Later CVA and behavioral studies must independently test those claims.
