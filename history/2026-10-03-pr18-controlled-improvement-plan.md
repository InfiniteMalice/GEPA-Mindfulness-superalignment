# Controlled improvement intake implementation plan

> Required workflow: superpowers:executing-plans, inline implementation with one fresh final reviewer.

**Goal:** Bind PEO diagnostics to recorded corrections before controlled candidate review.
**Architecture:** One read-only CoevolutionStore method and one opt-in diagnostic adapter.
**Tech stack:** Python dataclasses, existing SQLite stores, pytest, existing registry YAML.
**Spec:** history/2026-10-03-pr18-controlled-improvement-spec.md

## Global constraints

Python 3.10; 100-column Python; no new dependencies; exactly 17 canonical cases.
No default runtime, optimizer admission, reward or persistence changes. No bd/.beads operations.
Keep planning in history. Reuse the existing checkout on a feature branch.

## Review focus

- Mixed action/run/version/evidence identity must not pass intake through unrelated evidence.
- Mutable nested records or serializers must not turn malformed diagnostics into authority.
- Missing measurements and unsupported roots must not become zero uncertainty or supported cause.
- Catalog changes after intake must still face registration and existing acceptance gates.
- Extracted legacy records must not be described as retaining the envelope's training restriction.

### Task 1: Validated intake, research reconciliation and integration

**Files:** Create gepa_mindfulness/controlled_improvement.py and
tests/test_controlled_improvement.py; modify gepa_mindfulness/coevolution.py,
docs/controlled_evolution.md, docs/recommendations/{registry.yaml,references.yaml,
UNIFIED_RECOMMENDATIONS.md,RESEARCH_TRACEABILITY.md}, evaluation/recommendations.py and registry
tests; create docs/adr/0019-controlled-improvement.md.

**Interfaces:** Consumes CorrectionProposal, FailureGraph, EpistemicStateEstimate and catalog
source events. Produces correction_source_events(correction) -> tuple[EventEnvelope, ...],
TriageDiagnostics and assess_improvement(...) -> dict[str, Any], with exact spec fields/rules.

1. Write tests for the three routes with matched inputs, each threshold/missing dimension,
   exact typing, observable evidence, run/repeat/model/harness/action/prediction mismatch,
   mutated correction, wrong source epoch, source version mismatch and catalog tampering.
   Test no writes, complete-envelope admission rejection, extraction caveat, and explicit
   registration of sandbox result followed by unchanged receipt requirements.
2. Run ../venv/Scripts/python.exe -m pytest tests/test_controlled_improvement.py -q.
   Expected: RED because new adapter/store method are missing.
3. Implement the spec using existing snapshot, graph and coevolution validators.
4. Run the task suite and controlled-evolution integration tests. Expected: all pass.
5. Document routing/lifecycle boundary and independently verify primary metadata for ten named
   sources. Add absent sources, update REC-010 and exact reader/traceability mirrors. Distinguish
   CARE competence-aware length shaping from Qwen-Planner's different CARE acronym.
6. Run focused, registry, full offline CPU suite, Ruff/Black/scoped mypy, build and installed-wheel
   smoke with no Torch. Expected: all pass; new module statement coverage >=80%.
7. Commit; task-done uses ../venv/Scripts/python.exe -m pytest tests/test_controlled_improvement.py -q.
   Dispatch one whole-branch reviewer; rule on all declined judgments; fix important findings
   with RED/GREEN and full suite. Push feature branch and open draft PR18.

## Decisions

Ruling: Continue the established autonomous staged-PR workflow, using inline implementation and
a draft PR as the reviewable result — repeated next-PR instructions authorize this — cost if
wrong: design changes before merge.
Ruling: Use fixed documented conservative routing thresholds and host declarations, with no
empirical effectiveness claim — no calibrated local protocol exists — cost if wrong: excess
investigation or human review, or an under-sensitive diagnostic recommendation.
Ruling: Reuse catalog gates and leave runtime replay/deployment and private promotion to their
owners — avoids duplicate authority — cost if wrong: hosts still need runtime integration.
Ruling: Retain ignored workflow scratch because prior automatic approval review rejected its
cleanup — do not retry the restricted action — cost: local ignored files remain.

Self-review: All spec contracts map to Task 1; no inter-task dependencies or new lifecycle.

## Validation before independent review

- RED: new intake module missing, followed by 72 passing intake tests; adapter coverage 100%
  (78 statements). A fixture referenced an absent evidence enum; corrected to LATENT_STATE.
- Full offline CPU suite: 4697 passed, 18 skipped, 16 warnings in 270.98 seconds.
- Research/recommendation/reader checks: 90 passed; after moving the PR18 paragraph to its
  REC-010 section, the seven reader checks passed again.
- Ruff clean; Black 630 files unchanged; CI scoped mypy plus new adapter clean (10 files),
  logging schema separately clean. Five changed Python files pass Python 3.10 AST/100 columns.
- Wheel and sdist built. Installed-wheel smoke verifies exact module/guide/registry resources,
  disabled default, 17 cases, 75 sources and no Torch import; CLI help passes.
- Ten named primary sources checked; six added, four existing updated. CARE identity is
  disambiguated from the existing Qwen-Planner source. No claims of empirical local benefit.
- Documentation precision pass: explicit actors, routing thresholds, snapshot semantics,
  complete-envelope eligibility and host lifecycle responsibilities; no unresolved BLOCK.

## Independent review and single fix pass

Ruling: New reviewer creation hit the agent-thread limit; reuse the independent PR17 reviewer
for a new PR18 review — it had not seen PR18 implementation and remains separate from the author —
cost if wrong: prior-PR context may bias review.

The reviewer inspected all 13 files, source metadata and reused validators, independently ran
148 tests, and reported one Important finding and no Critical or Minor findings. Nested
CorrectionProposal evidence serializers could disguise private provenance and mutate triage
before snapshotting. Regraded Important: a misleading sandbox recommendation is possible,
although existing acceptance gates still prevent deployment authority.

Eleven regression cases failed before the fix: source/teacher evidence mutation across intake,
source-event inspection and direct serialization; callback mutation of triage; and substituted
nested evidence/graphs. The fix reconstructs raw CorrectionProposal fields through its existing
validators, serializes the detached canonical snapshot, and compares against the ORIGINAL
binding. Valid proposal serialization remains compatible. No second review is requested.

Final: Ruling: Calibration and empirical improvement remain unmeasured — this is opt-in
investigation guidance — cost if wrong: diagnostic recommendations may be unhelpful.
Final: Ruling: Evidence authenticity and causal sufficiency remain host-verified — declared
linkage is not truth — cost if wrong: a false diagnosis may reach sandbox review.
Final: Ruling: Auxiliary evidence beyond the required localized intersection is allowed —
hosts interpret its relevance — cost if wrong: unrelated auxiliary evidence can distract review.
Final: Ruling: Actual edit size and attribution remain host-reviewed — a component label does
not count edits — cost if wrong: a bundled change may be hard to attribute or regress.
Final: Ruling: Direct registration remains callable outside optional intake — durable gates
still own acceptance — cost if wrong: a host can omit these additional diagnostic checks.
Final: Ruling: Legacy extraction loses DEVELOPMENT metadata — retain the complete envelope
at admission — cost if wrong: stripped diagnostics can enter legacy training admission.
Final: Ruling: Catalog concurrency/ACLs and deployment/replay/retirement remain external —
registration revalidates snapshots and runtime owners control execution — cost if wrong:
compromised storage or missing runtime controls can invalidate the assurance boundary.
Final: Ruling: Arbitrary process-wide monkeypatching is outside the typed-record contract —
this API is not a Python sandbox — cost if wrong: same-process code can replace validators.

Deferred minors: none.

Final: fixed caller-controlled correction serializers — 11 regression cases RED to GREEN;
intake plus coevolution 111 passed. Final full offline suite: 4708 passed, 18 skipped,
16 warnings in 240.22 seconds. Ruff/Black clean; intake and coevolution mypy clean;
Python 3.10 syntax/100-column and whitespace checks clean. Rebuilt wheel and sdist;
installed-wheel smoke matches final modules/docs/registries and imports no Torch.
Task 1 complete; one independent review and one fix pass complete. No deferred minors.
