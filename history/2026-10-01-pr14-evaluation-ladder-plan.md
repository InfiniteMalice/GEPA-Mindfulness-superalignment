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

Task 1 completed at bf60939208a05b987d3bbf4f35486ea1d39e321c; task-done verified 123
focused tests. Fresh full suite at that commit: 4,443 passed, 18 skipped, 16 warnings in
262.90s. Rebuilt installed-wheel smoke passed exact-content and executable-example checks.

## Independent review and fix pass

One fresh gpt-6-astra/high reviewer examined the whole branch. Verdict: with fixes;
three Important findings, no Critical or Minor findings. All three remain Important after
effect-based grading; the publication-status contradiction is a documentation BLOCK.

- EvidenceReference instance `to_dict` overrides could run callbacks and replace validated
  evidence with private references. Reproduced by
  `test_evidence_serialization_never_invokes_instance_overrides`; replaced instance-method
  serialization with direct validated fields. RED to GREEN. The same fix pass checked
  evaluator serialization: shadowed `__dataclass_fields__` erased identity through asdict.
  `test_evaluator_serialization_ignores_instance_dataclass_metadata` reproduced that bypass
  RED to GREEN; evaluator serialization now also copies only validated fields directly.
  A fabricated source kind could additionally spoof isinstance through a `__class__`
  property and set-membership methods. `test_evidence_kind_cannot_spoof_enum_identity`
  reproduced the bypass RED to GREEN; source kinds now require the exact enum type before
  invoking the legacy validator. This closes the same nested-capture validation finding.
- String subclasses could disguise empty evidence/evaluator identifiers through overridden
  `strip`. Reproduced for the evidence ID and all three evaluator fields by
  `test_nested_identity_strings_cannot_execute_methods_or_hide_blank_values`; exact built-in
  string validation now precedes legacy validators. RED to GREEN. The same checks confirmed
  existing SystemIdentity validation already rejects subclasses without invoking them.
- Official arXiv:2609.25014 Comments states acceptance at SBSeg 2026. A new source-status
  contract test failed on null metadata; registry and reader now preserve that status.
  RED to GREEN. Final focused suite after fixes: 131 passed.

Final: Ruling: Detecting undeclared opportunities, authenticating captures and checking
semantic labels remain host responsibilities because the aggregator receives declared
measurements and references only. Cost if wrong: omitted opportunities or false labels
can produce misleading measured rates despite structurally valid reports.

Final: Ruling: Unfinished latencies retain censoring counts and completion-only summaries;
survival estimates remain outside this API. Cost if wrong: users ignoring censoring can
underestimate delay; the guide explicitly requires reading censoring and missingness.

Final: Ruling: Model effectiveness and population catastrophic-event rates remain unmeasured;
synthetic contract tests justify reporting behavior only. Cost if wrong: users may overgeneralize
fixture results, so the guide and PR make the empirical limitation explicit.

Deferred minors: none. No second reviewer is dispatched; regression tests and the final
full suite verify this single fix pass.

## Final verification

Final: fixed evidence serialization overrides, evaluator dataclass-metadata overrides,
spoofed evidence kinds, and nested identifier subclasses with RED-to-GREEN regressions.
Final: fixed publication-status metadata with a RED-to-GREEN primary-source contract test.
Final full suite after every fix: **4,453 passed, 18 skipped, 16 warnings in 232.03s**.
Focused feature/research/documentation consistency suite: **133 passed**. Feature module:
78 tests, **99% coverage**. Ruff and Black (620 files) passed. CI-scoped mypy plus the new
module, separate logging-schema mypy, Python 3.10 syntax and Python 100-column checks passed.
Wheel and sdist rebuilt after all code fixes. Outside-checkout installation verified exact
module/guide/four research documents, executable guide example, CLI, 17 cases, 65 references,
and import without Torch. No documentation BLOCK or review finding remains open.

Remote main remained 0dfdd59d44d533984d62c442ac44336ff9cb6700. The user authorized the next
draft PR; push targets only codex/evaluation-ladder-rare-events. Remote CI and maintainer
merge remain external. No PR-15 implementation, real-model effectiveness experiment or
runtime authority change is included.
