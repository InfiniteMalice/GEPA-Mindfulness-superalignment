# Multi-hypothesis state implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline and one fresh whole-branch review.

**Goal:** Preserve externally owned competing explanations with bounded, non-authoritative views.
**Architecture:** Strict immutable records retain hypotheses, assessments and trigger history.
Pure opt-in operations append, validate extensions, compute conservative Pareto diagnostics and page
the existing HypothesisSet overlay. The host retains storage and authority.
**Tech Stack:** Existing Python evidence/context/overlay contracts, standard library, pytest.
**Spec:** `history/2026-10-02-pr16-hypothesis-state-spec.md`.

## Global Constraints

- Exactly 17 canonical cases; Python >=3.10 and 100-column Python; no new dependency.
- Existing competing_hypotheses flag, disabled by default; no new reward or actor deletion tool.
- Full external history survives comparisons and projections. Unknowns/conflicts stay unresolved.
- Host authenticates evidence, scores, storage and callers; no empirical effectiveness claim.

## Review Focus

- Mixed live verifier assessments must not silently select the most convenient verdict.
- Restored or forged successor state must not delete/rewrite retained history or downgrade identity.
- Missing dimensions and ties must not produce false dominance or delete alternatives.
- Long Unicode text, page boundaries and dominated hypotheses must stay bounded and reachable.
- Mutable nested evidence/context records and serializer overrides must not bypass validation.

### Task 1: External history and Pareto projections

**Files:** Create `gepa_mindfulness/verification/hypothesis_records.py`,
`gepa_mindfulness/verification/hypothesis_state.py`, `tests/test_hypothesis_state.py`,
`docs/hypothesis_state.md`, `docs/adr/0017-multi-hypothesis-state.md`. Update pyproject packaged guide,
four recommendation files, source inventory in evaluation/recommendations.py and traceability tests.

**Interfaces:** Records Hypothesis(id, statement, evidence_refs), HypothesisScores(seven optional
dimensions), HypothesisAssessment(id, hypothesis_id, assessor_id, status, scores, evidence_refs,
supersedes=None), HypothesisTrigger(id, kind, evidence_refs), HypothesisState(state_id, context,
source_case_id, protocol_id, complexity_unit, compute_unit, hypotheses, assessments, triggers,
training_eligibility=DEVELOPMENT, revision=0). JSON to_dict/from_dict methods on state.
Operations append_hypotheses(state, *, hypotheses=(), assessments=(), triggers=(), config=disabled),
validate_extension(prior, successor), pareto_hypotheses(state, *, config=disabled),
project_hypotheses(state, *, diagnostic_uncertainty, offset=0, limit=4, config=disabled).

- [x] Write focused tests named by Review Focus plus before/after conflict and supersession cases,
  gate defaults, eligibility, exact numbers, unknown links, roundtrip and legacy overlay integration.
- [x] Run `../venv/Scripts/python.exe -m pytest tests/test_hypothesis_state.py -q`.
  Expected: RED because modules do not exist.
- [x] Implement records and pure operations. Retain exact history prefixes; compare only complete,
  agreed live vectors; projections use insertion order and explicit omission metadata.
- [x] Run focused tests/coverage. Expected: all pass and >=80% across both modules.
- [x] Add guide/example, ADR and five source mappings (three new refs, 68 total). Run traceability
  and reader-consistency tests. Expected: pass; unchanged 17-case/19-recommendation inventory.
- [x] Run Ruff/Black/scoped mypy/Python3.10/100-column checks. Build wheel/sdist before running
  full Torch-enabled suite; installed-wheel exact module/guide/registry/example/CLI smoke.
  Expected: all pass, import does not require Torch.
- [ ] Commit and run task-done with focused tests; review-package base
  `07ee9d7f034b62f50c0250dd81e40ac959af5a36`. Dispatch one gpt-6-astra/high fresh reviewer.
- [ ] Regrade, one Important/Critical fix pass with RED/GREEN and full suite if needed. Record every
  declined-to-judge ruling/cost, then push feature branch and open/attach draft PR.

## Rulings and evidence

Ruling: Proceed under established autonomous next-PR authorization. Cost: reversible branch/draft
may require revision. No merge is authorized.
Ruling: Preserve dominated, challenged and conflicting candidates; exact agreement is required for
comparable scores. Cost: larger frontiers, rather than unsupported preference aggregation.
Ruling: Storage/authentication/concurrency remain host-owned. Cost: callers must protect the
authoritative prior and enforce validate_extension before persistence.
Ruling: Reuse the legacy overlay's uncertainty field only as an explicit host diagnostic input.
Cost: hosts must not interpret it as an inferred aggregate posterior.
Pre-flight: one cohesive task; no cross-task interfaces.

## Validation evidence

- Initial missing-module RED; 42 focused tests GREEN after implementation.
- Separator-collision and nested custom-mapping regressions each RED then GREEN.
- Final focused suite: 53 tests; new-module coverage 94% (records 93%, operations 99%).
- Combined focused/registry/traceability: 121 pass after updating the existing REC-011
  implementation and acceptance-test inventory. The first full run found that stale expectation
  (4568 passed, one failure); the registry correctly includes the new modules and test.
- Ruff, Black, new-module and CI-scoped mypy, Python 3.10 syntax and 100-column checks pass.
- Wheel/sdist built; installed-wheel guide example, module/resource inventory and CLI pass.
  Import smoke confirms 17 canonical cases, 68 sources and no Torch dependency.

- Full Torch-enabled suite: 4569 passed, 18 skipped, 16 warnings in 234.72 seconds.
