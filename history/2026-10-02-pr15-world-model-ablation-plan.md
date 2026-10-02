# World-model value ablation implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, then one fresh whole-branch review.

**Goal:** Run a matched DIRECT/STRUCTURED/PEO experiment with auditable outcomes and costs.
**Architecture:** Typed contracts and strict preflight feed a bounded offline simulator loop.
The existing PEO exporter validates reconciliation; paired reporting preserves raw results.
**Tech Stack:** Python standard library, existing world/PEO APIs, pytest.
**Spec:** `history/2026-10-02-pr15-world-model-ablation-spec.md`.

## Global Constraints

- Exactly 17 canonical cases; no new dependency or default runtime/training/reward behavior.
- Python >=3.10 and 100-column Python lines. Opt-in, non-TRAIN, public-only actor data.
- One shared model/training contract and budget. Host authenticates identity and compute receipts.
- No real effectiveness or mechanism-recovery claim from deterministic fixtures.

## Review Focus

- Equivalent visible worlds with different latent metadata must yield identical actor inputs.
- A callback mutating caller-owned contracts must not alter later arms or reported matching.
- Resource exhaustion and denied actions must remain failures with complete denominators.
- PEO replay metadata must not expose evaluator labels through the actor projection.
- Paired arithmetic must preserve negative and zero gains; no selective row omission.

### Task 1: Matched world-model experiment

**Files:** Create `evaluation/world_model_contracts.py`, `evaluation/world_model_ablation.py`,
`tests/test_world_model_ablation.py`, `docs/world_model_ablation.md`,
`docs/adr/0016-world-model-value-ablation.md`; update `pyproject.toml`, the four recommendation
registry/reader files, and `tests/test_research_traceability.py` reciprocal link fixture.

**Interfaces:** Consume SyntheticWorld/from_dict, render_world, simulate, expected_judgment,
EpisodeStep/build_episode, EpistemicContext and strict JSON helpers.
Produce frozen slotted Arm, Budget, ModelContract, WorldCase, DecisionInput, Decision,
WorldBackend and `compare_world_models(cases, backend, budget, *, seed=0, enabled=False)`.
Return JSON report with all three arms, raw rows, severe/failure inventories and paired deltas.

- [x] Write tests for the Review Focus plus deterministic benefit/no-benefit controls, exact types,
  invalid catalogs before callback, all budgets, unknown/foreign/effectful actions, eligibility,
  prospective prediction requirements and validated PEO history.
- [x] Run `../venv/Scripts/python.exe -m pytest tests/test_world_model_ablation.py -q`.
  Expected: RED because the comparison API is absent.
- [x] Implement the two cohesive modules using strict snapshots before callbacks, public JSON
  inputs, read-only auxiliary actions and bounded rollout. Reuse PEO events without latent exports.
- [x] Run the focused suite and new-module coverage. Expected: PASS and >=80% coverage.
- [x] Add guide with runnable deterministic example, ADR, source mappings and packaged guide.
  Run focused plus traceability and documentation-consistency tests. Expected: PASS, 65 references.
- [x] Run Ruff/Black, scoped mypy, Python3.10 syntax/line checks, build then installed-wheel
  smoke and full Torch-enabled suite. Expected: all checks pass, 17 canonical cases unchanged.
- [x] Commit feature, run task-done with focused tests, create review package from base
  `6b0d11c642303bdb737719a96f36d01acc8c643b`, dispatch gpt-6-astra/high reviewer once.
- [x] Regrade findings; one Important/Critical fix pass if needed, with RED/GREEN regressions
  and final suite. Record every declined-to-judge ruling and cost.

Integration: push the feature branch and open/attach a draft PR under existing user authorization.

## Execution rulings and evidence

Ruling: Existing autonomous next-PR authorization supplies execution approval; implement inline
without another menu. Cost if wrong: a reversible feature branch and draft PR require revision.
Ruling: Bound this first ablation to reveal-only investigation and a target decision. Generalized
effectful planning needs a separate objective contract; cost is narrower external validity.
Ruling: Match declared resource caps and report actual usage plus local overhead separately.
Provider-wide hard quotas require the host; cost is reliance on authenticated host receipts.
Pre-flight: one cohesive task; no inter-task interface conflicts.

Initial verification: 50 focused tests, 98% new-module coverage. Full Torch-enabled suite:
4,516 passed, 18 skipped, 16 warnings (225.21 seconds). Ruff, Black (623 files), scoped mypy
(11 modules plus logging schema), Python3.10 syntax/100-column checks and diff whitespace pass.
Wheel/sdist and installed-wheel module/guide/registry bytes, executable example and CLI smoke pass;
17 canonical cases, 65 references and no Torch import in the new API. Documentation precision review
found a registry/reader venue punctuation mismatch; corrected and verified with all 8 reader checks.
No outstanding documentation BLOCK. Deterministic controls cover positive/negative/zero gains;
the documented two-world policy yields success in all arms and zero paired behavioral gain.

## Independent review and final quality gate

One fresh gpt-6-astra/high reviewer inspected 6b0d11c..5947185, the spec, plan and ledger.
No Critical, Important, Minor or documentation BLOCK findings. Independently ran 98 ablation and
research tests, a three-step privacy probe and whitespace checks. The probe preserved actor inputs
and remaining budgets when hidden truth, another actor's visibility, world/case IDs, provenance,
parent digest, evidence IDs/source kinds and eligibility changed. No fix pass was necessary.

Rulings on every declined-to-judge item:

1. Model/checkpoint/training identity and compute receipt authenticity: retain the explicit host
   audit boundary; no real adapter/provider evidence was supplied. Cost: dishonest declarations
   invalidate a model study even though local validation passes.
2. Provider quotas, callback timeouts and session isolation: retain host enforcement. Cost:
   callbacks can overspend, hang or share state unless the host applies controls outside this API.
3. Learned effectiveness, mechanism recovery and external dataset contamination: retain unmeasured
   status and external split audit. Cost: deterministic controls provide no empirical generalization
   or causal mechanism claim; real model studies remain separate work.
4. General planning with effectful auxiliary actions: retain the reveal-only target-decision scope.
   Cost: the experiment cannot establish value for general state-changing planning.

Deferred minor findings: none. No default training/reward/authority change, new dependency or
canonical case was introduced. Full suite and installed-wheel results above apply to the reviewed
implementation; this final update records the review only.
