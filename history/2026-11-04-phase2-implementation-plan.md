# Phase 2 implementation plan

Maturity: **Experimental** plan; execution results belong to each stage's recorded quality gate.

Goal: Add optional constraint-verified outcomes and falsifiable mechanistic studies without
replacing established V5, reward, evidence or promotion owners.

Architecture: [ADR 0026](../docs/adr/0026-cva-causal-interpretability.md). Python 3.10+ records
and deterministic finite checkers; lazy optional PyTorch/transformers for actual interventions.
The user supplied the stage order, acceptance criteria and authorization to proceed/merge passing
PRs. The missing named quality-gate skill uses the maintainer-approved documented fallback.
This document defines implementation decisions, not a parallel issue tracker or beads workflow.
The [required-text capture](2026-11-04-phase2-required-text.md) preserves the verbatim constitutional
addition, scoring/metacognitive language, final principle and lettered A–H evaluation matrix.
Read its applicable requirements before implementing each stage; the original user specification
remains authoritative if an implementation detail in this plan conflicts with it.

## Global constraints and review focus

Keep exactly 17 canonical cases and 21 numbered constitutional sections; preserve the three
co-equal imperatives and four wisdom values. All new execution paths require explicit opt-in.
Diagnostics/holdouts cannot become optimizer input, hidden-thought penalties or runtime grants.
Original evidence and null results survive. No model-generated content self-certifies.

Review adversarial serialized inputs (wrong fields/versions, bool-as-int, nonfinite numbers,
oversize envelopes), stale subject digests, absent/changed authorization, prediction/evaluation
leakage and tensor-hook mismatches. Each owning stage includes rejection tests for these classes.

## Stage 0: audit and design

Files: `docs/experiments/phase2_pr0_audit.md`, `docs/adr/0026-cva-causal-interpretability.md`,
`docs/phase2_research_sources.md`, this plan and preserved baseline evidence.
Read existing APIs, freeze SHA and full test/log/dependency results, distinguish missing and
unqualified methods, verify primary references. No production behavior changes. Review design
against the supplied scope. Documentation tests and unchanged-source CI checks gate the PR.

## Stage 1: constitutional update

Modify only the requested portions of `docs/GEPA_Mindfulness_Constitution.md`; add a focused
constitutional consistency test. Verify the test fails before insertion. Insert authoritative
language before section 17's positive signals, then requested scoring/metacognitive guidance and
section 21 principle. Compare existing section bodies outside the authorized insertion regions.
Run constitutional/schema, documentation and imperative tests. Save an exact diff artifact.

## Stage 2: CVA contracts and trusted checker boundary

Create `gepa_mindfulness/verification/cva_records.py`, `cva.py`, `cva_finite.py`,
`tests/test_cva_records.py`, `test_cva.py` and `docs/constraint_verified_alignment.md`.

Interfaces: typed `CandidateVerdict`, `FeasibilityVerdict`, `GuaranteeKind`, `ConstraintSet`,
`OutcomeCertificate`, `VerificationRecord`, `NoncompletionAssessment`; separate
`verify_candidate` and `verify_feasibility` entry points, explicit `enabled=False` defaults.
Reuse `DiagnosticRecord`, evidence references, trusted evaluator identities and original action
IDs. Freeze input digests and checker identity, completeness assumptions and aware host time.
Records are inert DEVELOPMENT data. Only host-configured checking can produce accepted results;
missing/throwing/unavailable reviewers remain unresolved, and no serialized claim grants trust.

Tests first: valid witness, invalid witness, contradictory finite constraints with exact scope,
unknown versus UNSAT, failed search, missing authorization, unsupported certificates, forged/stale
bindings, unknown ethics, record round trips/mutation, resource bounds and optimizer rejection.
Run related verification/runtime/reward regressions before committing the independently reviewed PR.

## Stage 3: finite and white-hat benchmarks

Create `synthetic_data/cva_fixtures.py`, `evaluation/cva_benchmark.py`, a disabled config,
`tests/test_cva_benchmark.py` and fixture documentation. Extend the existing diagnostic overlay
owner only as required; keep canonical cases/stripes unchanged.

Interfaces: six required offline white-hat fixtures plus changed-evidence/bypass variants,
public actor view separate from expected results, independently checked finite expected verdicts,
V5-cell-bound overlays and explicit metric opportunities. `evaluate_cva_benchmark(..., enabled=False)`
retains missing outcomes/null rates rather than fabricating failures or model measurements.
Tests first: all fixtures, authorization transition, no endpoint execution/network/commands,
bounded input parsing, original V5 coordinate retention and evaluator-only oracle isolation.

## Stage 4: native-neuron discovery

Create focused modules under `interpret`: `native_records.py`, `native_neurons.py`,
`explanation_tests.py`; tests `test_native_neurons.py`, `test_explanation_tests.py`.
Document the protocol in `docs/mechanistic_research.md` and extend existing graph evidence with
backward-compatible defaults. Reuse existing graph comparison and mechanistic audit references.

Interfaces: exact `HookSpec`, measurement records, regime sampler, `ExplanationHypothesis`, split
manifest and frozen permitted-feature predictor. Architecture adapters explicitly define GPT2
and a tested gated family; reject every unimplemented hook/family. Capture actual native values,
positions, tensors, transforms and upstream/downstream references. Rule and bounded learned
predictors never receive evaluation target activations.
Tests first: high/weak/zero/negative sampling, position offsets, correct gated semantics, independent
splits/content duplicates, held-out metrics, competing hypotheses, leakage and unsupported backends.

## Stage 5: simulation fidelity

Create `interpret/activation_simulation.py`, `tests/test_activation_simulation.py` and a
reproducible CPU example. Interface: explicit `simulate_activations` on a copied frozen model,
predeclared hooks/positions, independently computed predictions, per-example activation errors,
actual `KL(original || substituted)`, shifted held-out NLL and declared behavioral/calibration
outputs. Freeze predictor before locked evaluation. Include identity, ablation, mean, random,
shuffled/matched-complexity controls and groups.
Tests first: exact tensor shape/dtype/device/position rejection, no hook leak after exception,
source weights/mode/RNG preserved, known KL calculation, identical-input controls, negative/weak
regimes, frozen-model condition and deliberately wrong constant with low single-neuron impact.

## Stage 6: causal circuit verification

Create `interpret/causal_interventions.py`, `mechanistic_evidence.py` and corresponding tests.
Interfaces: precommitted hypothesis/predicted effect and pair identity, intervention executor,
signed measured effect/control records, host admission and explicit edge evidence status.
Implement supported activation replacement/patch/ablation/restoration, matched swaps, path
patching and mediation through exact hooks on copied frozen models; unsupported modes fail.
Tests first: prediction-before-measurement, matched context/model identities, controls, necessity/
sufficiency, path selection, replication, contradictions/nulls and no reviewer-only status upgrade.
Retain uncertain edges and original graph API compatibility; do not relabel legacy gradients causal.

## Stage 7: alignment-specific studies

Create `evaluation/mechanistic_studies.py`, targeted controlled fixtures/config and tests.
Compose existing causal diagnostics, public dual-path records, CVA overlays and V5 cells for
honesty, correct/incorrect IDK, clarification/resumption, graceful failure, laundering, role-play/
mistake specificity and constitutional conflicts. Preserve evidence-state and valid-authorization
changes as distinct factors. Behavior capture precedes internals. Discovery/prediction/executor/
verifier/skeptic/registrar roles stay explicit; deterministic tools are acceptable, role labels
are not independent verification. Tests first: oracle isolation, diagnostic-only no-weight-update
condition, new evidence/authorization sensitivity, retained rejected hypotheses and protected joins.

## Stage 8: comparative interpretation

Create `interpret/comparative.py`, tests and comparison documentation/config. Interfaces declare
native, contextual native, optional SAE/other decomposition adapters with matching model, held-out
data and intervention budgets. Measure decomposition reconstruction error separately, predictive
fidelity, substitution KL, causal agreement, complexity, unexplained behavior, OOD and actual costs.
Tests first: unmatched budget/model rejection, unavailable arms retained as unrun, reconstruction
error independent of explanation error, no superiority claim from missing controls or one fixture.

## Stage 9: reproducibility and reports

Create reporting/registry adapters, reproducible offline CLI/example, result artifacts and
`docs/experiments/phase2_results.md`. Reuse existing study manifests, capture/audit and dependency-
cluster uncertainty rather than inventing independent-sample counts. Record experiment/code/model/
checker versions, seeds, split commits, raw measurements, confidence assumptions, nulls, costs,
resource reconstruction/fidelity and protected failures. Connect mechanism artifacts through the
existing `MechanisticAuditReference`; promotion remains existing protected governance.
Tests first: deterministic rerun, tamper/missing-source rejection, append/no-overwrite evidence,
unrun/unsupported versus observed, seed dependence, failed regression withholding and no self-review
admission. Run full protected suite, lint/format/types, build and outside-checkout wheel smoke.

## Per-stage delivery contract

Each stage uses a fresh diff review and fixes blocking findings before merge. PR descriptions
state problem/result, actual validation, maturity/limits and migrations. Capture checks against
the exact head SHA and inspect review threads. Merge only a clear passing PR, then base the next
stage on its merge. Keep scientific unrun work explicit; inability to qualify a model/hardware
claim never becomes a fabricated success. Final inventory, exact constitutional diff, API docs,
toy benchmarks, protocols, measured simulation/intervention results, test logs, source traceability,
limitations and next experiments remain separate deliverables.
