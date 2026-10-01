# PR-12: contrastive causal hard negatives

## Task spec and reconciliation

The requested experiment trains broad semantics, semantic hard negatives, causal
hard negatives, laundering hard negatives, then PEO trajectory negatives. It
compares an ordinary classifier, JEV, CLM, and CLM with the curriculum, reporting
contrastive margins by family. It must remain opt-in and preserve the 17 cases,
runtime authority, rewards, and training admission.

Existing: CPT candidate/pair records, recursive TRAIN admission, deterministic
world transitions, relation counterfactuals, and validated PEO event generation.
Partial: CPT scalar losses and System-One backend callbacks; neither executes
this contrastive curriculum. Missing: a differentiable pair-ranking training
loop, ordered family exposure, world-derived negatives and matched comparison.
Redundant: a second public preference-record schema or a new world simulator.
Experimental: all learned-effectiveness claims and external JEV/CLM arms.

## Design

Reuse CPT records. A strict boundary snapshots only public prompt/final-answer
text plus family, source-group and admission metadata. Require a strict preference,
distinct answers and explicit admission. Never optimize reasoning summaries,
confidence, uncertainty or residuals. Validate the entire catalog before invoking
an encoder or optimizer. Generated world examples keep source admission and
cannot become TRAIN by relabeling their outer metadata.

Train caller-owned differentiable scorers with two-candidate cross entropy
(softplus of rejected minus chosen). Order all five families for curriculum;
provide a pooled shuffle control with identical per-record update counts and seed.
Use the existing optional torch dependency only inside training. No model downloads,
checkpoint loading, filesystem writes or runtime registration. This is a real
optimizer loop and an integration contract, not a shipped CLM/JEV checkpoint.

Build causal and laundering negatives from decisive relation pairs; broad and
semantic families use authored admitted CPT data. Build trajectory negatives from
the existing PEO simulator: contrast faithful public action/outcome summaries with
one falsified observation. Keep hidden world snapshots in retained provenance only.

Compare the four named callback arms on identical held-out public examples in both
candidate orders. Reject non-finite scores and train/heldout overlap. Report missing
arms and missing families, ties, order sensitivity, accuracy and per-family margins.
Raw score scales are backend-specific. No placeholder empirical results.

## Implementation sequence

1. RED/GREEN: strict CPT projection, admission, family schedule and actual CPU
   optimization with a tiny authored test scorer; reject late invalid records
   before model calls. Check pooled/curriculum exposure equality.
2. RED/GREEN: deterministic relation/laundering and PEO negative builders;
   provenance retention, no hidden evaluator data in prompts, no promotion.
3. RED/GREEN: matched four-arm comparison, position reversal, missing arms,
   finite-score checks, duplicate and split-leakage rejection.
4. Document API and runnable example, research mechanisms versus repository
   hypotheses, ADR, traceability links; run targeted and full CI-equivalent
   validation, wheel smoke and one fresh whole-branch review. Apply at most one
   reviewer fix pass, then commit, push and create a draft PR.

## Research interpretation

CLM's official trainer separates state/action embeddings and uses contrastive
ranking; this PR uses an explicit two-candidate objective rather than reproducing
its bidirectional, group-masked in-batch loss. P-TTT preference reversals and
MechBench interventions motivate causal negatives. VGCompiler and Physical
Representation Languages motivate structured latent generation and identifiability
limits. CHART motivates staged exposure; this fixed schedule is not its adaptive
GRPO harness. These are repository inferences, not replicated paper findings.

## Risks and validation

Test nested admission restrictions, CPT mutability, label and position leakage,
duplicate/heldout leakage, nonfinite tensors and gradients, disabled paths, exact
family ordering and exposure, world-oracle agreement, and observable-only prompts.
Learning evidence is a CPU wiring test: a tiny authored scorer's margin improves.
External backend efficacy remains unmeasured. Existing full-suite, formatting,
lint, typing, package build and installed-wheel checks protect compatibility.

Skills: Superpowers brainstorming, writing-plans, executing-plans, TDD,
verification-before-completion, requesting-code-review; repo-quality-gate.
Repository rules place this spec/plan in history and prohibit automated beads work.

## Pre-review validation evidence

- Focused tests: 41 passed in the torch-enabled CPU environment.
- Full suite: 4,292 passed, 18 skipped, 16 warnings (236.26 seconds).
- Ruff and Black passed across the repository (614 Python files).
- Mypy passed for the nine CI modules plus three new modules, and the separate
  logging-schema check. New code also passed Python 3.10 syntax and 100-column checks.
- Source/wheel build passed. Outside-checkout installed-wheel smoke verified exact
  module/guide hashes, the guide example, PEO generation, CLI, 17 cases and 62 refs.
  Importing the new APIs did not import torch.
- The authored scalar scorer's public preference margin increased from 0 to
  2.717833; per-update loss fell from 0.693147 to 0.067231. Both schedules received
  four updates per family and gave identical results in this deliberately simple
  wiring test. This does not measure curriculum effectiveness.
- Additional RED/GREEN checks bind dataset digests to full source provenance,
  reject shared prompts with different negatives across splits, and detect
  nonfinite parameters immediately after an optimizer step.
- Documentation precision review: API actor, inputs, opt-in conditions, outputs,
  failure behavior and host responsibilities are explicit; no unresolved BLOCK.
  Primary-source mechanisms, repository inference and unmeasured claims are separate.
