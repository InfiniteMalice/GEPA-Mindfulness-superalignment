# Unified research integration

## Status and scope

This guide records the dependency-ordered integration boundary for the October 2026 program.
PR 0 delivered a **Diagnostic** documentation audit, subsequently merged as PR #801.
PR 1's [experimental credit contracts](research_credit_accounting.md) add opt-in diagnostic
proposals, overlap accounting and durable host-reviewed event reservations. Optimizer credit
composition remains deferred; accounting receipts grant no runtime authority. PR 2 adds
[experimental ActionOPD/LoCD adapters and kernels](distillation_adapters.md), without a full
teacher service, training runner or empirical qualification. PR 3 adds a separate optional
[CPU SST adapter and numerical probe](subspace_tuning.md); canonical training attachment and
behavioral retention experiments remain unqualified. PR 4 adds the default-off
[DeLoop + PEO audit and recovery preparation](deloop_peo.md), with a separate DPO kernel;
live intervention/training attachment and effectiveness remain unqualified. PR 5 adds [knowledge/progress preparation](knowledge_progress.md): reviewed evidence transitions,
prior-state groups and milestone potentials with separate unresolved credit proposals. No trainer
is attached and no behavioral improvement is established. PR 6 adds [VPO set-score preparation](vector_policy.md),
with whole-rollout quarantine and independent review; full VPO training remains unsupported.
PR 7 adds [immutable dataset research preparation](dataset_research.md), reusing the existing
manifest and review contracts. Full AutoSFT orchestration, training admission and empirical
effectiveness remain unsupported.
PR 8 adds [structured decision routing and native-depth diagnostics](decision_routing.md),
reusing the existing guard and calibration metrics. Live Jev/LEV services, native loop
models and empirically qualified calibration remain unsupported.
PRs 9–11 remain unimplemented.
The [audit evidence](experiments/unified_research_pr0_audit.md) preserves the original local
baseline and primary-source limitations; PR #801's Python 3.12 CI subsequently passed.
The [method matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) covers all 19 requested
methods, existing owners, missing work, dependencies, trust boundaries and proposed verification.

The design follows the operator's staged plan: extend established contracts in small PRs,
keep experiments disabled by default, preserve serialized compatibility, and stabilize primary
methods before optional arms. A monolithic training pipeline would prevent independent ablations;
a parallel governance or reward stack would duplicate existing authority. Neither is proposed.

## Protected foundations and authoritative owners

The [Constitution](GEPA_Mindfulness_Constitution.md) remains normative: increase prosperity,
reduce suffering and increase understanding are co-equal and mutually constraining. Mindfulness,
Emptiness, Non-Duality and Boundless Care inform contextual reasoning. Honesty, autonomy,
non-coercion, non-deception, corrigibility, consent, humility, oversight, proportionality and
intelligent disobedience remain foundational. Interpretation can adapt; foundational revision
requires explicitly authorized human governance. No objective coefficient can erase a commitment.

Separate normative commitments, enforceable constraints, unresolved contextual judgments,
verified task outcomes, diagnostics and optimizable research objectives. Independent host checks
enforce applicable hard constraints before optimization or execution. Unresolved moral judgment
retains uncertainty and routes to appropriate review. A classifier cannot prove moral alignment.
Context-sensitive constitutional provisions must not become blanket deterministic violation labels.

[evaluation/cases/17_case_manifest.yaml](../evaluation/cases/17_case_manifest.yaml) alone
authors the 17 canonical cases. Case 0 remains noncanonical. Keep
`CASE × STRIPE × REPEAT`, confidence, calibration, IDK and clarification semantics intact.
New loop observations belong in experimental overlays; do not silently add subtype strings
that the current robustness registry rejects. Safety refusal remains outside the epistemic taxonomy.

[Verified process rewards](epistemic_process_rewards.md) retain positive-only, provenance-bound
process bonuses. Observable behavioral penalties remain separate. Teacher consensus, latent
memory, private thought, model explanations, fingerprints and unverified graphs create no credit.
Use the current `RewardProvenance`, `RewardRequest` and eligibility checks.
The [runtime authority contract](VERIFICATION_AND_RUNTIME_AUTHORITY.md) remains independent:
a selected action, progress marker, proposal or accepted catalog entry is never an execution grant.

## End-to-end integration

Observe task/evidence → bounded prediction → independently authorized action → execution and
observation → independent local/relational verification → canonical V5 record → supported failure
localization → typed improvement-surface selection → isolated eligible experiment → protected
baseline comparison → provenance/review record → governed admission.

Retain the distinction between record validation and host authentication/semantic verification.
Existing records preserve evidence identifiers but do not dereference sources or authenticate
verifier identities. Candidate code and tool outputs cannot own host policies or protected stores.

Use `CorrectionProposal`, `TrajectoryBinding`, `EvaluationEpochStore` and
`CoevolutionStore`; preserve model and harness versions separately. The current
`assess_improvement()` produces DEVELOPMENT diagnostics without authority. Extend it only
with typed evidence and existing human-review precedence. A no-change result is valid.

## Proposed PR 1 shared contract boundary

Before attaching a loss, specify versioned records that reference existing evidence, action,
run, model, harness, evaluator, teacher and policy identities. Records must retain source
dataset and split, configuration, exact student prefix/state digest, immutable outcome evidence,
independent verification and review disposition. Proposed records are not authorization tokens.

Credit accounting must identify a semantic verified event in its run/task/authority scope,
not merely compare evidence-reference IDs. The same source can support distinct facts;
different aliases can denote the same event. Preserve every raw component and its lineage,
report possible overlaps, and reject optimizer composites with unresolved overlap. Select an
explicit host-owned credit allocation before combining knowledge gain, milestone progress,
process bonuses or teacher-derived signals. The causal adapter already rejects unknown
accounting and already-counted adjustments; reuse that boundary instead of bypassing it.

An admissibility result must distinguish admitted, rejected and unresolved, with applicable
constraint, verifier/version, evidence, proposed action, decision route and reason. Model text
cannot fill an independent-verifier field. Authorization remains separately checked at execution.
Protect serializations against unknown versions, nonfinite values, contradictory evidence,
nested held-out metadata, replay, mutated snapshots and issuer substitution.

The current research-credit claim/report schemas implement the diagnostic subset of these
requirements. They remain DEVELOPMENT data with no optimizer accessor or admission authority.
Durable allocation reservations now require exact request binding, explicit prior-layer accounting
and fresh host review. Live composition and protected admissibility enforcement remain with
existing host mechanisms; these records cannot grant either. PR 0 did not introduce runtime schemas.

## Method-specific protocol boundaries

| Integration | Protocol required before runtime attachment |
| --- | --- |
| LoCD / ActionOPD | LoCD binds original prefix, supported error region and privileged-context teacher continuation; ActionOPD compares typed semantic actions at student states. Keep losses and eligibility separate. Student evaluation excludes privileged evidence. |
| LoCD + SST | Compare local correction alone and with projected applied optimizer delta at matched data/compute. Fixed reference basis first; explicit refresh, numerical leakage, unsupported shape handling and resume metadata. Protecting parameter directions is not moral preservation. |
| DeLoop + PEO | Ledger binds action, equivalent arguments, state and observation. Independent classification excludes productive retries, changed state, necessary verification and human rechecking. Mine recovery only after supported loop-breaking success from the original preceding state. |
| KnowGPO + PRM4RL | Versioned atomic knowledge conditions and dependency milestones use separate verified state and belief state. Corrected false belief can be progress. Shaping requires declared discount, potential, terminal handling and applicable invariance assumptions. Detect duplicate event credit before any composite. |
| VPO | Independently filter/quarantine before diversity optimization. No constitutional weight in a tradeable vector. Empty admitted set uses context-appropriate abstention, clarification, refusal, redirection or escalation. Never execute forbidden exploration. |
| AutoSFT | Immutable source examples, content-addressed derived datasets, declared operations, independent label/source review and host admission. Separate analysis, synthesis, execution and verification roles. No protected-test mutation or generator self-certification. |
| Jev / LEV | Extend existing bounded routing backend and guard. Preserve raw/guarded decisions, false pass/escalation and provenance. LEV capability must be established before adaptive-depth claims. Independent verification outranks deep agreement. |
| RSIFlow / harness | Failure evidence → verified diagnosis → proposal → isolated trial → independent evaluation → protected checks → governed admission. Data, memory, model, harness, skill, verification, human and no-change routes remain distinct. Admission does not deploy. |
| Optional memory / representation | Keep raw recoverable sources and existing access/trust checks. Latent recall, specialist bids, action decodability and text transfer probes are diagnostic until independently validated. |

## Compatibility and ablations

The portable `gepa rl` engine is the canonical weight-update path; legacy simulator outputs
are not trained models. Full-weight CPU PPO/GRPO is documented Supported upstream; CUDA/DDP
is Implemented; hardware unqualified. New methods cannot inherit those maturity claims.
CPT provides pair construction/loss scaffolding, not a substitute for canonical training.
LoRA has a backend API but no canonical configuration/CLI selector at the audited commit.
SST therefore needs explicit effective-update/adapter semantics before LoRA qualification.

| Study | Required comparison conditions | Additional decision evidence |
| --- | --- | --- |
| A: correction/retention | Baseline; ActionOPD; LoCD; LoCD+ActionOPD; LoCD+SST; LoCD+ActionOPD+SST | SFT/off-policy/on-policy context-distillation controls; both curriculum orders; new corrections versus forgetting. |
| B: knowledge/progress/diversity | Outcome-only GRPO/DAPO; KnowGPO; PRM4RL; VPO; KnowGPO+PRM4RL; VPO+KnowGPO; all three | Joint arms only after independent validation; duplicate credit and reward interference. |
| C: loops | PEO baseline; ledger; DeLoop; DeLoop+PEO memory; DeLoop+LoCD; all three | Also separate duplicate blocker, DPO and verified GRPO penalties; premature termination and useful repeat controls. |
| D: adaptive inference | Ordinary bounded; generative; Jev; compatible LEV | Fixed/adaptive depth, confidence-only, deep agreement, independent escalation; latency and verifier recall. |
| E: improvement selection | Fixed data/harness; AutoSFT; RSIFlow; harness evolution; AutoSFT+RSIFlow; all three | Same protected suites and independent intervention attribution. |
| F: evidence access | Structured retrieval; Librarian; optional latent memory; hybrid | Exact source recovery, false memory, contradictions and DeLoop completion. |

Every major intervention also uses the existing model/harness factorization: original/original,
original/candidate, updated/original and updated/candidate. Hold tasks, relevant budgets,
checkpoints and split policy fixed; compare model, harness and interaction effects.
Absent runtime support makes an arm Unsupported, not an implicit baseline substitute.

## Reproducibility and acceptance evidence

Record source revisions/licenses, data/source digests, split assignments, protected manifest
digests, all seeds, checkpoint/model/harness/evaluator versions, budgets, hyperparameters,
optimizer/precision/dependencies, exclusions, failed runs, raw metrics and review decisions.
Large training, paid APIs, deployment and irreversible actions need separately configured
operator authority. Do not infer it from an experiment configuration.

Report capability (verified success, OOD, best-of-N), epistemics (calibration, correctness,
abstention and source fidelity), alignment (constitutional/intent invariance, authorization,
correction), efficiency (tokens, latency, repeat actions, FLOPs), retention (prior/new skills,
KL drift, spectral diagnostics) and governance (rejections, protected integrity, rollback).
Include failure distributions, not only means.

Use paired tasks, multiple seeds, independently held-out tests and declared uncertainty methods.
Reuse existing improvement-audit/statistics contracts where suitable; repeats are not independent
sampling units. Separate selection results from final tests; missing outcomes remain missing.
Software checks do not establish research effectiveness. Reject promotion on protected
regression even if an aggregate improves.

For each adversarial fixture retain constraint, proposed action, independent verification,
evidence, route, disposition, rejection reason and authorized follow-up. Cover prohibited
teacher shortcuts, diluted VPO weights, unverified milestones/knowledge, mislabeled synthetic
data, answered clarifications, early termination, blanket refusal, forgetting, false memory,
verifier weakening, self-modified acceptance and self-certified review. Evaluator disagreement
remains unresolved evidence until handled by the host; never force it into a deterministic
moral-violation label for scoring convenience.

PR 0 advances only after baseline checks are recorded and source intake is resolved enough
for each next implementation. Later PRs add their own executable schemas, default-off configs,
negative tests, documentation and acceptance evidence. No unfinished stage is marked implemented.
