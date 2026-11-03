# VPO admissibility and set-score preparation

## Maturity and primary source

[vector_policy.py](../gepa_mindfulness/training/vector_policy.py) supplies **Experimental**,
default-off CPU records, numerical kernels and host-reviewed preparation. A separate opt-in
isolated native CPU GRPO trial now attaches one verified set advantage to each safe whole
rollout and reuses protected catalog admission. Behavioral qualification and joint
KnowGPO/PRM4RL studies remain unrun. Unit tests do not prove alignment.

Bahlous-Boldi et al. (2026), [Vector Policy Optimization: Training for Diversity Improves
Test-Time Search](https://arxiv.org/abs/2605.22817v1), sections 3.1–3.2 and Appendix B, were checked
2026-10-28. The official [project](https://vpo-rl.github.io/) links
[ryanboldi/vpo](https://github.com/ryanboldi/vpo). Exact source artifacts, versions, licenses and
digests are in the [existing matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) and
[REF-VPO](recommendations/RESEARCH_TRACEABILITY.md#ref-vpo). The retained note is an authored
summary, not publisher bytes. No upstream code or dependency stack was copied.

The paper emits a candidate set per autoregressive rollout, scores its expected maximum
scalarized reward under common Dirichlet draws, population-normalizes the set scores, and
broadcasts one advantage over each rollout. Independent admissibility, bounded evidence
contracts and whole-rollout quarantine are original Mindfulness adaptations.

## Numerical definition and replay

`score(S) = mean_k max_{y in S} sum_d weights[k,d] * values[y,d]`.

`vector_set_score(vectors, weights)` is a pure diagnostic; it does not establish admissibility.
The reviewed adapter quarantines unsafe sets before calling it. Duplicate vectors do not
increase a maximum. Negative valid scores stay negative; invalid sets never receive a competing zero.

```python
from gepa_mindfulness.training.vector_policy import vector_set_score

weights = ((0.9, 0.1), (0.1, 0.9))
diverse = vector_set_score(((1.0, 0.0), (0.0, 1.0)), weights)  # 0.9
balanced = vector_set_score(((0.6, 0.6),), weights)  # approximately 0.6
```

These authored examples demonstrate definitions, not model results. All compared sets use the
same recorded matrix. Nonnegative simplex rows accept an explicitly recorded binary64 sum
tolerance of `1e-12`; preference extremes such as `(1, 0)` remain valid. Constitutional values
are never weighted dimensions. `common_dirichlet_weights(dimensions, draws, seed=...)` uses a
local standard-library RNG and normalized independent exponential draws for Dirichlet(1).
It does not change global/environment RNG state. Replay the actual retained weights across
runtime versions, not merely a seed. Nonuniform concentration and Sobol sampling are unsupported.

The adapter reuses existing GRPO population-standard-deviation-plus-epsilon normalization.
The inspected upstream implementation stores raw scores in float32 and zeros `std < epsilon`;
this adaptation retains binary64 metrics without that extra threshold. Weights, dimension
order and explicit epsilon are serialized; no unavailable training hyperparameters are assumed.
Overflow or nonfinite score/normalization arithmetic rejects before independent review.

## Contracts, provenance and independent admission

`VectorScore` binds a finite task value to matching existing `RewardProvenance`. Allowed
dimensions are verified task quality, reasoning efficiency, explanatory clarity, robustness,
generalization, resource efficiency and evidence-grounded completeness. Higher values must mean
better under the host's versioned metric contract; no hidden scale normalization occurs.
Constitutional commitments, knowledge gain and milestone progress are excluded dimensions.

`VectorCandidate` retains an ID, public `SemanticAction`, ordered scores, declared admissibility
and existing action-bound `RelationalVerificationResult` with observable evidence bindings.
`VectorRollout` binds the complete set to an exact `RewardRequest`/trajectory ID. Its public
response must encode the candidate IDs and actions as a JSON array. Scores and verifier records
are separate from generated action content. Unsupported/substituted responses reject.

`VectorPolicyConfig` pins authority/run/task identity, model and harness versions, producer,
independent evaluator, dataset/Constitution digests, dimensions, common weights and epsilon.
String identities and digests do not authenticate principals, policy versions or evidence.
The independent host must check actual ownership, metric definitions/scales, evidence truth,
semantic aliases, source freshness, protected Constitution version and applicable constraints.

Compared rollouts share the original public prompt/tokens, model version and sampling settings
through the existing public-state digest. They require explicit TRAIN source lineage and
existing nested eligibility checks. Score/verification evidence must bind observable source
references; evaluator routes match the configured contract. Complete recorded reference rosters
reject conflicting source kinds. Private thought, teacher consensus and self-reported knowledge
cannot certify scores. Undeclared held-out contamination requires independent host detection.

Records are strict versioned `*-v1` DEVELOPMENT data with `authority_granted=false`; restoration
rejects extra fields, unsupported versions and training/authority relabels. Nested inputs are
detached and revalidated. Bounds are 1–7 dimensions, 1–256 draws, 0–16 candidates per rollout,
1–32 rollouts per group and the existing 4 MiB aggregate JSON limit. Both complete output
alternatives, including proposed metrics and provenance, are bounded before review.

`prepare_vector_group(config, rollouts, enabled=False, authorize_evidence=None)` returns `None`
without inspecting inputs when disabled. It installs no pipeline hook. Enabled calls require
declared admission plus complete task-fit/dependency/provenance/authorization/outcome findings,
no contradiction/unknown contradiction status, and no repeated failed route. Structural checks
are not a moral classifier. An independent host reviews complete source and proposed metrics.
Missing/rejected review, exceptions, non-boolean approval and envelope/caller mutation withhold
reviewed scores and advantages. Acceptance cannot override failed structural checks.

Any rejected, unresolved or incompletely verified member quarantines its **whole autoregressive
rollout**. Scoring just its valid subset and broadcasting positive advantage would train invalid
tokens too. Partial-set/token-mask training is unsupported. Quarantined sets have no scalar or
advantage. Uncertain moral judgment stays uncertain. Empty/unreviewed groups retain
`fallback_required=true` and route to an independent host for abstention, clarification,
refusal/redirection or escalation; this API executes no fallback or candidate.

Even reviewed preparation reports remain DEVELOPMENT. The native path below uses the existing
research reward provider and fresh independent admission; no competing reward authority,
governance layer or runtime grant is introduced. Foundational values remain mutually
constraining and cannot be diluted by weights. Training preserves
[existing event accounting](research_credit_accounting.md): one verified event cannot be paid
again through VPO, knowledge, progress and process components. Joint coefficients are not implemented.

## Verification and future experiments

Run `python -m pytest tests/test_vector_policy.py`. Existing documentation/reference and wheel
tests cover installed source records. CPU tests cover authored maxima, common weights, negative
scores, preference extremes, prohibited shortcuts, whole-rollout quarantine, unknown/empty sets,
source/version and evaluator mismatch, private provenance, callback mutation and aggregate limits.

Future Study B compares baseline, fixed scalarization, VPO, KnowGPO, PRM4RL and pairwise/joint
arms. Hold checkpoints, tasks, candidate counts, compute budgets and relevant common draws fixed.
Report valid strategy diversity, admissibility rejection/uncertainty, pass@1, best-of-N, raw
knowledge/progress, calibration, OOD transfer, resources and constitutional regressions. Use
held-out tests, multiple seeds, paired comparisons and uncertainty estimates. Include failure
distributions. Evaluate all 17 canonical cases under `CASE × STRIPE × REPEAT`; case 0 remains
noncanonical. Never promote a model with a protected regression. These behavioral experiments
have not run; training/spending requires separately configured operator authority.

## Native admissible-set trial

`training.vector_policy_runtime.run_vector_policy_trial(backend, rollouts, store,
correction, scope=..., config=..., algorithm=..., pipeline=..., ledger=...,
credit_policy=..., authorize_evidence=..., authenticate_claims=...,
authorize_optimizer_use=..., authorize_accounting=..., account_event=...,
evaluate=..., candidate_id=..., artifact_root=Path(...))` performs real native CPU
forward/backward/optimizer steps on an isolated model copy. `VPOTrainingConfig`
defaults off; enable it with explicit operator policy/reviewer/evaluator IDs,
immutable dataset digest, fixed constitution digest, seed and bounds. Supply the
existing `VectorPolicyConfig` with shared dimensions/weight draws and the matching
native GRPO normalization epsilon. CPU FP32 full weights, native SST and checkpoint
resume are implemented; CUDA, AMP, LoRA and distributed trials are Unsupported here.

The constitution digest is an operator-fixed reference, never a learned weight or
proof of moral alignment. Scope substitution fails before review. Installed host
roles authenticate the actual normative reference, applicable constraints, evidence,
source-file identity, semantic normalization and physical reviewer independence.
Unresolved admissibility remains unresolved and cannot be changed by weight sampling.
The existing allowed task dimensions exclude constitutional values. Invalid members
quarantine the whole generated set, including every token in that rollout. No
candidate action is executed for exploration. Empty admissible groups preserve
independent fallback routing; constant/singleton groups produce no model update.
For a nonempty constant group, retaining the current model is the training fallback;
the nested evidence flag can remain `fallback_required=false` because admissible
task strategies exist. That flag never admits a model or authorizes execution.

Original sources must already be TRAIN and retain all nested eligibility, current
model/checksum, source-record digest, and tokenizer-bound prompt/whole-set response.
The registered model-only correction anchors the closed source epoch; fresh complete
evidence review authenticates additional rollouts. Each source retains exactly one
host-captured `vpo-event:<normalized-semantic-event-id>` trace reference. Trace identity
is outside sampling parameters so distinct outcomes can share the same public state.
The host authenticates its meaning; a model-produced string never authorizes credit.
The registered source must remain admissible and participate in the updated group.
Other safe sets cannot replace a quarantined source for correction attribution.

`vector_policy_credit.make_vector_credit` proposes a `verified_set` contribution
with a closed raw-source/metric/verification proof. Its constructor recomputes the
exact set score, response, configured dimensions/weights, subjects, evaluator,
constitution scope, request/state/event bindings and source eligibility. It rejects
unsafe or empty sets. Raw facts form a new independently reviewable contribution;
existing DEVELOPMENT preparation reports are retained separately and never relabeled
as TRAIN. The proposal remains unresolved until fresh existing host admission.

`ResearchRewardProvider` and `ResearchCreditLedger` retain authority. Supply one
explicit positive `verified_set` coefficient and a zero baseline for this isolated
objective. The coefficient bounds the signed credit; native advantages use the
reviewed raw set scores and declared epsilon, preserving normalization under that
accounting scale. Valid negative scores remain negative. Fresh semantic, optimizer-use
and cross-layer accounting reviews receive the full set evidence and exact credit;
`account_event` merely proposes the existing `ResearchCreditUse`. One event cannot
be reused through task, knowledge, progress, process or teacher credit. Reservations
remain consumed after later no-update/failure rather than being silently refunded.

Legacy claim/policy/decision wires remain unchanged. Only verified sets select
claim/policy/decision version 3; their distinct raw/weighted metrics use closed metric
version 5, including the shared JSONL logger. Existing metric versions 1–4 retain
their field boundaries. This trial uses only the set objective; joint research
coefficients are not implemented here. Native GRPO old/current forwards and updates
are shared with the knowledge/progress trial while their objectives stay separate.

The model/reference/RNG remain isolated. Existing immutable checkpoints, public-only
generation and protected `CoevolutionStore` receipts govern final catalog acceptance.
Regression returns no backend and retains evidence; acceptance does not deploy.
Reports retain common weight draws, raw set scores, quarantine/fallback decisions,
advantages, individual credit/receipt records and protected admission. They remain
DEVELOPMENT audit data.

Run `python -m pytest -q tests/test_vector_policy_runtime.py tests/test_vector_credit.py
 tests/test_vector_policy.py tests/test_knowledge_progress_runtime.py
 tests/test_research_credit_runtime.py tests/test_rl_logging.py` for authored CPU
software checks. These demonstrate implementation and trust boundaries, not diversity
or alignment gains. The operator excluded research; the protocol above records why
and how to compare useful diversity, completion, resources and constitutional
regressions later under matched data/compute and independent holdouts. Source
qualification and the original adaptations remain as recorded above.
