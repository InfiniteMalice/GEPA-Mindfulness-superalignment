# VPO admissibility and set-score preparation

## Maturity and primary source

[vector_policy.py](../gepa_mindfulness/training/vector_policy.py) supplies **Experimental**,
default-off CPU records, numerical kernels and host-reviewed preparation. Full VPO training,
token advantage attachment, behavioral improvement, protected regression qualification and joint
KnowGPO/PRM4RL rewards remain **Unsupported** or unexecuted. Unit tests do not prove alignment.

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

Even reviewed metrics remain DEVELOPMENT and require host admission. No new reward provider,
governance layer, runtime grant or optimizer path is introduced. Foundational values remain
mutually constraining and cannot be diluted by weights. Future attachment must preserve
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
