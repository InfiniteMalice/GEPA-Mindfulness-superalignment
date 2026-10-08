# Unified epistemic integration: specification and implementation map

Base: `4322a0ad379a8940eee7feb8fe3770f7bedd7dbb` (2026-10-07).
Workflow: Superpowers design, writing-plans, native execution, TDD, Repo Quality Gate.
The maintainer requested inspection followed by sequential implementation. Each stage is a
separate dependent branch/commit and reviewable PR. No merge or deployment is part of this work.

## Specification

Integrate public commitments, claim decomposition, stakeholder robustness, external checks,
temporal updates, synthetic argument families and bounded offline search into the existing V5
architecture. Preserve exactly 17 canonical cases, noncanonical fallback Case 0, independent
verified-process reward, runtime role separation, and historical content-addressed serialization.
Experimental procedures are opt-in and diagnostic. No private reasoning, simulated preference,
synthetic label, majority vote, debate win or search score confers reward or authority.

The observable sequence is COMMIT, DECOMPOSE, CHALLENGE, CHECK, OBSERVE, VERIFY, UPDATE,
CLASSIFY. The existing action-bound event sequence remains authoritative for executed actions.
Training credit remains a separate `EpistemicProcessAssessment` with independently issued
`VerifiedProcessComponent` provenance. This project adds no reward component or promotion path.

## Inspection: existing, partial and absent capabilities

| Concern | Existing implementation | Extension and missing behavior |
| --- | --- | --- |
| Canonical identity | evaluation/cases/17_case_manifest.yaml; registry.py; V5 runner | Reuse manifest and cells; no new cases |
| Robustness | robustness_stripes.yaml; modules/semantic_intent_robustness | Reuse stripes; add documented compatible subtypes only |
| Public evidence | verification/state.py: EvidenceClaim, EvidenceState | Attach graph metadata by composition; preserve old serializers |
| Checks and authority | verification/interfaces.py; runtime_governance.py | Claim/check/evidence links; checks never issue grants |
| Temporal state | epistemic_state.py; temporal_estimator.py; epistemic_reconciliation.py | Select discriminating checks, feed existing validated reconciliation |
| PEO continuity | modules/semantic_intent_robustness/peo_continuity.py; action_bound_events.py | Link inquiry and revision episodes to existing event IDs |
| Hypotheses | hypothesis_records.py; hypothesis_state.py; semantic_exploration.py | Reuse external hypotheses and propose sensitivity-weighted checks |
| Synthetic data | synthetic_dataset_validation.py; synthetic_data; rich schema; adapters | Optional argument-family metadata, ordered sweeps, boundary summaries and pairs |
| Reward | core/epistemic_process.py; reward_provenance.py; training/eligibility.py | Preserve eligibility and provenance; diagnostic outputs DEVELOPMENT |
| Offline evolution | skill_bank.py; skill_lifecycle.py; coevolution.py; evaluation catalogs | Search-only ranking and budgets before existing grounded acceptance |

### Trust conflicts and decisions

PlurPO self-produced preferences cannot become verified process credit. Stakeholders describe
interests and never receive runtime grants. Evidence source kinds are provenance declarations,
not authentication: hosts still own capture, authorization and verifier identity. New record
validation cannot prove natural-language truth. Existing exact serializers cannot gain optional
fields without changing old digests; use new composed records instead. SIFT can propose a branch
for grounded evaluation but cannot manufacture durable evaluation receipts or install code.
Hypothesis agreement cannot clear a claim. Unknown findings remain unresolved.

## Implementation plan

Goal: implement the specification in eight sequential, independently testable stages.
Architecture: small typed records and pure diagnostic functions beside existing owners, with
explicit adapters at evidence, temporal, synthetic and evolution boundaries. Python 3.10+,
stdlib and existing dependencies only. Permanent architecture decisions go in docs/adr.

Global constraints: 100-character Python lines; Black/Ruff; exact typed inputs; finite numeric
values; versioned JSON round trips; no new canonical identities or private thought targets.
Review focus: forged provenance; partial rollout coverage; stale or duplicate observations;
nonmaterial framing changes; search/hidden-eval leakage. Each owning stage tests these inputs.

### Stage 1: contracts, types and invariants

Files: verification/claim_graph.py, verification/perspective_records.py,
verification/check_records.py, tests/test_epistemic_contracts.py, docs/adr/0020-unified-epistemic.md.
Compose EvidenceClaim in ClaimNode; ClaimDependency and ClaimDecomposition preserve provenance.
ClaimGraph validates unique identities, closure and cycles. Stakeholder and Perspective records
separate hard constraints, interests and preferences. CheckRequest/CheckResult retain factors and
links without granting authority. Test round trips, malformed types, graph closure, evidence
provenance, canonical count and old serialization. Run failing tests, implement, targeted suite,
Ruff/Black/mypy and commit. Sources: all five papers motivate separate contracts, not their truth.

### Stage 2: sensitive debate and perspectives

Files: verification/sensitive_debate.py, verification/perspective_robustness.py,
synthetic schema copies, tests/test_sensitive_perspectives.py.
Implement empirical ablation sensitivity and bounded recursive challenge selection over ClaimGraph;
generate perspective variants retaining semantic-core/fact identity; compare typed judgments and
report omissions/hallucinations and hard-constraint drift. Extend rich schema with optional typed
family metadata. Test irrelevant/material changes, unsupported preferences and graph limits.
Sources: Sensitive Debate (2610.02557), PlurPO (2610.02568). Proxies have no theorem guarantee.

### Stage 3: synthetic argument-phase curriculum

Files: synthetic_data/argument_families.py, synthetic_data/argument_pairs.py,
scripts/synthetic_dataset_tool.py, training/adapters/synthetic_cases.py, tests/test_argument_families.py.
Generate ordered host-authored sweeps and minimal hard negatives from rich source rows. Validate
single-parameter lineage, transition labels, boundary bands, response curves, all requested V5
boundary pairs, and source hashes into derived pairs. Preserve DEVELOPMENT eligibility. Extend
the existing summary CLI. Tests cover spurious, premature, delayed, missed and locked transitions.
Sources: Sensitive Debate, PlurPO; boundary curriculum is a repository hypothesis.

### Stage 4: external evidence verification

Files: verification/claim_verification.py, tests/test_claim_verification.py.
Partition multiple rollout values by requirement into disputed/consensus/omitted; preserve partial
coverage. Separate resolver requests from consensus falsifiers. Rank CheckRequest factors. Fresh
adjudication consumes typed findings plus existing local/relational results and authorized evidence;
returns explicit unresolved or evidence-backed revisions. Never commit EvidenceState or authorize
execution. Test unanimous falsehood, majority uncertainty, missing evidence, unauthorized sources,
non-independent/circular evidence and fresh-context separation. Source: VeriHarness (2610.00972).

### Stage 5: PEO uncertainty-directed checking

Files: verification/uncertainty_inquiry.py, tests/test_uncertainty_inquiry.py,
data/synthetic/gold/uncertainty_closing_v1.jsonl.
Select bounded checks, record hypothesis discrimination and stopping diagnostics, then bridge
verified observations to ScalarTemporalEstimator.reconcile without replacing its validation.
Test useful versus irrelevant checks, event ordering, duplicate evidence, missing provenance,
unsupported latent evidence and surprising observations. Source: EurekaBench (2610.00492).

### Stage 6: observable revision and V5 integration

Files: evaluation/epistemic_revision.py, tests/test_epistemic_revision.py.
Link commitments, graph, checks, counterevidence, updates and canonical V5 assessment in an episode.
Report revision/stability/underreaction/overreaction/rationale migration/unresolved separately.
Adapters cover all 17 cases, with explicit IDK/clarification/resume findings. Preserve evaluator
uncertainty and original evidence. Test forced revision, alternative support, unsupported replacement
premises and diagnostic reward rejection. Sources: VeriHarness and Sensitive Debate.

### Stage 7: controlled SIFT search

Files: gepa_mindfulness/sift_search.py, tests/test_sift_search.py.
Disabled-by-default offline candidate specifications, pairwise records, regularized Bradley-Terry
ranking, tree lineage, budgets and grounded-evaluation proposals. Reuse SkillCard metadata and
existing coevolution decisions; SEARCH_ONLY never means acceptance. Reject hidden-eval inputs and
declared leakage markers. Require objective improvement and configurable nonregression/cost gates
before consideration by durable acceptance. Test ranking, budgets, forbidden targets, leakage,
regression and optimizer rejection. Source: Self Improvement via Fast Tree-search (2609.19526).

### Stage 8: ablations, documentation and traceability

Files: evaluation/epistemic_ablations.py, scripts/run_epistemic_ablations.py,
docs/unified_epistemic_verification.md, docs/recommendations registries, docs/synthetic_dataset.md,
docs/controlled_evolution.md, tests/test_epistemic_ablations.py.
Reproducible A-K configuration matrix, separate cost/behavior metrics, family-disjoint splits,
deterministic fixture smoke run, threat model, maturity, limitations and research table. No model
benchmark improvement claim without experiments. Run full suite, repository lint/format/type and
documentation checks; independent whole-change review; report baseline limitations explicitly.

## Research traceability

| Source | Source-supported motivation | Repository inference | Experimental hypothesis |
| --- | --- | --- | --- |
| Mitigating Social Sycophancy via Pluralistic Preference Optimization, 2610.02568 | Stakeholder simulation reduces social sycophancy in evaluated settings | Perspective diagnostics with hard-constraint separation | Families expose semantic laundering |
| How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate, 2610.02557 | Stable decomposition admits sensitivity-related formal oversight guarantees | Public graph ablation proxy | Proxy prioritizes useful checks; theorem not inherited |
| EurekaBench: Measuring Agentic Ability to Discover New Scientific Insights, 2610.00492 | Predictive accuracy and scientific insight can diverge | Track uncertainty closure separately | Discriminating-check fixtures improve inquiry |
| VeriHarness: Scaling Agentic Verification for Long-Horizon Tasks, 2610.00972 | Disagreement resolution and consensus challenge use environmental evidence | Claim-level provenance and fresh adjudication | Combined V5 harness improves evidence fidelity |
| Self Improvement via Fast Tree-search, 2609.19526 | Pairwise comparison and tree search allocate expensive evaluations | Apply search to bounded verifier/generator procedures | Grounded-gated search improves cost without epistemic regression |

Primary sources: https://arxiv.org/abs/2610.02568, https://arxiv.org/abs/2610.02557,
https://arxiv.org/abs/2610.00492, https://arxiv.org/abs/2610.00972,
https://arxiv.org/abs/2609.19526. These papers do not empirically validate this combined design.
