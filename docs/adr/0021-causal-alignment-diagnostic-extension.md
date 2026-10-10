# ADR 0021: Extend causal diagnostics through existing V5 contracts

Status: PR-1/PR-2 approved on 2026-10-08; PR-3 design and native plan approved on 2026-10-09; PR-4 design, implementation plan and native execution approved on 2026-10-09; PR-6 design and PR-5 deferral approved on 2026-10-10, with PR-6 implementation plan and execution pending approval.
Date: 2026-10-08.

## Context

V5 already has canonical results, strict single-variable relation pairs, public claim graphs,
stakeholder records, evidence memory, diagnostic reporting and separately governed rewards.
The requested causal/debate series needs richer observations without duplicating these systems.
The [PR-0 audit](../recommendations/CAUSAL_ALIGNMENT_AUDIT.md) maps current interfaces and gaps.

## Decision for PR-1

Compose new opt-in diagnostic records with existing V5 and evidence identities. Preserve the
strict relation-pair evaluator as a complete-capture API. Use a separate, explicitly named
diagnostic entry point for planned multi-turn interventions and incomplete adjudication.
Retain correct per-arm cases, observed action classes, declared changed factors and independent
verification. Preserve missing, unresolved and censored states in explicit denominators.

For public argument analysis, reuse `ClaimGraph` and `CheckRequest`/`CheckResult`. For pluralistic
analysis, reuse `Stakeholder` and `Perspective`. These records remain declarations requiring
host verification, not optimizer signals or runtime capabilities. Reuse the existing evidence
state, memory adapter, evaluation epochs and curriculum admission boundaries in later PRs.

Do not modify the 17 canonical cases, fallback Case 0, stripe meanings, normative commitments,
reward formulas or authority gates in the diagnostic PRs. Separate semantic equivalence from
required sensitivity. An unchanged wrong decision is not correct invariance. A justified
decision update is not an invariance failure. Human adjudication remains available when evidence
or action acceptability is disputed.

## Decision for PR-2

Compose public debate snapshots with existing claim graphs and check records. Predeclare check
slots and semantic opportunities, retain incomplete capture, and authenticate exact verifier
envelopes independently. Bound sessions to eight rounds. Keep literal graph/action transitions
separate from independently judged semantic changes and preserve the PR-1 ablation boundary.
The [public debate guide](../sensitive_debate.md) specifies callbacks and denominators.

Compute exact fractional block sensitivity only for complete Boolean functions of one to four
inputs, with rational primal/dual certificates. Natural-language priority remains heuristic;
these fixtures do not establish the paper's recursive-decomposition or judgment-oracle assumptions.

## Decision for PR-3

Compose authored laundering pairs, public stakeholder plans, explicit role permutations and
independently authenticated pluralistic judgments. Preserve unverified source assertions and
simulated preferences separately from derived verification. Exact receipts bind complete public
captures, semantic rubrics and evaluator identities; reanalysis authenticates them again.
The [pluralistic guide](../pluralistic_robustness.md) defines missingness and denominators.

Keep sycophancy, overcriticism, perspective robustness and third-party interests separate. Retain
PR-1 metric contracts and PR-2 session bindings. Three-condition comparisons require declared
matching family/split/model-family/harness/seed/repeat/case/stripe coordinates plus identical
evaluation content and rubric contracts. They retain absent
observations and preserve distinct checkpoint/curriculum versions. Exports remain DEVELOPMENT
with stricter source restrictions nested intact; no training effectiveness follows from fixtures.

## Decision for PR-4

Compose exact artifact/version records, source fragments and digest-bound interpretations with
the existing evidence state, memory gate and claim graph. Recheck current host access for every
retrieval and withhold any derived content with an unavailable ancestor. Preserve original source
times, identities, statuses and training restrictions. Supply only an allowlisted producer view.

Keep structural support routes separate from freshly authenticated semantic verdicts. Retain
planned-but-missing observations in source-removal and three-condition retrieval comparisons.
Match exact evaluation content and scope, and preserve explicit measured cost units. The
[artifact topology guide](../evidence_topology.md) specifies these contracts and the offline example.
The maintainer approved the PR-4 written design, implementation plan and native execution through
a separate draft PR. PR-5 reward-policy and PR-7 independent empirical-evidence gates remain.

## Decision for PR-6

The maintainer approved the PR-6 written design and explicit deferral of PR-5 on 2026-10-10.
PR-6 will compose an opt-in offline evaluation report with existing diagnostic contracts.
The report will separate five data partitions by scenario family and transformation ancestry,
retain every attempted candidate, and report paired cluster-aware uncertainty, cost and failures.
Selection progress, evaluated behavior, independent improvement evidence and deployment
eligibility remain separate. The reporting module will issue no training or deployment authority.

This approval permits implementation planning; PR-6 implementation plan and execution approval
remain pending. The sequencing exception does not waive PR-5's independent evidence or reward
accounting requirements. Repository defaults govern the completed PR-5 pilot and the additional
causal penalty remains withheld. PR-7 retains its earlier-stage and independent-evidence gates.

## Alternatives considered

Broadening `RelationPair` would weaken its exact-intervention/oracle contract. Replacing existing
modules with a new causal, GraphRAG or memory framework would duplicate validation and authority.
Composition adds explicit joins, but preserves backward compatibility and separate review units.

## Verification and approval

PR-0's existing registry, documentation, V5 and reward tests verify metadata consistency and
unchanged behavior. The audit specifies PR-1's positive, negative and incomplete-observation
fixtures and metric denominators. Each later PR needs its own tests and review before acceptance.
The maintainer approved the PR-1 written design, then its implementation plan and native execution.
The [causal diagnostics guide](../causal_diagnostics.md) specifies the opt-in API and host contract.
The maintainer also approved the PR-2 written design, plan and native execution. These approvals
cover diagnostic PR-1/PR-2. The maintainer also approved the PR-3 written design, implementation
plan and native execution through a separate draft PR. Later architectural stages retain their
review gates.

The optional causal reward adapter additionally requires explicit reward-policy review,
authenticated evidence, a duplicate-accounting policy and exact baseline equality when disabled
or unverified. PR-7 additionally requires independent withheld-scenario results. No benchmark
result, debate victory or diagnostic record establishes training or deployment eligibility.
