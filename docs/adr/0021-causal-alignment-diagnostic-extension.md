# ADR 0021: Extend causal diagnostics through existing V5 contracts

Status: PR-1 and PR-2 designs and native implementation plans approved on 2026-10-08.
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
cover diagnostic PR-1/PR-2 only; later architectural stages retain their review gates.

The optional causal reward adapter additionally requires explicit reward-policy review,
authenticated evidence, a duplicate-accounting policy and exact baseline equality when disabled
or unverified. PR-7 additionally requires independent withheld-scenario results. No benchmark
result, debate victory or diagnostic record establishes training or deployment eligibility.
