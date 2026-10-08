# ADR 0021: Extend causal diagnostics through existing V5 contracts

Status: Proposed for written-spec review; no behavioral implementation in PR-0.
Date: 2026-10-08.

## Context

V5 already has canonical results, strict single-variable relation pairs, public claim graphs,
stakeholder records, evidence memory, diagnostic reporting and separately governed rewards.
The requested causal/debate series needs richer observations without duplicating these systems.
The [PR-0 audit](../recommendations/CAUSAL_ALIGNMENT_AUDIT.md) maps current interfaces and gaps.

## Proposed decision

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

## Alternatives

Broadening `RelationPair` would weaken its exact-intervention/oracle contract. Replacing existing
modules with a new causal, GraphRAG or memory framework would duplicate validation and authority.
Composition adds explicit joins, but preserves backward compatibility and separate review units.

## Verification and approval

PR-0's existing registry, documentation, V5 and reward tests verify metadata consistency and
unchanged behavior. The audit specifies PR-1's positive, negative and incomplete-observation
fixtures and metric denominators. Each later PR needs its own tests and review before acceptance.
The maintainer reviews the PR-1 written design before implementation planning. This ADR remains
proposed until that review; a successful test run does not accept an architectural decision.

The optional causal reward adapter additionally requires explicit reward-policy review,
authenticated evidence, a duplicate-accounting policy and exact baseline equality when disabled
or unverified. PR-7 additionally requires independent withheld-scenario results. No benchmark
result, debate victory or diagnostic record establishes training or deployment eligibility.
