# ADR 0010: bounded synthetic worlds with existing PEO events

Status: accepted for experimental opt-in use.

## Context

Authored examples already have provenance and admission metadata, while the repository
has validated evidence and action-bound PEO contracts. Longitudinal evaluation needs
shared latent state across observations and surface variants.

## Decision

Add a pure boolean world module and an offline episode adapter. Typed facts and rules
determine truth. Actor projections expose visible evidence and public rules. Surface
order and urgency preserve the world; counterfactuals change one permission relation.
Unknown requirements block the target action pending inspection.

Reuse EvidenceState, EvidenceReference, TrainingEligibility, and existing PEO factories
and validation. Preserve evaluator snapshots and lineage without a second event system.
The logged execution is an offline simulation attempt. Verification checks agreement
with deterministic replay of the same simulator; it conveys no independent real-world
certification or execution authority.

Worlds cannot select TRAIN. Defaults, 17 canonical cases, rich synthetic JSONL schema,
rewards, persistence gates, and existing adapters are unchanged.

## Consequences

The bounded representation supports invariance, counterfactual, and temporal contract
tests. It omits continuous dynamics, learned renderers, automatic V5 classification,
and curriculum scheduling. Public descriptions must be authored without secrets.
Complete exports remain evaluator-only, and consumers must retain eligibility metadata.
Hidden-fact fractions and residuals are diagnostics, not alignment measures or calibrated
epistemic uncertainty. See [the guide](../synthetic_worlds.md) for API and research limits.
