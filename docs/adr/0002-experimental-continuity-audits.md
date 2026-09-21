# ADR 0002: Experimental public-evidence continuity audits

## Status

Experimental; disabled by default. 2026-09-21.

## Context

Existing trajectory, semantic decomposition, representation provenance, memory safety and
verification interfaces provide the required boundaries. REF-SOT suggests compact state-conditioned
historical support, but its findings do not establish semantic robustness or motivated-forgetting
detection. The implementation plan is in
[history](../../history/2026-09-21-sot-continuity-plan.md).

## Decision

Add pure diagnostic functions to semantic_intent_robustness and explicit pipeline/evaluator entry
points. Reuse existing records and validators rather than create a controller or a parallel evidence
architecture. All four flags default to disabled, with fixed research/shadow maturity. No runtime
activation, optimizer input, policy action, automatic logging or network access is added.

Bound state snapshots to four normalized scalars plus provenance and measurement metadata.
Comparisons require compatible measurement spaces; unavailable telemetry remains null. Backend
extraction and calibration belong to separately validated adapters, not the synthetic fixtures.

Bind public commitments and terminal update declarations to the existing action-bound event
sequence. Preserve original evidence and memory trust. Hash the evaluated public records, current
state and event prefix to prevent replay against a different analysis context. State similarity may
rank independently relevant evidence but cannot withdraw its relevance or confer authority.

Require independent pressure and favored-action evidence before reporting possible motivated
forgetting. Preserve omission, recovery, update, uncertainty and missing-data observations as
separate outputs. Report metrics with explicit denominators; do not aggregate an alignment score.

## Alternatives and consequences

A transcript-derived proxy is available only with its explicit proxy label. Treating it as measured
internal state would conceal measurement limitations. Automatically installing a controller or
reward term would exceed both source evidence and this task's scope.

Callers must authenticate event producers and independently assess semantic relevance, active
context and pressure. Cryptographic digests bind records, not their truth. Legacy untyped reference
IDs cannot establish source kinds independently. The host must keep public summaries free of
private reasoning. Simultaneous terminal replacement chains are conservatively rejected.

## Verification

The four continuity test modules verify default-off behavior, incompatible measurements, matched
controls, provenance/authority preservation, replay rejection, typed verifier boundaries and metric
denominators. The synthetic suite runs with
`python -m evaluation.suites.robustness.sot_continuity`. It tests software contracts only. Empirical
model experiments remain unrun and are described in the module README.
