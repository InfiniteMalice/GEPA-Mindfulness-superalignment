# ADR 0009: Skill metadata and failure-layer hypotheses

Status: Accepted for experimental opt-in use.

## Context

FailureGraph distinguishes causal roles; SkillLifecycleStore controls durable skill
versions. PR-8 needs failure-layer diagnosis and separate routing/guidance metadata
without letting diagnoses grant repair, training or persistence authority.

## Decision

Add an explicit-call diagnostic sidecar over a complete reconciliation window and an
existing failure graph. Verify event/evidence linkage, retain unknowns and multiple
hypotheses, and preserve the original graph. Recorded verifier findings and externally
reported claims have distinct basis labels. Residuals remain descriptive.

Add immutable host-owned SkillCard metadata tied to lifecycle artifact identity and
digest. SkillBank returns review proposals only. Routing and knowledge gaps target
different text fields. Other layers require other review surfaces. Foundational changes
and retirement are blocked regardless of operational performance.

## Consequences

Existing defaults, case count, reward computation and authority protocols remain stable.
Hosts authenticate evidence and review conclusions. A metadata/artifact binding does not
certify the text. Human governance owns foundational classification and any replacement
of trusted configuration. No automatic application, training or retirement is introduced.

See [the guide](../skill_failure_localization.md) for the interface, source mechanisms,
repository hypotheses and executable validation contract.
