# ADR 0006: Qualitative eligibility before numeric evidence use

Status: accepted for experimental explicit use; no runtime integration.

## Context

The scalar estimator and fusion APIs cannot infer whether a referenced claim is stale, superseded
or tainted. The repository already owns evidence claims, memory trust checks and write authority.

## Decision

Extend the existing EvidenceClaim status vocabulary without changing its serialized fields or
governed commit policy. Add EvidenceUseAssessment as an adapter over existing records, with
explicit host quality thresholds and content/influence labels. Withhold numeric inputs on failed
qualitative checks while retaining their original values in audit output. Keep display summaries
alongside immutable originals and security metadata.

## Alternatives and consequences

Do not multiply variance by arbitrary trust scores: their units and calibration are unrelated.
Do not silently erase rejected inputs or substitute superseding claims. Do not duplicate the
evidence store or change legacy memory retrieval semantics. The adapter requires explicit calls,
so hosts can bypass it; host integration remains necessary. Complete snapshots increase audit size.
Caller declarations and summary semantics remain unauthenticated. The adapter provides no durable
storage, process isolation, automatic freshness refresh, reward or authorization.

The [guide](../evidence_memory.md) defines eligibility, input ownership and validation commands.
