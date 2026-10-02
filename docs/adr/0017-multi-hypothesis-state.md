# ADR 0017: Retain hypotheses independently of their actor projection

Status: Accepted for experimental opt-in diagnostics.

## Context

The existing HypothesisSet overlay stores alternative strings, while uncertainty updates retain
unresolved names. Neither represents changing evidence, conflicting verifiers or multiple diagnostic
trade-offs. Reducing these alternatives to one scalar or one incumbent would lose unresolved structure.

## Decision

Add immutable external records for candidates, assessments and declared triggers, reusing observable
EvidenceReference, EpistemicContext and non-TRAIN labels. Append history only. Explicit same-assessor
supersession revises a live assessment while retaining its original; it cannot remove another verifier.
Validate every successor against the host's authoritative prior before host-managed atomic storage.

Use strict seven-dimensional Pareto comparison only for complete, agreed vectors. Unknown or
conflicting candidates remain incomparable. Dominance is a diagnostic and never deletion. Bound actor
payloads with insertion-order pages, text limits and omission counts. Reuse the disabled
competing_hypotheses flag and HypothesisSet output, with its scalar uncertainty supplied explicitly by
the host rather than inferred from the candidate set.

## Consequences

The API retains diverse explanations and supports external history inspection without adding an
actor write capability. Authentication, evidence semantics, normalized measurement protocols and
storage concurrency remain host responsibilities. Conservative exact agreement can enlarge frontiers;
full history and Pareto computation require host resource budgets. No durable database, automatic
trigger detector, reward, Gaussian collapse or learned effectiveness result is introduced.

The [guide](../hypothesis_state.md) specifies the API and review responsibilities.
`tests/test_hypothesis_state.py` verifies retention, comparison, projection and serialization boundaries.
