# Metacognitive assessment contracts

`gepa_mindfulness.verification.metacognitive_records` supplies inert, immutable diagnostic
records. Import and construction install no monitoring hook, execution path, reward component,
evidence writer or model dependency. Computational enablement belongs to subsequent stages.

`SourceSnapshot` retains detached original `EvidenceState`, optional `HypothesisState`, optional
`EpistemicStateEstimate`, and existing `EpistemicContext`/`V5EvaluationCell` identity. It also
requires source event IDs and provenance. `AssumptionBinding` connects an existing hypothesis
to an existing claim, applicability-condition claim IDs and unresolved alternative IDs.
The original evidence statuses and conflicting assessor histories remain unchanged.

`SharedDependency` records a common declared assumption or evidence reference for at least two
retained hypotheses. Its links must exist in the supplied sources. It does not establish that
the common source is false, or that the hypotheses are independent.

`MonitoringFinding` retains source events, evidence, affected claims/hypotheses, uncertainty and
a proposed next diagnostic step. Findings are explicitly unverified or unavailable; changing
a wire label cannot turn them into verified evidence. `MetacognitiveAssessment` binds those
findings to sources, method/version, separate uncertainty dimensions and monitoring lifecycle.
`AssessmentStatus.UNDER_REVIEW` describes monitoring review, not evidence truth.
`unassessed_domains` explicitly retains missing diagnostic coverage. An assessment with any
unassessed domain cannot declare its monitoring status complete.

All records use the versioned `metacognitive-v1` wire schema with an exact record type and exact
fields. `to_dict()` revalidates recursively and returns detached JSON. `from_dict()` rejects
unknown/missing fields, invalid identities, unsupported schemas and unresolved links. Unknown
uncertainty remains `None`; booleans and nonfinite/out-of-range numeric values are rejected.
Observable reference kinds remain host declarations and do not authenticate their contents.
Private reasoning and latent-source references cannot certify these diagnostics.

Bounds are 512 evidence claims, 64 hypotheses, 512 hypothesis assessments, 64 triggers,
256 assumption bindings, 1,024 source event IDs, 32 provenance entries and 4,096 distinct evidence
references. An assessment retains at most 512 findings and shared dependencies. These bounds
limit records accepted by this diagnostic view; they do not prune the host's original history.

The contract layer validates declared run, case, repeat and system-version consistency. Event
existence, causal chronology, and association of a source history with a robustness stripe must
still be validated against the host's protected episode by the PEO adapter. A string event ID
alone proves neither chronology nor authenticity. A recorded `supported` claim remains a source
declaration; construction of a source snapshot is not independent verification of that claim.

Run `python -m pytest -q tests/test_metacognitive_records.py` for schema, unknown-state,
identity, missing-link, conflict/history, snapshot, self-certification and reward-admission
rejection checks. These tests establish software contracts, not alignment effectiveness.
