# Verification and Runtime Authority

## Status and scope

The `gepa_mindfulness.verification` package provides immutable validation records, verifier
protocols, failure graphs, runtime-authority decisions, and bounded recovery state. These contracts
validate caller-supplied records and preserve explicit evidence links. They do not observe external
systems, authenticate identities, dereference evidence, execute actions, or prove that a claim is
true.

`gepa_mindfulness.runtime_governance` is a compatibility import path. The implementation is in
`gepa_mindfulness.verification.runtime_governance`.

## World state and evidence state

`ArtifactObservation` binds an observation ID, artifact identity, SHA-256 digest, RFC 3339 time,
and observable evidence. `WorldStateChange` is an observed artifact transition that binds one
action to an exact after observation and, when available, an exact before observation.

`EvidenceClaim` records a proposition, its canonical evidence references, and one explicit status.
`EvidenceState` preserves superseded claims and resolves only validated, acyclic
`superseded_by` links. `EvidenceClaim` does not create or prove a `WorldStateChange`. For example,
the statement “I fixed the bug” can be an unverified claim without establishing an observed
artifact change.

Verification: `tests/test_world_evidence_state.py` checks construction, supersession, corruption
revalidation, and exact JSON round trips.

## Local and relational verification

`LocalVerificationResult` records execution-bound findings such as argument validity,
authorization validity, and observation of the intended operation.

`RelationalVerificationResult` records findings against wider evidence state, including task fit,
dependency satisfaction, provenance, contradictions, and support for the claimed outcome.
`LocalVerificationResult` does not imply `RelationalVerificationResult`. A relational result does
not establish that local execution occurred.

Every affirmative finding requires a field-keyed binding to observable `EvidenceReference`
records. Event adapters emit level-tagged `verification_result` envelopes and retain both verifier
references and the complete typed result. They do not reduce verification to one success scalar.

Verification: `tests/test_verifier_interfaces.py` checks both result contracts and their event
adapters. `tests/test_action_bound_event_sequence.py` checks the causal action, outcome, evidence,
and verifier links in complete event sequences.

## Failure-graph epistemics

`FailureGraph` keeps `causal`, `contributing`, `preceding`, `correlated`, and `hypothesized`
relations distinct. A causal edge requires verifier references. Event order alone does not create
a causal edge. Only explicit causal edges participate in the supported topological order and
causal-cycle check.

`FailureLocalization` keeps the first anomaly, decisive failure, root cause, symptoms, and recovery
boundary as separate references. These roles must satisfy explicit supported or hypothesized
directed paths; `preceding` and `correlated` edges cannot support localization. A serialized root
cause is accepted only when an explicit path supports it or labels it as `hypothesized`.

Verification: `tests/test_failure_graph.py` checks graph structure, causal support, hypothesized
status, deterministic ordering, and exact JSON round trips.

## Authority and time boundaries

`AuthorityGrantRegistry.enroll()` stores defensive grant and `ActionAuthorityPolicy` snapshots
outside the public registry handle.
`AuthorityGrantRegistry.enroll()` does not authenticate grant issuers. The runtime owner must
authenticate each issuer before enrollment. The registry is process-local and does not provide
signatures, durable storage, revocation distribution, or an identity provider.

`authorize_action()` derives the required operation from an enrolled action policy, then resolves
only explicitly named grants and matches the exact runtime role, capability, principal, action,
and authorization scope. Every irreversible write or execution also requires a policy-bound human
authorization grant and observable approval evidence. The returned decision does not execute the
action. `consume_authorization()` accepts only the exact unpredictable decision object issued by
that registry and consumes it once atomically; reconstructed, replayed, mutated, cross-registry,
or expired decisions fail closed.

`action_author_id` and `action_executor_id` are runtime-owner-authenticated inputs used to enforce
verifier independence. This package validates their relationships but does not authenticate those
identities. Authority and recovery ledgers are process-local. POSIX child processes receive fresh
locks and empty ledgers, so inherited handles and decisions cannot be used after `fork()`.

`TrustedClock` is an injected trust boundary. The runtime owner must supply a clock whose `now()`
returns an aware `datetime`. The module validates the returned value's type and timezone awareness;
it cannot establish that the clock source is accurate or tamper-resistant.

Verification: `tests/test_runtime_authority.py` checks least-authority matching, independence,
irreversible approval, registry enrollment, trusted-clock validation, expiry, and consumption.

## Recovery enrollment and consumption

`RecoveryStateStore.enroll()` creates a process-local, runtime-owned ledger for one plan, action,
failure, and finite budget. Authoritative state stays outside the public store handle. The module
does not persist or replicate this state.

Caller-created `FailureClassificationBinding` and `RepeatedRouteFinding` records are enrollment
candidates. `RecoveryStateStore.enroll_classification()` does not authenticate verifier output.
The runtime owner must authenticate the complete verifier result and evidence links before calling
`enroll_classification()` or `enroll_route_finding()`. Enrollment validates and snapshots record
shape and identity; it does not dereference evidence or authenticate verifier identities.

`select_recovery()` resolves enrolled record identifiers at an exact store revision.
`select_recovery()` reserves a proposal but does not consume its budget transition.
`consume_recovery()` performs the authoritative budget transition after it revalidates the exact
pending proposal. Stale revisions, concurrent proposals, replayed decisions, cross-store records,
and unauthenticated repeated-route claims fail closed. A forked child receives a fresh empty
ledger and cannot use an inherited store handle. Retry, replan, and revision integers are
limited to `0..9_007_199_254_740_991`.

Verification: `tests/test_bounded_recovery.py` checks policy mapping, finite budgets, enrollment,
revision conflicts, repeated-route provenance, replay rejection, and exact JSON round trips.
