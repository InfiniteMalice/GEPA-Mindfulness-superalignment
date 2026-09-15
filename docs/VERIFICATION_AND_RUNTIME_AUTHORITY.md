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
Claims marked `supported` or `contradicted` must retain at least one evidence reference; an
evidence-free claim can remain `unverified` but cannot be promoted by mutating its status.
`EvidenceState` preserves superseded claims and resolves only validated, acyclic
`superseded_by` links. `EvidenceClaim` does not create or prove a `WorldStateChange`. For example,
the statement “I fixed the bug” can be an unverified claim without establishing an observed
artifact change.

Verification: `tests/test_world_evidence_state.py` checks construction, supersession, corruption
revalidation, and exact JSON round trips.

### Host commit adapter: propose → verify → commit

`gepa_mindfulness.verification.state.commit_verified_claim()` connects the existing evidence
records, local and relational verifier results, and one-use runtime WRITE authorization.
Constructing or deserializing `EvidenceState` remains a record operation and conveys no authority.

The trusted host performs this sequence:

1. Preserve a proposed `EvidenceClaim` separately from published evidence state. A generated claim
   can remain `unverified`; the commit adapter rejects that status even with accepted references.
2. Authenticate the evidence-producing action and its local and relational verifier results.
   Independently classify captured evidence into accepted references and quarantine. Rejected,
   injected, or evaluator-manipulated outputs belong in quarantine, including their derivatives.
3. Bind an enrolled WRITE grant and commit action policy to `evidence_update_scope(state, claim,
   source_action_id=..., supersedes=...)`. This digest covers the complete previous state,
   proposed update, and evidence-producing action ID. Both verifier results identify that source
   action. The separate `action` argument identifies the proposed memory write; local verification
   does not claim that the memory write has already occurred. The host verifies that each finding
   concerns this exact claim; evidence IDs alone cannot establish entailment.
4. Call `commit_verified_claim()` with host-owned state, results, registry, clock, and evidence
   lists. The adapter validates the update before consuming authorization. The adapter accepts
   only supported claims with affirmative support or contradicted claims with a contradiction
   finding. Every used reference needs host acceptance and an observable source kind. Quarantine
   wins by reference ID even if a caller relabels its source kind.
5. Publish the returned snapshot under the host's state lock or compare-and-swap transaction.
   Serialize authorization consumption and publication with other writers. Retain rejected
   proposals and outputs in a separate diagnostic store; do not append them to accepted state.

The adapter does not authenticate callers, verifier identities, or evidence contents. It does not
detect injection text or establish semantic truth. A caller with arbitrary Python execution can
construct its own registry and evidence allowlist; this is not a sandbox against that caller.
The host must keep these inputs outside the model/tool surface. Deployment review verifies that
ownership and the publication transaction. The package supplies no durable evidence store,
cross-process transaction, automatic quarantine propagation, or commit-to-disk recovery.

Failed validation returns no new state and does not consume a write decision. Successful
authorization consumption is one-use; a host publication failure requires explicit recovery and
a fresh authorization, not replay of that decision. Irreversible memory writes retain the existing
human-approval requirement in `consume_authorization()`.

With `supersedes=(old_id,)`, a verified contradiction reopens the current claim by appending a
`contradicted` successor. The older claim keeps its proposition and original references.
`EvidenceState.merge_equivalent()` proposes one unverified claim for whitespace-equivalent
propositions and retains every distinct source reference. It leaves original records intact.
The merge requires at least two distinct current claims after resolving supersession links.
It does not infer paraphrase or translation equivalence. Broader equivalence and promotion
require an independently verified host decision. Embedding similarity confers no authority.

Legacy `EvidenceClaim` and `EvidenceState` JSON fields and accepted status values are unchanged.
Verification: `tests/test_governed_evidence_commit.py` exercises rejection, reference binding,
quarantine precedence, exact-update authorization, replay, contradiction reopening, and retained
equivalence provenance. Existing supersession DAG checks remain in `tests/test_world_evidence_state.py`.

### Ontology context and provenance graphs

The ontology workbench's `buildContextBundle()` retains the existing one-hop semantic
neighborhood, assessments, independent support/opposition, conditions, and invariants. Semantic
relations may contain cycles. Unknown requested targets fail instead of producing an empty view.

An optional `EvidenceItem.derivedFrom` array names parent evidence IDs in the supplied assessments.
When derivation links are supplied, the exporter rejects dangling parents, conflicting evidence
identities, and provenance cycles. The exporter includes the selected evidence and all its
ancestors in parent-first `provenanceEvidence`, even when an ancestor belongs to an assessment
outside the semantic neighborhood. This is a separate directed acyclic graph (DAG); semantic
ontology relations do not participate in its cycle check. Legacy inputs without derivation links
keep their existing bundle shape. Unstructured provenance strings are not inferred into edges.

JSON, YAML, and Markdown exports preserve explicit derivation provenance. The browser continues
to produce `generated_noncanonical_bundle` artifacts and has no authoritative commit endpoint.
Verification: `apps/gepa-ontology-workbench/app/lib/bundles.test.ts` checks semantic cycles,
provenance rejection, ancestor retention, context targets, and export behavior.

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
