# PR-5 Verification and Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`. Keep
> verification levels and authority roles distinct in types and serialized output.

**Goal:** Separate world state from evidence state, add two verifier levels, localize failures, and
enforce bounded recovery and runtime authority.

**Architecture:** A focused `gepa_mindfulness.verification` package owns immutable state,
verification results, failure graphs, and recovery. A separate runtime-governance module defines
who may propose, execute, verify, audit, or authorize actions.

**Tech Stack:** Python dataclasses, enums, protocols, graph adjacency maps, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Separate world and evidence state

**Files:**

- Create: `gepa_mindfulness/verification/__init__.py`
- Create: `gepa_mindfulness/verification/state.py`
- Test: `tests/test_world_evidence_state.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class WorldStateChange:
    change_id: str
    action_id: str
    artifact_ref: str
    before_digest: str | None
    after_digest: str
    observed_at: str


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    proposition: str
    evidence_refs: tuple[EvidenceReference, ...]
    status: Literal["unverified", "supported", "contradicted", "superseded"]
    superseded_by: str | None = None


@dataclass(frozen=True)
class EvidenceState:
    claims: tuple[EvidenceClaim, ...]
```

1. Write failing tests proving the statement `I fixed the bug` creates an unverified claim but no
   `WorldStateChange`.
2. Add tests requiring an observed artifact digest and action ID for world-state changes.
3. Add tests proving supersession preserves the original claim and rejects dangling or cyclic
   `superseded_by` links.
4. Run tests and verify module-not-found failure.
5. Implement the records and `EvidenceState.resolve(claim_id)` without a universal world object.
6. Run `python -m pytest tests/test_world_evidence_state.py -q`.

## Task 2: Add separate local and relational verifier contracts

**Files:**

- Create: `gepa_mindfulness/verification/interfaces.py`
- Test: `tests/test_verifier_interfaces.py`

**Interfaces:**

```python
class VerificationLevel(str, Enum):
    LOCAL_EXECUTION = "local_execution"
    RELATIONAL_EVIDENCE = "relational_evidence"


@dataclass(frozen=True)
class VerificationEvidenceBinding:
    field_name: str
    evidence_refs: tuple[EvidenceReference, ...]


@dataclass(frozen=True)
class LocalVerificationResult:
    action_id: str
    executed: bool
    arguments_valid: bool
    schema_valid: bool
    authorization_valid: bool
    intended_operation_observed: bool
    irreversible_action_permitted: bool | None
    evidence_refs: tuple[EvidenceReference, ...]
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] = ()


@dataclass(frozen=True)
class RelationalVerificationResult:
    action_id: str
    task_fit: bool
    dependencies_satisfied: bool
    contradiction_status: str
    provenance_intact: bool
    authorization_scope_valid: bool
    claimed_outcome_supported: bool
    repeated_failed_route: bool
    evidence_refs: tuple[EvidenceReference, ...]
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] = ()


class LocalExecutionVerifier(Protocol):
    def verify_local(self, action: ActionRecord) -> LocalVerificationResult: ...


class RelationalEvidenceVerifier(Protocol):
    def verify_relational(
        self,
        action: ActionRecord,
        evidence_state: EvidenceState,
    ) -> RelationalVerificationResult: ...
```

1. Write failing tests proving local success does not imply relational success and that a
   relational verifier cannot fabricate local execution.
2. Require a field-keyed binding to observable evidence for every affirmative boolean field. Treat
   contradiction status `unknown` as no finding; require its own observable binding for `none` or
   `contradicted`. Permit optional diagnostic bindings for negative boolean findings.
3. Run tests and implement distinct result types and protocols.
4. Add adapters that convert both result types into `VERIFICATION_RESULT` envelopes without
   collapsing them into one scalar. Require an explicit keyword-only, nonempty verifier-reference
   sequence in each adapter and bind the references in both the payload and envelope.
5. Run verifier and action-bound event tests.
6. Extend canonical action-bound sequence validation to dispatch the exact legacy schema or the
   exact level-tagged schema and validate the new result, causal parent, action, evidence, and
   verifier links.

## Task 3: Build epistemically qualified failure graphs

**Files:**

- Create: `gepa_mindfulness/verification/failure_graph.py`
- Test: `tests/test_failure_graph.py`

**Interfaces:**

```python
class FailureRelation(str, Enum):
    CAUSAL = "causal"
    CONTRIBUTING = "contributing"
    PRECEDING = "preceding"
    CORRELATED = "correlated"
    HYPOTHESIZED = "hypothesized"


@dataclass(frozen=True)
class FailureNode:
    failure_id: str
    event_id: str
    summary: str
    observed_at: str
    evidence_refs: tuple[EvidenceReference, ...]


@dataclass(frozen=True)
class FailureEdge:
    source_id: str
    target_id: str
    relation: FailureRelation
    verifier_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FailureLocalization:
    first_anomaly: str
    root_cause: str | None
    decisive_failure: str | None
    symptoms: tuple[str, ...]
    recoverable_until: str | None
```

1. Add failing tests for a chain where the first anomaly differs from the decisive failure and the
   last symptom differs from the root cause.
2. Add tests rejecting causal edges without verifier references, dangling nodes, cycles in causal
   edges, and unsupported root-cause labels.
3. Allow hypothesized roots when evidence is insufficient and serialize that epistemic status.
4. Implement deterministic topological localization for supported causal subgraphs. Do not infer a
   causal edge from event ordering alone.
5. Run failure graph tests.

## Task 4: Enforce runtime authority

**Files:**

- Create: `gepa_mindfulness/runtime_governance.py`
- Test: `tests/test_runtime_authority.py`

**Interfaces:**

```python
class RuntimeRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    AUDITOR = "auditor"
    HUMAN = "human"


class RuntimeCapability(str, Enum):
    READ = "read"
    PROPOSE = "propose"
    WRITE = "write"
    EXECUTE = "execute"
    VERIFY = "verify"
    AUDIT = "audit"
    AUTHORIZE_IRREVERSIBLE = "authorize_irreversible"
```

1. Add failing tests for Planner write denial, Auditor execute denial, Executor action with explicit
   authorization, Verifier independence, and Human irreversible authorization.
2. Add a test proving a Prime/coordinator role cannot acquire capabilities not listed in its typed
   grant.
3. Implement `AuthorityGrant` and `authorize_action()` with least-authority exact matching.
4. Require an observable authorization reference for irreversible execution.
5. Run runtime authority tests.

## Task 5: Add bounded recovery

**Files:**

- Create: `gepa_mindfulness/verification/recovery.py`
- Test: `tests/test_bounded_recovery.py`

1. Add failing table-driven tests mapping transient runtime error to retry, argument error to
   argument repair, strategy/dependency failure to replan, missing user fact to clarification, and
   unresolved consequential ambiguity to escalation or abstention.
2. Add tests for explicit `max_retries` and `max_replans`, exhaustion, and repeated known-failed
   route rejection.
3. Implement `RecoveryBudget`, `FailureCategory`, `RecoveryAction`,
   `FailureClassificationBinding`, and `RepeatedRouteFinding` as immutable exact records. Bind each
   classification to the canonical failure event, complete `ActionRecord` digest, observable
   failure evidence, verification event, nonempty verifier references, and a complete verifier
   result. These caller-created records remain untrusted enrollment candidates.
4. Implement `RecoveryStateStore.enroll()` as the runtime-owned ledger boundary, then require the
   runtime owner to authenticate and explicitly enroll exact classification and repeated-route
   snapshots. Keep the store identity, authenticated records, revisions, counters, pending
   proposals, and consumed decisions outside the public store handle.
5. Make `select_recovery()` resolve only store-enrolled classification and optional route-finding
   IDs and reserve one proposal at the current store revision without consuming a retry or replan
   count. Make `consume_recovery()` atomically revalidate and consume only the exact pending
   proposal. Reject stale revisions, concurrent proposals, replayed decisions, cross-store or
   caller-recomputed records, and repeated routes without an authenticated link to a prior consumed
   decision.
6. Limit every count, maximum, and revision to the inclusive JSON-safe integer range
   `0..9_007_199_254_740_991`. Forbid unlimited sentinels, negative counts, booleans, floats, and
   larger integers.
7. Run bounded recovery and failure graph tests.

## Task 6: Verify and commit PR-5

1. Run world/evidence state, verifier, failure graph, authority, recovery, action-bound event, and
   reward provenance tests.
2. Run Black, Ruff, and mypy on the verification package and runtime governance module.
3. Run JSON round-trip smoke tests for all public records.
4. Run `git diff --check`, inspect the diff, and commit with message
   `feat: separate verification state and runtime authority`.
