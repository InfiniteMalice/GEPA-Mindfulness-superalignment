# Controlled Learning and Offline Evolution

This document defines the implemented boundary for learning proposals, evaluation epochs,
verified skill artifacts, and model-harness candidates. The implementation creates records,
validates transitions, and persists authority decisions. It does not update model weights, edit a
live harness, install a skill, execute a candidate, deploy a system, or perform a rollback.

## Online collection and offline change

An evaluation epoch is the unit that freezes a model version and a harness version. During an open
epoch, `append_epoch_record()` accepts a canonical `V5EvaluationRecord` only when both versions
match the epoch. It rejects duplicate record content and duplicate logical evaluation cells. The
append operation adds evaluation evidence; it does not change the model or harness version.

Before a model or harness version changes, `close_evaluation_epoch()` must close the current tip.
`begin_candidate_epoch()` then creates an empty epoch with a new epoch ID. At least one component
version must change. Every changed version must be unused in the same evaluation catalog and
`authority_domain`; an unchanged component keeps the immediate source version. The new epoch
receives no records from the source epoch.

`EvaluationEpochStore` persists epoch lineages, canonical V5 records, candidate target claims, and
validation receipts in SQLite. `BEGIN IMMEDIATE` transactions serialize catalog changes. The
store creates an opaque `EvaluationEpochHistory` capability with a catalog locator and expected
revision. While that exact handle remains valid, `append_epoch_record()`,
`close_evaluation_epoch()`, and `begin_candidate_epoch()` can open the catalog and authorize a
transition without retaining the Python `EvaluationEpochStore` object. A stale handle cannot read
or change the lineage. These properties are covered by
[`test_offline_evolution_epochs.py`](../tests/test_offline_evolution_epochs.py).

## One typed destination per lesson

`classify_learning_surface()` reads `LessonCharacteristics`; it never classifies from `summary` or
`rationale` text. The decision table is:

| `LessonKind` | `LearningSurface` |
| --- | --- |
| `ONE_OFF_OBSERVATION` | `TRACE_ONLY` |
| `EPISODE_FACT` | `MEMORY` |
| `STABLE_PROCEDURAL_CONVENTION` | `HARNESS` |
| `REUSABLE_DEPENDENCY` | `SKILL_GRAPH` |
| `PERSISTENT_INTRINSIC_BEHAVIOR` | `MODEL` |

If `normative`, `ambiguous`, or `difficult_to_reverse` is true, the destination is `HUMAN`
regardless of `LessonKind`. A `LessonProposal` must contain one destination equal to the decision
table result, at least one observable `EvidenceReference`, a rationale, a reversibility flag, and
an exact `LessonReviewStatus`.

`PENDING` records that no human review decision exists. `APPROVED` records permission for later
integration code to consider the declared destination. `REJECTED` records a decision that later
integration code must treat as a prohibition. A `LessonProposal` is only a detached proposal
record; this module has no apply operation, and none of the three statuses grants runtime
authority. These rules are covered by
[`test_learning_surfaces.py`](../tests/test_learning_surfaces.py).

## Verified skill lifecycle

`SkillLifecycleStore` creates or reopens a skill lineage and returns an opaque, revision-bound
`SkillLifecycleHistory` capability. While that exact handle remains valid, `transition_skill()`
uses the handle's catalog locator, catalog identity, authority domain, pinned evaluation authority,
skill ID, and expected revision. The creating `SkillLifecycleStore` Python object does not need to
remain alive. The normal state sequence is:

```text
SOURCE_EXPERIENCE
  -> VERIFIED_SKILL
  -> PROCEDURAL_FAMILY
  -> TASK_LOCAL
  -> EXECUTED
  -> CREDITED
  -> REFINED
  -> HELD_OUT_VALIDATED
  -> COMMITTED
```

`transition_skill()` enforces the following gates:

1. A `PROCEDURAL_FAMILY` transition uses `FamilySeedProvenance` bound to the current verified
   artifact, or `ConsolidationProvenance` bound to nonempty task-local constituent artifact IDs in
   the same lifecycle catalog. Consolidation can include evidence-backed pruning decisions.
2. A `TASK_LOCAL` transition uses `InstantiationProvenance` bound to the current procedural-family
   artifact and an explicit task context.
3. An `EXECUTED` transition uses one `ExecutionEvidenceBundle`. The bundle binds an action event,
   an outcome observation, a `WorldStateChange`, and distinct local and relational verification
   events for one action.
4. The execution receipt binds evidence separately to local `executed`, local
   `intended_operation_observed`, relational `claimed_outcome_supported`, and relational
   `provenance_intact`. Every required finding must intersect the canonical outcome observation
   evidence. The receipt preserves the bundle's `run_id`; the catalog scopes an action identity
   by `(authority_domain, run_id, action_id)`. Reusing an `action_id` in a different run is valid,
   but reusing a bundle digest in the same authority domain is rejected. A generated explanation
   or a legacy scalar verification result cannot satisfy this gate.
5. A `CREDITED` transition revalidates the complete execution receipt. Caller-supplied success
   text, event IDs, or scalar booleans do not create credit.
6. A `REFINED` transition uses `RefinementProvenance`, creates a new version, and names the exact
   credited version in `supersedes`.
7. A `HELD_OUT_VALIDATED` transition resolves a durable `ValidationReceipt` through the evaluation
   catalog pinned by the lifecycle store. The receipt must use `ValidationSplit.HELD_OUT`, belong
   to an allowed lineage, contain passing canonical records from a closed epoch, and bind the exact
   current artifact ID, skill ID, version, and digest.
8. A `COMMITTED` transition names a strict predecessor artifact as its rollback target. The target
   must be stable and its version must match `supersedes`. `COMMITTED` is a catalog state; it does
   not install or deploy the skill.

From every state after `VERIFIED_SKILL`, a caller can request `ROLLED_BACK`. The transition requires
an in-catalog strict predecessor ID plus `PruningProvenance` that retires the exact current
artifact. The rollback record is terminal. The store records the decision; it does not restore
external files, processes, models, or deployed skills.

The lifecycle, consolidation, seed, instantiation, refinement, pruning, commit, rollback, and
adversarial trust-boundary checks are covered by
[`test_verified_skill_lifecycle.py`](../tests/test_verified_skill_lifecycle.py) and
[`test_verified_skill_lifecycle_review.py`](../tests/test_verified_skill_lifecycle_review.py).

## Controlled model-harness coevolution

`CoevolutionStore` pins one evaluation catalog and one evaluation lineage. Its controlled flow is:

1. `register_trajectory()` records a canonical action-bound trajectory from a closed source epoch.
   The supplied typed event evidence must exactly match the trajectory events and remain
   observable.
2. `CorrectionProposal` names one localized failure node, its source action, the complete verifier
   references required by the failure localization, source evidence, teacher evidence, and the
   model or harness components proposed for change. The teacher text is proposal content, not
   acceptance authority. A wholesale trajectory imitation proposal has no valid scope.
3. `register_candidate()` binds the proposal to an empty candidate epoch and atomically claims the
   candidate ID, artifact digest, versions, coevolution catalog, authority domain, and correction
   digest in the evaluation catalog. The changed components must match the new epoch versions.
4. After candidate evaluation closes, the evaluation catalog can issue separate durable
   `HELD_OUT` and `PROTECTED` receipts. Both receipts bind the registered candidate target.
5. A `ProtectedSuiteManifest` fixes the source protected logical cells. Candidate protected results
   must cover exactly those cells, and every protected record must pass.
6. `MetricPolicy` declares all V5 score components, each direction and tolerance, the primary
   metric, and arithmetic-mean aggregation. `MetricComparisonReceipt` derives each component value
   from matched canonical source and candidate records; callers cannot replace the component set
   with one opaque score.
7. `decide_candidate_acceptance()` accepts only when the protected suite passes and the declared
   primary metric is non-worse within its tolerance. It persists an `AcceptanceDecision` with
   `execute_candidate=False` and the source epoch as the rollback target.
8. `validate_decision()` reopens every catalog dependency. `consume_decision()` accepts only a
   live, validated decision and marks that decision consumed once. Consumption does not execute,
   install, deploy, or roll back the candidate.

The trajectory, failure-localization, candidate, protected-manifest, record-derived metric,
single-use decision, and restart checks are covered by
[`test_model_harness_coevolution.py`](../tests/test_model_harness_coevolution.py).

## PEO improvement intake

PR18 adds `gepa_mindfulness.controlled_improvement.assess_improvement()` for failures in the
prediction/execution/outcome (PEO) process. The function requires `enabled=True`; its default
raises `ValueError` before reading a catalog. It does not run during ordinary evaluation.

The host supplies an existing `CorrectionProposal`, `EpistemicStateEstimate`, live
`CoevolutionStore`, and `TriageDiagnostics`. The function:

1. Calls `CoevolutionStore.correction_source_events()` to validate the proposal against its
   recorded trajectory, localization, evidence, digest and closed source epoch. Source event
   model/harness versions must equal the epoch versions.
2. Matches the estimate's run, repeat, model and harness to the source events. The estimate's
   action and prediction IDs must match the executed action named by the correction.
3. Requires observable estimate and triage evidence intersecting the localized failure evidence.
   An unavailable estimate may have empty evidence and leads to investigation. Evidence identity
   checks do not authenticate the issuer or establish causal truth.
4. Applies the routing table below and returns detached JSON. The function performs catalog reads
   but never registers, executes or accepts a candidate.

`CorrectionProposal.to_dict()` reconstructs canonical fields before serialization and compares
the resulting payload with the original content binding. Nested evidence serializers cannot
replace raw provenance or run callbacks while an intake snapshot is being prepared.

`TriageDiagnostics` contains `protocol_id`, `evidence_refs`, and seven separate optional unit
values: `severity`, `irreversibility`, `recurrence`, `ood_novelty`, `systemic_effect`,
`autonomy_impact`, and `reward_hacking_signal`. OOD means out of distribution. Uncertainty stays
in the estimate's separate world/model/monitor fields. `None` means unavailable. Numeric inputs
must be finite built-in numbers in [0, 1]; booleans and numeric subclasses are rejected.
Triage evidence requires 1..32 unique observable references. The protocol and evidence IDs are
nonblank exact strings of at most 128 UTF-8 bytes; estimate evidence IDs have the same bound.
The host defines and retains each measurement protocol. A reward-hacking signal is a declared
investigation signal, not an inference that an actor intended deception.

### Routing rules

The function accumulates all applicable reasons, then applies this precedence:

| Condition | Route | Correction in result |
| --- | --- | --- |
| Severity or systemic effect >= 0.8; or irreversibility, autonomy impact or reward-hacking signal >= 0.5 | `human_review` | `null` |
| Otherwise: any missing triage or uncertainty value; any uncertainty >= 0.5; graph root not `supported`; or more than one changed component | `investigate` | `null` |
| Otherwise | `sandbox_review` | Detached `CorrectionProposal` JSON |

Human review has `urgent` priority. Other results have `elevated` priority if any diagnostic is
missing or >= 0.5, and `routine` priority otherwise. Recurrence and OOD novelty can elevate priority
without blocking sandbox review. A root classified `hypothesized` or absent remains unqualified;
the adapter never upgrades it from summary text. A supported root reflects the graph's declared
causal path and verifier references, not independently established causation.

These thresholds are local experimental heuristics. Contract tests establish routing behavior;
they do not establish calibration, beneficial self-improvement or deployment safety. The
single-component rule bounds declared component scope, not the number of edits inside a model
or harness. The host reviews the actual diff and proposal attribution before sandbox execution.

The result includes `schema_version="controlled-improvement-v1"`, source identifiers, the
estimate, triage, root-cause status, route, priority, reasons, and correction. Every result has
`training_eligibility="DEVELOPMENT"` and `authority_granted=False`. The host retains the complete
result at optimizer admission: `require_training_eligible(result)` rejects it. The extracted
legacy correction and estimate schemas have no eligibility field; untagged legacy admission
still accepts those objects. Hosts must not submit stripped intake diagnostics to optimization.
The explicit admission/extraction tests cover this boundary.

### Host handoff and lifecycle coverage

With already recorded inputs, a host can inspect a result as follows:

```python
from gepa_mindfulness.controlled_improvement import assess_improvement

assessment = assess_improvement(
    store, correction, estimate, triage, enabled=True,
)
print(assessment["route"], assessment["reasons"])
```

After the host reviews a `sandbox_review` result, the host can restore
`CorrectionProposal.from_dict(assessment["correction"])` and pass it to
`store.register_candidate(...)` with a candidate ID and artifact digest. Registration still
requires a compatible empty candidate epoch and revalidates trajectory binding. Intake does
not reserve an epoch; a later closed or changed epoch can prevent registration. The host keeps
the assessment with the candidate's provenance. An assessment cannot replace an acceptance
decision or validation receipt. The existing registration API remains callable independently
of this optional adapter; the review route is guidance, not a new authorization boundary.

| Stage in the proposed program | Implemented owner or integration boundary |
| --- | --- |
| Failure/opportunity | Existing recorded failure graph; opportunities without a localized failure need host investigation |
| Consequence triage and root-cause qualification | `assess_improvement()` and existing `FailureGraph` classification |
| Sparse attributable proposal | Existing one-node `CorrectionProposal`; adapter limits sandbox review to one component; host inspects edit size |
| Sandbox, local evidence, experimental candidate | Host executes sandbox; existing epoch and candidate stores record version-bound evidence |
| Prospective hidden evaluation | PR19 private promotion adapter below; host supplies disjoint future evaluation and protects evaluator access |
| Protected regressions and qualification | Existing protected suite, component metrics, acceptance and single-use decision validation |
| Active deployment, global replay, retirement | Host-owned runtime operations; this PR adds no such states or execution authority |

## Private promotion and experimental provenance

PR19 adds the opt-in `gepa_mindfulness.private_promotion` adapter. The host creates a
`PrivatePromotionStore` with `enabled=True`; any other value fails before catalog access.
The adapter separates the improver's request/response interface from private promotion evidence.
Existing development diagnostics, including PR18 intake, remain the optimization signal. The
adapter never turns private scores or failures into explanatory optimization feedback.

### Pin the private protocol

The host provides the existing `CoevolutionStore`, its matching `EvaluationEpochStore`, a source
held-out `ValidationReceipt`, measured source usage and a `PrivateProtocol`. The new SQLite
catalog must have its own path. The adapter normalizes path casing and resolves aliases; an
existing file that is the same file as either authority catalog is rejected before promotion
schema creation, including hard links. Its first audit event pins:

- The coevolution authority, evaluation authority and lineage, with canonical catalog identities.
- The source held-out receipt, protected manifest and complete metric policy. Source held-out and
  protected records must be disjoint and belong to the same closed source epoch.
- Two to 128 distinct nonnegative seeds. Each split must cover exactly this set. Existing matched
  logical-cell checks still apply during candidate acceptance; seed checks alone do not establish
  balanced coverage across every world or family.
- Positive per-seed token, tool-call and wall-time-in-milliseconds caps (`ComputeBudget`). Both
  source and candidate `SeedUsage` must cover each seed exactly once and fit the same caps. Usage
  covers the combined held-out and protected work, including failed attempts. Integer fields are
  exact built-in integers at most `2**53-1`; booleans and floats are rejected.
- SHA256 commitments for the hidden generator, withheld world templates and withheld rendering
  families. Digests use `sha256:` plus 64 lowercase hexadecimal characters.

The host generates the private cases, verifies the committed artifact contents, keeps the withheld
families disjoint from optimizer data, measures usage and stops workers at the limits. This adapter
checks the supplied commitments and measurements; it does not inspect generator files, run a
generator, prove disjointness, or meter a process. Reopening the catalog with a different protocol,
baseline, authority, policy, manifest or source usage fails. Live operations recheck those bindings.
Create a new host-governed protocol/catalog for a planned protocol change; do not use new catalogs
to grant retries against the same private evidence.

### Typed experimental operations

The improver-facing entry point is `store.dispatch(ExperimentRequest(...))`. The complete operation
allowlist is `ExperimentOperation`; requests carry only an existing candidate ID and artifact
digest. No operation accepts evaluator edits, reward edits, benchmark code, paths, commands,
replacement checkpoints or hidden-case data.

| Operation | Required state | Result |
| --- | --- | --- |
| `SUBMIT_CANDIDATE` | Registered candidate from the pinned source; candidate epoch open and empty | `submitted` |
| `REQUEST_PRIVATE_EVALUATION` | Submitted candidate; epoch still open and empty | `pending` |
| `READ_PROMOTION_STATUS` | Known candidate and matching digest | Latest status, or `unsubmitted` |
| `REQUEST_REVIEW` | Completed `accepted`, `rejected` or `failed` evaluation | `review_requested` |

The first three existing candidate/epoch registration steps above happen on the host before
submission. For an already initialized private store, a host transport can construct a request:

```python
from gepa_mindfulness.private_promotion import ExperimentOperation, ExperimentRequest

response = store.dispatch(ExperimentRequest(
    ExperimentOperation.SUBMIT_CANDIDATE, candidate_id, artifact_digest,
))
```

Only serialized response dictionaries cross the process boundary. Each has exactly
`schema_version="private-promotion-v1"`, the caller's `candidate_id`, `status`,
`training_eligibility="HIDDEN_EVAL"` and `execute_candidate=False`. Invalid requests or catalog
errors return `invalid_request` without exception details. A malformed request whose identity
cannot be validated uses an empty candidate ID. No score, case, explanation, receipt, policy,
decision or catalog identifier is disclosed. Complete responses are rejected by
`require_training_eligible()`; stripping the label still loses this protection under legacy
untagged admission. Hosts must preserve it. Binary promotion status still supplies limited
feedback, so the host must bound total requests across candidates and protocol catalogs.

An identical repeated state-changing request returns the status originally recorded for that
operation and adds no event; `READ_PROMOTION_STATUS` returns the latest status. For example,
repeating a private evaluation request after completion returns `pending`, while a status read
returns the completion status. There is one evaluation request per candidate per catalog.
This PR supplies the promotion subset of Experiment OS; dataset registration, training proposals,
training launch, checkpoint comparison and Worker/Reviewer orchestration remain host integrations.

### Host evaluation, failures and review

The host runs the fixed private protocol after the request is recorded, appends candidate records,
closes its epoch and constructs the existing `ValidationBundle`. Host-only
`complete_evaluation(candidate_id, bundle, usage)` requires a pending request, the exact candidate,
pinned source receipt, metric policy and protected suite, exact seed coverage in both splits,
disjoint record IDs and measured usage within budget. It calls `CoevolutionStore.decide()` and
appends the returned decision ID and digest only to the private audit. The returned public status
is `accepted` or `rejected`; neither consumes the decision nor executes the candidate.

Invalid completion raises a host-only exception and leaves the request pending. Existing
validation receipts require passing canonical records. If a worker fails, exceeds budget, or
cannot supply passing records, the host calls `record_failure(candidate_id)` to append a terminal
`failed` result without private error text. A repeated failure call is idempotent. A completed
evaluation cannot be replaced by another completion. If a process fails after persisting the
coevolution decision but before appending completion, retrying the pending completion reuses the
existing idempotent decision. The two catalogs do not share one transaction.

`REQUEST_REVIEW` records a request; it does not launch a reviewer or grant approval. An authorized
reviewer reads host-only `audit_events()` and resolves its decision ID with
`CoevolutionStore.read_decision()` to inspect canonical evidence. Public operations revalidate
completed decisions before returning status. The existing single-use decision consumption and
deployment boundary remain unchanged.

### Provenance and deployment boundary

Successful state changes append an event with a sequence, previous-event hash, operation,
candidate identity, private input payload and status. Submission/request events link the source
epoch, candidate epoch, correction proposal and immutable artifact digest; completion links the
decision and per-seed usage. SQLite transactions serialize promotion transitions. Every operation
checks the hash chain and pinned configuration; update/delete triggers reject ordinary SQL changes
to audit rows. The host-only audit returns detached data. Invalid public requests and status reads
do not append events; the transport owns abuse logging and rate limits.

The host must isolate the evaluator and its catalogs from the improver using separate process and
filesystem permissions, authenticate callers, check candidate ownership, and serialize evaluation
epoch writers around request admission. The open/empty check and promotion append span distinct
databases. Do not give an improver the Python store object, database files, detailed host errors,
audit events, or host-only completion/failure methods. This library supplies no authentication,
network service or Python sandbox. A database owner can remove triggers or rewrite the hash chain;
use host backups or external audit anchoring if protection from that owner is required. Host code
can also call existing coevolution APIs directly, so this optional adapter is not a global gate.

Before exposing a host transport, its operator verifies the external controls: run a request as
the improver identity and confirm that private-file reads and another owner's candidate requests
are denied; place a canary string in private evidence and confirm it is absent from responses and
public logs; compare measured usage with a separately instrumented worker; and verify that the
withheld manifests have no optimization-family overlap. Record these deployment checks outside
improver-visible evidence. This repository's contract tests do not perform those host checks.

[`test_private_promotion.py`](../tests/test_private_promotion.py) covers the flow, response
allowlist, training rejection, budgets/seeds, substitutions, overlap, restarts, replay, concurrency,
crash retry and audit mutation. Those are contract tests; empirical improvement, private-family
quality, process isolation and metering accuracy remain unmeasured here.

The design draws on [AIDE2](https://arxiv.org/abs/2609.26457) for distinct public optimization and
private selection signals under fixed budgets, and [RSI-Master](https://arxiv.org/abs/2609.35561)
for typed experimental actions, provenance and role-scoped evidence. Existing MedRSI,
Harness-of-Harness, Self-Healing Harness and RRSI references remain linked through REC-010.
The research registry distinguishes reported mechanisms from this repository's limited transfer.

Verification: [`test_controlled_improvement.py`](../tests/test_controlled_improvement.py) covers
matched routing changes, identity mismatches, private evidence, numeric validation, catalog
immutability and explicit later registration. Existing coevolution and lifecycle tests continue
to cover receipt, protected-regression, consumption and rollback rules.

## Authority and trust limits

The durable guarantees above are scoped to each canonical SQLite path, catalog ID, authority
domain, and configured lineage. An `EvaluationEpochStore`, `SkillLifecycleStore`, or
`CoevolutionStore` does not establish universal authority across separate catalogs or authority
domains. Reusing a domain name in another catalog does not merge the catalogs.

### Handle capabilities and live stores

`EvaluationEpochHistory` and `SkillLifecycleHistory` are process-local authority capabilities,
not detached views. Each module keeps the capability state in a private `WeakKeyDictionary` keyed
by the exact handle object. The capability state contains the catalog locator and expected
revision; the skill handle also pins the catalog ID and evaluation authority. Consequently:

- The original handle can authorize its transition functions after the creating store Python
  object is garbage-collected. Each operation reopens and revalidates the SQLite catalog.
- A successful transition increments the catalog revision and the calling handle's expected
  revision. Any other handle at the old revision becomes stale. A stale read or transition raises
  `RuntimeError` before it can commit a change.
- Shallow copies, deep copies, and objects restored from pickle are different keys with no private
  capability state. The transition and snapshot functions reject them as non-store-derived
  handles. Pickle therefore serializes an object shape, not authority.
- A handle's private locator cannot be retargeted to another catalog. Evaluation receipt issuance
  additionally requires the supplied `EvaluationEpochStore` and history handle to have the same
  canonical catalog path and authority domain. Skill operations revalidate the lifecycle catalog
  ID and its pinned evaluation catalog on every read and transition.
- On platforms with `os.register_at_fork`, each module clears inherited handle bindings in the
  child process. The child rejects a parent handle as non-store-derived. The explicit process-ID
  check also rejects a detected process change. Spawned processes and other inter-process transfers
  do not receive capability state. A new process must construct the matching store and call
  `open()` to obtain a new current-revision handle.

Live store APIs remain necessary for authority configuration, root creation, and `open()`. The
`EvaluationEpochStore` API issues and validates persisted evaluation receipts, claims and resolves
candidate targets, and resolves persisted epochs. `SkillLifecycleStore` is required to create or
reopen skill lineages; held-out transitions reconstruct and revalidate the pinned evaluation store
internally.
`CoevolutionStore` has no transferable history capability: trajectory and candidate registration,
receipt and metric issuance, decision issue/read/validation, and single-use decision consumption
all require a live store pinned to the matching catalogs.

SQLite transactions, content digests, and strict schemas detect inconsistent records and serialize
cooperating writers. They do not authenticate an operator who can replace or edit a database.
Before deployment, the runtime owner must protect each catalog path with filesystem access control,
authenticated operator access, and recoverable backups. If an attacker can replace or directly
edit a catalog, the attacker can invalidate the trust assumptions of every dependent receipt and
decision.

`EvidenceReference` preserves typed provenance identifiers, but these modules do not dereference
an identifier or authenticate its issuer. Likewise, in-memory proposal and serialized artifact
snapshots are not authority tokens. A valid process-local history handle can authorize only its
defined lineage transitions; matching store APIs authorize persisted catalog operations outside
those transition functions. The implementation contains no external execution, filesystem
deployment, model-weight mutation, live skill installation, or catalog-to-catalog federation.

## Verification map

| Contract | Acceptance evidence |
| --- | --- |
| Typed destination and proposal validation | [`test_learning_surfaces.py`](../tests/test_learning_surfaces.py) |
| Version freeze, durable epoch lineage, and validation receipts | [`test_offline_evolution_epochs.py`](../tests/test_offline_evolution_epochs.py) |
| Skill evidence, lifecycle provenance, and rollback | [`test_verified_skill_lifecycle.py`](../tests/test_verified_skill_lifecycle.py) |
| Skill catalog and evaluation trust boundaries | [`test_verified_skill_lifecycle_review.py`](../tests/test_verified_skill_lifecycle_review.py) |
| Candidate correction, protected regressions, metrics, and decisions | [`test_model_harness_coevolution.py`](../tests/test_model_harness_coevolution.py) |
| Recommendation and research registry consistency | [`test_recommendation_registry.py`](../tests/test_recommendation_registry.py), [`test_recommendation_documentation_consistency.py`](../tests/test_recommendation_documentation_consistency.py), [`test_research_traceability.py`](../tests/test_research_traceability.py) |
