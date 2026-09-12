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
validation receipts in SQLite. `BEGIN IMMEDIATE` transactions serialize catalog changes, and each
opaque `EvaluationEpochHistory` handle carries an expected revision. A stale handle cannot append,
close, or begin another candidate epoch. These properties are covered by
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

`SkillLifecycleStore` is the sole transition authority for a skill lineage. It persists artifacts
in SQLite and returns an opaque, revision-bound `SkillLifecycleHistory`. The normal state sequence
is:

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
   evidence. A generated explanation or a legacy scalar verification result cannot satisfy this
   gate.
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

## Authority and trust limits

The durable guarantees above are scoped to each canonical SQLite path, catalog ID, authority
domain, and configured lineage. An `EvaluationEpochStore`, `SkillLifecycleStore`, or
`CoevolutionStore` does not establish universal authority across separate catalogs or authority
domains. Reusing a domain name in another catalog does not merge the catalogs.

SQLite transactions, content digests, and strict schemas detect inconsistent records and serialize
cooperating writers. They do not authenticate an operator who can replace or edit a database.
Before deployment, the runtime owner must protect each catalog path with filesystem access control,
authenticated operator access, and recoverable backups. If an attacker can replace or directly
edit a catalog, the attacker can invalidate the trust assumptions of every dependent receipt and
decision.

`EvidenceReference` preserves typed provenance identifiers, but these modules do not dereference
an identifier or authenticate its issuer. Likewise, in-memory proposal and serialized artifact
snapshots are not authority tokens. Only the matching live store can authorize a persisted
transition or validate a receipt. The implementation contains no external execution, filesystem
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
