# Research credit proposals and diagnostic accounting

The opt-in native ActionOPD adapter adds separately versioned verified-decision proposals and
decision-enabled policies/reports (v2), using this same host admission and durable accounting
path. Raw teacher-agreement and diagnostic claims remain ineligible. It retains typed comparison
inputs and independent observable verification, reproduces the semantic score, and requires fresh
host semantic/use/accounting reviews. Metrics use closed schema v4; existing wire/metric schemas
remain unchanged. See [native verified ActionOPD](distillation_adapters.md#native-verified-actionopd)
for source capture, teacher isolation, event normalization, grouping, checkpoint and authority
requirements. This software integration supplies no empirical effectiveness claim.

## Status and scope

PR 1 introduces **Experimental** opt-in contracts in
[research_credit.py](../gepa_mindfulness/training/research_credit.py). These records and reports
do not grant optimizer credit, certify evidence, enforce constitutional admissibility or authorize
actions. The shared-contract stage now includes diagnostic accounting and a host-reviewed
durable event-reservation journal. An optional live host adapter now connects independently
reviewed knowledge/progress/process contributions to canonical CPU PPO/GRPO updates. Reservation
alone never pays a reward, and exported proposals and reports remain optimizer-ineligible.

This is original infrastructure derived from the operator's October 2026 integration requirements,
not a faithful reproduction of ActionOPD, KnowGPO, PRM4RL or another research method. Primary papers have since been checked for the bounded ActionOPD, KnowGPO and PRM4RL adapters;
this shared accounting layer remains an original adaptation. See the
[integration guide](unified_research_integration.md) and
[method matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md).

## API and default configuration

`ResearchCreditClaim` is a frozen, versioned proposal. It binds a claim, method, component and
source reference to authority/run/task/event/action identity, model and harness versions,
producer identity, evaluator ID/version/contract, dataset and student-state SHA-256 digests,
observable evidence, finite raw score, declared admissibility and retained source metadata.
`credit_kind` distinguishes knowledge, progress, process, teacher agreement, diagnostics and
separately verified decisions. The last kind requires the retained ActionOPD comparison and
independent observable verification; raw agreement still cannot pass eligibility alone.
Process values cannot be negative; the record does not create a new reward scale.

The existing `RewardProvenance`, `EvidenceReference` and `TrustedEvaluatorContract` contracts
and trajectory provenance codecs are reused. Private evidence, component mismatch, provenance
outside the declared evidence set, contradictory route fields, and mismatched evaluator identity
are rejected. The producer cannot name itself as the independent evaluator. String inequality
does not authenticate independence; the host must verify principals, versions and evidence meaning.

Host code calls the module explicitly; importing it installs no hook or default pipeline change:

```python
from gepa_mindfulness.training.research_credit import audit_research_credit

assert audit_research_credit(object()) is None  # default-off; input untouched
# claims is a tuple of host-captured ResearchCreditClaim proposals.
report = audit_research_credit(claims, enabled=True)
```

The audit configuration remains the strict boolean `enabled=False` function parameter.
Enabling this audit does not enable the separate live adapter, training, external calls,
candidate execution or spending. No new runtime dependency is introduced.

## Serialization and evidence integrity

Original kinds serialize as `research-credit-claim-v1`; verified decisions use v2. Reports use
`research-credit-audit-v1` with explicitly versioned nested claims. Both
carry `training_eligibility="DEVELOPMENT"`. Claim restoration rejects missing/extra fields,
unsupported versions and relabeling to TRAIN. Source metadata remains immutable canonical JSON
and retains nested provenance, including held-out restrictions. Full-source metadata parsing
reuses the existing depth, size, finite-number and duplicate-key checks.

Serialization and audit intake revalidate nested records, including all provenance fields before
route-specific serialization. Returned dictionaries are detached. These safeguards detect
corrupted caller records; they are not a security sandbox against arbitrary host Python code.

Each claim requires 1–256 unique observable references. An enabled batch accepts 1–1024 unique
claim IDs. Claims and reports retain the existing 4 MiB bounded-JSON envelope. A cumulative
serialized snapshot budget is checked during ingestion before restoring further claims.
Malformed input raises ValueError; absence of evidence is not silently filled with a verifier.

## Overlap accounting and admissibility

`event_overlaps` groups claims with the same host-normalized semantic event key:
`(authority_domain, run_id, task_id, event_id)`. Such claims must agree on action, model,
harness and state digest. Distinct methods and components cannot conceal a shared event by
using different credit names. Raw component scores and all proposals remain available.

`shared_evidence` separately warns when claims reuse a scoped evidence reference. One source
can support several distinct facts, so shared evidence alone does not merge events or establish
duplicate credit. Conversely, different reference IDs may alias one event. Hosts must normalize
semantic event identity, authenticate evidence and resolve both cases before any composition.
Two declared observable kinds for one ID currently produce a shared-evidence warning; source-kind
conflict resolution remains a host responsibility and is not treated as successful verification.

`prerequisite_gaps` records unresolved/rejected declared admissibility, absent provenance,
diagnostic/teacher-agreement signals and missing or nested-ineligible TRAIN source declarations.
The shared `require_training_eligible()` check is reused. An empty gap list means only that these
structural prerequisites were supplied. It is not evidence of correctness or reward eligibility.
An unresolved judgment is not converted into a deterministic moral violation.

Every report sets `requires_host_review=True` and `authority_granted=False`. There is no total,
optimizer accessor, admission token or replay-restorable live decision. The existing eligibility
checker rejects reports, claims and TRAIN wrappers retaining them. Provenance stripping cannot
be detected by this module or by the legacy untagged-admission allowance; retain full records.

## Host-reviewed event reservations

[research_credit_ledger.py](../gepa_mindfulness/training/research_credit_ledger.py) adds
`ResearchCreditUse` and `ResearchCreditLedger`. The journal records reviewed allocations under
existing host authority. Keep its path, callback and policy outside model/candidate control.

The host binds a use proposal to an exact `RewardRequest` digest. Each selected claim must retain
that digest as `target_request_digest` in its source metadata, match the target policy version,
and use only the request's observable references. Source eligibility, independent provenance and
resolved declared admissibility are required. Diagnostic/teacher-agreement signals cannot be
selected. Unselected proposals remain in the audit. At most one contribution per semantic event
can be selected, even when knowledge and progress methods both claim the same event.

Explicit overlap declarations for `provider`, `reward_integrity` and `causal_reward` are mandatory,
including inactive layers. Every declared layer must be `not_counted`. Missing, unknown and
already-counted statuses reject reservation. Additional declared layers are checked too; the host
must enumerate new active layers and independently verify their overlap. Declarations are not proof.

The existing causal adapter's mutation-safe `_accepted` boundary calls the host reviewer with
the complete target, audit, selection and accounting proposal. Only literal True without envelope
mutation passes. Missing callbacks, exceptions, false/truthy non-booleans and altered envelopes
reject without writing. The host authenticates review independence, evidence meaning and semantic
event identity. The package cannot sandbox arbitrary caller-provided Python.

```python
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger

ledger = ResearchCreditLedger("/host-owned/credits.sqlite", authority_domain="host-v1")
assert ledger.reserve(object(), object(), object()) is None  # actual default: enabled=False
# All following inputs are separately configured, trusted host objects.
receipt = ledger.reserve(
    target, claims, use_proposal, authorize_use=trusted_host_review, enabled=True
)
```

The journal pins its schema and authority domain. SQLite `BEGIN IMMEDIATE` and unique run/task/event
keys reserve all selected events atomically across connections and reopenings. Changing method,
use or policy labels cannot reserve the same recorded event again. Conflicting capture kinds for
one scoped evidence reference in a submitted batch reject before review. Distinct facts may share
a source when independently approved; aliases across event IDs still require host normalization.

Proposals use `research-credit-use-v1`; receipts use `research-credit-reservation-v1`. Both remain
DEVELOPMENT data. Receipts retain the reviewed envelope and SHA-256 digest with
`authority_granted=False`. `read()` returns detached, digest-checked evidence. Serialized proposals
and receipts cannot restore a live grant or change RewardPipeline. No reward scale or loss is added.

A reservation is conservative: downstream training failure leaves its events reserved. There is
no automatic refund, rollback, payment or retry token. Operator recovery must inspect original
evidence; deleting the journal is not supported recovery. Raw database errors roll back the entire
reservation. This is per-journal accounting: the host must protect one authoritative journal per
domain. Copying/rewinding a journal, changing domains or exposing its file to candidate code defeats
durable accounting. It is not a replacement for authorization or protected candidate promotion.

## Integration and remaining boundaries

This audit does not replace or mutate RewardPipeline, CausalRewardAdapter, runtime governance,
the Constitution, the canonical 17 cases, protected suites or CoevolutionStore. No aggregate
optimizer score is computed by the audit. The separate live adapter must resolve semantic
overlap using host-owned policy and existing admission mechanisms. The journal now prevents
cross-call reservation replay within its domain; it does not enforce duplicate credit in other
journals or reward paths. Raw evidence storage, evaluator authentication and protected promotion
remain host responsibilities. The explicitly configured live adapter below reuses these owners;
it never upgrades a restored report into authority.

## Software evidence and limitations

Run the deterministic CPU lane from a full checkout:

```bash
PYTHONPATH=src:modules:reasoning-generalization-tracer/src \
  python -m unittest discover -s tests -p 'test_research_credit*.py' -v
```

These unittest tests are also discoverable by the repository's ordinary pytest CI. They exercise
actual package imports without replacing modules. Tests cover serialization, provenance mutation,
resource short-circuiting, evidence boundaries, overlaps, metadata restrictions and diagnostic
admission rejection. The combined lane passes 42 tests, including host-review rejection/mutation,
request bindings, restart/replay, competing writers, rollback, corrupted receipts and multi-event
allocation. Software correctness evidence does not establish research effectiveness.

The approved repo-quality-gate fallback was used: spec/design, failing tests, implementation,
verification, independent review and evidence. The design is recorded in
[the implementation plan](../history/2026-10-26-research-credit-contracts.md). Independent review
identified route-field loss and late aggregate-size checking; both received failing regression
tests before fixes. An unused import was removed. Evidence-kind disagreement is explicitly
documented above; the diagnostic warning remains compatible while reservation rejects batch
capture-kind conflicts. The continuation design and independent review evidence are recorded in
[the allocation plan](../history/2026-10-26-research-credit-admission.md).

Local work uses a 1,019-file hash-verified API snapshot, not a Git clone; excluded binary fixtures,
notebooks, apps, history and beads prevent a full baseline claim. The local stdlib test lane and
Black 24.10.0's sequential formatting API were run. Black and development dependencies were
retrieved into an external workspace tools directory through GitHub because pip/clone networking
remained unavailable; they are not vendored or added as runtime dependencies. Full pytest,
lint/type and packaging CI for this change remain independent acceptance requirements.

The [KnowGPO + PRM4RL protocol](knowledge_progress.md) describes task-local evidence transitions
and separate, unresolved knowledge/progress proposals. Their same-event overlap is reported by
this audit; evidence review does not replace normative admission or durable event reservation.


## Live host composition and canonical CPU training

Maturity: **Experimental** software integration. The operator requested working features while
omitting research studies; no empirical improvement, calibrated admissibility classifier or
research reproduction is claimed. The why/how is to preserve separate raw research signals while
making fresh host admission and duplicate-event accounting unavoidable at the training boundary.

`research_credit_runtime.ResearchRewardProvider` wraps the existing `RewardPipeline` and exact
`ResearchCreditLedger`. Its immutable `ResearchRewardPolicy` pins policy/run/task/harness identity,
dataset SHA-256, explicit positive coefficients for `knowledge`, `progress` and `process`, and an
absolute adjustment budget in `(0,1]`. These coefficients never weight constitutional commitments.
Process values retain the existing nonnegative rule; signed progress retains observable evidence.
Raw teacher agreement and diagnostics remain ineligible, even with an admitted declaration.

Each enabled score performs this sequence:

1. Snapshot the actual source request; require explicit TRAIN and recursive source eligibility.
2. Snapshot the proposed claims/use. Reuse the ledger's existing structural preflight, including
   exact target/policy version/evidence binding, prior-layer accounting and selected event overlap.
   Also pin the action, run, task, harness, dataset, host policy and evaluator independence.
   Unselected held-out material is excluded from training review too.
3. Score the existing baseline once and preserve its existing negative-evidence validation and
   optional integrity breakdown. Retain every raw and weighted research component separately.
4. Reserve the complete bounded diagnostic export, then require fresh `authenticate_claims` and
   `authorize_optimizer_use` host reviews. Only literal True on an unchanged envelope passes.
5. Use the existing ledger's fresh `authorize_accounting` review and durable event reservation.
6. Return a live target-bound decision. The canonical engine explicitly requests its optimizer
   value and its validated scored trajectory; a report or duck-typed diagnostic cannot substitute.

The host owns and authenticates all callbacks, actual source records, reviewer principals and
applicable constraints. The evidence review must establish semantic support and admissibility,
including constitutional uncertainty routing; a producer's `admitted` string is insufficient.
The optimizer-use review must authorize the specific policy, baseline, source and allocation.
Unknown or rejected judgment withholds optimizer access; it is not automatically a moral violation.
Identity strings and unit tests cannot prove alignment or independent factual truth.

The scalar is `baseline_total + sum(raw_score * configured_coefficient)` for the selected,
independently eligible contributions. Unconfigured kinds, unresolved accounting, overlap, nonfinite
values, excessive adjustments and component storage bounds fail explicitly; no clipping erases a
problem. Negative progress has its own observable references and verified evaluator provenance.
Its sign describes verified milestone dynamics, never private thoughts or virtuous wording.
Binding retains the request's exact observable-reference selection, including a restricted
subset or a different selection order; it never broadens that selection. Structured baselines
retain their own references alongside the original request references;
conflicting source kinds for one reference identity are rejected before review.

Canonical setup uses an operator-controlled Python host; callbacks are not imported from
candidate-controlled YAML. For example, after configuring the real existing source observer,
pipeline, proposer and independent review functions:

```python
from gepa_mindfulness.training.engine import RLTrainingEngine
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger
from gepa_mindfulness.training.research_credit_runtime import ResearchRewardProvider
from gepa_mindfulness.training.runtime_config import load_rl_config

config = load_rl_config("operator-run.yaml")
runtime = ResearchRewardProvider(
    existing_pipeline,
    ledger=ResearchCreditLedger("run-credit.sqlite", authority_domain="operator-domain"),
    policy=config.research_credit,
    propose=propose_for_request,
    authenticate_claims=authenticate_evidence_and_admissibility,
    authorize_optimizer_use=authorize_this_training_use,
    authorize_accounting=review_prior_credit,
)
engine = RLTrainingEngine(
    config,
    backend_factory=host_observed_backend_factory,
    reward_factory=lambda _: runtime,
)
result = engine.train(max_steps=1)
```

`host_observed_backend_factory` uses the existing backend contract to retain admitted source
metadata and real outcome/evidence references in generated trajectories. A raw model response is
not an independently verified outcome. This example names required operator bindings; it does not
provide fake approving verifiers or imply that the ordinary CLI installs a verifier service.
The CPU acceptance test exercises real model updates with an independently defined local fixture.

The optional `research_credit` configuration section contains the policy fields (without its wire
`schema_version`), for example:

```yaml
research_credit:
  enabled: true
  policy_id: operator-policy-v1
  run_id: declared-run
  task_id: declared-task
  harness_version: declared-harness-v1
  dataset_digest: <sha256-of-the-exact-training-file>
  coefficients:
    - [knowledge, 0.5]
    - [progress, 0.25]
  max_abs_adjustment: 1.0
```

The coefficients are illustrative operator choices, not validated method hyperparameters. The
engine checks the captured dataset digest and the exact enabled host provider/policy. Enabled
policy settings participate in checkpoint/resume identity; absent/inactive settings preserve the
old hash. Legacy translation retains an explicit request. CPU FP32 portable training is the
implemented path; CUDA/AMP/distributed/hybrid credit attachment is Unsupported. Protected held-out
evaluation must use its existing evaluation path, without the training-credit adapter.

Successful scored trajectories retain weighted component evidence and reviewer provenance.
Metric-record version 3 retains baseline, adjustment and raw/weighted knowledge/progress/process
metrics. Versions 1 and 2 keep their existing closed names; SST diagnostics remain separate v2 rows.
Keep the full detached review/decision export in the operator's existing audit store as well.
`research-reward-policy-v1` and `research-reward-submission-v1` are versioned inputs. Decision
exports use `research-reward-decision-v1`, remain DEVELOPMENT and cannot restore live authority.

The durable reservation survives process restart. Replaying a semantic event with another use ID
fails. A downstream optimizer/checkpoint failure does not refund the event automatically; preserve
the journal with rollback references and route recovery to the existing host procedure. Accepted
catalog entries and accounting receipts do not deploy models, authorize tools or revise policy.
No new authority registry, canonical case, reward authority or runtime dependency is introduced.
