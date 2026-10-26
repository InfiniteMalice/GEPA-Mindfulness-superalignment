# Research credit proposals and diagnostic accounting

## Status and scope

PR 1 introduces **Experimental** opt-in contracts in
[research_credit.py](../gepa_mindfulness/training/research_credit.py). These records and reports
do not grant optimizer credit, certify evidence, enforce constitutional admissibility or authorize
actions. They are the diagnostic portion of the shared-contract stage. Live host admission,
composite-reward attachment and durable credit consumption remain unimplemented.

This is original infrastructure derived from the operator's October 2026 integration requirements,
not a faithful reproduction of ActionOPD, KnowGPO, PRM4RL or another research method. Primary-paper
verification remains required before those adapters are implemented. See the
[integration guide](unified_research_integration.md) and
[method matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md).

## API and default configuration

`ResearchCreditClaim` is a frozen, versioned proposal. It binds a claim, method, component and
source reference to authority/run/task/event/action identity, model and harness versions,
producer identity, evaluator ID/version/contract, dataset and student-state SHA-256 digests,
observable evidence, finite raw score, declared admissibility and retained source metadata.
`credit_kind` distinguishes knowledge, progress, process, teacher agreement and diagnostics.
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

The actual configuration is the strict boolean `enabled=False` function parameter. There is no
new CLI or implicit YAML loader. Enabling this audit does not enable training, external calls,
candidate execution or spending. No new runtime dependency is introduced.

## Serialization and evidence integrity

Claims serialize as `research-credit-claim-v1`; reports use `research-credit-audit-v1`. Both
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

## Integration and remaining boundaries

This audit does not replace or mutate RewardPipeline, CausalRewardAdapter, runtime governance,
the Constitution, the canonical 17 cases, protected suites or CoevolutionStore. No aggregate
optimizer score is computed from the proposals. Future live composition must resolve semantic
overlap using host-owned policy and existing admission mechanisms. Cross-call replay prevention,
durable evidence storage, evaluator authentication and protected candidate promotion remain with
their existing host owners. Transaction-local duplicate detection is not a persistent ledger.

## Software evidence and limitations

Run the deterministic CPU lane from a full checkout:

```bash
PYTHONPATH=src:modules:reasoning-generalization-tracer/src \
  python -m unittest discover -s tests -p test_research_credit.py -v
```

These unittest tests are also discoverable by the repository's ordinary pytest CI. They exercise
actual package imports without replacing modules. Tests cover serialization, provenance mutation,
resource short-circuiting, evidence boundaries, overlaps, metadata restrictions and diagnostic
admission rejection. Software correctness evidence does not establish research effectiveness.

The approved repo-quality-gate fallback was used: spec/design, failing tests, implementation,
verification, independent review and evidence. The design is recorded in
[the implementation plan](../history/2026-10-26-research-credit-contracts.md). Independent review
identified route-field loss and late aggregate-size checking; both received failing regression
tests before fixes. An unused import was removed. Evidence-kind disagreement is explicitly
documented above rather than being claimed resolved.

Local work uses a 1,019-file hash-verified API snapshot, not a Git clone; excluded binary fixtures,
notebooks, apps, history and beads prevent a full baseline claim. The local stdlib test lane and
Black 24.10.0's sequential formatting API were run. Black and development dependencies were
retrieved into an external workspace tools directory through GitHub because pip/clone networking
remained unavailable; they are not vendored or added as runtime dependencies. Full pytest,
lint/type and packaging CI for this change remain independent acceptance requirements.
