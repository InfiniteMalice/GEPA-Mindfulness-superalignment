# Native verified ActionOPD

The operator requests working opt-in features and excludes research runs. Existing
REF-ACTIONOPD source verification, semantic action contracts, Eq. 6 reward and
Eq. 7 batch whitening are reused without merging LoCD's KL objective.

The authorized repo-quality-gate fallback applies: specification/design,
implementation, tests, one independent review and evidence. Superpowers workflows
apply; repo-quality-gate remains unavailable. No beads commands or records change.

## Design and acceptance

Introduce a separately versioned verified-decision credit kind. Raw teacher
agreement and diagnostics remain ineligible. A verified-decision proposal must
retain an exact TRAIN target, typed semantic action, teacher identity/sampling
records, independent observable relational verification and reproducible Eq. 6
score. Existing host semantic/use reviews and the durable credit ledger remain
the only optimizer admission/accounting path. Protected constraints are not
scalarized, and unsupported, prohibited or uncertain actions fail closed.

An explicitly configured black-box teacher callback receives only the student's
public state and opaque source fingerprint, never sampled student answers,
private ground truth or evaluation labels. Fresh host source authorization occurs
before teacher access. Bounded state/sample caches retain teacher provenance.
No API client, credential, external model download or spending is introduced.
Host-installed parsing/verifier roles authenticate observed decisions; teacher
agreement is recorded separately and cannot establish correctness.

Add a strict default-off canonical configuration binding teacher generation,
independent verifier identity, sampling, budgets and global-whitening settings
into resume identity. A native batch preparer uses only live authorized decisions,
groups by actual public state, skips unsupervised/equal-reward groups, and applies
Eq. 7 across retained groups before the existing clipped GRPO objective. The
canonical optimizer/checkpoint/SST path remains unchanged. Ordinary GRPO and old
configuration hashes/wire schemas are preserved. New metrics have a separate
closed schema; raw and weighted components remain inspectable.

CPU-safe functional tests must demonstrate actual weight changes, black-box
teacher sampling without teacher probabilities, full-batch whitening, source
provenance restrictions, prohibited-teacher/uncertain-verifier rejection, denied
or mutating reviews, event replay/overlap rejection, exact teacher/config binding,
native checkpoint identity and unchanged default behavior. No empirical alignment,
calibration, transfer or efficiency improvement is claimed. Research protocols
remain documented with rationale and future qualification instructions.

## Implementation evidence before review

The new claim kind was reproduced as rejected on a separate clean baseline after
correcting an initial fixture setup error. Missing native batch/config attachment
was observed RED before integration. Existing schemas/default hashes remain
covered; new verified decisions and policy/report versions are separate, and
raw teacher-agreement claims remain ineligible.

A real weight-sampling CPU fixture runs canonical GRPO, updates parameters and
resumes its native checkpoint, using source capture over an explicitly admitted
DatasetProvider and no teacher probabilities. The broader initial suite passes
513 checks. New negative checks cover prohibited teacher shortcuts, missing and
changed method binding before queries, teacher identity/state/config/duplicates,
source output/origin mutation and retained held-out provenance. The interface
rejects cross-microbatch whitening rather than mislabeling per-microbatch math
as full-batch Eq. 7. All checks qualify software behavior only; independent
review and final-head remote CI remain required.

## Independent review

One fresh read-only reviewer inspected all 13 paths and independently passed all
514 focused checks, plus temporary cache-approval and state-budget probes. No
Critical or Important finding was confirmed. The one Minor finding was a stale
v1-only claim-reader docstring; it now describes supported versioned DEVELOPMENT
claims. This documentation fix changes no runtime behavior or wire format.

Every declined-to-judge item is explicitly resolved. Empirical gains remain
unclaimed because research was excluded. Semantic truth, event normalization
and real role independence remain installed host responsibilities, not certified
by identifiers. No external service client/spending is installed; service
reliability and deterministic external sampling require configured host authority.
CUDA/AMP/distributed/LoRA execution remains unsupported with rejection boundaries.
Final remote CI supplies isolated package/Gloo coverage, and publication must
verify all remote blobs/modes against the full base tree. Earlier credit/LoCD
redesign is excluded; their changed ActionOPD interfaces were reviewed here.

## GitHub review: cache lifetime

The public cache-lifetime finding was verified: logical student policy versions
change each optimizer step, so retained obsolete entries made memory budgets act
as a run-length cap. Two native runs with max_states=1 failed on their second
optimizer step before the fix, with and without SST. Cache scope now includes the
student model/version and rotates after fresh source approval, resetting only
ephemeral capacity. Existing evidence and ledger reservations remain unchanged.
Both runs now train beyond that limit and resume another native update. A separate
negative test confirms the same-policy state limit still rejects before extra
teacher calls. All 33 ActionOPD checks pass; final-head remote checks remain required.
