# Phase 2 stage 9 refinement

Base `c92ba24`; stage 8 passed 7,280 tests. This refines the approved implementation plan,
not an issue tracker. The maintainer-approved quality-gate fallback persists. Filesystem writes
require scoped escalation in the current read-only environment.

Reuse ArtifactObservation/ArtifactRecord/ArtifactSnapshot for a host-only content-addressed raw
research archive. New source bytes and catalog versions are exclusive/no-overwrite; original
versions remain readable. Catalogs contain no accepted claims or optimizer eligibility. Host source
custody authentication and existing action-scoped one-use WRITE authorization bind exact before/
after catalog, bytes, logical store and physical root. Deserialized records are not credentials.
Use a single-writer host object with pinned catalog on reopen; durable concurrent coordination is
not claimed. Preserve orphans on I/O failure for inspection; never certify partial writes.

Reporting uses immutable source selectors and versioned paired protocols, not model self-reported
scores. Resolve source bytes under the pinned archive, validate digests, parse bounded JSON/gzip,
and derive scalar values only at frozen pointers. Reuse existing PairedObservation, ResamplingPolicy
and estimate_paired for dependency-cluster confidence intervals. Host authentication must cover
source custody, rubric/cluster declarations, original protocol timing and actual reviewer
independence. Missing verification, missing pairs, unknown dependencies and insufficient clusters
remain explicit; repeats do not become independent samples. A software regression pass does not
stand in for model capability regression. Authenticated protected model failures withhold any
improvement interpretation even when the observed benchmark delta is positive. General alignment
and promotion remain unestablished/unauthorized; existing coevolution stores remain authoritative.

Implement an actual finite-CVA resource trial: retain original source bytes and audit metadata,
compare cached/full serialized state with a compact source/verification index, reconstruct eligible
results from original evidence, independently rerun the finite checker, measure CPU time and
serialized retained-state bytes, and check exact verification fidelity and unchanged source hashes.
Do not claim peak-memory savings, a transformer bound or a mathematical time-space theorem.

An offline CLI audits committed stage manifests against their actual historical Git source blobs,
loads existing retained experiments, records seeds/model/checker/code versions and produces a
structured integrated summary. Historical hashes must be checked at each stage commit, not against
later modified README/config files. Preserve A-H observed/unsupported/unrun distinctions, nulls,
falsified hypotheses and protected failures. Generate final API/inventory/migration/PR notes,
source traceability, exact constitutional-diff links and proposed next experiments.

Tests first: append-only version/source immutability, missing/wrong authority, self-review and
mutating callback rejection, stale catalog/root binding, tamper/missing/symlink detection,
source-based paired statistics and deterministic intervals, incomplete/dependent sources,
protected failure withholding, software/model regression separation, immutable source reconstruction
and exact finite verification. Fresh review and full protected/package/type/lint/format checks
close this stage. Trained-model studies remain explicitly unrun unless separately executed;
software correctness and alignment improvement remain separate claims.
