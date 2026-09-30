# Evidence semantics and memory influence

PR-5 adds an experimental, explicitly called diagnostic adapter in
`gepa_mindfulness.verification.evidence_use`. Importing the module installs no runtime hook.
The adapter connects the existing `EvidenceState`, `EpistemicMeasurement` and `RetrievedMemory`.
It neither writes memory nor changes confidence fusion, rewards, routing or execution permission.

## Evidence status

`EvidenceClaim.status` accepts these labels. Existing JSON fields and labels remain unchanged.

| Status | Meaning of the declaration | Required evidence |
| --- | --- | --- |
| `observed` | A captured observation | At least one observable reference |
| `inferred` | A conclusion derived from evidence | At least one reference |
| `supported` | Supporting evidence has been declared | At least one reference |
| `unverified` | Support remains unverified | May be empty |
| `unavailable` | Evidence is unavailable | May retain historical references |
| `stale` | Evidence is no longer current for this use | At least one reference |
| `contradicted` | Contradicting evidence has been declared | At least one reference |
| `superseded` | A replacement claim exists | Exactly one `superseded_by` link |

These constructors validate records, not truth or issuer identity. `EvidenceState` still validates
the supersession graph. `commit_verified_claim()` still accepts only `supported` or `contradicted`
claims with its existing verifier and WRITE-authorization checks. The new labels grant no write
permission. Older consumers may reject the new labels; producers should use them only with
consumers that support this extension. Existing labels need no migration.

## Construct an assessment

The trusted host supplies `EvidenceUseAssessment` with:

- `state` and `claim_id`: the complete current evidence snapshot and the exact original claim.
- `measurement`: the numeric record whose typed evidence references belong to that claim.
- `memory`: the existing retrieved-memory boundary record. Its ID and content equal the claim's
  ID and proposition; `source_identity` is required. Existing representation provenance remains.
- `kind`: `MemoryKind.FACT`, `PROCEDURE` (including tips), `NORM`, or `EPISODE` (episode/evidence).
- `target_influence`: `MemoryInfluence.IGNORE`, `BOUND`, or `CONTROL`.
- `quality`: `EvidenceQuality(recorded_at, source_reliability, compression_distortion, integrity,
  authority, provenance)`.
- `policy`: `EvidenceUsePolicy(max_age_seconds, min_source_reliability,
  max_compression_distortion)` with explicit host thresholds.
- `assessed_at`: an RFC3339 timestamp with an offset.

Reliability and compression distortion are optional finite scores in `[0, 1]`. Reliability is
larger for better declared performance; distortion is larger for more declared information loss.
Their definitions, measurement methods, domain applicability and thresholds belong to the host.
Zero distortion is a host declaration, not an automatic default for raw or summarized content.
Unknown values remain `None`. The adapter does not learn these scores or convert them to variance.
Age limits are finite and nonnegative; booleans are rejected as numbers.

Integrity is `intact`, `tainted`, or `unknown`. Authority status is `information_only`,
`external_policy_reference`, or `unknown`. A policy-reference label retains the declared origin
of normative information; it neither authenticates policy nor supplies a runtime grant. Trust
uses the existing `MemoryTrustLevel`. Reliability, trust and authority remain separate fields.

`IGNORE` means no answer-specific use is intended. `BOUND` means local supporting use. `CONTROL`
means material influence on a conclusion or constraint. The host declares the target for the task;
the adapter does not measure actual model influence. The returned `influence` applies only to
numeric use: it is `IGNORE` when a check fails, otherwise it equals `target_influence`. A norm or
procedure may retain a `CONTROL` target for another consumer while remaining ineligible as a
numeric observation here. Actual influence and continuity diagnosis belong to PR-6.

## Numeric eligibility

`measurement_for_update()` returns a detached, unchanged measurement only if every check passes:

1. Effective claim status is `observed`, `inferred`, or `supported`.
2. The measurement is available and all its references are observable.
3. `assess_retrieved_memory()` returns `USE_WITH_PROVENANCE`.
4. Integrity is `intact` and authority status is known.
5. Reliability is known and at least the policy minimum.
6. Compression distortion is known and at most the policy maximum.
7. Memory kind is `FACT` or `EPISODE`, and target influence is not `IGNORE`.

The adapter marks a current observed/inferred/supported claim `stale` when age **exceeds**
`max_age_seconds`. Equality remains current. It preserves other statuses, including contradiction
and supersession. A future `recorded_at` raises. The adapter looks up the named original claim;
it never substitutes a replacement claim for an old measurement.

Any failed eligibility check makes `measurement_for_update()` raise `ValueError` with the reasons.
Raw numbers remain in the audit record. Even variance `1e-12` cannot override a stale or
contradicted claim. Admission does not certify semantic calibration, causal ancestry or source
independence. The host must map the numeric target to the proposition and authenticate declarations.
PR-2 causal checks, PR-3 estimator restrictions and PR-4 correlation checks still apply downstream.
The host must construct a new assessment with current state and time before later use.

## Reports and transformations

`to_dict()` exports `STATUS`, `EVIDENCE`, `LIMITATION`, and `NEXT_ACTION`, followed by all input
snapshots. `NEXT_ACTION` is `review_evidence` if any limitation exists, otherwise
`consider_measurement`; these are diagnostic suggestions. The JSON report is not an authenticated
import format or an authorization token.

`assessment.summarize(text, transformation_id="summary-1")` appends an unverified display view.
It preserves the complete original state, measurement, memory trust/source/representation fields,
quality authority/integrity/provenance, policy and assessment time. Repeated summaries retain earlier
views. IDs are unique and the adapter allows at most 128 views. A summary cannot refresh age,
change variance, clear taint or become the measurement. Returned dictionaries are detached.

This operation records caller-supplied text; it does not run a summarizer or verify equivalence.
The host must retain the complete report when transmitting views. Sending only summary text drops
the labels outside the adapter's boundary and can let a recipient mistake untrusted content for
instructions. Durable storage, confidentiality, transport integrity and caller authentication remain
host responsibilities. Existing standalone `EvidenceState.merge_equivalent()` creates a provisional
claim and has no memory-quality metadata; it does not transform an `EvidenceUseAssessment`.

## Research and validation

The [research registry](recommendations/RESEARCH_TRACEABILITY.md#ref-memcalib) records source
findings separately from repository inferences. MemCalib motivates explicit target influence;
JitMem motivates retaining originals while adding task-specific views. CompKV motivates tracking
distortion only as an analogy: its attention-cache error bound is not a semantic-memory variance.
Qwen-Planner-Agent motivates retaining execution evidence. FTA motivates explicit report fields.
Share-Borne AI Virus and A2M motivate retaining trust and integrity across artifact/tool boundaries.
No paper's benchmark, curator training, token reward or attack implementation is reproduced.

Run `python -m pytest -q tests/test_evidence_use.py tests/test_world_evidence_state.py
tests/test_governed_evidence_commit.py tests/test_memory_mediated_laundering.py
tests/test_scalar_fusion.py tests/test_research_traceability.py` from the repository root.
These tests verify status/identity checks, policy boundaries, source preservation, refusals and
fusion of admitted measurements. Host authentication and score calibration require deployment
review; synthetic contract tests do not establish them.
