# PR-4: scalar correlation-aware fusion and frozen Round-0 judgments

## Task spec and design

Base: merged PR-3, 836ca32. Implement only PR-4 of the two user-supplied program specifications.
The user authorized autonomous staged implementation; routine design and execution continue
without another approval. Superpowers brainstorming/writing-plans/executing-plans and the repository
quality gate apply. Store this combined specification/plan in history/ per AGENTS.md. No Beads.

Current code has scalar temporal estimation and causal reconciliation, but no multi-source fusion
or peer-exposure collector. Reuse EpistemicMeasurement, EpistemicStateEstimate, ConfidenceSource,
CorrelationTreatment and the existing sequence validator. Do not change default runtime confidence,
rewards, authority, canonical cases, PR-3 filtering or PR-5 evidence/memory behavior.

Add a pure scalar fusion API over 1..16 measurements of one context, representation and dimension.
Source labels and distinct correlation-group names never establish independence. Preserve all
input measurements, evidence and provenance. The result contains a diagnostic estimate, method,
normalized weights, covariance/provenance and peer-exposure flag, with detached JSON serialization.
It does not create a raw observation, causal reconciliation or authority token.

Default unknown-correlation mode: Covariance Intersection (CI), using normalized nonnegative
declared weights (uniform by default). Precision is sum(weight/variance); mean is the corresponding
information-weighted sum. CONSERVATIVE_BOUND uses a convex mean and maximum marginal variance.
UNRESOLVED_CORRELATION returns an unavailable estimate, never a zero-uncertainty substitute.

KNOWN_COVARIANCE requires a full ordered covariance matrix and its provenance. Compute the
declared convex mean and w-transpose C w; weights are not claimed optimal. This supports positive
semidefinite singular matrices, including perfectly correlated clones. Check exact symmetry,
matching marginal variances, and positive semidefiniteness with rational LDL decomposition of
the input floats. Use standard-library Fraction arithmetic for small batches, avoiding inverses,
overflow and tolerance-based acceptance of indefinite covariance. Reject positive variances that
underflow on float conversion. Inputs require positive marginal variances for numeric fusion.

Known zero cross-covariance is rejected for shared evidence/group pairs and all post-discussion
pairs. Explicit nonzero known correlations remain usable. Covariance arguments in unknown modes
raise instead of being ignored. Physical validity/calibration of supplied covariance remains a
host responsibility; source names or absence of peer exposure are not a statistical certification.

Add VerifiedJudgmentPanel: declare a unique participant cohort and panel ID; collect a measurement
from each participant's existing fully validated reconciliation event with explicit successful
verifier binding. Reject duplicates, conflicting reused history events and incompatible targets.
open_discussion(released_at) requires all judgments and a timestamp at/after their reconciliations;
it freezes detached Round-0 snapshots and seals additions before returning any peer packet.
No partial-peer accessor. fuse_round_zero uses retained originals; fuse_post_discussion requires
the opened panel, fresh measurement IDs in cohort order and the same target/context, and forces
peer_exposed=True. The host must actually withhold external peer exposure until release; the
collector has no transport or authentication authority. Calls are sequential, in memory only.

## Alternatives

Avoid a matrix state estimator or an optimized generalized least-squares solver: only the source
error covariance is a matrix, and a declared convex mean has a transparent variance even for
singular covariance. Avoid adding NumPy as a mandatory dependency. Bound source count at 16 to
keep exact rational validation practical. Avoid connecting fused estimates to raw observation
bindings or applying a temporal prior twice. Temporal/fusion composition needs a later explicit
statistical contract; this stage returns a standalone diagnostic estimate.

## Implementation plan

1. tests/test_scalar_fusion.py: analytical independent/correlated/CI/bound/unresolved examples,
   singular and indefinite covariance, source/identity/evidence compatibility, strict numeric and
   shape validation, extreme arithmetic, permutation invariance and serialization snapshots.
   Run RED before adding verification/scalar_fusion.py; implement and run GREEN.
2. tests/test_judgment_panel.py: complete/partial cohort, failed or unbound verification, altered
   history, freeze immutability, chronology, duplicate submissions, post-exposure independence
   rejection and preservation of pre-discussion dissent. Run RED before judgment_panel.py; GREEN.
3. Add guide, ADR 0005, package resource and REC-002 research traceability for CI, Unanimity Without
   Persuasion and Weakly Supervised Quantum Error Mitigation; link existing GRUET. Explain the
   latter is a structural analogy, not transferred quantum/LLM calibration evidence.
4. Run targeted tests, full offline suite, Ruff/Black, applicable mypy and installed-wheel smoke.
   Build before repository-wide discovery checks to avoid temporary sdist-tree collisions.
5. Fresh independent whole-branch review as required by executing-plans; one RED/GREEN fix pass
   for material findings, then commit, push, open and attach a draft PR. Do not merge or start PR-5.

## Review focus

Check covariance PSD (including zero pivots), extreme-scale rounding/underflow, actual fusion
weights versus reported weights, duplicate/shared evidence, peer-exposure independence bypasses,
immutable Round-0 histories, failed-verifier acceptance, conflicting context/representation,
unknown-correlation defaults, no synthetic raw-observation laundering, and source/authority limits.

## Execution evidence

Initial RED: each new test module failed on its missing production module. After implementation,
the numerical/panel suite passed 47 tests, then 56 with additional strict-policy and snapshot
checks. The focused record/reconciliation/temporal/fusion/panel/research suite passed 326 tests.

Fresh whole-branch review: review_pr4 inspected 836ca32..67f37e8 and relevant dependencies/specs.
One Important finding: separate valid participant histories could reuse causal IDs behind fresh
envelope IDs. The five parameter cases of
test_causal_identity_reuse_across_separately_valid_histories_is_atomic failed before the fix.
The collector now retains detached event JSON and validates the deduplicated combined history
before storing a participant. This reuses existing prediction/action/observation/verifier/update
identity rules and preserves atomic failure and valid overlapping submissions. All five cases
then passed; the focused suite passed 331 tests. No second reviewer or deferred minors.

Documentation precision: updated retained-history semantics to describe full detached JSON and
combined validation, including memory/cost growth. No unresolved documentation BLOCK or WARN.

Final: Ruling: physical covariance validity and semantic calibration remain host responsibilities;
the scalar algorithms validate declared numeric contracts. Invalid assumptions can make the
reported uncertainty misleading.

Final: Ruling: participant authentication and actual external peer isolation remain host boundaries;
the collector enforces its own release sequence only. Undeclared exposure can invalidate a claimed
blind round.

Final: Ruling: optimal covariance/CI weights remain excluded; declared convex weights are used
consistently. This can sacrifice useful precision but avoids adding an optimization policy.

Final: Ruling: temporal-prior composition and runtime routing remain deferred. Treating source
fusion as an independent new observation without a later contract could double-count evidence.

Final: Ruling: real LLM calibration and reproduction of published experiments are not claimed.
Synthetic arithmetic and protocol tests alone do not establish semantic performance.

Final: Ruling: persistence, concurrent calls and transport remain outside this in-memory sequential
API. A host that ignores those limits can lose audit records or race the release boundary.

Final: Ruling: full-suite and installed-wheel validation are the implementer's responsibility;
the reviewer ran the focused tests and read source. An unchecked distribution could differ from
checkout behavior, so the final built wheel is exercised outside the repository.

The first full run hit test_invalid_pipe_write_progress_fails_closed[0]: its 0.1-second deadline
expired before the expected invalid-write-progress error. The test and Mojo coordinator source
are unchanged from the base; the deadline check precedes the progress check. All four parameter
cases passed on an isolated rerun (1.97 seconds). Final full-suite results are recorded below.

Final validation on the corrected branch:
- Full offline suite: 3,555 passed, 18 skipped, 16 warnings in 196.97 seconds. The earlier
  invalid-write-progress deadline failure did not recur; no unrelated RL code or tests changed.
- Focused contracts/reconciliation/temporal/fusion/panel/research suite: 331 passed.
- New fusion/panel tests: 61 passed; combined statement coverage 97% (208 statements, 6 missed).
- Repository Ruff passed; Black reported 576 files unchanged; applicable CI-selected mypy checks
  and checks of the new modules and research registry passed. git diff --check passed.
- Rebuilt sdist and wheel, reinstalled the wheel, and ran the smoke test outside the checkout.
  Verified panel release, independent versus CI variance, correlated post-discussion handling,
  JSON export, all 33 research references and the packaged guide.

Final: fixed the review's cross-submission causal identity issue with five RED-to-GREEN cases.
The corrected full suite and installed distribution both passed. No unresolved review findings
or deferred minors remain. PR-5 evidence/memory integration is the next stage.
