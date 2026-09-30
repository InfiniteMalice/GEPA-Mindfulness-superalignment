# PR-5 evidence and memory integration

Uses Superpowers brainstorming, writing-plans, executing-plans, TDD, independent final review,
and repo-quality-gate. The user's staged program authorizes narrow implementation and a draft PR.
This task-local checkout remains isolated from other user projects; branch starts at 9db9f3a.

## Reconciliation and spec

Existing: EvidenceClaim/EvidenceState provide immutable claims, supported/contradicted/unverified/
superseded status and a validated supersession DAG. Governed commits require independent verifier
records and a host WRITE grant. RetrievedMemory and assess_retrieved_memory preserve trust,
provenance and representation derivation checks. PRs 1-4 supply numeric measurements and estimates,
causal validation, temporal estimation and conservative source fusion. Exactly 17 cases remain.

Partial: claims lack observed/inferred/unavailable/stale labels. Numeric records retain evidence
references but cannot assess qualitative eligibility. Memory boundaries lack explicit content kind
and target influence. Representation provenance already protects one transformation path.

Missing: an explicit adapter binding the existing claim, measurement and memory, host-declared
quality data, eligibility checks and audit views that preserve originals through summarization.

Redundant: no new evidence store, evidence reference vocabulary, memory trust scale, authority
system, temporal estimator or fusion implementation. Learning-surface LessonKind routes lessons;
it does not classify retrieved content, so its purpose remains distinct.

Experimental: explicit calls only, no runtime hook, routing, reward, trained curator or persistence.
PR-6 continuity/influence-failure diagnosis remains next. The adapter cannot authenticate host
declarations or establish that a numeric target represents the proposition.

## Design

Extend EvidenceStatus with observed, inferred, unavailable and stale, retaining all existing labels
and exact serialization shape. Observed requires observable references; inferred and stale require
references. Only supported/contradicted still pass the existing governed commit path.

Add verification/evidence_use.py. EvidenceQuality holds recorded_at, source_reliability and
compression_distortion (optional unit diagnostics), integrity (intact/tainted/unknown), authority
(information_only/external_policy_reference/unknown) and nonempty provenance. EvidenceUsePolicy
requires explicit max_age_seconds, min_source_reliability, max_compression_distortion. All numbers
are finite and reject bool. No calibrated variance conversion is inferred from those scores.

EvidenceUseAssessment retains a detached EvidenceState, exact claim ID, EpistemicMeasurement,
RetrievedMemory, MemoryKind (fact/procedure/norm/episode), MemoryInfluence (ignore/bound/control),
quality, policy and assessment timestamp. Claim and memory IDs/content must match; measurement
references must be a subset of claim references without ambiguous reference kinds. Preserve full
supersession state; do not silently resolve an old measurement onto a replacement claim.

Effective status retains terminal/negative labels; current supported/observed/inferred claims
become stale when age exceeds the host threshold. Future records raise. Numeric eligibility also
requires available data, observable evidence, accepted memory retrieval, known authority, intact
integrity, known quality values meeting thresholds, fact/episode kind and non-ignore influence.
Unknown/failed checks retain raw numbers in the audit but measurement_for_update() raises.
No output creates authorization. Policy-reference labels are declarations, not validated grants.

The report returns STATUS/EVIDENCE/LIMITATION/NEXT_ACTION plus all input snapshots. Summarize
appends a labeled view while retaining originals and all boundary metadata. Views are unverified
display text, never substitute measurements; no sanitization or semantic-equivalence claim.

## Implementation plan

1. Add status tests to tests/test_world_evidence_state.py; run RED; extend state.py; run GREEN.
   Verify serialization, evidence requirements, old contracts and governed commit rejection.
2. Add tests/test_evidence_use.py. RED before implementing evidence_use.py. Cover all statuses,
   tiny-variance stale/contradicted refusal, exact expiry, unavailable values, supersession,
   trust/taint/authority/quality failures, kinds/influence, malicious summary preservation,
   source-kind swaps, strict numbers, snapshots and actual fusion of admitted measurements.
3. Add docs/evidence_memory.md, ADR 0006, navigation/package resource and research traceability.
   Read primary sources for MemCalib, JitMem, CompKV, Qwen-Planner-Agent, Share-Borne AI Virus and
   A2M; reuse existing FTA registry. Distinguish source findings from repository inferences.
4. Run focused tests, full offline suite, Ruff/Black, applicable mypy and installed-wheel smoke.
   Build before broad repository discovery. One fresh whole-branch review, one RED/GREEN fix pass,
   record results, commit, push the feature branch, open and attach draft PR-5. Do not merge.

## Review focus

Reference identity/kind confusion; superseded originals treated as current; low variance bypassing
qualitative failure; trust/authority/taint lost through repeated summaries; nested mutation and
representation-derived memory bypasses; unknown quality defaults; accidentally granting runtime
or reward authority; backwards compatibility of evidence serialization and commit policy.

Pre-flight: status extension feeds the adapter; existing serialization remains exact and unchanged.
Research metadata and package resources depend on final public names and guide path above.

## Execution evidence

Baseline focused evidence/commit/memory/fusion tests: 143 passed. Four new status round-trip cases
failed before the extension; status/commit suite then passed 87 tests. New adapter tests initially
failed on its missing module. The first implementation passed 49 of 50; the remaining test expected
plural "references" while the earlier ambiguity check correctly raised "reference". Corrected the
test's diagnostic match. All 50 adapter tests pass. The focused suite including research registries
and compatibility tests passes 378 tests. New modules/state type checks pass.

Research metadata test failed with six missing references before adding the primary-source
records. Reader synchronization initially missed evidence_use.py in the canonical evidence list;
corrected that list. Ruff found one test import ordering issue; fixed it. Build produced wheel and
sdist successfully. Final whole-suite, distribution and independent-review results follow.

Independent final review: review_pr5 examined 9db9f3a..4dbe79c read-only. No Critical, Important
or Minor findings; documentation precision PASS with no BLOCK/WARN. Reviewer ran 141 focused
tests and adversarial nested-representation mutation checks. No fix pass or second review needed.

Final: Ruling: host authentication, score calibration and semantic target mapping remain host
responsibilities, as specified by the explicit diagnostic boundary. Incorrect declarations can
admit misleading measurements despite valid record structure.

Final: Ruling: actual model influence and continuity diagnosis remain PR-6 work. This stage records
target influence and numeric eligibility; treating either as observed causal influence would
overstate the evidence.

Final: Ruling: full-suite and distribution acceptance remain the implementer's responsibility.
The installed wheel passed an outside-checkout smoke covering admission, stale refusal, summary
preservation, fusion, JSON export, 39 references and the packaged guide. Skipping such validation
could hide a discrepancy between source and distribution.

Repository Ruff passed; Black reported 578 files unchanged. CI-selected mypy and checks of the
adapter, state and registry passed. The 50 adapter tests cover 96% of its statements. The full
offline suite result is recorded below. No deferred minors remain.

Final full offline suite: 3,617 passed, 18 skipped, 16 warnings in 215.10 seconds. No full-suite
failures occurred. git diff --check passed. PR-5 scope is complete; PR-6 is next after merge.
