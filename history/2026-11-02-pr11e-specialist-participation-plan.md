# PR 11e advisory specialist participation: design, plan and evidence

## Intent and constraints

Implement the master prompt's optional MPT-inspired advisory coordination arm. Reuse
PublicDecisionContext, SystemConfig, TrustedEvaluatorContract, ImprovementRecord and the
existing bounded JSON envelope. All records/reports remain DEVELOPMENT, disabled by default,
and confer no runtime, optimizer, verification, dataset-admission or promotion authority.
No model, provider, action executor, training loop, counterfactual environment or reward
allocator is added. Constitutional judgments and current runtime grants remain external.

The maintainer already authorized staged implementation, ready PRs, review follow-up and
merge. Use superpowers and the maintainer-approved repo-quality-gate fallback; that skill is
unavailable. Native inline execution with one fresh whole-branch reviewer. No bd/.beads edits.
Plans live in history per AGENTS.md; no duplicate issue tracker or checkbox task list.

## Approaches and design

A native MPT trainer needs compatible specialists, restored environments and independently
qualified outcome feedback. A saved-bid analyzer alone cannot exercise eligibility-before-
selection. Choose a bounded advisory selector with host callbacks: review all candidates
before requesting any bid, quarantine rejected/uncertain candidates, and rank only eligible
positive bids. This is the implementer's choice within the authorized optional research arm.

ParticipationPacket binds request ID, public context, scope and 1..16 SpecialistCandidate
records. Candidates bind specialist/principal IDs, SystemConfig, checkpoint digest, description
and public provenance. ParticipationReview binds packet/candidate digests, independent host
evaluator/principal, review ID, eligible/rejected/uncertain status, reason and evidence.
SpecialistBid binds packet/candidate, specialist/system and finite [0,1] willingness, never
confidence or competence. ParticipationConfig defaults off, with seed and output-byte cap.

select_specialist(packet, *, current_state_digest, evaluator, reviewer_principal_id, review,
propose_bid, config=ParticipationConfig()) returns an inert report. It rejects stale state,
duplicate/alias identities and self-review before callbacks. Each callback receives a fresh
detached packet/candidate; mutation, exceptions, malformed or misbound outputs abort selection
with fixed failure categories. Host callbacks are trusted integrations, not a sandbox.

Review and bid records each have a 64 KiB serialized cap. Reserve the complete report with
both record capacities per candidate before any callback. Output cap is configurable from
4096 bytes through the existing 4 MiB maximum. Conservative reservation can reject small
budgets even when a particular short response would fit. Exact positive maxima only; seeded
uniform choice among sorted exact ties; zero or no eligible bids abstains. No jitter or floor
reopening. Reviewer authentication, current grants and enforcement of moral constraints remain
host work; declarations and syntactic checks do not establish them.

## Source and adaptation boundary

Read Xu et al., arXiv:2610.22322v1, 2026-10-15, sections 2.1–2.3 and Appendix A; v2 method
also inspected. Paper license: arXiv.org perpetual non-exclusive. Author repository
Embodied-Minds-Lab/MPT pinned 3b56087a2e436e04f077672cf8cffd91b8820b08; README, auction.py,
Apache-2.0 LICENSE and NOTICE read through GitHub plugin. No code/data copied or executed.
Paper says skip all-zero turns; pinned auction.py instead reopens at a floor and uses jitter
for ranking. This original adaptation follows paper abstention and exact-tie sampling, adding
independent host eligibility and advisory-only output. Native credit propagation, REINFORCE,
Bradley–Terry training, task-specific environments and empirical effectiveness are Unsupported.

## Implementation plan and review focus

1. Write tests for default-off no inspection; strict versioned records; stale state; alias/
   duplicate candidates; independent review; complete output reservation. Observe RED, then
   add participation_records.py and specialist_participation.py under factuality_observability.
2. Exercise quarantine, all-zero abstention, positive exact ties, near ties, changed model/state,
   callback mutation/errors, partial failures and malformed outputs. Observe missing behavior
   fail before implementation, then run the complete related CPU suite GREEN.
3. Add disabled config, guide, authored source note/digest, matrix and existing source registry
   traceability. Include guide/source note in wheel resources and outside-checkout smoke.
4. Run formatter, syntax/line/import checks, related tests, wheel build/resource checks and
   fresh independent whole-branch review; resolve verified findings. Publish ready PR through
   GitHub plugin, babysit reviews and final-head Python matrix, merge only when clear.

Review focus: principal aliases versus authenticated identity; callback substitution/mutation;
uncertain eligibility versus deterministic rejection; exact tiny positive bid ordering; report
reservation including both serialized reviews and bids. No live spending or model effectiveness
is established by CPU controls. Existing protected evaluation and authority files are unchanged.

## Baseline and rulings

Base main 945b7aa6114fadc6c74ab8ee8b465c96d3f2b333; tree
722140fb00127f0dfb1ba4431ef91cc7e95db962. No usable local Git metadata; use an isolated source
snapshot and GitHub Git-data publication. All 1118 locally present tracked blobs match main.
The 126 pre-existing omitted blobs (unrelated app/history/binary/.beads files) are preserved
by overlaying changed paths on the complete 1244-blob remote base tree; none is deleted.

Ruling: proceed with the documented bounded advisory adaptation under standing implementation
authority, without repeating design/push/merge approval questions. Native MPT remains Unsupported;
incorrectly treating this adaptation as a reproduction would overstate its research value.

Baseline: 203 related routing/runtime/eligibility/workflow/source/link tests passed. Record
tests observed missing-module RED, then 14 passed; one test fixture initially named a
nonexistent evidence enum and was corrected to existing PRIVATE_REASONING. Selector tests
observed missing-module RED, then all 40 participation cases passed. Source inventory observed
missing REF-MPT RED, then 91 combined participation/source/documentation checks passed.
No protected test expectation was weakened. Source note: 1537 UTF-8/LF bytes, SHA256
1f2be4b2d4fe85867cf5410262b7a36802a746c9f4f9177981667dc792b84984.

Combined related routing/runtime/eligibility/workflow/source/link/participation suite: 243
passed. Initial wheel build and isolated config/default-off/registry/source smoke passed,
with 42 required resources and 3 changed runtime files matching source; no NumPy import.
CLI Black multiprocessing is blocked by sandbox socket permissions; use the same installed
Black formatter API in-process and require the normal CLI check in remote CI. No rule relaxed.

Final review/integration correction: broader existing controls and the independent reviewer
reproduced REC-012 reader/registry drift. The ordered Repository evidence, Acceptance checks
and backticked Research lists now mirror the registry. The third failing control was
test_experimental_recommendations_name_current_code_docs_and_tests: its REC-012 inventory was
additively extended for the new implementation/test paths, retaining strict equality, all prior
entries, status and targets. These three controls changed from 3 failed / 84 passed to 87 passed.
The full related suite passed 330 tests in 13.39 seconds. Runtime files were unchanged.

Final wheel SHA256 b8278d5bbc7cc6cf102a92a28128430199b5675720d32433c233fd7092b599e8:
42 required resources and 3 runtime files byte-match source; isolated default-off/config/
registry/source smoke passed with no NumPy import. Full Torch/Ruff/Python matrix and fresh
isolated builds remain required in remote CI. No model experiment or empirical gain claimed.

Fresh whole-branch reviewer: no Critical/Minor findings and one Important reader/registry
integration finding, corrected above. Reviewer independently ran 292 passing tests plus the
two reproduced reader failures, max-roster/mixed-eligibility/exact-bid/capacity probes and
outside-checkout wheel smoke; all 42 resources and 3 runtime files matched. The author
validated the correction with the existing failing tests and the green 330-test suite.
No second whole-branch review was dispatched. No deferred minor finding remains.

Final: Ruling: declared principals/checkpoints/state and evidence are not authenticated by
this selector. Existing host identity, verifier and evidence controls remain mandatory;
otherwise eligibility could be falsely asserted. ASCII alias checks do not prove identity.

Final: Ruling: callback timeout/spending/isolation and asynchronous or restored transient
mutation are host responsibilities. The selector detaches and checks observed snapshots,
not arbitrary Python execution; omitting host isolation can permit external side effects.

Final: Ruling: recommendations never issue/consume grants, admit optimizer data, allocate
reward or promote a model. Existing host mechanisms alone perform those operations; treating
a report as authority would violate the documented boundary and is not a supported integration.

Final: Ruling: native MPT, learned competence and empirical gains remain Unsupported/unmeasured.
CPU tests establish software contracts only; extrapolating them could justify a harmful
promotion without evidence. Native model work requires separate configured operator authority.

Final: Ruling: primary paper and pinned repository were verified by the implementer through
web/GitHub tools, not independently re-fetched by the reviewer. Exact authored-note and
upstream blob identities are retained; an intake error would misstate source fidelity.
Remote omitted base blobs will be checked against the complete Git tree before publication
and merge; local snapshot completeness alone is not evidence of remote preservation.

Final: Ruling: local tests/Black API/wheel evidence do not substitute for Torch, Ruff and
both supported Python versions. Required final-head CI must pass before merge; skipping it
would leave compatibility and full regression behavior unqualified.
