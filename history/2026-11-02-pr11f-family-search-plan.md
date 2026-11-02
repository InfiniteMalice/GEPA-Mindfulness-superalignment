# PR 11f evaluator-guided family search: design, plan and evidence

## Intent and scope

Implement the master prompt's optional multi-family scientific-search pattern as a bounded
CPU adapter over existing improvement contracts. Fixed task/protected/budget commitments,
separate initial candidate families, explicit ancestry, scheduled donor transfers and retained
failed attempts improve experimental inspectability. No quantum code, geometry, orchestration
stack, model/API execution, reward allocation, admission policy or normative change is added.
Everything remains default-off DEVELOPMENT metadata/diagnostics, without promotion authority.

Use superpowers and the approved unavailable repo-quality-gate fallback: spec/design,
implementation, tests, independent review, evidence. Standing maintainer authority covers
implementation, ready PR publication, review follow-up and merge; no repeated permission menu.
No bd/.beads or checkbox issue tracker. Plans/decisions live in history per AGENTS.md.

## Approaches and selected design

A new autonomous search engine would duplicate orchestration and require unconfigured model
budgets. A new evaluator would compete with tested evidence authority. Choose a pure family
plan validator and journal reporter around existing CandidateSpec/ImprovementProtocol and
AttemptJournal/summarize_journal. Existing audit_improvement remains the only outcome-audit
path; the adapter never grades, selects, executes or promotes a candidate. This bounded
adaptation is the implementer's choice within the authorized staged research scope.

New evaluation/family_search_records.py provides strict v1 FamilySearchConfig (enabled=False),
SearchFamily(family_id,surface,hypothesis,evidence_refs), FamilyMembership(candidate_id,
family_id,round_index,round_id), and FamilySearchPlan(protocol,task_spec_digest,
protected_manifest_digest,parent_checkpoint_digest,inference_budget_digest,
training_budget_digest,families,members,evidence_refs,independent_rounds=1,
sharing_interval=1,max_candidates_per_family=16,max_evaluations_per_family=512).

Surfaces are model/data/memory/harness hypothesis labels, never eligibility judgments.
Allow 2..8 families, 1..64 candidates, rounds 0..15, at most 128 task cases, 16 metrics,
2048 slots and 4096 journal events. IDs/text/public references are bounded; exact version,
field, primitive and array types preserve existing serialization. No new dependency.

prepare_family_search(plan,manifest,*,config=FamilySearchConfig()) returns None when disabled
before inspecting task inputs. When enabled it snapshots exact records, checks the existing
manifest/protocol, exact candidate membership, common baseline and complete paired task/
metric/repeat/seed rosters with 2..16 shared seeds and common inference-budget digest.
Every family has an independent round-zero root. Parent rounds/freezes precede descendants;
cross-family single-parent edges are permitted only at the predeclared sharing schedule.
Per-family candidate and evaluation-slot caps are common upper bounds, not proof of equal
actual compute. A family label cannot prove structural novelty or admissibility.

FamilyMembership uses the existing CandidateSpec.parent_candidate_id; no second ancestry
graph. A donor transfer creates a descendant whose sole parent is the donor. Multi-parent
fusion is unsupported. This preserves the existing transitive exposure-history checks.
All memberships at a round index share a unique round ID. Journal proposal round IDs must
match memberships, chronological proposal rounds must not go backward, and proposed children
must follow their recorded parents. Evaluation rounds stay independently recorded.

summarize_family_search(plan,manifest,journal,*,config=FamilySearchConfig()) uses existing
summarize_journal globally and per family. It retains failed/rejected/cancelled/pending and
unattempted candidates, incomplete evaluations, original events and unknown/partial costs.
It reports supplied terminal statuses as declarations, never governed acceptance. Metrics
remain decomposed in independently authenticated audit_improvement reports; no winner,
composite reward, new admissibility checker or experimental success claim is emitted.
Inputs/outputs use the existing 4 MiB envelope; plans are limited to 1 MiB. Pure processing
has no callback, store mutation or live side effect; oversized reports fail closed.

## Source and adaptation boundary

Gu, Kemeny, Yelin and Lukin, arXiv:2610.22367v1 (2026-10-15), checked 2026-11-02:
opening framework/Figure 1, Validation and statistical guarantees, Cost model and scoring,
Evolutionary search. Paper CC BY-NC-SA 4.0. Author-linked andigu/uncompute pinned
64b5a13aa249f1360846657392e6c66817b26d0f: README and Apache-2.0 LICENSE read via GitHub.
This release contains circuits/task definitions/evaluator; no search-runtime source was
established by README/root inspection. Nothing copied/imported/executed from upstream.

The retained lesson is separate structural families, independent feedback, ancestry/failure
retention and periodic sharing. Synchronous round scheduling, fixed caps, complete paired
rosters and single-parent transfer are original GEPA adaptations. Circuit correctness,
failure budgets, TQ scoring, MAP-Elites geometry and quantum code are not imported.
Native search orchestration and comparative effectiveness remain Unsupported/unmeasured.

## Implementation sequence and review focus

First add failing strict-record/default-off/graph/roster/budget tests, then the minimal records
and prepare function. Next add failing journal/parent/order/failed-attempt/unknown-cost tests,
then the reporter and a real existing-audit/exposure integration control. Add the source note,
digest, disabled config, protocol guide and existing registry/matrix/reader/package links.
Run related suites, Black/syntax/line/import checks, wheel resources/outside-checkout smoke
and one fresh whole-branch review. Resolve verified findings; publish a ready PR, babysit
final-head review/CI, merge only when required checks and threads are clear.

Review focus: candidate-family versus task-family identity; hidden cross-family/multi-parent
influence; chronological proposal round and parent binding; incomplete or differently seeded
comparison rosters; partial costs and declared selection mistaken for verified improvement.
Host authentication, source novelty, actual metering, protected evaluation and governed
promotion remain independent existing responsibilities; CPU tests cannot establish them.

## Baseline and rulings

Base main735f0bbaff542061d385588c8f40505a0bd85d66/tree
c2474ca9b4ddce7a4240636ae2bcecab555d1527. Isolated snapshot at
/workspace/gepa-pr11f-family-search; no usable local Git metadata. All 1125 present tracked
blobs match merged main; 126 pre-existing remote-only app/history/binary/.beads files must
be preserved by overlaying changed paths on the full 1251-blob base tree.

Ruling: use bounded metadata/journal adaptation rather than autonomous search or a second
evaluator. This preserves authority and can be tested here; treating it as native search
or empirical improvement would overstate its research maturity.

Baseline: 145 related existing checks passed. New preparation tests observed missing-module
RED, then 23 passed; report tests observed missing-function RED, then 32 passed. A real
existing-ladder/audit composition control retains negative outcomes and donor contamination;
33 family controls pass. Missing source inventory observed RED before adding the reference.
The combined records/audit/integration/statistics/study/registry/source/link suite passed
217 checks in 18.66 seconds. No protected test expectation was weakened.

A metadata writer stopped at its own formatting assertion before changing registry.yaml:
its section-end search also matched list indentation. Restricting the boundary to actual
four-space YAML field headers resolved it. Reader evidence is derived in the registry's
existing order, preserving old links; source/recommendation consistency checks pass.

Black formatting used the installed in-process API because local CLI multiprocessing sockets
are sandbox-restricted. Python 3.10 syntax and 100-column checks pass. Local Torch/Ruff remain
unavailable; normal full remote CI is mandatory before merge. No dependency was installed.

Initial wheel SHA256 50286b4dfa175e6e1faf6447a1934ab16a413b3ca19cea4ebc3a91653de6d6d0:
44 required resources and 3 changed runtime files match source. Outside-checkout config,
default-off, registry and source smoke passed without importing NumPy. Authored source note
1594 UTF-8/LF bytes, SHA256 3a5037b1b8d505b288ecd87e5aa2a5f04b957e5d16eb9e0508cf0fc5c55e02bb.

Fresh whole-branch review completed: no Critical, Important or Minor code findings in the
20 changed paths. Independent 84-test family/source/link run passed. Additional probes rejected
mismatched paired conditions and unequal seed coverage; cancellation retained its incomplete
evaluation. All 20 hashes, wheel SHA256, 44 resources and 3 runtime files matched. The reviewer's
isolated wheel test stopped during dependency installation, before project build/assertions.
Author post-format related run passed 217 tests in 22.18 seconds. No deferred minor remains.

Final: Ruling: hidden donor inputs, structural novelty and actual checkpoint/task/budget
correspondence require host provenance/authentication. These declarations cannot prove them;
incomplete host evidence could otherwise create misleading lineage or experimental claims.

Final: Ruling: common caps/planned rosters and supplied journal costs do not establish equal
actual compute or complete billing. Existing host metering remains necessary; treating partial
costs as complete would invalidate efficiency comparisons.

Final: Ruling: live generation, autonomous search and comparative effectiveness remain
Unsupported/unmeasured. CPU validation establishes software behavior only; extrapolation would
justify unjustified promotion. External runtimes and spending need configured operator authority.

Final: Ruling: primary and pinned source verification was performed by the implementer, not
independently refetched by the reviewer. Exact authored-note/upstream identities and adaptation
limits remain recorded; source-intake mistakes would misstate scientific fidelity.

Final: Ruling: full Ruff/Torch/Python-version compatibility and isolated rebuilding require
normal final-head CI before merge. Local checks cannot replace that gate; skipping it would
leave full regression and environment compatibility unqualified.

Final: Ruling: remote-only files are preserved by a complete base-tree overlay and hash/mode
verification before publication and after merge. Local snapshot checks alone cannot establish
remote preservation; an incorrect overlay could delete unrelated/protected files.
