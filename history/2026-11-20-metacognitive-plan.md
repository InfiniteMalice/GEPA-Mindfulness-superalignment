# Metacognitive Monitoring and Control Implementation Plan

Execution: native implementation in this chat using Superpowers executing-plans, TDD, and the
maintainer-approved quality-gate fallback. The user approved the design and authorized creation,
review, fixes, and merge of all eight staged PRs on 2026-11-20.

Goal: source-linked diagnostic monitoring and bounded strategy proposals over existing evidence.

Architecture: derive immutable records from existing verification contracts, preserve original
evidence and event history, and expose an explicitly enabled shadow/replay runner separately
from the fixed overlay registry. Python 3.10+; no new heavy dependency or model download.

Spec: `history/2026-11-20-metacognitive-design.md` and the full user-supplied implementation brief.

## Global constraints

Preserve the constitution, exactly 17 canonical cases, CASE × STRIPE × REPEAT identities,
default reward behavior, permission boundaries, accepted evidence, and existing registry IDs.
New computational entry points default to disabled. No private reasoning as independent
verification, no automatic rewards, no execution in the controller, no hidden-label leakage.
Unknown values remain unknown. Python lines have an absolute 100-character limit. Do not invoke
`bd` or modify `.beads/*`. Store design/planning notes under `history/`.

## Review focus

1. Source IDs reused across runs, versions, repeats, or stripe coordinates must be rejected.
2. Forged or future verification inputs must not establish support or influence an earlier choice.
3. Shared provenance and conflicting live assessments must remain visible through supersession.
4. Disabled operation and unknown metrics must never silently become successful measurements.
5. Resource limits, partial diagnostic inputs, and no-problem episodes must have bounded behavior.

## Task 0: Architectural audit and baseline

Files: design/plan here, `docs/metacognitive_monitoring.md`, and a baseline record under
`docs/experiments/`. Document existing interfaces, limitations, baseline commands/results, and
all eight ablations. Run existing focused tests, complete offline tests, lint/format/types, and
the existing synthetic calibration response scorer and V5 planner. No runtime source changes.
Commit and open PR-0 for review. Record source revision and actual execution environment.

## Task 1: Derived assessment contracts

Create `gepa_mindfulness/verification/metacognitive_records.py` and
`tests/test_metacognitive_records.py`.

Produces immutable `MetacognitiveAssessment`, `MonitoringFinding`, `SharedDependency`, and
`AssumptionBinding` contracts plus explicit monitoring availability/uncertainty records.
Assessments retain existing `EpistemicContext`, `V5EvaluationCell`, `EvidenceReference`, existing
claim status and typed claim/hypothesis source records, method/version, source event IDs,
provenance, and diagnostic-only status. New dependency bindings link existing hypothesis and
claim IDs. Records have strict versioned `to_dict`/`from_dict` and detached snapshots.

Write failing tests for exact JSON round trips, unknown/missing fields, unsupported schemas,
identity mismatch, missing references, conflicting assessments, shared assumptions, supersession,
mutable-bypass revalidation, explicit null uncertainty, private-source rejection, and inability
to pass a diagnostic assessment as verified evidence or optimizer input. Implement minimally,
run the focused suite plus existing epistemic contracts, then commit/open PR-1.

## Task 2: Monitoring engine

Create `gepa_mindfulness/verification/metacognitive_monitoring.py` and
`tests/test_metacognitive_monitoring.py`.

Consumes Task 1 records and existing source snapshots. Produces
`monitor_episode(inputs: MonitoringInput, *, enabled: bool = False) -> MetacognitiveAssessment | None`.
`MonitoringInput` binds evaluation identity, cutoff, original evidence/hypothesis/uncertainty
records, explicit assumption links, and supplied existing diagnostics. Input validation checks
source IDs, types, histories, and bounds. Implement assumption/status audit, shared evidence and
assumption detection, contradiction retention, and adapters for existing continuity, calibration,
repetition, and verified-outcome diagnostics. Missing diagnostics are unassessed, never negative.

Write failing tests covering each finding, a no-problem input, missing sources, conflicts,
same-source false consensus, legitimate revision/repetition, and private/self-reported evidence.
Run focused and compatibility suites before committing/opening PR-2.

## Task 3: Cognitive repertoire

Create `gepa_mindfulness/verification/metacognitive_repertoire.py` and corresponding tests.
Produces immutable `RepertoireStrategy` entries and `get_repertoire() -> tuple[RepertoireStrategy, ...]`.
Each of the eight briefed strategies declares ID, description, applicability, required evidence,
expected output, verification, resources, prohibited effects, and research source. Clarification
and abstention remain distinct outputs of the relevant strategy. Write failing tests for complete
contracts, applicability decisions, malformed IDs, unsupported effects, and deterministic export.
Implement, verify, commit/open PR-3.

## Task 4: Shadow control proposals

Create `gepa_mindfulness/verification/metacognitive_control.py` and corresponding tests.
Consumes Task 1 assessments and Task 3 repertoire. Produces immutable `ControlBudget`,
`ControlProposal`, and `propose_control(assessment, *, budget, enabled=False)`.
Selection follows explicit finding priority and strategy applicability. It returns diagnostic
proposals only, including IDs, reason, uncertainty, estimates, budget, stopping condition, and
required authorization. No evaluator, executor, mutable store, or network callback is accepted.

Write failing tests for all supported proposal kinds, no intervention, unknown benefit,
clarification versus abstention, exhausted budget, unjustified stopping, unsupported strategy,
disabled behavior, deterministic ties, and no evidence/reward mutation. Implement and verify
before committing/opening PR-4.

## Task 5: PEO and evidence integration

Create `gepa_mindfulness/verification/metacognitive_peo.py` and corresponding tests.
Produces `monitor_recorded_episode(...)` adapter using existing action-bound event validation,
reconciliation, evidence-use, continuity, hypotheses and verification records. It validates a
recorded causal prefix at a cutoff and supplies detached original records to Task 2. Later
observations and evaluation-only labels are excluded from prospective control inputs.

Write failing tests for matched identity/chronology, future parents, missing verification,
prediction-error ambiguity, hypothesis revision, preserved conflicting/superseded evidence,
motivated-forgetting continuity controls, and semantic laundering. Implement and verify before
committing/opening PR-5.

## Task 6: Evaluation battery and ablations

Create `evaluation/metacognitive.py`, deterministic fixture helpers and
`tests/test_metacognitive_evaluation.py`.
Produces `evaluate_metacognitive(...)` over explicit matched V5 cells with four conditions and
eight ablations. Separate policy-visible episode prefixes from trusted evaluation labels. Arm D
is replay/simulation only; unavailable model outcomes remain NOT_RUN. Reuse calibration and
abstention metrics, with null/reason pairs for metrics without independently labeled observations.

Write failing tests for all ten challenge families, false positives, case/stripe coverage,
repeat determinism, held-out isolation, corrupted inputs, matched budgets, hindsight leakage,
missing outcomes, and immutable baseline/rewards. Implement and verify before committing/opening
PR-6. Record raw synthetic results and coverage, with no empirical alignment claim.

## Task 7: Runner, demonstrations and documentation

Create `evaluation/run_metacognitive.py`, two CPU demonstration fixtures, corresponding CLI tests,
and update permanent documentation, package data and experiment records where needed.
The CLI requires explicit opt-in and produces structured source-linked diagnostics for an
unsupported/shared assumption episode and a no-problem episode. No model download, external API,
reward update, accepted-evidence write, or execution is performed.

Write failing tests for the disabled CLI, enabled demos, strict malformed-input rejection,
deterministic JSON, installed-wheel resources, and absence of unintended authority effects.
Implement, run full offline tests, lint, formatting, applicable types, wheel build, outside-checkout
smokes and remote CI. Obtain a fresh whole-branch Codex review, address findings with regression
tests, and commit/open PR-7.

## Review and merge sequence

PRs can be stacked while earlier reviews run; each diff must remain scoped to its stage.
Before merging each stage, inspect fresh PR head, checks, all comments and unresolved threads.
Fix actionable findings and request/await review of the changed head. Merge only when CodeRabbitAI
or Codex has no outstanding problems and required CI passes. Use the expected head SHA and
preserve source history. Retarget/rebase dependent PRs as necessary without force-overwriting
another contributor's changes. The user's merge authorization already covers these operations.

## Verification commands

Use the existing `../.venv312/bin/` tools with offline environment flags. Focused suites run after
each behavior change. Stage acceptance runs the relevant new and reused-contract tests. The full
repository suite runs before claiming stage completion and before merging; lint/Black and CI type
targets run for every final head. Record any environment limitation as a limitation, never a pass.
No markdown TODO tracking is introduced; task completion and rulings live in the scoped execution
ledger, as allowed by the approved fallback, without touching Beads.
