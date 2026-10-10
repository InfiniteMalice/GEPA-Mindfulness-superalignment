# ADR 0021: Extend causal diagnostics through existing V5 contracts

Status: PR-1/PR-2 approved on 2026-10-08; PR-3 design and native plan approved on 2026-10-09; PR-4 design, implementation plan and native execution approved on 2026-10-09; PR-6 design and PR-5 deferral approved on 2026-10-10, with PR-6 implementation plan and native execution also approved on 2026-10-10; PR-5 design/plan approved on 2026-10-09 and corrected native implementation approved on 2026-10-10.
Date: 2026-10-08.

Amendment: On 2026-10-10 the maintainer corrected the PR-5/PR-7 gates as specified below.
Earlier approvals remain recorded; the amendment supersedes the blanket implementation hold.

## Context

V5 already has canonical results, strict single-variable relation pairs, public claim graphs,
stakeholder records, evidence memory, diagnostic reporting and separately governed rewards.
The requested causal/debate series needs richer observations without duplicating these systems.
The [PR-0 audit](../recommendations/CAUSAL_ALIGNMENT_AUDIT.md) maps current interfaces and gaps.

## Decision for PR-1

Compose new opt-in diagnostic records with existing V5 and evidence identities. Preserve the
strict relation-pair evaluator as a complete-capture API. Use a separate, explicitly named
diagnostic entry point for planned multi-turn interventions and incomplete adjudication.
Retain correct per-arm cases, observed action classes, declared changed factors and independent
verification. Preserve missing, unresolved and censored states in explicit denominators.

For public argument analysis, reuse `ClaimGraph` and `CheckRequest`/`CheckResult`. For pluralistic
analysis, reuse `Stakeholder` and `Perspective`. These records remain declarations requiring
host verification, not optimizer signals or runtime capabilities. Reuse the existing evidence
state, memory adapter, evaluation epochs and curriculum admission boundaries in later PRs.

Do not modify the 17 canonical cases, fallback Case 0, stripe meanings, normative commitments,
reward formulas or authority gates in the diagnostic PRs. Separate semantic equivalence from
required sensitivity. An unchanged wrong decision is not correct invariance. A justified
decision update is not an invariance failure. Human adjudication remains available when evidence
or action acceptability is disputed.

## Decision for PR-2

Compose public debate snapshots with existing claim graphs and check records. Predeclare check
slots and semantic opportunities, retain incomplete capture, and authenticate exact verifier
envelopes independently. Bound sessions to eight rounds. Keep literal graph/action transitions
separate from independently judged semantic changes and preserve the PR-1 ablation boundary.
The [public debate guide](../sensitive_debate.md) specifies callbacks and denominators.

Compute exact fractional block sensitivity only for complete Boolean functions of one to four
inputs, with rational primal/dual certificates. Natural-language priority remains heuristic;
these fixtures do not establish the paper's recursive-decomposition or judgment-oracle assumptions.

## Decision for PR-3

Compose authored laundering pairs, public stakeholder plans, explicit role permutations and
independently authenticated pluralistic judgments. Preserve unverified source assertions and
simulated preferences separately from derived verification. Exact receipts bind complete public
captures, semantic rubrics and evaluator identities; reanalysis authenticates them again.
The [pluralistic guide](../pluralistic_robustness.md) defines missingness and denominators.

Keep sycophancy, overcriticism, perspective robustness and third-party interests separate. Retain
PR-1 metric contracts and PR-2 session bindings. Three-condition comparisons require declared
matching family/split/model-family/harness/seed/repeat/case/stripe coordinates plus identical
evaluation content and rubric contracts. They retain absent
observations and preserve distinct checkpoint/curriculum versions. Exports remain DEVELOPMENT
with stricter source restrictions nested intact; no training effectiveness follows from fixtures.

## Decision for PR-4

Compose exact artifact/version records, source fragments and digest-bound interpretations with
the existing evidence state, memory gate and claim graph. Recheck current host access for every
retrieval and withhold any derived content with an unavailable ancestor. Preserve original source
times, identities, statuses and training restrictions. Supply only an allowlisted producer view.

Keep structural support routes separate from freshly authenticated semantic verdicts. Retain
planned-but-missing observations in source-removal and three-condition retrieval comparisons.
Match exact evaluation content and scope, and preserve explicit measured cost units. The
[artifact topology guide](../evidence_topology.md) specifies these contracts and the offline example.
The maintainer approved the PR-4 written design, implementation plan and native execution through
a separate draft PR. PR-5 live reward use and PR-7 training/empirical claims retain their
evidence requirements, distinct from software implementation approval.

## Decision for PR-6

The maintainer approved the PR-6 written design and explicit deferral of PR-5 on 2026-10-10.
PR-6 will compose an opt-in offline evaluation report with existing diagnostic contracts.
The report will separate five data partitions by scenario family and transformation ancestry,
retain every attempted candidate, and report paired cluster-aware uncertainty, cost and failures.
Selection progress, evaluated behavior, independent improvement evidence and deployment
eligibility remain separate. The reporting module will issue no training or deployment authority.

The maintainer subsequently approved the PR-6 implementation plan and native execution on
2026-10-10. The [evaluation guide](../reliable_improvement_evaluation.md) defines the implemented
opt-in contracts, host authentication obligations and offline example. Repository defaults govern
the completed PR-5 pilot and the additional causal penalty remains withheld. The subsequent
amendment below distinguishes implementation readiness from live evidence and activation.

## Correction for PR-5 and PR-7

The maintainer clarified that the incorrect-answer penalty and the additional causal penalty
address distinct properties. The former penalizes an incorrect outcome; the latter requires an
independently verified unjustified decision change or missed required update under the approved
causal predicates. An incorrect-answer penalty alone does not establish duplicate causal
accounting. An active provider or overlay already penalizing the same causal failure does;
unknown causal accounting still withholds the additional deduction. Component renaming cannot
authorize a second deduction for the same causal failure.

The maintainer explicitly requested correction of the overly broad gates on 2026-10-10:

- PR-5 may implement and test its default-disabled adapter under the existing approved plan and
  native execution method. A new admissible live failure example is not required before coding.
- A real nonzero deduction still requires independently authenticated causal evidence, unchanged
  training admission, complete semantic accounting and separately authorized experiment settings.
  The pilot's stochastic flip does not establish those facts. No live penalty is activated.
- PR-7 design and offline curation may proceed through their normal review stages. Implementation
  follows prerequisite software work, not activation of PR-5's penalty. Curriculum admission,
  live training, withheld-scenario usefulness and deployment retain their separate requirements.

The corrected [PR-5 design](../../history/2026-10-09-verified-causal-reward-design.md) and
[implementation plan](../../history/2026-10-09-verified-causal-reward-plan.md) specify the runtime
conditions and the regression cases for incorrectness-only versus duplicate causal accounting.
Authored tests demonstrate software behavior; they do not certify a live failure or training gain.

## Implemented PR-5 boundary

The approved native implementation adds a separately invoked, default-disabled reward adapter.
The [adapter guide](../causal_reward_adapter.md) defines exact evidence records, fresh host
verification, semantic overlap accounting and an explicit optimizer accessor. Existing provider
and integrity formulas remain intact; an incorrectness-only reduction does not suppress a
separately verified causal failure. Severe or unresolved gates block enabled optimizer access.
Authored integration fixtures validate software behavior, not empirical gains or live permission.
No live activation, paid experiment, training run or deployment has been approved here.

## Alternatives considered

Broadening `RelationPair` would weaken its exact-intervention/oracle contract. Replacing existing
modules with a new causal, GraphRAG or memory framework would duplicate validation and authority.
Composition adds explicit joins, but preserves backward compatibility and separate review units.

## Verification and approval

PR-0's existing registry, documentation, V5 and reward tests verify metadata consistency and
unchanged behavior. The audit specifies PR-1's positive, negative and incomplete-observation
fixtures and metric denominators. Each later PR needs its own tests and review before acceptance.
The maintainer approved the PR-1 written design, then its implementation plan and native execution.
The [causal diagnostics guide](../causal_diagnostics.md) specifies the opt-in API and host contract.
The maintainer also approved the PR-2 written design, plan and native execution. These approvals
cover diagnostic PR-1/PR-2. The maintainer also approved the PR-3 written design, implementation
plan and native execution through a separate draft PR. Later architectural stages retain their
review gates.

The optional causal reward adapter preserves exact baseline equality when disabled or unverified.
Applying a live deduction requires reviewed policy, authenticated evidence and accounting for
the distinct causal failure. PR-7's empirical acceptance requires independent withheld-scenario
results; their absence does not block design or offline software work. No benchmark result,
debate victory or diagnostic record establishes training or deployment eligibility.
