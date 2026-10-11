# PR-7 design: synthetic curriculum and cumulative knowledge

Status: Proposed written design; awaiting maintainer review. No PR-7 implementation or live run is approved by this document.
Date: 2026-10-11.
Base: merged PR-5, commit `6d14ca0d4289207c6d2da3e2a05b1b90d320c88f`.
Workflow: superpowers brainstorming (architectural), with repo-quality-gate.

## Intent and scope

Implement the original assignment's PR-7 as an opt-in offline extension of existing lessons,
synthetic examples, and improvement reports. A curator should be able to inspect completed
episodes, retain supported lessons with their complete provenance, identify useful curriculum
contrasts, and prepare reproducible comparisons. Software acceptance and empirical usefulness
are separate outcomes. The absence of a live PR-5 penalty does not block this implementation.

The initial implementation will not call a model, execute training, admit records to training,
change reward formulas, write a learning catalog, or promote a model. Experimental exports
remain DEVELOPMENT. A later empirical study needs independent withheld scenarios and its own
execution authorization. Preserve the 17 canonical cases, Case 0 fallback, normative commitments,
abstention/clarification semantics, and existing provenance and action-authorization boundaries.

## Alternatives and recommendation

| Approach | Benefit | Cost or limitation |
| --- | --- | --- |
| Compose offline curation over existing lesson, generator, curriculum, and audit contracts | Reuses existing identities and admission boundaries; recommended | The host still supplies independently authenticated evidence and later execution |
| Extend the durable coevolution lifecycle with a knowledge store and automatic promotion | Direct end-to-end automation | Adds authority and duplicates lifecycle concerns; outside the initial PR-7 scope |
| Curate manually in documents only | Small initial change | Cannot enforce exact evidence bindings, stale-bank checks, or reproducible comparison plans |

## Existing components and proposed ownership

Reuse these existing interfaces without changing their authority:

- `gepa_mindfulness/learning_surfaces.py`: `LessonProposal`, `LessonCharacteristics`, human
  review status, and destination classification. A new curation result does not change a
  proposal's `LessonReviewStatus` or become a lifecycle receipt.
- `synthetic_data/causal_interventions.py`, `debate_interventions.py`,
  `laundering_families.py`, and `pluralistic_curriculum.py`: deterministic intervention
  construction, full source retention, and evaluator-only contrastive exports.
- `gepa_mindfulness/training/peo_curriculum.py`: curriculum units, deterministic mixtures,
  anchor retention, and explicit train/resume admission. No new DatasetProvider is required.
- `gepa_mindfulness/training/eligibility.py`: unchanged recursive training restrictions.
- PR-6's `evaluation/improvement_records.py`, `improvement_audit.py`, and
  `improvement_statistics.py`: partition manifests, candidate/attempt identities, paired
  comparison and uncertainty. Do not create a second statistics engine or candidate journal.

Proposed additions:

| Module | Responsibility |
| --- | --- |
| `synthetic_data/curriculum_records.py` | Immutable bounded episode, lesson-review, bank-snapshot, utility, milestone, and curriculum-recipe inputs |
| `synthetic_data/lesson_curation.py` | Authenticate complete review payloads, derive offline curation dispositions, and report utility and milestone evidence |
| `synthetic_data/cumulative_curriculum.py` | Join existing diagnostic/generator outputs into provenance-preserving offline candidate exports |
| `evaluation/curriculum_comparison.py` | Validate seven-arm component/budget manifests and assemble references to existing PR-6 comparisons |

Names are proposed, not existing APIs. The implementation plan will specify exact schemas and
test commands after this design is approved. Keep functions and records local to these four
responsibilities; do not introduce a general plugin framework or a new dependency.

## Episode provenance and lesson review

The host supplies complete public episode snapshots: episode identity, source-family and
ancestor identities, model/harness configuration, terminal status, observed outputs, evidence
references, original eligibility labels, and content digests. Only a completed episode can
support a curated lesson. Completed failure episodes may support negative lessons; their failed
outcomes and restrictions remain visible. Pending, cancelled, or incomplete episodes are
retained as excluded inputs rather than silently treated as successful experience.

Lesson content is an existing `LessonProposal`. Host- or model-authored summaries remain
proposals until reviewed. PR-7 does not extract truth from a rationale or private reasoning.
Public source text is data and is never executed as a command or treated as an instruction.

The host supplies a review that binds the full proposal, all source snapshots, the exact
existing bank snapshot, evaluator identity and evidence references. Separate review fields
cover factual support, contradictions, freshness, reproducibility, information preservation,
and redundancy. Each check has pass/fail/unknown status, evidence references, and a reason.
Human-required or disputed reviews remain unresolved. Typed status fields and hashes do not
authenticate themselves.

An explicit host callback receives a detached complete payload, including the curation policy
and as-of time. Only built-in `True` with an unchanged payload authenticates it. Missing callbacks,
exceptions, mutation, and truthy non-booleans leave the review unverified. The host is responsible
for evaluator independence, provenance completeness, and identifying semantically equivalent
or contradictory statements that have different text hashes.

The offline dispositions are `curated`, `withheld`, and `rejected`:

- `curated` requires authenticated, undisputed passing support, contradiction, freshness,
  reproducibility, information-preservation and nonredundancy checks, plus an existing approved
  human lesson review. It means included in this offline curation report only.
- `withheld` covers missing, unauthenticated, disputed, expired or incomplete evidence, pending
  human review, and reviews bound to a different bank or source revision.
- `rejected` covers an existing rejected proposal or authenticated failed review checks.

All applicable reasons remain visible, even when one reason already prevents curation. Unknown
utility does not become zero utility or prove uselessness; utility is a separate diagnostic.

Freshness uses caller-supplied UTC as-of time and explicit review validity bounds, not a hidden
wall clock. Reproducibility binds replay specification/configuration, seed policy, externally
captured results and independent check evidence. No replay is executed by the curator.

The bank is an immutable input snapshot of existing lesson versions and retained sources.
Reviews bind its digest. Returning a proposed next snapshot does not mutate an active knowledge
bank. Contradictions or newly expired evidence require a fresh review; adding a lesson never
silently deletes or overwrites a conflicting older entry. Exact duplicate IDs/content are
detected locally; semantic redundancy remains a separately authenticated review claim.

## Utility and information diagnostics

Report Knowledge-Weaver-inspired criteria separately:

1. Standalone utility: externally measured task outcomes with the lesson alone, together with
   an explicit no-knowledge comparison when supplied.
2. Marginal utility: outcomes with the exact existing bank plus the lesson minus outcomes with
   that same bank alone.
3. Information preservation and nonredundancy: authenticated assessments with source evidence;
   these are neither truth certificates nor optimizer rewards.

Pair utility observations only when task/content identity, executor configuration, rubric,
budget, repeat/seed policy, and knowledge condition match the declared comparison. Include raw
opportunities, successes/failures and missingness. Mismatched or unauthenticated evidence remains
descriptive and cannot support a verified utility comparison. No mutual-information estimator
or invented proxy truth score is added in this PR.

Uncertainty and withheld-scenario claims use PR-6's existing reporting machinery for supported
source schemas. Do not relabel a new curation report as a supported causal/debate/ladder report.
Where utility evidence has no supported PR-6 diagnostic source, report raw paired observations
and mark uncertainty unavailable; extending statistical source adapters requires explicit tests.

## Observable milestones

A milestone binds a completed episode, public step identity, declared legitimate goal, exact
before/after observations, evidence references, and an independent review. Report four distinct
nullable judgments: step completed, legitimate-goal progress, appropriate epistemic update,
and verified final success. A completed step cannot imply any of the other three.

Retain goal authorization, ethical-constraint violations, and severe events separately. A
milestone with unknown legitimacy or a verified constraint violation cannot count as verified
legitimate progress, even if its step completed. Invalid or unresolved final-success evidence
cannot be repaired by accumulating intermediate milestones. Duplicate milestone IDs reject;
the host authenticates alias identity and checks that repeated representations do not multiply
progress.

This initial PR adds milestone diagnostics only. It emits no numeric shaping reward and changes
no training objective. Potential-based or process-sensitive shaping remains a separate proposal
requiring explicit assumptions and policy-invariance analysis; no invariance guarantee is
inferred from the WPM results.

## Boundary examples and generation validation

Use existing causal and debate diagnostics to identify candidate decision boundaries, retaining
the exact source row and complete report/provenance. A failure identifies an opportunity for
curation, not a trusted target answer. Missing or disputed labels remain unresolved.

Reuse existing generators to construct or retain declared single-factor pairs and explicitly
identified compound controls. Each complete curriculum bundle includes correct invariance and
correct sensitivity examples, with benign assistance and legitimate refusal/authorization
counterexamples. An incomplete bundle is reported as incomplete and cannot be described as
balanced. Benign wording changes, two acceptable answers, or a justified decisive update are
not failure labels.

Validate generated examples separately from the generator and the lesson author. The review
binds both prompts, expected actions, changed factors, exact V5 identities, source family and
ancestry, generator/configuration version, and evaluator contract. Validate V5 cases using the
existing manifest; each arm keeps its correct case when a decisive intervention changes it.

Keep the subject-visible prompts separate from evaluator-only labels and expected answers.
Retain the complete evaluator-side source envelope in the offline export. No formatter may
strip a restricted origin to manufacture TRAIN provenance. Inputs from hidden evaluations are
retained as excluded audit entries, never proposed examples or lessons for future training;
their exposure still remains visible in the partition/accounting report.

All resulting exports contain `training_eligibility="DEVELOPMENT"`, `optimizer_input=False`,
and `confers_authority=False`, alongside every nested source restriction. Existing curriculum
collect/evaluate use can inspect them; train/resume must continue rejecting them. A future
admissible training dataset requires a separate source/admission process, not a switch that
rewrites DEVELOPMENT exports to TRAIN.

## Seven-arm experiment specification

Produce a reproducible comparison manifest for:

1. Existing training baseline.
2. Synthetic data alone.
3. Causal paired curriculum.
4. Causal curriculum plus semantic laundering.
5. Causal curriculum plus Sensitive Debate.
6. Causal curriculum plus pluralistic perspective checks.
7. Verified knowledge accumulation, when the preceding evidence gates permit it.

Record each arm's actual component configuration, data/content/family digests, starting model,
optimizer and reward configuration, seed/repeat policy, training budget and evaluation budget.
Every comparison names its control and intended component difference. Arms 4-6 use arm 3 as
their declared control. Arm 7 names an explicit frozen control; its comparison is unresolved
if that control is missing. No label alone proves an isolated ablation.

Planned budgets bind exact units, limits and data counts; measured compute, data consumption,
cost and completion status remain separate. A plan with equal limits does not prove matched
actual use. Budget mismatch, missing actual measurements or unintended component differences
prevent a matched-ablation claim and remain visible as reasons. Failed/cancelled attempts and
unused planned arms remain distinguishable through the PR-6 journal.

Reuse the five PR-6 partitions, their family/ancestry/content leakage checks, candidate freezes,
audit-exposure accounting and final-test authorization. Curation and lesson-bank inspection are
exposures: a case used to form a lesson cannot later be claimed as an independent withheld case
for the consuming candidate. Keep the entire source ancestry, including non-evaluated ancestors.

PR-7 constructs plans and joins externally supplied results; it does not execute the seven arms.
Reports distinguish software validation, curated evidence, empirical utility, and deployment
eligibility. Deployment remains `not_assessed`. Positive utility or progress does not suppress
alignment regressions or severe events.

## Validation and completion criteria

| Contract | Required evidence |
| --- | --- |
| Exact inputs | Round trips and detached snapshots; malformed types, duplicate IDs, changed digests, nonfinite values, cycles, deep/oversized input rejected before recursive parsing |
| Curation | Valid supported lesson; contradiction, staleness, failed reproducibility, changed bank, absent source, pending/rejected review and forged authentication controls |
| Utility | Hand-calculated standalone/marginal examples, duplicate bank content, missing observations, executor/budget/seed mismatch, utility without truth support |
| Milestones | Completed step without progress, justified epistemic update, unknown goal authority, severe event despite progress, false final success and duplicate IDs |
| Curriculum | Invariance/sensitivity plus benign/refusal controls; independent generation validation; missing counterexamples, expected-answer leakage and changed-factor mismatch |
| Admission | Nested DEVELOPMENT/REGRESSION/HIDDEN_EVAL and failed V5 records survive exports; existing train/resume rejection remains intact |
| Ablations | All seven manifests, explicit controls, exact one-component differences, unmatched budgets, missing actual usage, reused withheld scenarios and failed attempts |
| Compatibility | Canonical 17-case, fallback, causal/debate/pluralistic, PEO curriculum, learning-surface, reward and PR-6 regression suites |
| Authority | No model/trainer/network/catalog execution, optimizer score, admission receipt or deployment receipt; disabled opt-in entry points fail before side effects |
| Documentation | Executable offline example; every normative gate maps to a test or an explicit host-verification requirement; research claims marked proposed/not reproduced |

The existing code tree equals the PR-5 tree validated with 5,466 passing tests and 18 skips.
That is compatibility baseline evidence, not PR-7 implementation evidence. Establish focused
baselines again at implementation, then use failing regression tests, repository lint/format/type
checks and the full suite. No new product code has been written for this design.

Software completion requires the implemented contracts, tests, documentation and review.
Empirical acceptance additionally requires independently authenticated usefulness on withheld
scenarios at matched budgets without loss of original case behavior. Missing empirical results
are reported as not established and do not block software implementation.

## Research traceability

Primary abstracts were checked on 2026-10-11. None of these studies is reproduced here.

Zhao, Y., Liao, L., Shen, L., Zhao, X., Zhang, Y., Feng, F., & He, X. (2026).
*Learning to accumulate knowledge with mutual information*. arXiv.
[2610.10042](https://arxiv.org/abs/2610.10042).
The paper reports curator training on ALFWorld/WebShop combining standalone and marginal
success with MI-inspired feedback. PR-7 adapts the separation of diagnostic criteria, not its
training rewards or an implication that measured utility establishes truth.

Zhao, J., Tang, J., Shu, Y., Wu, J., Lu, Y., Tong, J., Xu, H., Ge, W., & Zhang, Q. (2026).
*World potential model: Pretrained world knowledge as progress potentials*. arXiv.
[2610.09560](https://arxiv.org/abs/2610.09560).
The paper reports milestone-anchored progress evaluation and policy-optimization results on
ALFWorld/ScienceWorld. The proposed external milestone checks and ethical/authorization gates
are repository requirements, not demonstrated consequences of that study.

Du, W., & Chung, S. (2026). *Can AI agents make open-ended scientific discovery? Evidence from
Station*. arXiv. [2610.08927](https://arxiv.org/abs/2610.08927).
The source studies discovery coverage with supervisor and reflection mechanisms in three
oracle-paper tasks and two additional tasks. PR-7 uses it as motivation for reproducible lesson
retention; it does not infer reliable unrestricted self-improvement.

Update the existing [traceability registry](../docs/recommendations/RESEARCH_TRACEABILITY.md)
and [ADR 0021](../docs/adr/0021-causal-alignment-diagnostic-extension.md) during the approved
implementation, preserving earlier decisions and the corrected PR-5/PR-7 gates.
