# Knowledge-utility follow-up study design

Date: 2026-10-13. Status: proposed design for maintainer review.
Preparation is authorized. Implementation, packet freeze, and live execution remain separate stages.

## Purpose and evidence

Determine whether the proposed knowledge improves task answers beyond improving adherence to
an answer format. Preserve the four conditions: `none`, `lesson_only`, `bank_only`, and
`bank_plus_lesson`. The intended deliverable at this stage is a reviewable study specification.

The first pilot completed 72 generations. The frozen rubric scored 71 as successful:
17/18 for `none` and 18/18 for each knowledge condition. The sole unsuccessful response was
`{"answer":"VIA costs less."}` where the required value was `VIA`. It identified the correct
route but violated the exact-answer requirement. This observation does not establish a
reasoning benefit, nor does the zero marginal delta establish that the lesson has no utility.
The tasks were near ceiling and the comparisons remain unverified.

The offline closeout reproduced both the capture export and the complete scoring report.
The report digest is `9615a041722389f6f290b8825117f92c64cbf1ff7b854e2566a3a7c4674c54f5`.
The capture implementation was `57959f1e03dd3427aa7a7605946e32583aa774f4`;
PR #794 merged as `c6d7a147b0812ac967dd778734941ec20e1319fb`.
Raw run evidence remains outside the repository. No prior response, rubric, or score changes.

Relevant contracts: [original pilot](../docs/experiments/knowledge_utility_pilot.md),
[capture decision](../docs/adr/0022-knowledge-utility-capture.md),
[utility evaluation](../docs/cumulative_curriculum.md), and
[improvement evaluation](../docs/reliable_improvement_evaluation.md).

## Approaches considered

1. Keep exact-match scoring and only make tasks harder. This is inexpensive but continues
   to combine answer correctness and canonical formatting in one result.
2. Use deterministic canonical scoring plus blinded review of noncanonical completed answers.
   This separates the outcomes without introducing an automated semantic judge. Recommended.
3. Add a model judge for every answer. This adds model dependence, cost, and another validation
   problem. It is unnecessary for this finite-choice study.

## Proposed roster and difficulty

Author 48 tasks: two variants for each of the following 24 candidate mechanisms. A mechanism
label is an authoring group, not evidence of statistical independence. The variants must differ
in a decisive boundary or constraint, rather than only replacing names or numbers.

| Candidate mechanism | Contrast between its two variants |
| --- | --- |
| Inventory reconciliation | Reservation precedes a shipment versus release precedes shipment |
| Tiered pricing | Threshold reached exactly versus crossed with a minimum fee |
| Unit conversion | Mixed units with an exact boundary versus a strict inequality |
| Flow conservation | Explicit loss before a split versus loss after one branch |
| Weighted allocation | Fixed overhead before division versus a binding capacity |
| Rounding policy | Per-item rounding versus rounding only the aggregate |
| Constrained route cost | Cheapest route violates eligibility versus a fee reverses ranking |
| Time-window scheduling | Inclusive endpoint versus exclusive endpoint |
| Task precedence | Resource constraint adds an order versus an apparent order is unnecessary |
| Resource packing | Best value violates capacity versus unused capacity is optimal |
| Matching constraints | Unique feasible assignment versus one locally attractive dead end |
| Queue discipline | Priority changes after arrival versus stable tie breaking |
| Access conjunction | One missing necessary condition versus an explicit overriding exception |
| Policy precedence | Specific rule overrides general rule versus no override applies |
| Quorum with abstentions | Denominator includes abstentions versus excludes them |
| Quantified rules | All members satisfy a condition versus one counterexample is decisive |
| Conditional statements | Necessary versus sufficient conditions with a converse distractor |
| Exception scope | Exception applies to one object versus an entire category |
| State transitions | Guard tested before update versus after update |
| Transaction order | Noncommuting updates versus rollback of only the latest operation |
| Table joins | Duplicate keys affect counts versus unmatched rows are retained |
| Aggregation filters | Filter before grouping versus group condition after aggregation |
| Set operations | Complement relative to a stated universe versus intersection precedence |
| Interval composition | Touching half-open bounds versus a nonempty overlap after clipping |

Each task specifies all necessary facts, three or more interacting steps or constraints, four
distinct options, and exactly one correct option. Distractors must correspond to documented
errors such as ignoring a guard, reversing precedence, or applying a threshold incorrectly.
An evaluator-only derivation must establish the answer and explain each distractor. No task
may depend on current events, external knowledge, unstated conventions, or a hidden trick.

The maintainer appoints an independent reviewer to solve every task and review its ambiguity,
dependency group, and source overlap before freeze. Unresolved tasks prevent freeze. A replacement
creates a new version and preserves its predecessor and rejection reason.

Keep the first pilot's proposed lesson and proposed bank content unchanged in this study;
retain their pending review status and authored-demonstration lineage. Changing either proposal
requires a separately identified study version. None of these tasks becomes training material.

### Worked authoring examples

These three examples illustrate the required depth; they are draft development material,
not a frozen audit roster. Final option positions are assigned by the procedure below.

| Example | Public facts and options | Evaluator derivation |
| --- | --- | --- |
| Inventory | Start with 23 items. Receive 8, discard 3 damaged items, reserve 7 of the remainder, then ship 9 from unreserved stock. Reservations remain unavailable. How many are available? A: 12; B: 19; C: 9; D: 16. | Physical stock ends at 19; 7 remain reserved; available stock is 12 (A). |
| Route | Direct costs 15 and takes 20 minutes. Via costs 4+6 plus a toll of 2 and takes 28 minutes. Express costs 10 plus a fee of 4 and takes 22 minutes. Routes taking more than 25 minutes are ineligible. Choose the cheapest eligible route. A: Direct; B: Via; C: Express; D: None. | Via is ineligible. Express costs 14, below Direct's 15; answer C. |
| Schedule | One machine runs jobs without gaps, starting at time 0. A takes 2 units, B takes 3, C takes 1. A precedes C. B must finish no later than time 4. Which complete order is feasible? A: A-B-C; B: A-C-B; C: B-A-C; D: C-B-A. | B finishes at 5 and 6 in A and B; D violates precedence. C finishes B at 3 and preserves A before C. |

Do not claim that authoring complexity guarantees lower baseline accuracy. A separately approved
calibration could use eight additional development tasks excluded from the final 48, with one
`none` generation each. Calibration is optional and is not part of the 384-request roster.
If used, retain all calibration versions and outcomes, finish revisions before final freeze,
and report its tiny sample descriptively. Never select or replace final tasks using their live outcomes.

## Allocation and capture proposal

There are 48 tasks × 4 conditions × 2 repeats = 384 generation slots, 96 per condition.
Two repeats characterize repeat variability; they do not create independent tasks.
Every slot is a fresh conversation. Actor views contain public task facts, options, the common
format instruction, and the assigned knowledge context only. Answer keys, derivations, reviewer
records, condition labels, and metadata remain evaluator-only.

For reproducibility, use schedule key `knowledge-utility-followup-order-v1`. Order task IDs by
the hexadecimal SHA-256 of the UTF-8 string `key + "|task|" + task_id`, breaking ties by task ID.
Assign correct-answer
positions A, B, C, D cyclically in that order, giving 12 tasks per position. Preserve authored
distractor order in the remaining positions. Options stay identical across conditions and repeats.

Create 96 `(task_id, repeat)` blocks with repeat values 0 and 1. Order blocks by the hexadecimal
SHA-256 of `key + "|block|" + task_id + "|" + str(repeat)`. Ties use task ID then repeat.
Enumerate the 24 permutations of the four condition names in lexicographic order; assign the
permutation at `block_index % 24`. Each permutation occurs four times. This reproducible,
balanced schedule does not establish random sampling, independent assignment, or a provider seed.

Propose the same model snapshot, reasoning effort `none`, temperature 1, input ceiling 2,048,
output ceiling 256, no tools, and no retries as the first pilot. Compatibility and prices must
be checked again before execution. Scheduling keys must never be passed off as generation seeds.
Unknown generation seeds keep the existing utility comparisons unverified and unpaired.

The existing runner accepts only the original frozen packet and 72 slots. It must reject this
new study until a separately reviewed capture extension exists. Do not pass a changed packet
through the old runner, alter its digest allowlist, or silently increase its roster bound.
There are no provider calls, credential access, or spending authorizations in preparation.
The first pilot's US$0.20 allowance applied to that run only.

## Separate outcome definitions

Every completed output receives these distinct fields:

| Field | Definition |
| --- | --- |
| `format_valid` | Boolean. Exactly one JSON object, exactly the key `answer`, with a string value exactly A, B, C, or D. Allow outer JSON whitespace only. Reject duplicate keys, extra keys, fences, prose, and altered case. |
| `selected_option` | A/B/C/D from a valid canonical response; otherwise a consensus value from blinded review, or null while unresolved. |
| `answer_correct` | Boolean equality of `selected_option` with the frozen answer key; null while the selected option is unresolved. |
| `joint_success` | `format_valid` and `answer_correct` for canonical output; false for noncanonical completed output even while semantic review is unresolved. |

For provider/transport failure, declared refusal, truncation, or an unsent slot, retain the
original status and set all four fields null. Never turn a censored partial answer into success.
Never infer refusal merely from answer wording. Preserve original response bytes and provenance.

Canonical wrong answers have `format_valid=true` and `answer_correct=false`. Noncanonical
correct answers can have `format_valid=false` and `answer_correct=true`. No substring matcher,
whitespace repair, or broad natural-language parser is permitted to fabricate semantic labels.

### Blinded review of noncanonical completed answers

Assign two independent reviewers who did not author the task or produce the capture. Each sees
the public task, its options, and exact response text under an opaque review ID. Hide condition,
repeat, knowledge context, model identity, answer key, aggregate outcomes, and the other review.
Map review IDs to slots in a separate retained file. Response text can itself reveal contextual
clues; disclose that limitation rather than claim perfect blinding.

Each reviewer records the one option explicitly asserted as the final answer, or `unresolved`
if none, multiple conflicting options, or an ambiguous statement remains. Negated mentions are
not selections. A response such as `{"answer":"C is the cheapest eligible route."}` maps to C
only if the reviewer finds an unambiguous assertion of C. Do not grade hidden reasoning or infer
an intended answer from a plausible fragment.

Accept a noncanonical `selected_option` only when both reviewers independently name the same
option and their records bind the exact task, response, rubric, and reviewer identities.
Disagreement, absent review, or an unresolved judgment remains unresolved; no tie-breaking
model is used. Complete and freeze review records before revealing condition assignments.
Review-record integrity and claimed reviewer identity are not authentication. Preserve the
existing distinction between operator assertions and independently authenticated evidence.

## Reporting and interpretation

Report 96 expected slots per condition, with separate completed, failed, censored, and missing
counts. Among completed outputs report canonical-format counts, correct/incorrect/unresolved
answer counts, and the full cross-tabulation of format and correctness. An unresolved semantic
answer must remain visible and must not be silently scored wrong or dropped.

Report format and joint-success rates over all completed outputs, with their denominators.
If a denominator is zero, its rate and any contrast that uses that rate are null.
Report the semantic correctness rate over completed outputs only when every completed output
has a resolved semantic label; otherwise the rate and semantic contrast are null. Also report
the number resolved so reviewers can distinguish low correctness from incomplete adjudication.

For each metric, descriptive standalone and marginal contrasts use `lesson_only - none` and
`bank_plus_lesson - bank_only`, respectively. Report all three metrics together, rather than
selecting the most favorable contrast after seeing results. Show status counts and task-level
results alongside aggregates, since missing or censored outcomes can differ between conditions.
No observed-rate contrast is an estimate corrected for such missingness.

As a coverage diagnostic for semantic correctness, report an all-roster interval
`[known_correct / 96, (known_correct + unknown_slots) / 96]` per condition. Unknown slots include
unresolved completed outputs and every failed, censored, or missing slot. This is a logical
coverage bound, not a confidence interval or a substituted performance estimate.

The 24 candidate mechanisms may collapse into fewer independent dependency groups after review.
Do not claim 24 independent clusters from these labels. The current PR-6 interval procedure
requires a supported source and at least 20 reviewed clusters; this study design does not supply
either automatically. This version reports no confidence intervals, significance claims, or
verified improvement. All artifacts remain `DEVELOPMENT`, `confers_authority=false`, and
`optimizer_input=false`. No reward, optimizer, curation, or training-admission policy changes.

## Implementation boundary and acceptance

Implement the follow-up in stages. The first implementation would prepare the offline task
roster, actor/evaluator separation, rubric, ordering, and empty report/review templates under a
new `knowledge-utility-followup-packet-v1` schema. It would also document the protocol. It would
not implement live capture or a semantic review authority. A later reviewed scorer would use
new capture, adjudication, and report schemas without changing the v1 pilot entry points.

Likely new preparation files are `scripts/prepare_knowledge_utility_followup.py`, a dedicated
authored task-data module, `tests/test_knowledge_utility_followup.py`, and
`docs/experiments/knowledge_utility_followup.md`. Separate data and preparation logic so each
task's derivation is reviewable. The implementation plan must specify exact fields and limits
before code is written. Update an ADR when the new scoring/capture interfaces are implemented.

Preparation acceptance requires deterministic content digests and schedule, 48 unique cases,
384 unique slots, 96 slots per condition, 12 correct options in each position, four occurrences
of every condition permutation, and exactly two variants per candidate mechanism. Every
case needs a derivation and distractor explanations. Every task review starts pending; no
generated fixture claims an independent review occurred. All output paths refuse overwrite.

The later scoring tests must distinguish canonical correct, canonical wrong, noncanonical
correct, noncanonical wrong, duplicate-key JSON, contradictory prose, unresolved review,
missing review, disagreement, stale response bindings, and censored partial answers. Tests
must prove unresolved labels withhold semantic rates and that every expected slot survives.
The original packet digest, export, and scoring tests must remain unchanged and passing.

## Design self-review and review request

This specification fixes the study size, authoring requirements, ordering, proposed settings,
measurement definitions, unresolved-outcome policy, and authority boundary. The 48 final tasks
and their independent review are deliverables of preparation, not claims made by this document.
No task set is frozen and no live allowance is inferred from this design.

The maintainer should review the 384-generation size and the two-reviewer requirement for
noncanonical answers before the implementation plan is written. These are the main added
costs compared with the first pilot. A preparation-only implementation is the next stage.
