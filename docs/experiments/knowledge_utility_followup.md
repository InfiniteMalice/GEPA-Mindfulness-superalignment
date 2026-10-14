# Knowledge-utility follow-up preparation

The [offline preparer](../../scripts/prepare_knowledge_utility_followup.py) authors review
material for a follow-up to the [original pilot](knowledge_utility_pilot.md). It writes 48 tasks,
384 planned requests, a proposed rubric, and unfilled review/report templates. It makes no
provider requests, reads no credentials, and grants no spending, execution, or admission authority.
Task answers, source overlap, and independence still require independent review before freeze.

The first pilot's sole exact-answer failure selected the right route with extra prose inside
the answer string. The follow-up therefore separates answer correctness, canonical format, and
joint success. It preserves the original pilot's packet, rubric, and results. Authored complexity
is not evidence that the follow-up is harder for a model; no follow-up outcomes exist yet.

## Prepare offline

From the repository root in its configured Python environment:

```bash
python -m scripts.prepare_knowledge_utility_followup --output followup-packet.json
python -m pytest tests/test_knowledge_utility_followup.py tests/test_documentation_links.py -q
```

Supply a new output path with an existing parent directory on a filesystem supporting hard
links. The CLI accepts only `--output` (and standard help). It writes canonical UTF-8 JSON
using private staging and exclusive hard-link publication. Existing destinations and competing
writers are preserved; there is no overwrite fallback. Invalid authored data, missing parents,
write failures, unsupported hard links, and existing destinations produce exit code 2 with
`cannot prepare follow-up: invalid task data or file operation`. Staging is cleaned up without
unlinking the destination. If cleanup fails after publication, the complete destination may
remain despite exit code 2; inspect it before selecting another path.

The output is an evaluator packet. Only a request's `messages` constitute actor input.
Never send the complete packet or the containing request record to a model.
The existing v1 capture and scoring commands reject this schema. This release adds no live
capture or scoring command for the follow-up. Calibration is excluded from this packet.

## Roster and ordering

The [task data](../../scripts/knowledge_utility_followup_tasks.py) contains two structural
variants for each of 24 candidate mechanisms: inventory, pricing, conversion, flow, allocation,
rounding, route eligibility, time windows, precedence, packing, matching, queues, conjunction,
rule precedence, quorum, quantification, conditional logic, exception scope, state transitions,
transaction order, joins, aggregation, sets, and intervals. Each task has four distinct options,
one authored answer, at least three derivation steps, and an explanation for each distractor.
Most pairs change a decisive rule or constraint; the matching exception is described below.

Each task appears in `none`, `lesson_only`, `bank_only`, and `bank_plus_lesson`, twice each:
48 × 4 × 2 = 384 slots, 96 per condition. Repeats do not create independent tasks. The proposed
lesson, bank, and source episodes exactly match the original frozen packet. Both knowledge
proposals retain their pending review status and authored-demonstration lineage.

The matching pair uses the same unique feasible assignment in both variants. The second
variant adds cost summation and a locally cheap but infeasible distractor; it does not compare
multiple feasible assignments. The first access task closely reuses the source and bank's
badge-and-reservation denial pattern. Treat this as near transfer; success on that task alone
cannot establish distant transfer. These limitations do not demonstrate answer-key leakage
or statistical independence. Task review and dependency assessment remain required before freeze.

The schedule key is `knowledge-utility-followup-order-v1`. Sort task IDs by the hexadecimal
SHA-256 of UTF-8 `key + "|task|" + task_id`, breaking ties by ID. Cycle the correct option
through A/B/C/D in this order, giving 12 tasks per correct position. Fill other positions in
authored distractor order, keeping each explanation with its option. Labels remain fixed across
conditions and repeats. The common instruction requests exactly one JSON `answer` string with
value A, B, C, or D; the user message lists the public prompt and options in label order.

Order 96 task/repeat blocks by SHA-256 of `key + "|block|" + task_id + "|" + str(repeat)`;
break ties by task ID then repeat (0 or 1). Enumerate all 24 permutations of the condition
names in lexicographic order and use permutation `block_index % 24`. Each occurs four times.
This is a balanced deterministic schedule, not randomized assignment or a generation seed.

## Packet reference

The packet has exactly the following fields:

| Field | Contents |
| --- | --- |
| `schema_version` | `knowledge-utility-followup-packet-v1` |
| `study_id` | `knowledge-utility-followup-v1` |
| `as_of` | Authored metadata timestamp `2026-10-13T00:00:00Z`, not a capture time |
| `status` | `pending_independent_review` |
| `execution_authorized`, `packet_frozen`, `optimizer_input`, `confers_authority` | false |
| `training_eligibility` | `DEVELOPMENT` |
| `model` | `model`, `reasoning_effort`, `temperature`, `generation_seed`, `availability_checked` |
| `limits` | `generation_requests`, `automatic_retries`, `max_input_tokens`, `max_output_tokens`, `tools`, `approved_generation_budget_usd` |
| `knowledge` | `origin_packet_digest`, `source_episodes`, `lesson`, `bank` |
| `rubric` | Proposed outcome and adjudication declarations below |
| `schedule` | `key`, `assignment_policy`, `generation_seed_policy`, `block_count`, `slots` |
| `evaluator_cases` | Task text, options, keys, derivations, and pending review metadata |
| `actor_requests` | Slot identity plus only two public `messages` |
| `review_templates` | `task_reviews` and one unfilled `response_review` form |
| `report_template` | An unexecuted report with null outcomes |
| `packet_digest` | `content_digest(packet_without_packet_digest)` |

The model proposal is snapshot `gpt-5.4-mini-2026-03-17`, effort `none`, temperature 1,
`generation_seed=null`, and `availability_checked=false`. Limits are 384 generations, zero
retries, 2,048 input and 256 output tokens, no tools, and `approved_generation_budget_usd=null`.
These settings require fresh provider compatibility and price review before a separately
approved execution. The first pilot's US$0.20 allowance does not authorize this study.

The origin digest is `637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1`.
The preparer verifies that digest and recomputes the original packet's content digest before
copying knowledge. A digest binds content, not approval, identity, or empirical validity.

Each evaluator case has `task_id`, `mechanism_id`, `variant`, `prompt`, `options`,
`expected_option`, `derivation`, `distractor_errors`, `variant_rationale`, `task_digest`,
`dependency_group=null`, `dependencies_known=false`, and `review_status="pending"`.
Options map A/B/C/D to text. Distractor errors map the three incorrect labels to explanations.
The task digest binds `{prompt, options}`; the packet digest also binds all evaluator annotations.
Derivations refer to option text rather than preassignment letters.

Each schedule slot has `sequence` (0–383), `slot_id`, `task_id`, `repeat` (0/1), `condition`,
and `generation_seed=null`. Slot IDs are `{task_id}:{repeat}:{condition}`. An actor request adds
`messages` to exactly those fields. Messages contain a common system instruction plus assigned
knowledge and a user task with options. Bank text precedes lesson text in the combined condition.
Schedule metadata, expected answers, derivations, and review records stay outside messages.

Task-review forms have `task_id`, `task_digest`, `status="pending"`, `reviewer_id=null`,
`evidence_refs=[]`, and `checks`. Check names `answer`, `ambiguity`, `variant_structure`,
`source_overlap`, and `dependency_group` all start null. The response-review form has
`review_id=null`, `task_digest=null`, `response_digest=null`, `rubric_digest`, `reviewer_id=null`,
`selected_option=null`, `status="pending"`, and `evidence_refs=[]`. It is not a completed review
or a blinded-response export. Future review tooling must generate opaque IDs and retain the
slot mapping separately. The rubric digest binds the complete rubric declaration.

The report template has `execution_state="not_started"`, `expected_slots=384`, `conditions`,
`comparisons`, `observations=[]`, `verified=false`, `paired=false`, `interval=null`, and reasons
`not_executed`, `seed_policy_unknown`, `study_unauthenticated`. Each condition has `expected=96`;
`completed`, `failed`, `censored`, `missing`, `correct`, `incorrect`, `unresolved`, `format_valid`,
and `joint_success` are null. Its `rates` maps all three metric names to null.
Both comparison objects also map all three metrics to null. Null counts avoid fabricating
observations or failures before a run. The template is not an evaluation result.

Authored text is bounded by UTF-8 bytes: prompt 6,000; each option 256; each derivation step,
distractor explanation, and variant rationale 1,024. Derivations contain 3–12 steps. Blank text,
NULs, and options differing only in surrounding whitespace are rejected. The complete canonical
packet is limited to 4 MiB. These byte bounds do not verify the proposed input-token limit.

## Proposed scoring contract

The rubric's `version` is `followup-separate-outcomes-v1`, with `allowed_options=[A,B,C,D]`,
`format_rule="single_json_answer_option"`, `duplicate_keys="reject"`,
`noncanonical_rule="two_blinded_reviews_agree"`, `unresolved_rule="withhold_semantic_rate"`,
and `noncompleted_rule="null_metrics"`. Its `metrics` are `answer_correct`, `format_valid`,
and `joint_success`. Its `contrasts` declare standalone `[lesson_only, none]` and marginal
`[bank_plus_lesson, bank_only]`. These declarations describe a later scorer; they do not run one.

| Outcome | Proposed interpretation for a completed response |
| --- | --- |
| `format_valid` | Exactly one JSON object with exactly key `answer` and string value exactly A/B/C/D. Reject duplicate/extra keys, fences, prose, and altered case. Allow outer JSON whitespace. |
| `selected_option` | The canonical option, or consensus option from two blinded reviewers; null if unresolved. |
| `answer_correct` | Selected option equals the frozen key; null if no resolved selection. |
| `joint_success` | Correct canonical response; false for any noncanonical completed response. |

For failed transport/provider calls, declared refusal, truncation, or an unsent slot, preserve
the original status and leave all four outcomes null. Do not infer refusal from wording or
score censored partial answers. Canonical wrong answers are valid format but incorrect;
noncanonical answers can be correct after review while still failing format.

Two independent reviewers who neither authored the task nor produced the capture must review
each noncanonical completed answer before conditions are revealed. Each sees only an opaque ID,
public task/options, and exact response text, without the key, condition, repeat, context,
model identity, aggregates, or the other judgment. Text itself may reveal clues, so perfect
blinding is not claimed. Each identifies one unambiguously asserted final option or records
`unresolved`. Negated mentions are not selections; contradictory assertions remain unresolved.
Matching judgments must bind the same task, response, rubric, and distinct reviewer identities.
Disagreement or absent review remains unresolved; no automated substring matcher or model judge
fills the gap. Claimed reviewer identity and file integrity do not authenticate a reviewer.

A later report must show all expected/completed/missing/failed/censored counts and the cross-tab
of format and correctness, including unresolved outputs. Format and joint rates use completed
outputs as denominator. Semantic rates are null until all completed outputs in that condition
have resolved labels. Zero denominators yield null rates, and any contrast needing a null rate
is null. Publish all three standalone and marginal contrasts together, with task-level results.
Observed-rate contrasts do not correct for differential missingness or censoring.

An all-roster semantic coverage bound is `[known_correct/96, (known_correct+unknown_slots)/96]`.
Unknown slots include unresolved completed answers plus all missing, failed, and censored slots.
This is a logical coverage bound, not a confidence interval or an alternative performance estimate.

## Review and authority boundary

Before freeze, the maintainer appoints an independent reviewer to solve every task and check
ambiguity, paired structural variation, source overlap, and dependency grouping. Keep every
review pending until evidence exists. A reviewer must not infer independence from the 24 labels:
related mechanisms may need to collapse into fewer groups. Revisions require new packet versions
and retained predecessors; final task selection must not use final-roster model outcomes.

The [existing utility contract](../cumulative_curriculum.md#utility-and-milestones) withholds
verification with unknown generation seeds. This study also supplies neither an authenticated
study nor a supported uncertainty source. It reports no confidence intervals or significance
claims; the [PR-6 minimum cluster rule](../reliable_improvement_evaluation.md) does not make
these authored groups independent. Preparation grants no reward, optimizer, curation, or
training-admission authority. A separately reviewed capture/scoring extension and exact live
execution approval are necessary before any follow-up run.
