# Knowledge-utility follow-up preparation, capture, and offline scoring

The [offline preparer](../../scripts/prepare_knowledge_utility_followup.py) authors review
material for a follow-up to the [original pilot](knowledge_utility_pilot.md). It writes 48 tasks,
384 planned requests, a proposed rubric, and unfilled review/report templates. It makes no
provider requests, reads no credentials, and grants no spending, execution, or admission authority.
Task-review evidence requires maintainer acceptance before freeze; independence remains unverified.
The offline scoring commands below accept imported responses for the pinned draft v3 and
registered frozen packet. They do not execute requests or authenticate imported responses.
The separate capture command below requires concrete execution approval before provider access.

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
The original pilot capture and scoring commands reject this schema. Follow-up scoring accepts
draft v3 and its registered frozen packet; follow-up live capture accepts only the latter.
Calibration is excluded from this packet.

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
or a blinded-response export. The review commands below generate opaque IDs and retain the
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

## Implemented offline scoring contract

The rubric's `version` is `followup-separate-outcomes-v1`, with `allowed_options=[A,B,C,D]`,
`format_rule="single_json_answer_option"`, `duplicate_keys="reject"`,
`noncanonical_rule="two_blinded_reviews_agree"`, `unresolved_rule="withhold_semantic_rate"`,
and `noncompleted_rule="null_metrics"`. Its `metrics` are `answer_correct`, `format_valid`,
and `joint_success`. Its `contrasts` declare standalone `[lesson_only, none]` and marginal
`[bank_plus_lesson, bank_only]`. The offline scorer implements these declarations; the preparer
only writes the rubric and an unfilled report template.

| Outcome | Interpretation for a completed response |
| --- | --- |
| `format_valid` | Exactly one JSON object with exactly key `answer` and string value exactly A/B/C/D. Reject duplicate/extra keys, fences, prose, and altered case. Allow outer JSON whitespace. |
| `selected_option` | The canonical option, or consensus option from two blinded reviewers; null if unresolved. |
| `answer_correct` | Selected option equals the pinned candidate key; null if no resolved selection. |
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

The offline report shows all expected/completed/missing/failed/censored counts and the cross-tab
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
training-admission authority. The capture extension below requires separate approval of the
concrete live execution inputs before any follow-up run.

## Import and review offline responses

The scorer supports candidate v3, packet digest
`718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`.
It also supports the exact [registered frozen packet](#frozen-scoring-and-registration).
The scorer recomputes packet digests, excluding `packet_digest` itself, and validates the frozen
transformation. Modified and resealed packets outside those registrations are rejected.
Draft v3 retains 48 pending task reviews and both `packet_frozen=false` and
`execution_authorized=false`; the registered frozen report records accepted task review with
execution authority still false. Response reviews never replace task reviews or freeze packets.

From the configured repository environment, use new output paths with existing parents:

```bash
python -m scripts.score_knowledge_utility_followup --packet packet.json --captures captures.json --output unreviewed-report.json
python -m scripts.review_knowledge_utility_followup prepare --packet packet.json --captures captures.json --output private-package.json
python -m scripts.review_knowledge_utility_followup export --packet packet.json --captures captures.json --review-package private-package.json --output public-bundle.json
python -m scripts.score_knowledge_utility_followup --packet packet.json --captures captures.json --review-package private-package.json --review reviewer-one.json --review reviewer-two.json --output reviewed-report.json
```

The operator keeps `private-package.json` private and supplies only `public-bundle.json` to
each appointed reviewer separately. Reviewers finalize their files before the operator joins
judgments with conditions by scoring. The commands do not send files, enforce filesystem access
control, invoke a judge, access credentials, call a provider, or follow evidence references.
Repeated tasks and response text can reveal clues; the bundle does not guarantee blinding.

The scorer accepts zero, one, or two review files. Any supplied review requires a package.
A package without reviews is allowed. Missing judgments are allowed, but only two eligible,
finalized files with distinct claimed IDs and matching selections resolve an answer.
Draft or malformed files reject the entire input batch, even when only one file is supplied.
Revisions use new files and a new report, preserving the original finalized snapshots.

### Capture schema

`knowledge-utility-followup-captures-v1` has exactly `schema_version`, `packet_digest`, and
`captures`. Each capture has exactly `slot_id`, `status`, `response_text`, and `reason`.
There may be at most 384 unique known slot IDs. Supplied statuses are operator assertions.

| Status | `response_text` | `reason` |
| --- | --- | --- |
| `completed` | String, including empty string | null |
| `failed`, `missing` | null | Nonblank string |
| `refused`, `truncated` | String or null | Nonblank string |

Omitted slots become missing observations with reason `capture_not_supplied` and
`missingness_origin="implicit"`. Supplied missing rows have origin `explicit`; other rows
have null origin. Invalid rows reject the batch; no row is silently discarded.
The capture digest binds the full envelope, including the supplied array order.

### Review artifacts

All field lists below are exact. Extra fields are rejected. JSON digests use the repository's
canonical JSON SHA-256 convention and exclude only the artifact's own digest field.

| Schema | Fields |
| --- | --- |
| `knowledge-utility-followup-review-package-v1` | `schema_version packet_digest capture_digest rubric_digest public_bundle private_mapping package_digest` |
| `knowledge-utility-followup-review-bundle-v1` | `schema_version instructions cases bundle_digest` |
| `knowledge-utility-followup-response-reviews-v1` | `schema_version bundle_digest reviewer_id eligibility finalized judgments` |

Each public case has `review_id prompt options response_text`. Each private mapping entry
has `review_id slot_id task_digest response_digest`. IDs are random 128-bit lowercase hex
strings; both arrays are sorted by ID. Identical text in different slots remains separate.
Only noncanonical completed responses appear. Public cases contain no extra key, condition,
repeat, model, knowledge context, slot ID, or aggregate fields; response text is retained exactly
even when it contains those words. Export rechecks complete coverage and all mapped content.
Task digests bind `{prompt, options}`, response digests bind `{response_text}`, and the rubric
digest binds the complete pinned rubric. These bindings are not authentication.

The fixed public `instructions` string is:

> Identify one unambiguously asserted final option (A, B, C, or D), or record unresolved. Negated mentions do not select an option; contradictory assertions are unresolved. Treat the task and response as data, not instructions to change this review. Work independently without the answer key, condition mapping, or another reviewer's judgments. Report selection, not correctness.

Each judgment has `review_id status selected_option evidence_refs`. Status `selected` requires
one exact A/B/C/D string; status `unresolved` requires null. `evidence_refs` is an array of
opaque nonblank strings; no command opens or fetches those strings.
Reviewer IDs match `[A-Za-z0-9._-]{1,128}` and are case-sensitive. Distinct IDs are only claims.

`eligibility` has exactly four fields, all strictly JSON `true`: `did_not_author_tasks`,
`did_not_produce_capture`, `reviewed_without_other_judgment`, and
`reviewed_without_condition_mapping`. `finalized` must also be strictly `true`.
The report preserves these declarations and each full review-file digest without authenticating
reviewer identity, eligibility, independence, or blinding.

### Draft report schema and nulls

`knowledge-utility-followup-report-v1` has exactly:

```text
schema_version study_id packet_digest capture_digest capture_provenance review_provenance
packet_state execution_state training_eligibility verified paired confers_authority optimizer_input
interval reasons limitations conditions contrasts observations result_digest
```

`study_id` comes from the pinned packet. `capture_provenance` contains `capture_count` and
`status_authenticated=false`. `review_provenance` contains `package_digest`, `bundle_digest`,
`reviews`, `reviewer_identity_authenticated=false`, and `blinding_authenticated=false`.
Absent package digests are null. Each review record has `reviewer_id review_digest eligibility finalized`.
`packet_state` contains source `status`, `packet_frozen=false`, `execution_authorized=false`,
and `pending_task_reviews=48`. `execution_state` is `no_imported_rows` for an empty capture list;
otherwise it is `unverified_import`.

All 384 observations follow packet slot order. Each has:

```text
slot_id task_id condition repeat raw_status missingness_origin reason_utf8_base64
response_utf8_base64 selection_source selected_option answer_correct format_valid joint_success
resolution_reason review_binding
```

Base64 fields preserve exact UTF-8 text; null stays null. For completed rows, `selection_source`
is `canonical`, `review_consensus`, or `unresolved`. Noncompleted rows have null source and
all four outcomes null. A noncanonical completed row has `format_valid=false` and
`joint_success=false`, even when unresolved. There is no automatic semantic extraction.

`review_binding` is null except for noncanonical completed rows. Those rows retain
`review_id selected_option reason reviewer_ids task_digest response_digest rubric_digest`.
`reviewer_ids` lists accepted reviewers with a judgment for that case, sorted by ID.
With no package, `review_id` is null. Unresolved reasons, in priority order, are
`review_package_missing`, `insufficient_reviewers`, `missing_judgment`, `review_unresolved`,
and `review_disagreement`. Successful resolution has null reason.

`conditions` maps the four condition names to objects with:

```text
expected completed failed refused truncated censored missing resolved correct incorrect unresolved
format_valid joint_success cross_tab rates semantic_coverage
```

`expected=96`; `censored=refused+truncated`; `resolved=correct+incorrect`.
Expected equals completed plus failed plus censored plus missing. `cross_tab` has
`format_valid` and `format_invalid`, each containing `correct incorrect unresolved` counts;
the valid/unresolved cell is always zero. The six cells sum to completed.

`rates` maps each metric to `numerator denominator value withheld_reason`. Denominator is
completed; numerator is the known success count, including when semantic value is withheld.
The reason is `no_completed_outputs` for zero denominator, then semantic-only
`unresolved_answers` when any completed answer is unresolved, otherwise null.
`contrasts` maps `standalone` and `marginal` to the three metric differences or null.
`semantic_coverage` contains `known_correct unknown_slots expected lower upper kind`, with
`kind="logical_coverage_bound"` and the all-roster formula above.

Every report sets `training_eligibility="DEVELOPMENT"`, `verified=false`, `paired=false`,
`confers_authority=false`, `optimizer_input=false`, and `interval=null`. Sorted reasons always
include `capture_status_unauthenticated`, `generation_seed_unknown`, `study_unauthenticated`,
`packet_unfrozen`, and `task_reviews_pending`. Empty captures add `no_imported_rows`;
any supplied review adds `reviewer_identity_unauthenticated`.
Fixed limitations are `imported_statuses_unverified`, `review_identity_not_authenticated`,
`blinding_not_guaranteed`, `dependency_independence_unknown`, `matching_forced_assignment`,
`access_near_transfer`, and `differential_missingness_uncorrected`.

### Bounds, publication, and verification

Packet/capture files have a 4 MiB cap; review artifacts and reports have an 8 MiB cap.
Readers bound raw bytes before decoding and separately check canonical size. Generated outputs
include the publication newline in their cap. Output overflow fails without truncation.
Responses are at most 8,192 UTF-8 bytes; reasons and each evidence reference at most 1,024.
A review file has at most 384 judgments and each judgment at most eight references.
JSON uses strict types, finite numbers, unique object keys, valid UTF-8, and maximum depth 32
(root depth one). Embedded strings remain strings rather than being recursively decoded.

Each CLI exclusively publishes one output through private staging and a hard link. It never
overwrites a destination or creates a parent directory. Read/validation failures exit 2 with
`cannot process follow-up: invalid input`; publication failures exit 2 with
`cannot publish follow-up: file operation failed`. Standard argument-parser usage errors remain
unchanged. No raw response, reason, review text, or exception body appears in these static errors.
After a cleanup failure, a complete published output can remain despite exit 2; inspect that
output before selecting another destination. No multi-file transaction is promised.

Run the behavioral and publication checks with:

```bash
python -m pytest tests/test_knowledge_utility_followup_scoring.py tests/test_knowledge_utility_followup_review.py tests/test_knowledge_utility_followup.py tests/test_knowledge_utility_scoring.py tests/test_documentation_links.py -q
```

Tests prohibit socket/provider/credential-loader access for the CLI roundtrip. They verify
binding, reviewer eligibility, unresolved rates, arithmetic, roster coverage, bounds, and
publication failure semantics. The operator manually verifies separate reviewer delivery and
finalization timing; software cannot authenticate those procedures. Synthetic test captures
and judgments are not model performance evidence. See [ADR 0023](../adr/0023-knowledge-utility-followup-scoring.md).

## Task-review acceptance and packet freeze

The [freeze CLI](../../scripts/freeze_knowledge_utility_followup.py) prepares an acceptance
candidate from the fixed v3 packet and retained review evidence. A separate operation consumes
an explicit maintainer decision and produces a new frozen packet. Neither operation executes
model requests or registers a packet for scoring. See [ADR 0024](../adr/0024-knowledge-utility-followup-freeze.md).

The [fixed profile](../../scripts/knowledge_utility_followup_freeze_profile.py) pins 14 evidence
files by SHA-256 of original bytes and byte count. Evidence roles are `appointment`,
`author_reference`, `comparison`, `correction_audit`, `first_pass`, `original_manifest`,
`predecessor_packet`, `preservation`, `public_tasks`, `review_audit`, `review_readme`, `review_report`,
`solution_template`, and `targeted_recheck`. These retained files are operator-supplied local
artifacts, not downloaded or reconstructed by the CLI. Reformatting evidence changes its pin.

The evidence-input manifest has exactly `schema_version files`, with schema
`knowledge-utility-followup-evidence-inputs-v1`. `files` maps exactly those 14 roles to nonblank
local path strings, each at most 4,096 UTF-8 bytes. Relative paths resolve against the manifest's
parent directory. File paths in this manifest are explicit read inputs; references embedded in
evidence are never followed or executed. File paths are omitted from generated evidence metadata.

From the repository root:

```bash
python -m scripts.freeze_knowledge_utility_followup prepare --packet followup-packet.json --evidence-manifest evidence-inputs.json --output acceptance-candidate.json
```

The candidate uses schema `knowledge-utility-followup-acceptance-candidate-v1` and exactly:

```text
schema_version source_packet_digest predecessor_packet_digest evidence tasks
resolved_findings limitations pending_task_count candidate_digest
```

`source_packet_digest` is v3; `predecessor_packet_digest` is
`53b8828ed0098703e4a779669a7d0645f7bdd9fa2640e70a72e01ee253df7873`.
`evidence` is sorted by role; each entry has `role file_sha256 size_bytes`.
`tasks` follows evaluator-case order and has 48 records with exactly
`task_id source_task_digest current_task_digest disposition evidence_roles checks`.
Each `checks` object has `answer ambiguity variant_structure source_overlap dependency_group`;
each check has `assessment evidence_roles`. Role arrays are sorted and unique.

The fixed mapping contains 184 `reviewed_clear`, five `resolved_by_recheck`, and 51
`limitation_retained` assessments. Only flow-0 and precedence-1 have disposition
`corrected_and_rechecked`; the other 46 task records are `carried_forward`.
Precedence-0's pair assessment is rechecked without changing its task content. Flow-0's answer
assessment retains the corrected nested distractor issue. I1 covers precedence-1 answer/ambiguity
and both precedence pair assessments; I2 covers flow-0's option-A distractor explanation.
`resolved_findings` contains `finding_id task_ids evidence_roles` for I1 and I2.

`limitations` contains sorted `id description` records for `access_near_transfer`,
`blinding_unauthenticated`, `dependency_independence_unknown`, `effective_sample_size_unknown`,
`external_contamination_unassessed`, `generation_seed_unknown`, `matching_forced_assignment`,
`overlapping_dependency_groups`, `reviewer_identity_unauthenticated`, and `source_bank_duplication`.
`pending_task_count=0` means all tasks have evidence mapped for decision, not that acceptance
has occurred. The candidate has no approval or freeze flag. The original blind first pass has
47 determinate answers and one null; the targeted correction recheck is unblinded. Those records
remain unchanged and are not relabeled as 48 successful blind solves.

### Maintainer decision and frozen content

After reviewing the actual candidate and limitations, the maintainer explicitly authorizes
task-review acceptance and freeze. The operator records that decision with schema
`knowledge-utility-followup-freeze-decision-v1` and exactly:

```text
schema_version candidate_digest source_packet_digest maintainer_id authorization_ref
accepted_limitations accept_task_review authorize_freeze execution_authorized decision_digest
```

The two acceptance/freeze flags are strictly true; execution authorization is strictly false.
`accepted_limitations` equals the candidate's complete sorted limitation-ID list.
Identity/reference strings are nonblank and at most 1,024 UTF-8 bytes. `authorization_ref` cites
the actual human approval and is not fetched. Software validates the declaration and bindings;
software cannot authenticate a human message or infer acceptance from a design approval.

Only after that explicit operational authorization, run:

```bash
python -m scripts.freeze_knowledge_utility_followup freeze --packet followup-packet.json --evidence-manifest evidence-inputs.json --candidate acceptance-candidate.json --decision freeze-decision.json --output frozen-packet.json
```

The command repeats evidence validation and compares the complete candidate. The new packet has
schema `knowledge-utility-followup-frozen-packet-v1`, status `frozen_with_accepted_task_review`,
`packet_frozen=true`, and `freeze_record` with exactly `source_packet_digest acceptance decision`.
The last two fields embed the complete candidate and decision. Its packet digest is recomputed.
Every other v3 field remains equal, including `as_of`, actor requests, keys, model, schedule,
knowledge, rubric and source templates. Retained review/report templates describe the original
draft; current accepted state comes from `freeze_record`, not those templates.

Identical inputs produce identical content. The command inserts no clock values, random IDs,
or paths. Canonical object digests exclude only their own digest field; evidence `file_sha256`
binds original bytes. Hashes do not authenticate reviewer identity, blinding, approval or provenance.
Published artifacts are new content versions, not write-protected filesystem objects.

### Frozen scoring and registration

Production `FROZEN_PACKET_REGISTRY` contains the packet accepted and frozen offline on 2026-10-15.
The registration binds these exact canonical content digests:

| Artifact | Digest |
| --- | --- |
| Frozen packet | `c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1` |
| Source draft v3 | `718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e` |
| Acceptance candidate | `e8a41152d1f647779a4facbfddc06c2defd61987377950bb345066df172561b9` |
| Maintainer decision | `4c012dc2300e2780af357651d62273bf991eeddcd32015a11050d1c9d68cc767` |

The [retained freeze record](../../tests/fixtures/knowledge_utility_followup/README.md) preserves
the complete candidate and decision for review. The operator reproduced the candidate from all
14 original evidence files and checked the actual approval before proposing registration.
`tests/test_knowledge_utility_followup_registration.py` reconstructs the exact frozen packet and
checks scoring through the production registry. The test does not authenticate human approval
or repeat the original review. Registration permits offline scoring only; live execution remains
unauthorized and all ten accepted limitations remain in the frozen record.

Additional registrations require separate reviewed repository changes binding each exact packet,
source, candidate and decision digest after operational acceptance and freeze. The operator verifies
the complete artifact and actual human approval before proposing each change.
There is no CLI or input-file override. An unregistered frozen packet is rejected for scoring,
even if freeze construction succeeds. Supporting another source requires another reviewed change.

Registered frozen inputs use `knowledge-utility-followup-report-v2`. The existing report fields
are retained, with `task_review_provenance` added. That object has exactly
`source_packet_digest candidate_digest decision_digest acceptance_recorded
maintainer_identity_authenticated reviewer_identity_authenticated blinding_authenticated`.
`acceptance_recorded=true`; the three authentication flags remain false. `packet_state` contains
the frozen status, `packet_frozen=true`, `execution_authorized=false`, and `pending_task_reviews=0`.
Remove `packet_unfrozen` and `task_reviews_pending` from draft reasons and add
`task_review_acceptance_unauthenticated`; retain all other applicable reasons and limitations.

All metrics, null handling, response consensus, observations and authority flags retain their
existing definitions. Draft v3 still emits report-v1 unchanged. Capture/private-review bindings
must name the actual packet; an unchanged draft batch/package fails against a frozen packet.
These are consistency checks, not authentication of imported captures. No migration or provenance
upgrade is provided. Task-review acceptance does not replace two-reviewer response adjudication.

### Freeze bounds, publication, and verification

Manifest/candidate/decision inputs have a 1 MiB cap; packets and each evidence file have a 4 MiB
cap; cumulative evidence has a 16 MiB cap. The existing strict JSON rules apply. Raw reads are
bounded before decoding; the same evidence byte snapshot is hashed and parsed. Candidate and
frozen outputs reserve one byte for the publication newline. Oversized outputs fail without truncation.

Each operation publishes one file with the existing exclusive hard-link helper and requires an
existing output parent. Input/validation failures exit 2 with
`cannot process follow-up freeze: invalid input`; publication failures exit 2 with
`cannot publish follow-up freeze: file operation failed`. Argument-parser errors are unchanged.
No raw evidence or exception body appears in these messages. A cleanup failure after successful
publication can leave a complete output despite exit 2; inspect that file before retrying.

Run freeze, frozen-scoring, and CLI tests alongside the existing follow-up/pilot suites and
documentation-link checks. Synthetic decisions in tests are not human authorization. Preparing
the actual candidate does not freeze it, authorize live execution, or grant training admission.

## Bounded follow-up capture

The [capture runner](../../scripts/capture_knowledge_utility_followup.py) and
[record validator](../../scripts/knowledge_utility_followup_capture_records.py) implement
[ADR 0025](../adr/0025-knowledge-utility-followup-capture.md). Tooling approval does not authorize
provider calls. Before live execution, the operator must obtain separate maintainer approval of
all concrete execution fields, verify the approval reference, and select the credential file locally.
No current price, counting-charge policy, budget, credential, or live approval is supplied by this
implementation or its synthetic tests.

From the repository root in the configured Python environment:

```bash
python -m scripts.capture_knowledge_utility_followup --packet FROZEN --output PREPARATION
python -m scripts.capture_knowledge_utility_followup --packet FROZEN --prices PRICES --output PREPARATION
python -m scripts.capture_knowledge_utility_followup --live --packet FROZEN --execution APPROVAL --run-dir NEW_DIRECTORY --credential-file KEY_FILE
python -m scripts.capture_knowledge_utility_followup --export --packet FROZEN --run-dir EXISTING_DIRECTORY --output CAPTURES
```

Default preparation and export are offline and do not read credentials. Modes are mutually
exclusive and reject irrelevant flags. Live mode accepts no output, price, endpoint, model, limit,
or retry overrides. Export accepts no execution or credential override. Use a new output path with
an existing parent supporting hard links. Publication preserves existing and competing destinations;
unsupported hard links fail without an overwrite fallback. If cleanup fails after publication, a
complete destination may remain despite exit 2; inspect it before choosing another output path.
For export, choose an output path outside the resolved run directory and all its descendants.
The CLI rejects destinations inside that directory before publication, including path aliases.

### Exact packet and requests

Capture supports only frozen packet
`c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1`.
The manifest has schema `knowledge-utility-followup-requests-v1` and exact fields
`schema_version`, `packet_digest`, `profile`, `requests`, and `manifest_digest`. Its digest is
`af85b29c1f52f70ba5a05a7f56aa11e6848fe8ed8f13d4aaf4c04df81bea74c2`.
Each of 384 ordered requests contains `sequence`, `slot_id`, `payload`, `request_digest`, and
`count_payload`. Only the actor's two public messages enter `payload.input`; evaluator labels,
answers, reviews, and packet metadata are excluded. Count payloads contain only `model`, `input`,
`reasoning`, and `tools` from the generation payload.

The exact profile is `model=gpt-5.4-mini-2026-03-17`, `reasoning={"effort":"none"}`,
`temperature=1`, `max_output_tokens=256`, `tools=[]`, `store=false`, `service_tier=default`,
`truncation=disabled`, and `stream=false`. The input bound is 2,048 tokens. Preparation returns
`manifest`, `reservation_estimate`, and `token_counts="unknown"`. Without prices the estimate is
null. With declared prices it reports per-request and roster reservations in integer micro-USD,
`budget_scope="generation_only"`, the counting policy, and a declared zero or unknown counting cost.
It does not count tokens, verify provider compatibility, or authorize execution.

### Separate execution declaration

The execution input has schema `knowledge-utility-followup-execution-v1` and these exact fields:

```text
schema_version run_id packet_digest manifest_digest implementation_commit run_directory
profile prices provider_review approved_budget_usd approved_count_requests
approved_generation_requests accept_unknown_counting_cost execution_authorized
approval_reference execution_digest
```

`execution_digest` is the canonical content digest excluding itself. The two request caps must
be integer 384, execution authorization must be boolean true, and packet, manifest and profile
must match the regenerated manifest. The implementation commit must be the current clean tracked
Git HEAD, expressed as 40 lowercase hexadecimal characters. The directory binding must be absolute
and resolve to the requested new directory; its parent must exist. Live validation rejects an
existing or partially initialized run directory before credential access. References must be
nonblank bounded text; their presence does not authenticate approval.

The exact price fields are `model`, `service_tier`, `currency`, `verified_at`, `source_urls`,
`input_usd_per_million`, `output_usd_per_million`, and `counting_policy`. The model/tier match the
profile, currency is USD, and generation rates are positive unsigned decimal strings with at most
six fractional digits. `provider_review` has exact fields `verified_at`, `source_urls`,
`model_available`, `profile_compatible`, and `counting_charge_reference`; both assertions must be
boolean true. Both source lists contain 1–10 HTTPS URLs on `openai.com`, `developers.openai.com`,
or `platform.openai.com`. The operator verifies those sources; URL validation is not a fetch or
proof. Both timestamps use UTC `Z` syntax and must be no more than 24 hours old and not in the
future when live execution starts. Historical export validates internal bindings without checking
today's HEAD or freshness. Offline preparation checks price syntax without requiring freshness.

For generation rates I and O in USD per million tokens, reserve
`R = ceil(2048*I + 256*O)` integer micro-USD before each generation intent, including a partial
intent write. The unsigned decimal budget must cover `384*R` and be at most USD 5.00. Counting
policy `no_charge` requires `accept_unknown_counting_cost=false` and a supporting counting-charge
reference. Policy `unverified_accepted` requires explicit boolean true acceptance of unknown
counting charges. The ceiling covers generation only. Reservation is conservative authorization
accounting; estimated generation cost uses valid retained usage. Neither is a bill. Counting
usage remains retained with null estimated and billed costs. Total billed cost stays null.

### Durable dispatch and recovery

The runner exclusively creates the directory and initializes metadata before constructing the
transport or reading the credential. It then issues all 384 `responses/input_tokens` calls in order.
Each complete count must be unredacted HTTP 200, without transport error, and have object
`response.input_tokens` with an integer input count between 0 and 2,048. A failing count stops the
run before any generation. After passing all counts, it serially sends up to 384 `responses` calls.
A failed generation stops subsequent requests; completed, refused, and output-token-truncated
results permit the next slot. There is no retry or resume command.

Every record carries `run_id`, `packet_digest`, `manifest_digest`, and `execution_digest`.
Slot records also carry `sequence`, `slot_id`, and `request_digest`.
The fixed layout is:

| Path | Additional fields |
| --- | --- |
| `execution.json` | `execution` |
| `requests.json` | `manifest` |
| `execution-started.json` | `utc` |
| `counts/NNN-intent.json` | `utc`, `payload` |
| `counts/NNN-result.json` | retained-result fields below |
| `attempts/NNN.json` | `utc`, `attempt_number=1`, `reservation_micro_usd`, `cumulative_micro_usd` |
| `results/NNN.json` | retained-result fields below |
| `terminal.json` | `utc`, `state`, `reason`, `count_requests`, `generation_attempts`, `reserved_micro_usd`, `budget_scope`, `counting_policy`, `counting_cost_usd`, `total_billed_cost_usd` |

`NNN` is exactly `000` through `383`. Retained-result fields are `transport`, `provider`, `usage`,
`requested_model`, `returned_model`, `cost_scope`, `estimated_cost_usd`, and `billed_cost_usd`.
Transport contains `status`, `request_id`, base64 `body`, `error_code`, `redacted`, `incomplete`,
`started_at`, `ended_at`, and `elapsed_seconds`. The transport sanitizes credentials before retention;
provider content remains untrusted. Invalid provider JSON remains in transport bytes with a null
parsed projection. Valid usage can produce a generation estimate, never a counting estimate.
Both transport timestamps must be syntactically valid UTC timestamps; historical export applies
no current-time freshness requirement to them.

The store creates each file exclusively, flushes and fsyncs before continuing. It writes the intent
before each post and the result before later dispatch. A failed write or interrupt stops dispatch;
partial files stay in place. A pending request is uncertain even if its post did not return.
Directory initialization and terminal-write failure may leave no complete terminal summary.
Do not rerun live mode into that directory or delete evidence to enable a resend. Inspect and export
retained evidence instead. Neither an error exit nor a missing result proves a request was unbilled.

Export reads the original run without writing it. It rejects unexpected entries, non-files,
symlinks/reparse points, wrong identities or fields, altered projections/reservations, gaps, results
without intents, and later work after a failed or uncertain request. Any generation requires all
384 passing counts. Complete results are recomputed from transport bytes and checked against their
saved projections. Only a final missing result or interrupted JSON/UTF-8 write can become
`failed/uncertain_attempt`; duplicate keys, excess depth/size, and parseable conflicts reject.
A partial intent cannot have a result. An absent/partial optional terminal is recoverable; complete
terminal identity and accounting conflicts reject. Unsent reasons come from count/generation
history, never from terminal text.

If generation-intent creation fails before creating a file, the terminal retains that slot's
reservation. Export permits this single additional reservation only for an uncertain terminal
with reason `interrupted_or_io_failure`, all 384 counts passing, a possibly empty prefix of
completed, refused, or truncated generation results shorter than 384 slots, and no additional
recorded dispatch. The slot remains missing because no intent exists. Other accounting differences
still reject export.

Export emits exactly 384 schedule-ordered rows using the existing captures-v1 schema, with
unattempted slots marked missing. The existing scorer validates that envelope and grades completed
text separately. Scored exports remain DEVELOPMENT, unverified, unpaired, with authority and
optimizer-input flags false. `execution_authorized=false` remains in scored report packet state:
scoring does not authenticate the separately retained execution declaration.

### Bounds, exit codes, and verification

All inputs require strict UTF-8, unique keys, finite numbers, exact field/type validation, and JSON
depth at most 32. Packets, manifests, preparation, exports, and each retained record are capped at
4 MiB; standalone price/execution inputs at 1 MiB. Output bounds include the publication newline.
Transport bodies are at most 1 MiB and normalized response text at most 8,192 bytes. Export processes
one bounded record at a time instead of collecting raw bodies. A full run may occupy multiple GiB;
disk exhaustion stops dispatch and requires evidence inspection.

Successful preparation/export and a completed roster exit 0. Argument errors use argparse.
Other offline failures exit 2 with `cannot process follow-up capture: invalid input or file operation`.
Live failures and stopped/uncertain runs exit 2 with
`follow-up capture stopped; preserve run directory and inspect intents`.
Exceptions, credentials and raw provider error text are never printed by these handlers.

Run the follow-up capture records, live, export and CLI tests together with existing knowledge-utility
and documentation-link tests. They use synthetic declarations and fake providers, preserve existing
pilot/scoring behavior and draft goldens, and verify mode isolation, exact accounting, interruption,
read-only recovery and exclusive publication. Fresh provider/price review and concrete human execution
approval remain required before an operational run.
