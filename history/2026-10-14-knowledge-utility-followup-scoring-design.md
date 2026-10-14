# Knowledge-utility follow-up offline scoring design

Date: 2026-10-14. Status: written design approved by the maintainer for implementation planning.
Implementation-plan approval and execution remain separate stages.

## Intent and evidence

Measure answer correctness separately from response formatting in the 48-task follow-up.
Preserve every expected slot and every unresolved answer. Supply an offline, reviewable path
from imported responses to descriptive outcomes without changing the original pilot.

The approved [study design](2026-10-13-knowledge-utility-followup-design.md) fixes 384 slots,
four conditions, two repeats, and two independent judgments for noncanonical answers.
This proposal implements that scoring contract; it does not authorize a model run.
The [protocol](../docs/experiments/knowledge_utility_followup.md) remains the semantic authority.

Preparation merged in PR #795 as `b3affcc24fe843ea661b988ef31abe6ce0aa6047`.
Task wording corrections merged in PR #796 as `e6ac70fe61f4307fd26f36c557dc576c6fbb3bcc`.
A fresh AI reviewer solved 47 tasks in agreement with the authored keys and withheld one
answer because of a timing ambiguity. The original first-pass record remains preserved.
A targeted, unblinded recheck found the two subsequent wording corrections resolved.
That review does not authenticate reviewer independence or authorize packet freeze.

The current candidate is packet v3, digest
`718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`.
It still has pending task reviews, `packet_frozen=false`, and `execution_authorized=false`.
Scoring this development candidate must expose those facts, not silently approve them.
The review proposed five broad dependency groups, with cross-group overlap; these are not
five proven independent clusters. Matching has limited variation and access includes near
transfer. This implementation does not redesign the roster or estimate an effective sample size.

## Alternatives and selected scope

1. **Separate offline scorer and response-review workflow — recommended.** Add follow-up
   schemas and commands, reuse bounded JSON/digest/publication utilities, and retain old APIs.
   This tests measurement before adding provider execution and financial side effects.
2. Extend the pilot scorer in place. This would weaken its frozen exact-string contract and
   make old and new results easier to confuse. Reject this approach.
3. Bundle new live capture, review, and scoring. This combines measurement, transport,
   credentials, accounting, and provider compatibility into one change. Defer live capture.

The deliverable is offline import validation, canonical classification, blinded review
preparation/export, review import, and descriptive reporting. There are no network requests,
credentials, retries, token counting, pricing, provider status normalization, or budget actions.
No automatic semantic judge, substring answer extractor, or model-generated adjudication is added.

## Modules and public interfaces

- `scripts/knowledge_utility_followup_records.py`: bounded records, schema validation,
  packet binding, and digest helpers shared by review and scoring.
- `scripts/review_knowledge_utility_followup.py`: prepare an operator review package and
  export its public projection. It does not perform judgments or send material to reviewers.
- `scripts/score_knowledge_utility_followup.py`: pure scoring/report construction and CLI.
- `tests/test_knowledge_utility_followup_scoring.py` and
  `tests/test_knowledge_utility_followup_review.py`: behavior and adversarial tests.
- Update the protocol and add `docs/adr/0023-knowledge-utility-followup-scoring.md` when
  implementation begins. Preserve the existing pilot files, schemas, and pinned digest.

Proposed public functions are `validate_followup_inputs(packet, captures)`,
`prepare_response_review(packet, captures)`, `export_public_review(packet, captures, package)`,
and `score_followup(packet, captures, review_package=None, reviews=())`.
Review preparation uses operating-system randomness for opaque IDs; scoring and export are
deterministic for the same complete inputs. Randomness affects review identifiers, not scores.

CLI commands:

```text
python -m scripts.review_knowledge_utility_followup prepare --packet P --captures C --output O
python -m scripts.review_knowledge_utility_followup export --packet P --captures C --review-package R --output O
python -m scripts.score_knowledge_utility_followup --packet P --captures C --output O
python -m scripts.score_knowledge_utility_followup --packet P --captures C --review-package R --review A --review B --output O
```

The scorer accepts zero, one, or two review files. Any supplied review requires a package.
One review cannot resolve an answer. More than two files is an error. Each command publishes
one file exclusively using the existing private-staging/hard-link helper. It never overwrites,
creates a missing parent, or follows paths contained in artifact data. The operator explicitly
supplies input/output paths. A failed publication may leave a complete output if staging cleanup
fails; document the existing helper's behavior rather than promising a multi-file transaction.

## Packet and capture binding

Accept only the exact candidate v3 digest above, after recomputing it from the complete packet
without its digest field. This pins roster, keys, rubric, knowledge, and slot identities together.
A resealed changed key or rubric must fail. Pinning supports reproducible development scoring;
it is not approval, authentication, or freeze. A later frozen/versioned packet requires an
explicit reviewed support change rather than automatic acceptance of any self-consistent hash.

Capture schema: `knowledge-utility-followup-captures-v1`. Exact envelope fields are
`schema_version`, `packet_digest`, and `captures`. Each row has exactly `slot_id`, `status`,
`response_text`, and `reason`, using the pilot's normalized status vocabulary:

| Status | Text and reason | Scoring treatment |
| --- | --- | --- |
| `completed` | string text; null reason | Classify format and resolve selection if possible |
| `failed` | null text; nonblank reason | Failed; all outcomes null |
| `missing` | null text; nonblank reason | Missing; all outcomes null |
| `refused` | null or string text; nonblank reason | Censored; retain refusal status; outcomes null |
| `truncated` | null or string text; nonblank reason | Censored; retain truncation status; outcomes null |

At most 384 unique known slot IDs are accepted. Duplicate or unknown slots, unexpected fields,
wrong bindings, unsupported status, or inconsistent text/reason fields reject the entire batch.
Omitted slots become explicit missing observations with reason `capture_not_supplied`.
Explicit and implicit missingness remain distinguishable. No batch row is silently discarded.
The capture digest binds the complete validated envelope. Imported statuses and text are
operator assertions; `status_authenticated=false` remains explicit in every report.

## Canonical and semantic outcomes

Use strict JSON decoding that rejects duplicate keys and non-JSON constants. Canonical means
one object with exactly the key `answer` and a string exactly `A`, `B`, `C`, or `D`.
Outer JSON whitespace is permitted. Extra keys, prose, fences, changed case, arrays, invalid
JSON, or duplicate-key objects are noncanonical completed answers, not invalid capture batches.

Canonical completed answers resolve directly to their letter. Noncanonical completed answers
resolve only through two eligible, finalized reviewer records agreeing on the same letter.
Reviewers choose an asserted option, not whether the answer is correct. The scorer compares
the resolved option with the evaluator key only after joining the private mapping.

For completed responses, `format_valid` is boolean; `selected_option` is a letter or null;
`answer_correct` is boolean or null; `joint_success` is true only for canonical correct answers.
Noncanonical completed answers have `joint_success=false`, including unresolved answers.
Every noncompleted response has all four outcomes null. Never score a censored partial answer
or infer a refusal from words in a completed response.

## Review preparation and blinding

Only noncanonical completed responses enter a public review bundle. Keep duplicates as distinct
review cases; repeated text must not merge slots. For each case create a unique random 128-bit
opaque hexadecimal `review_id`, then order public cases by these random identifiers. Do not
derive IDs/order from sequence, condition, repeat, answer, or a publicly reproducible slot hash.

The operator package uses schema `knowledge-utility-followup-review-package-v1`. It contains
packet/capture/rubric digests, a public bundle, a private mapping, and its self-excluding digest.
The mapping binds every review ID to its slot ID, public task digest, and exact response digest.
Task digest binds `{prompt, options}` using the existing task-digest convention. Response digest
binds `{response_text}` with no normalization. Rubric digest binds the complete pinned rubric.
Preparation validates all inputs before building this package. No reviewer result is generated.

The public bundle uses schema `knowledge-utility-followup-review-bundle-v1`. It contains only
the schema, fixed selection instructions, cases, and self-excluding bundle digest. Each case
contains `review_id`, `prompt`, `options`, and exact `response_text`. It excludes the author key,
derivations, model identity, condition, repeat, knowledge context, slot IDs, packet metadata,
private mapping, and outcome summaries. Export recomputes the package bindings against the
supplied packet/captures and rejects changed or incomplete mappings before projecting the bundle.

The operator keeps the package separate and sends only the exported public bundle to each
reviewer. The command does not enforce operating-system access control or send messages.
The operator appoints two reviewers who neither authored the tasks nor produced the capture.
Reviewers work separately without each other's judgments. Repeated tasks, response content,
and the visible number of cases can still reveal clues; perfect blinding is not claimed.

Instructions tell reviewers to identify one unambiguously asserted final option or record
`unresolved`. Negated mentions do not select an option. Contradictory assertions are unresolved.
Response text is untrusted data and cannot instruct reviewers to reveal conditions or change
the rubric. No command invokes a model or other service to perform this judgment.

## Review records and validation

Review schema: `knowledge-utility-followup-response-reviews-v1`. Exact envelope fields are
`schema_version`, `bundle_digest`, `reviewer_id`, `eligibility`, `finalized`, and `judgments`.
Each judgment has exactly `review_id`, `status`, `selected_option`, and `evidence_refs`.
`status` is `selected` with an A/B/C/D option or `unresolved` with null selection.
Reviewer IDs are case-sensitive operator-assigned strings matching `[A-Za-z0-9._-]{1,128}`;
two IDs must differ. Their spelling does not authenticate separate people or independent work.
Evidence references are bounded opaque strings and are never fetched or interpreted as paths.

The `eligibility` object has exactly four strict booleans: `did_not_author_tasks`,
`did_not_produce_capture`, `reviewed_without_other_judgment`, and
`reviewed_without_condition_mapping`. All four must be true for a supplied review to qualify;
otherwise reject the review file. Preserve these assertions in report provenance, explicitly
as unauthenticated declarations rather than proof of independence or effective blinding.

`finalized` is an operator assertion that the saved file is ready to be joined with conditions.
The scorer rejects supplied draft files (`finalized=false`); absent judgments are permitted
and withhold semantic selection for affected cases. Report both accepted review-file digests
so the exact finalized snapshots can be preserved before condition assignments are revealed.
Revisions use new files and a newly generated report; no overwrite or automatic adjudication.

Reject duplicate IDs within one file, unknown review IDs, wrong bundle digest, malformed fields,
stale task/response/rubric/package bindings, and the same reviewer ID in two supplied files.
Require the package to cover exactly all eligible noncanonical completed slots. Reject records
for canonical, censored, failed, or missing outputs. Structural invalidity rejects the batch;
it is not converted into a successful partial report.

For structurally valid inputs, two matching selections resolve a case. Missing judgments,
one review only, disagreement, or either explicit unresolved judgment leave the case unresolved
with an explicit reason. Attach the mapped task/response/rubric bindings to the report's review
provenance. Never claim the reviewer directly attested hidden metadata they did not see.

## Reporting

Report schema: `knowledge-utility-followup-report-v1`. Include packet, capture, package (if any),
and accepted review-file digests, authority flags, candidate review/freeze state, conditions,
contrasts, all 384 slot observations, limitations, and a self-excluding result digest.
Each observation retains raw status, missingness origin, reason, exact response as UTF-8 base64,
selection source (`canonical`, `review_consensus`, or `unresolved`), and the four outcomes.
Noncompleted rows have no selection source. Retained raw strings are data, never executable text.

Each condition has 96 expected slots. Show completed, failed, refused, truncated, censored
(refused plus truncated), and missing counts. Among completed slots show correct, incorrect,
unresolved, format-valid, and joint-success counts. Include the six-cell cross-tabulation of
format valid/invalid against correct/incorrect/unresolved; counts must sum to completed.
Expected equals completed plus failed plus censored plus missing. Missing provenance does not
create an extra slot. Also expose resolved counts for every condition.

Format and joint rates divide by completed count. Semantic correctness divides by completed
only when all completed answers are resolved. Zero denominators yield null. Any unresolved
completed answer makes that condition's semantic rate null, not zero. Every metric has both
standalone `lesson_only - none` and marginal `bank_plus_lesson - bank_only` contrasts. A contrast
is null whenever either component rate is null. Use the same finite numeric representation
consistently; retain integer numerators and denominators alongside rates.

For each condition, report the logical coverage bound
`[known_correct/96, (known_correct+unknown_slots)/96]`, where unknown slots are unresolved
completed outputs plus failed, censored, and missing slots. Label it a coverage diagnostic,
not a confidence interval or a replacement semantic estimate. No significance tests or
uncertainty estimates are added. Missingness differences remain visible, not corrected away.

All reports remain `training_eligibility="DEVELOPMENT"`, `verified=false`, `paired=false`,
`confers_authority=false`, and `optimizer_input=false`. Always include reasons
`capture_status_unauthenticated`, `generation_seed_unknown`, `study_unauthenticated`,
`packet_unfrozen`, and `task_reviews_pending` for this pinned candidate. Add
`no_imported_rows` for an empty capture list and `reviewer_identity_unauthenticated` when
any review file is supplied. Nonempty imported captures are not called authenticated execution.
The five proposed dependency groups are contextual review evidence; do not manufacture cluster
confidence intervals or promote evidence into training, curation, reward, or optimizer authority.

## Bounds and errors

Packet and capture files are bounded at 4 MiB before decoding. Generated operator packages,
public bundles, review files, and reports are bounded at 8 MiB. These are independent caps;
an otherwise valid input that produces a larger output fails rather than truncating content.
Per response: at most 8,192 UTF-8 bytes. Per noncompleted reason: 1,024 UTF-8 bytes.
Each review file has at most 384 judgments; each judgment has at most eight nonblank evidence
references of at most 1,024 UTF-8 bytes each. Require explicit JSON types, including strict
booleans, and reject duplicate keys, invalid UTF-8, non-finite numbers, and excessive nesting
using the existing bounded-envelope validation rules with these follow-up-specific limits.
The current shared helpers have a fixed 4 MiB cap; preserve those helpers and implement the
parameterized reader/validator in the new follow-up records module. No unexpected field is ignored.

CLI validation/read/write failures exit 2 with a bounded static error message identifying
validation versus publication. Never include raw responses, reviewer text, or exception bodies.
Pure functions raise ValueError for invalid records. Filesystem failure stays distinct from
semantic noncanonicality. Existing destinations and competing publishers are preserved.

## Verification and acceptance

Test canonical correct/wrong output; whitespace; duplicate/extra keys; malformed JSON; prose;
fences; changed case; contradictory response text requiring reviewer unresolved status;
matching noncanonical correct/wrong judgments; missing, single, disagreeing, duplicate-identity,
draft, and stale reviews; repeated identical response text in different slots; all noncompleted
statuses; explicit and implicit missingness; zero denominators; unresolved rate withholding;
all six cross-tab cells; null contrasts; coverage-bound arithmetic; complete roster retention.

Adversarial tests must demonstrate that known conditions/answers/slot IDs cannot appear as
extra public export fields, that extra mapping entries fail, and that response strings stay
data even when they resemble artifact JSON or instructions. Public content can coincidentally
contain label words; tests verify projection structure rather than claim content redaction.
Test bounded input/output, invalid types, duplicate JSON keys, no-overwrite races and failed
publication cleanup. Offline tests prohibit socket/provider and credential-loader use.

Run the existing pilot preparation, capture and scoring tests unchanged. Their frozen packet
digest, capture schema, answer interpretation and reports must remain unchanged. Run new tests,
documentation checks, Ruff, Black, focused mypy, and the full repository suite for this new
subsystem. Exercise a synthetic 384-slot run offline and verify the report from independent
integer calculations. Mark synthetic evidence explicitly; never represent it as model results.

## Handoff and boundaries

Written-design approval permits an implementation plan, not native execution. The plan will
split strict records, canonical/report arithmetic, review workflow, and CLI/docs into reviewable
steps and specify exact report fields and reason constants. Select execution method only after
that plan is reviewable. No new dependency is proposed.

Task-review acceptance, candidate freeze, provider compatibility, prices, counting-charge policy,
credentials, live transport, and exact execution spending remain later decisions. This offline
stage can be implemented and tested while those decisions remain pending. Do not silently carry
the first pilot's allowance or the preparer's pending flags into a live authorization.

## Self-review

This proposal preserves the approved study's metrics and two-reviewer rule, pins draft semantics
without granting freeze, separates private mappings from public review material, preserves
missingness and unresolved outcomes, and leaves v1 interfaces intact. Digests provide binding,
not authentication. Reviewer distinctness and blinding remain explicitly limited. The design
has no provider or financial side effects and does not assume that any live run has occurred.
