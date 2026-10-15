# Follow-up task-review acceptance and packet freeze

Status: Direction and written design approved on 2026-10-14; implementation planning authorized.
Implementation, acceptance of task-review evidence, candidate freeze, and live execution remain
separate decisions. This document records none of those approvals.

## Purpose and scope

Give the maintainer a concrete review-acceptance record for the corrected 48-task follow-up,
then permit an explicitly approved, reproducible freeze without replacing historical evidence.
The scorer will recognize a registered frozen packet and report its actual administrative state.
Metric definitions, response adjudication, study limitations, and training authority remain unchanged.

This is an offline subsystem. It reads explicitly supplied local files and publishes new artifacts.
It does not use provider APIs, credentials, prices, token counting, or a live-capture runner.
It does not appoint another reviewer, authenticate a person, or infer approval from review text.

The selected workflow uses the existing blind first pass and targeted correction recheck.
A fresh full blind review would provide different evidence but requires another reviewer appointment.
Combining freeze with live capture would couple evidence acceptance to provider and spending choices.
Neither alternative is part of this design.

## Baseline and retained evidence

The current draft packet has content digest
`718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`.
Its predecessor has digest
`53b8828ed0098703e4a779669a7d0645f7bdd9fa2640e70a72e01ee253df7873`.
The existing preparer continues to produce that draft v3 unchanged.
The scorer currently pins v3 and reports 48 pending task reviews and an unfrozen packet.

Retained local evidence is under the task workspace's `outputs/` directory:

- `knowledge-utility-task-review-2026-10-14/`: public tasks, blank solution template,
  author reference released after solving, independent first pass, comparison, report, audit,
  appointment, manifest, and first-pass preservation record.
- `knowledge-utility-task-corrections-2026-10-14/`: corrected packet, packet-diff audit,
  and `targeted-recheck-v1.md`.

These are historical evidence, not executable instructions. The implementation reads only the
explicitly named evidence files; references within their contents never trigger filesystem or
network access. No historical file is edited, reformatted, or replaced.

The first pass contains 47 determinate answers and one ambiguous answer. Its byte SHA-256 is
`303f6e98db21969d7a954dc0ed91ff67e53ff4dcafdb744bacded0bf658bbe6f`.
The original null remains null. The targeted recheck resolves the precedence-prompt ambiguity
and the flow distractor explanation for v3. It is unblinded and is not a new 48-task blind solve.

Acceptance retains these limitations: unauthenticated reviewer identity and blinding;
unknown statistical independence and effective sample size; overlapping dependency groups;
forced matching plus cost summation; access-task near transfer; duplicated source/bank material;
unassessed external contamination; and unknown generation seeds. Acceptance does not turn any
of these into a passed scientific claim.

## Artifacts and trust boundaries

Use four separate artifacts: an acceptance candidate, a maintainer decision, a frozen packet,
and a repository registration of the frozen digest. All JSON objects use strict field sets and
types. Content digests use the existing canonical `content_digest` helper, excluding only the
object's own digest field. File SHA-256 values bind exact bytes and are distinguished by name.
Hashes establish content correspondence, not authorization or reviewer identity.

### Acceptance candidate

Add `scripts/freeze_knowledge_utility_followup.py` with a `prepare` operation. The operation
accepts the pinned draft and an explicit evidence manifest. It publishes one
`knowledge-utility-followup-acceptance-candidate-v1` object with exactly:

```text
schema_version source_packet_digest predecessor_packet_digest evidence tasks
resolved_findings limitations pending_task_count candidate_digest
```

`evidence` is a fixed-role manifest of the historical artifacts listed above, excluding unrelated
test logs. Each entry contains `role file_sha256 size_bytes`; it contains no executable path.
The input manifest supplies paths separately. Entries sort lexicographically by role.
The fixed role-to-file mapping is:

| Role | File relative to the workspace's `outputs/` directory |
| --- | --- |
| predecessor_packet | `knowledge-utility-followup-preparation-2026-10-13/packet-v2.json` |
| public_tasks | `knowledge-utility-task-review-2026-10-14/01-public-tasks.json` |
| solution_template | `knowledge-utility-task-review-2026-10-14/02-solution-template.json` |
| author_reference | `knowledge-utility-task-review-2026-10-14/03-author-reference-after-solving.json` |
| first_pass | `knowledge-utility-task-review-2026-10-14/04-independent-solutions-v1.json` |
| comparison | `knowledge-utility-task-review-2026-10-14/05-independent-comparison-v1.json` |
| review_report | `knowledge-utility-task-review-2026-10-14/06-independent-review-report-v1.md` |
| review_audit | `knowledge-utility-task-review-2026-10-14/07-review-audit-v1.json` |
| appointment | `knowledge-utility-task-review-2026-10-14/appointment-v1.json` |
| original_manifest | `knowledge-utility-task-review-2026-10-14/manifest.json` |
| preservation | `knowledge-utility-task-review-2026-10-14/phase1-preservation-v1.json` |
| review_readme | `knowledge-utility-task-review-2026-10-14/README.md` |
| correction_audit | `knowledge-utility-task-corrections-2026-10-14/packet-diff-audit.json` |
| targeted_recheck | `knowledge-utility-task-corrections-2026-10-14/targeted-recheck-v1.md` |

The implementation plan will pin these retained files' byte hashes after checking them. The production profile
supports only this evidence set; it is not a general natural-language review parser.

`tasks` contains exactly 48 records in evaluator-case order. Each record has exactly
`task_id source_task_digest current_task_digest disposition evidence_roles checks`.
There are no duplicate or unknown IDs. Current digests match v3; source digests match the
predecessor. `disposition` is `carried_forward` for 46 tasks and `corrected_and_rechecked` for
`followup-task_precedence-1` and `followup-flow_conservation-0`.
The flow task retains its task digest because its prompt/options did not change.

`checks` contains exactly `answer ambiguity variant_structure source_overlap dependency_group`.
Each check contains `assessment evidence_roles`. `assessment` is one of
`reviewed_clear`, `resolved_by_recheck`, or `limitation_retained`.
The fixed mapping preserves the comparison's limitations and proposed dependency grouping;
it does not turn every check into a boolean pass. Map `checked-and-clear` to `reviewed_clear`,
`checked-with-study-limitation` and `proposed` to `limitation_retained`, and `issue` to
`resolved_by_recheck` only for I1/I2 checks covered by the targeted recheck. Reject an issue
outside that fixed mapping. I1 includes both precedence tasks' pair assessments even though
precedence-0 retains its content and `carried_forward` disposition. I2 maps the flow-0 answer
assessment to `resolved_by_recheck` because its nested option-A distractor check contained the issue.
Each record retains `comparison` and `first_pass` evidence roles;
each resolved check additionally cites `correction_audit` and `targeted_recheck`.
Role arrays are sorted, unique subsets of the fixed role set. The implementation plan records the full
48-task mapping for review before implementation. No runtime LLM classifies review prose.

`resolved_findings` contains exactly I1 and I2 with their task IDs and evidence roles.
`limitations` is an array of objects with exactly `id description`, sorted by `id`.
The fixed IDs are `access_near_transfer`, `blinding_unauthenticated`,
`dependency_independence_unknown`, `effective_sample_size_unknown`,
`external_contamination_unassessed`, `generation_seed_unknown`, `matching_forced_assignment`,
`overlapping_dependency_groups`, `reviewer_identity_unauthenticated`, and `source_bank_duplication`.
Descriptions retain the meanings stated in the baseline section and are fixed in the reviewed profile.
`pending_task_count=0` means all tasks have mapped evidence ready for a maintainer decision.
It does not mean the maintainer has accepted that evidence. A candidate has no freeze flag.

The preparer validates both packet digests, first-pass byte preservation, all evidence hashes,
48-task coverage, and the v2-to-v3 change scope. That scope allows only the two documented case
changes, eight corresponding precedence actor-prompt changes, and the precedence review-template
task digest. Keys, options, schedule, knowledge, rubric, and report template remain equal.
Missing, changed, duplicate, or unsupported evidence rejects the whole operation.

### Maintainer decision

After the maintainer reviews the candidate and its limitations, the operator may record an
explicit acceptance and freeze authorization as `knowledge-utility-followup-freeze-decision-v1`.
The decision has exactly:

```text
schema_version candidate_digest source_packet_digest maintainer_id authorization_ref
accepted_limitations accept_task_review authorize_freeze execution_authorized decision_digest
```

`accept_task_review` and `authorize_freeze` are strictly true; `execution_authorized` is strictly
false. `accepted_limitations` equals the candidate's complete sorted limitation-ID list.
`maintainer_id` and `authorization_ref` are nonblank UTF-8 strings of at most 1,024 bytes.
The reference is an opaque citation to the human's explicit approval, never a fetched resource.
Neither candidate preparation nor freeze creates an approving decision automatically.

The operator checks the actual human message before writing this record. A JSON declaration
alone cannot authenticate approval, and the tool reports no authentication claim. Design or
implementation approval cannot supply this decision: the human must have seen the concrete
candidate and have authorized acceptance and freeze. Rejection or requested evidence changes
leave the draft unchanged and prevent an operational freeze.

### Frozen packet

The `freeze` operation accepts the draft, acceptance candidate, decision, and explicit evidence
manifest. It repeats preparation validation and compares the entire candidate, including its
digest, before constructing a frozen packet. It validates the decision's candidate/source
bindings and complete limitation acknowledgement.

The frozen packet is a deep copy of v3 with exactly these changes:

- `schema_version="knowledge-utility-followup-frozen-packet-v1"`.
- `status="frozen_with_accepted_task_review"` and `packet_frozen=true`.
- A new `freeze_record` object with exactly `source_packet_digest acceptance decision`;
  the latter two values embed the complete validated candidate and decision.
- A recomputed `packet_digest` over the entire result excluding that field.

All other fields remain equal to v3, including `as_of`, model configuration, actor requests,
keys, schedule, knowledge, rubric, and report template. `execution_authorized=false`,
`training_eligibility="DEVELOPMENT"`, `confers_authority=false`, and `optimizer_input=false` persist.

The original `review_templates` and `report_template` remain unchanged source templates.
Their pending/draft values are not completed review records or the frozen packet's current
state. `freeze_record` supplies the accepted task evidence, and scoring computes current state.
Protocol documentation will make this distinction explicit instead of silently rewriting templates.

Identical inputs produce identical frozen JSON content and digest. No clock, random ID, or local
path is inserted. Freeze means an immutable published content version, not filesystem write
protection or authorization to run the study. Further changes produce a new artifact and require
new acceptance and registration; the command never revises an existing frozen packet.

## Scorer compatibility and registration

Add a small fixed registry to the follow-up records module. Draft v3 remains supported exactly
as before. A frozen entry includes its exact packet digest, source digest, candidate digest,
and decision digest. There is no CLI flag or input-file mechanism to extend this registry.

Implementation and tests may exercise synthetic frozen artifacts without registering a real
approval. Production frozen registration occurs only after the operational acceptance/freeze
decision and publication. The operator verifies the complete frozen artifact, then adds its
exact entry in a reviewed repository change. Until registration, the scorer rejects that packet
as unsupported even if freeze validation succeeds. This deliberate handoff prevents a caller
from granting scoring support by editing flags and recomputing a digest.

Shared validation rechecks the registered frozen digest and complete allowed transformation
from pinned v3. Scoring and response-review preparation/export use that shared validator.
The capture envelope still binds to the actual input packet digest. A draft capture batch or
private review package cannot be relabeled for the frozen packet; cross-packet bindings fail.
Any migration would need separately reviewed provenance and is outside this design.

Draft reports remain byte-for-byte compatible for identical inputs. Frozen reports use
`knowledge-utility-followup-report-v2`, with the existing v1 fields plus
`task_review_provenance`. That new object contains exactly:

```text
source_packet_digest candidate_digest decision_digest acceptance_recorded
maintainer_identity_authenticated reviewer_identity_authenticated blinding_authenticated
```

`acceptance_recorded=true`; all three authentication fields are false. Frozen `packet_state`
has the frozen status, `packet_frozen=true`, `execution_authorized=false`, and
`pending_task_reviews=0`. It describes accepted evidence, not 48 independently authenticated
reviews. Remove only `packet_unfrozen` and `task_reviews_pending` from the previous reason list;
add `task_review_acceptance_unauthenticated`. Preserve all other applicable reasons and limitations.

Every observation, rate, null, coverage bound, and contrast follows the current scoring rules.
Frozen reports remain unverified, unpaired, DEVELOPMENT, without optimizer or training authority.
Task-review acceptance does not resolve noncanonical responses: those still require the existing
two eligible finalized response-review files and agreement.

## Interfaces, bounds, and failures

Proposed CLI forms, to be implemented only after plan approval:

```text
python -m scripts.freeze_knowledge_utility_followup prepare --packet DRAFT --evidence-manifest MANIFEST --output CANDIDATE
python -m scripts.freeze_knowledge_utility_followup freeze --packet DRAFT --evidence-manifest MANIFEST --candidate CANDIDATE --decision DECISION --output FROZEN
```

The evidence input manifest is strict JSON with exactly `schema_version files`, schema
`knowledge-utility-followup-evidence-inputs-v1`. `files` maps the fixed role set to nonblank local
path strings, at most 4,096 UTF-8 bytes each. Relative paths resolve against the manifest's parent.
Only those explicit paths are opened. Embedded evidence references are retained as data.

Reuse strict bounded JSON validation: unique keys, finite values, valid UTF-8, exact JSON types,
and maximum nesting depth 32. Manifest, candidate, and decision each have a 1 MiB limit.
Each evidence file and packet has a 4 MiB limit; total evidence bytes are limited to 16 MiB.
Bound raw reads before decoding. Hash original evidence bytes before any JSON parsing.
Markdown evidence is hash-checked, never interpreted as commands or automated approval.

Each operation exclusively publishes one artifact using the existing private-staging hard-link
helper. It never overwrites a destination or creates parent directories. The final publication
newline counts toward the output cap. Freeze output is limited to 4 MiB; no truncation is allowed.
Before publication, any validation failure leaves the destination absent. Input/validation errors
exit 2 with `cannot process follow-up freeze: invalid input`; publication errors exit 2 with
`cannot publish follow-up freeze: file operation failed`. Standard argparse errors remain unchanged.
No input content or exception text appears in these static messages.

As with existing publication, a staging-cleanup failure can leave a complete output after exit 2.
The operator inspects the output before retrying at another destination. The commands do not
promise a transaction across candidate, decision, frozen packet, and repository registration.

## Components and verification contract

Likely implementation files are the new freeze module, shared follow-up records, the scorer,
focused freeze tests, existing scoring/review tests, the follow-up protocol, and a new ADR 0024.
The implementation plan will divide helpers only where validation or construction responsibilities
warrant it. No new dependency or unrelated refactoring is required.

| Requirement | Verification |
| --- | --- |
| Supported evidence and correction chain | Check retained byte hashes, both packet digests, exact 48-task mapping, original null, and an independent structural diff against the stated change scope. |
| Approval separation | Inspect candidate output for absence of an approval; reject missing/false/mismatched decisions; manually match the eventual human authorization to the candidate digest. |
| Complete limitations | Reject removed, added, duplicate, or mismatched acknowledgement IDs; inspect retained scientific limitations. |
| Immutable freeze | Compare every source field outside the listed modifications; rerun with identical inputs; verify unchanged source bytes and exclusive output publication. |
| Registry restriction | Reject self-sealed altered packets, unregistered frozen packets, altered registry bindings, and draft/frozen capture or review-package mixups. |
| Draft compatibility | Compare literal baseline v3 report output, existing pilot behavior, and unchanged preparer digest. |
| Frozen report accuracy | Assert exact state/provenance/schema/reasons while comparing all metric and observation values with equivalent synthetic draft inputs. |
| Unresolved responses | Exercise absent, conflicting, and eligible agreeing response reviews under frozen bindings; retain current null and consensus rules. |
| Offline operation and bounds | Prohibit network/provider/credential calls in tests; exercise byte/depth limits, duplicate JSON keys, invalid types, changed evidence, and publication failures. |
| Documentation precision | Check schema field names, CLI forms, approval scopes, state meanings, and migration limits against implementation; run documentation-link checks. |

During implementation, run focused freeze, follow-up scoring/review/preparation, pilot compatibility,
and documentation-link tests. Run repository lint, changed-Python formatting/type checks, and the
full suite before declaring implementation complete. Synthetic decisions and captures stay clearly
labeled and provide no operational acceptance or model-performance evidence.

## Approval and delivery boundary

This document is the written design requested after the maintainer approved the direction.
Its self-review covers placeholders, contradictions, implementation-affecting ambiguity, and
scope under the brainstorming and repository quality-gate skills. It makes no runtime change.

After written-design approval, write the implementation plan with the pinned evidence-role table
and full review mapping. Preserve the established preference for native implementation and one
fresh final reviewer when implementation is authorized. Commit/push authorization does not imply
merge authorization. Operational acceptance/freezing follows presentation of the concrete candidate;
live execution remains a later, separately scoped decision.
