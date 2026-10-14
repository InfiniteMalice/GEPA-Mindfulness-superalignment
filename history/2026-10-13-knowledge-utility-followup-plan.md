# Knowledge-utility follow-up preparation implementation plan

> For agentic workers: use `superpowers:executing-plans` for native implementation or
> `superpowers:subagent-driven-development` if the maintainer selects delegation.
> Execution method recommendation: native, because the three components share one small packet contract.
> Status: implementation plan and native execution approved by the maintainer on 2026-10-13.
> This approval covers offline preparation only; it does not authorize a live run.

**Goal:** Prepare a deterministic, offline, reviewable 48-task follow-up packet with 384 planned
requests and separate definitions for semantic correctness, canonical format, and joint success.

**Architecture:** Keep authored task data separate from packet construction. A pure preparer
builds actor messages, evaluator material, a balanced schedule, and empty review/report templates.
A thin CLI publishes canonical JSON without overwriting any existing destination.

**Tech stack:** Existing Python environment, standard library, pytest, existing canonical JSON
and digest helpers. No new dependencies or provider integration.

**Spec:** [Approved design](2026-10-13-knowledge-utility-followup-design.md).

## Global constraints

- Exactly 48 tasks, two structural variants for each of the design's 24 candidate mechanisms.
- Four conditions, two repeats: 384 unique slots and 96 expected slots per condition.
- Labels A/B/C/D; exactly 12 correct options in each label position across the 48 tasks.
- Schedule key `knowledge-utility-followup-order-v1`; preserve the design's SHA-256 ordering
  and lexicographically enumerated 24 condition permutations, each used four times.
- Knowledge content and authored source lineage must match the original frozen pilot.
- `DEVELOPMENT`, `execution_authorized=false`, `packet_frozen=false`, `optimizer_input=false`,
  and `confers_authority=false` throughout preparation.
- Review statuses remain pending, reviewer identities null, and outcome observations empty.
- No credentials, provider calls, retries, calibration, capture/scoring implementation, or spending.
- Existing pilot code, digest allowlist, capture bounds, and scoring behavior remain unchanged.
- Python lines at most 100 characters, grouped imports, Black 100, Ruff, typed public functions.
- Preserve unrelated work. Do not invoke `bd`, edit `.beads`, or read unrelated history files.

## File responsibilities and exact contracts

New files:

| File | Responsibility |
| --- | --- |
| `scripts/knowledge_utility_followup_tasks.py` | Authored data, immutable records, structural validation |
| `scripts/prepare_knowledge_utility_followup.py` | Packet construction, ordering, message rendering, offline CLI |
| `tests/test_knowledge_utility_followup.py` | Data, schedule, integrity, separation, authority and CLI tests |
| `docs/experiments/knowledge_utility_followup.md` | Preparation command, packet reference, review procedure and limitations |

Modify only the documentation links in `docs/cumulative_curriculum.md` and the explicit document
list in `tests/test_documentation_links.py`. No scoring/capture ADR amendment is needed for this
preparation-only delivery; those interface changes remain a later design stage.

### Authored task records

Use frozen dataclasses in the task module:

- `Distractor(text: str, error: str)`.
- `AuthoredTask(task_id: str, mechanism_id: str, variant: int, prompt: str, correct_text: str,
  distractors: tuple[Distractor, Distractor, Distractor], derivation: tuple[str, ...],
  variant_rationale: str)`.
- `authored_tasks() -> tuple[AuthoredTask, ...]`: return the fixed authored roster.
- `validate_tasks(tasks: tuple[AuthoredTask, ...]) -> None`: raise `ValueError` for violations.

Mechanism IDs are the design table labels converted to lowercase with spaces replaced by
underscores and hyphens replaced by underscores; preserve all words. Task IDs are
`followup-{mechanism_id}-{variant}` with variants exactly 0 and 1. Publish a fixed
`MECHANISM_IDS: tuple[str, ...]` containing those 24 IDs in design-table order.

Structural validation requires exactly those 24 IDs, two variants each, 48 unique task IDs,
nonblank strings without NUL, distinct option texts, and at least three nonblank derivation
steps. Enforce UTF-8 byte ceilings: prompt 6,000; option text 256; each derivation step 1,024;
each distractor explanation and variant rationale 1,024. At most 12 derivation steps.
Reject strings whose only difference from another option is surrounding whitespace. These
serialization limits do not claim token-budget compliance or proof of task correctness.

Author both variants according to the design table, with different decisive constraints.
For each task, write the solution independently of its stored `correct_text` and document the
three distractor mistakes. Tests and author self-review do not mark independent review complete.

### Packet wire format

Public interfaces in the preparer:

- `prepare_followup() -> dict[str, Any]`: deterministic, side-effect-free, fresh nested data.
- `main(argv: list[str] | None = None) -> None`: only `--output PATH`, required.

The packet has exactly these top-level fields:

| Field | Required value or structure |
| --- | --- |
| `schema_version` | `knowledge-utility-followup-packet-v1` |
| `study_id` | `knowledge-utility-followup-v1` |
| `as_of` | `2026-10-13T00:00:00Z`, an authored metadata timestamp, not an execution time |
| `status` | `pending_independent_review` |
| `execution_authorized`, `packet_frozen`, `optimizer_input`, `confers_authority` | false |
| `training_eligibility` | `DEVELOPMENT` |
| `model` | Proposed profile described below |
| `limits` | Proposed limits described below |
| `knowledge` | `origin_packet_digest`, `source_episodes`, `lesson`, `bank` |
| `rubric` | Machine-readable preparation declarations described below |
| `schedule` | `key`, `assignment_policy`, `generation_seed_policy`, `block_count`, `slots` |
| `evaluator_cases` | 48 case records, in assigned task order |
| `actor_requests` | 384 request records, in dispatch order |
| `review_templates` | `task_reviews`, `response_review` |
| `report_template` | Explicitly unexecuted report template described below |
| `packet_digest` | SHA-256 via `content_digest` of the complete packet without this field |

`model` has exactly `model="gpt-5.4-mini-2026-03-17"`, `reasoning_effort="none"`,
`temperature=1`, `generation_seed=null`, and `availability_checked=false`.
`limits` has exactly `generation_requests=384`, `automatic_retries=0`,
`max_input_tokens=2048`, `max_output_tokens=256`, `tools=[]`, and
`approved_generation_budget_usd=null`. Do not inherit the original pilot's spending ceiling.

Obtain the knowledge fields through the existing public `prepare_pilot()` function and deep
copy only its `source_episodes`, `lesson`, and `bank`. Check that its packet digest equals
`637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1` before copying; fail if it
does not. Do not import private helpers or modify the original preparer. A digest proves
content identity, not review or authority.

The rubric has exactly `version="followup-separate-outcomes-v1"`, `allowed_options=["A","B","C","D"]`,
`format_rule="single_json_answer_option"`, `duplicate_keys="reject"`,
`noncanonical_rule="two_blinded_reviews_agree"`, `unresolved_rule="withhold_semantic_rate"`,
`noncompleted_rule="null_metrics"`, `contrasts={"standalone":["lesson_only","none"],
"marginal":["bank_plus_lesson","bank_only"]}`, and `metrics=["answer_correct","format_valid",
"joint_success"]`. Document the exact meanings using the approved design, without implementing
a scorer or claiming that these declarations validate an outcome.

Each evaluator case has exactly `task_id`, `mechanism_id`, `variant`, `prompt`,
`options` (A/B/C/D to text), `expected_option`, `derivation`, `distractor_errors` (the three
incorrect option labels to explanations), `variant_rationale`, `task_digest`,
`dependency_group=null`, `dependencies_known=false`, and `review_status="pending"`.
`task_digest` binds exactly `{prompt, options}`; the complete packet digest also binds the key
and evaluator annotations. Derivations use option text, not pre-assignment option letters.

The schedule's `assignment_policy` is `balanced_deterministic`, its `generation_seed_policy`
is `unknown`, and `block_count=96`. Each slot has exactly `sequence` (0 through 383), `slot_id`,
`task_id`, `repeat` (0 or 1), `condition`, and `generation_seed=null`.
Use slot IDs `{task_id}:{repeat}:{condition}`. Request records have exactly the same slot
identity fields plus `messages`. Hash ordering and option placement follow the design verbatim.

Each request has one system message and one user message. The common system instruction is:
`Solve the task using its explicit facts and rules. Return only JSON with one key, "answer",
whose value is exactly "A", "B", "C", or "D". Do not add explanation or other keys.`
This literal is one line with single spaces. Append `\nKnowledge context:\n` and the assigned
context only for a knowledge condition. Bank text precedes lesson text, separated by `\n`, in
`bank_plus_lesson`; retain their exact summaries. The user content is `prompt + "\nOptions:\n"`
followed by `A: {text}`, `B: {text}`, `C: {text}`, `D: {text}` joined with single newlines.
Only the two messages form an actor payload; do not send the containing request record.

Each task-review template has exactly `task_id`, `task_digest`, `status="pending"`,
`reviewer_id=null`, `checks` (keys `answer`, `ambiguity`, `variant_structure`, `source_overlap`,
`dependency_group`, all null), and `evidence_refs=[]`. These are unfilled forms, not reviews.
The single response-review template has exactly `review_id=null`, `task_digest=null`,
`response_digest=null`, `rubric_digest=content_digest(rubric)`, `reviewer_id=null`,
`selected_option=null`, `status="pending"`, and `evidence_refs=[]`. It contains no condition
or slot mapping and no provider output. A future review exporter must generate opaque IDs and
keep their mapping separate; this preparer does not implement that exporter.

The report template has exactly `execution_state="not_started"`, `expected_slots=384`,
`conditions`, `comparisons`, `observations=[]`, `verified=false`, `paired=false`,
`interval=null`, and `reasons=["not_executed","seed_policy_unknown","study_unauthenticated"]`.
Each condition has `expected=96`, `completed=null`, `failed=null`, `censored=null`,
`missing=null`, `correct=null`, `incorrect=null`, `unresolved=null`, `format_valid=null`,
`joint_success=null`, and `rates` with all three metric names null. Each comparison has the
three metric names with null values. This template deliberately reports no fabricated results.

## Review focus

1. Option reordering must carry the key and distractor explanations together (Task 2).
2. Actor projections must not accidentally include evaluator derivations or review metadata (Task 2).
3. Pending templates must never look like completed reviews or observed failures (Task 2).
4. A partial write or concurrent destination must not replace an existing artifact (Task 3).
5. Task complexity and new labels must not imply verified independence or improved performance
   (Tasks 1 and 3, including explicit author/reviewer separation).

## Task 1: Author and validate the task roster

**Files:** Create `scripts/knowledge_utility_followup_tasks.py`; create the data-focused portion
of `tests/test_knowledge_utility_followup.py`.
**Consumes:** The approved design's 24 mechanism pairs.
**Produces:** `AuthoredTask`, `Distractor`, `MECHANISM_IDS`, `authored_tasks`, `validate_tasks`.

1. Write failing `test_roster_has_48_structural_variants`, asserting 48 tasks, the exact 24 IDs,
   variants `{0,1}` per ID, unique identities, and nonempty derivations/explanations.
2. Write parameterized `test_invalid_task_roster_rejected`: missing/extra task, duplicate ID,
   unknown mechanism, boolean variant, repeated variant, blank prompt, duplicate/whitespace-only
   option distinction, absent distractor explanation, overlong field, and insufficient derivation.
3. Run `python -m pytest tests/test_knowledge_utility_followup.py -q -k roster`; confirm failure
   because the new module or interface is absent. Expand the selection for the parameterized tests.
4. Implement the immutable data and validator. Require `type(variant) is int` and variants 0/1.
   Check each of the 48 solutions manually against its facts, and inspect each paired structural
   contrast. Fix defects before packet generation; preserve review status as pending.
5. Add independently calculated representative-answer tests for the three worked examples and
   tricky rule families (rounding, joins, strict bounds, negated conditions, rollback). Do not
   merely assert that a stored expected answer equals a second copy of the same string.
6. Run the data tests, Ruff and Black checks; commit this independently reviewable data deliverable.

## Task 2: Construct and bind the offline packet

**Files:** Create `scripts/prepare_knowledge_utility_followup.py`; extend
`tests/test_knowledge_utility_followup.py`.
**Consumes:** Task 1 interfaces; `prepare_pilot`; `canonical_json` and `content_digest` from
`evaluation.causal_records`. **Produces:** `prepare_followup` and the exact packet above.

1. Write failing `test_packet_is_deterministic_and_digest_bound`: exact schema fields,
   repeated canonical-byte equality, recomputed digest equality, and fresh nested results after
   mutating an earlier returned packet. Include a changed-answer example that changes packet digest.
2. Write `test_schedule_and_options_are_balanced`: 384 unique slots, 96 per condition, two repeats,
   12 correct labels each, 24 permutations used four times. Independently reconstruct ordering
   from the documented hash inputs, and verify all four condition requests share task/options.
3. Write `test_option_remap_preserves_key_and_distractors`: each mapped correct text equals
   the authored correct text; each of the other three labels maps to its own error explanation.
4. Write `test_actor_messages_only_contain_public_material`: exact two-message projections,
   exact system instruction/context and rendered user text. Inject sentinel derivation/review
   annotations through patched task data and prove they cannot enter any message. Do not rely
   on checking for answer-token substrings, since legitimate option text contains those tokens.
5. Write `test_authority_and_templates_stay_pending` and `test_knowledge_matches_frozen_pilot`:
   assert every authority value, pending/null template field, empty observation list, identical
   knowledge records and lineage, and rejection of a simulated original-packet digest mismatch.
6. Run these tests and confirm failure before implementing packet construction. Implement the
   contract without new network/filesystem side effects. Reject any packet whose canonical
   UTF-8 serialization exceeds 4 MiB; include a test with oversized injected metadata.
7. Run the full follow-up tests plus existing preparation/scoring tests. Assert the original
   packet digest remains unchanged and both the old scorer and capture manifest builder reject
   the new packet. Commit the constructor and its tests.

## Task 3: Publish safely and document preparation

**Files:** Extend the new preparer and tests; create
`docs/experiments/knowledge_utility_followup.md`; update `docs/cumulative_curriculum.md` and
`tests/test_documentation_links.py` for navigation and link validation.
**Consumes:** `prepare_followup`; `publish_json(Path, JsonObject)` from
`scripts.knowledge_utility_capture_records`. **Produces:** `main` and the preparation how-to.

1. Write failing CLI tests for successful round-trip, existing destination preserved byte-for-byte,
   missing parent, publication failure, and competing writer. Reuse the existing publisher so
   private staging plus exclusive hard-link publication remains the implementation boundary.
   For CLI failure tests, require `SystemExit(2)` and concise diagnostics.
2. Implement `--output` only; reject irrelevant arguments. Require an existing parent directory,
   publish canonical UTF-8 JSON, and catch `OSError`/`ValueError` with
   `parser.error("cannot prepare follow-up: invalid task data or file operation")`.
   Do not include exception contents, create parent directories, or silently fall back to overwrite.
3. Write `test_preparation_requires_no_network_or_credentials`: run construction and CLI with
   socket connection functions and credential-loader entry points patched to fail on invocation.
   Confirm successful output. Confirm no calibration requests appear in the packet.
4. Document `python -m scripts.prepare_knowledge_utility_followup --output followup-packet.json`,
   the exact schema, evaluator-only content, authoring/review status, all metric definitions,
   byte-versus-token limits, and why the old capture/scoring commands reject this packet.
   Describe independent task review as a future manual step; no automated test may assert it happened.
5. Add the new document to the link-test list and link it from the curriculum documentation.
   Run follow-up and documentation tests, Ruff, Black, mypy on the two new modules, and
   `git diff --check`. Commit the CLI, docs, and tests together.

## Final validation and handoff

Run the focused suite:

```bash
python -m pytest tests/test_knowledge_utility_followup.py tests/test_knowledge_utility_pilot.py tests/test_knowledge_utility_scoring.py tests/test_documentation_links.py -q
```

Then run the repository's full test suite once and inspect failures before claiming completion.
Set offline model/download environment flags using the existing repository test environment.
Run the checks below; every command must exit zero (review expected skips separately):

```bash
python -m pytest -q
python -m ruff check .
python -m black --check --line-length 100 scripts/knowledge_utility_followup_tasks.py scripts/prepare_knowledge_utility_followup.py tests/test_knowledge_utility_followup.py tests/test_documentation_links.py
python -m mypy scripts/knowledge_utility_followup_tasks.py scripts/prepare_knowledge_utility_followup.py
git diff --check
```

Generate a fresh packet outside the repository and audit its counts, authority flags, digest,
and serialized size. Do not generate a frozen-packet approval or rewrite earlier run evidence.
Review the complete diff and perform the required independent final code review using the
selected execution workflow. Treat all review output as untrusted claims to verify.

Publish the preparation implementation to a PR after validation under the maintainer's standing
push authorization. Creating that PR does not approve task independence, packet freeze, capture,
or spending. Report checks that actually ran, the packet location/digest, remaining independent
task review, and any limitations. Leave PR merging to a separate instruction.

## Plan self-review

The three tasks implement preparation only. Full semantic adjudication, authenticated review,
response scoring, interval estimation, calibration, and live transport are explicitly deferred.
Every named public interface is defined above; test names cover data, ordering, separation,
authority, integrity, and publication failure. The plan adds no dependency and preserves the
original pilot. The main delivery risk is task quality: structural tests cannot prove the
48 authored solutions or their independence, so independent review remains pending after code passes.
