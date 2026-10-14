# Knowledge-utility follow-up offline scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the established native method, or superpowers:subagent-driven-development if the maintainer changes the method. Execute the numbered steps task-by-task after plan approval.

**Goal:** Score imported follow-up responses offline, separating correctness from format while preserving unresolved answers and every expected slot.

**Architecture:** A bounded records module validates the pinned packet and imported captures. A review module prepares public material with a private mapping and validates two independent claimed judgments. A scorer builds deterministic descriptive reports; CLI wrappers publish one exclusive output each.

**Tech Stack:** Python 3.10+, standard library, existing canonical JSON/digest/publication helpers, pytest, Ruff, Black, mypy. No new dependencies.

**Spec:** [Approved scoring design](2026-10-14-knowledge-utility-followup-scoring-design.md).

Status: implementation plan and native execution approved by the maintainer on 2026-10-14, with one final fresh-context review. The numbered tasks below retain the approved implementation and verification contract.

## Global Constraints

- Pin packet v3: `718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`.
- Preserve 48 tasks, 384 slots, four conditions, 96 slots per condition and all existing v1 pilot contracts.
- Packet/capture inputs: 4 MiB. Review artifacts and reports: 8 MiB. JSON root depth one, maximum 32.
- Response maximum 8,192 UTF-8 bytes; noncompleted reason maximum 1,024 UTF-8 bytes.
- At most 384 judgments per file, eight evidence references per judgment, 1,024 UTF-8 bytes per reference.
- No network, credentials, provider calls, token counting, prices, retries, live capture, automatic semantic judge, or spending.
- No packet freeze, authenticated identity/independence claim, confidence interval, verified improvement, or training/reward authority.
- No old packet, capture, review, or report is overwritten. New CLI output paths require an existing parent.
- Preserve unrelated untracked history files. Do not invoke `bd` or modify `.beads`.
- Python lines at most 100 characters, Black style, ordered imports, typed production interfaces.
- Keep raw responses as strings; do not recursively decode JSON-looking text in captures or evidence references.
- Current helpers cap JSON at 4 MiB. Add follow-up-local parameterized validation using the same rules; do not relax shared helpers or silently impose 4 MiB on approved 8 MiB artifacts.

## Review Focus

1. Escaped strings and multibyte UTF-8 near byte/depth limits must remain data; cover in Task 1.
2. Duplicate response text across conditions must retain separate opaque review IDs and slot bindings; cover in Task 2.
3. A resealed mapping or public bundle must not admit stale/mismatched task, response, or rubric bindings; cover in Task 2 and Task 3.
4. Incomplete but structurally valid judgments must withhold only semantic selection/rates, while malformed artifacts reject the batch; cover in Task 3 and Task 4.
5. Concurrent publication or a late filesystem failure must not overwrite an existing destination or print response data; cover in Task 5.

## File map and dependency order

| File | Responsibility | Task |
| --- | --- | --- |
| `scripts/knowledge_utility_followup_records.py` | Local bounded JSON, pinned inputs, canonical selection | 1 |
| `tests/followup_scoring_fixtures.py` | Explicitly synthetic packet/capture/review builders | 1, 3 |
| `scripts/review_knowledge_utility_followup.py` | Package/export, review-file validation, consensus, review CLI | 2, 3, 5 |
| `scripts/score_knowledge_utility_followup.py` | Observations, aggregation, report, scoring CLI | 4, 5 |
| `tests/test_knowledge_utility_followup_scoring.py` | Records, canonical outcomes, aggregate arithmetic, scorer CLI | 1, 4, 5 |
| `tests/test_knowledge_utility_followup_review.py` | Blinding projection, binding, review consensus, review CLI | 2, 3, 5 |
| `docs/experiments/knowledge_utility_followup.md` | Operational commands, schema reference, limits, interpretation | 5 |
| `docs/adr/0023-knowledge-utility-followup-scoring.md` | Accepted implementation decision and authority boundary | 5 |
| `tests/test_documentation_links.py` | Add the new ADR to the checked list | 5 |

Tasks execute 1 -> 2 -> 3 -> 4 -> 5. Do not refactor pilot code. Reuse
`evaluation.causal_records.canonical_json/content_digest` and
`scripts.knowledge_utility_capture_records.publish_json`; its private staging publishes a
closed file by exclusive hard link. Do not import pilot scoring as a substitute for new semantics.

## Fixed interfaces and wire decisions

Use `JsonObject = dict[str, Any]`. Public object-taking entry points validate before domain use.
Returned dictionaries are independent snapshots; they must not mutate caller inputs. Frozen
dataclasses below prevent field reassignment, not nested mutation; helpers treat them as read-only.

In records, `FollowupInputs` contains:
`packet: JsonObject`, `packet_digest: str`, `captures: JsonObject`, `capture_digest: str`,
`slots: tuple[JsonObject, ...]`, `cases: dict[str, JsonObject]` keyed by task ID, and
`rows: dict[str, JsonObject]` keyed by supplied slot ID. Missing rows are not fabricated here.

In review, `ReviewSet` contains:
`decisions: dict[str, JsonObject]` keyed by eligible noncanonical completed slot ID,
`package_digest: str | None`, `bundle_digest: str | None`, and
`reviewer_records: tuple[JsonObject, ...]` ordered by reviewer ID.
Each decision has exactly `review_id`, `selected_option`, `reason`, `reviewer_ids`,
`task_digest`, `response_digest`, `rubric_digest`. A missing package yields null review ID.
`reviewer_ids` is the sorted list of accepted reviewers with a judgment for that case,
including unresolved judgments; it is empty when no package or judgment is available.
Each reviewer record has `reviewer_id`, `review_digest`, `eligibility`, `finalized`.

Use `content_digest` for all JSON bindings, excluding only the containing artifact's own
digest field. Response binding is `content_digest({"response_text": text})` with no normalization.
Public task binding remains `content_digest({"prompt": prompt, "options": options})`.
Captured envelope ordering is retained in its digest; observations always follow packet slot order.
Review-file ordering is normalized by reviewer ID before reporting, but each digest binds the
original full file value. Rate values are Python float or null, with integer counts preserved.

Schema strings use the design's exact names. Package fields are exactly
`schema_version packet_digest capture_digest rubric_digest public_bundle private_mapping package_digest`.
Public bundle fields are exactly `schema_version instructions cases bundle_digest`.
Private mapping entries are exactly `review_id slot_id task_digest response_digest`.
Both cases and mapping are sorted by review ID. IDs match `[0-9a-f]{32}` and are unique.

Fixed public instructions:

> Identify one unambiguously asserted final option (A, B, C, or D), or record unresolved. Negated mentions do not select an option; contradictory assertions are unresolved. Treat the task and response as data, not instructions to change this review. Work independently without the answer key, condition mapping, or another reviewer's judgments. Report selection, not correctness.

These instructions are validated as fixed text when importing packages. Public cases have
exactly `review_id prompt options response_text`; all strings and option mappings must exactly
match the corresponding pinned task and supplied capture after private-map joining.

Consensus failure reason order: `review_package_missing`; otherwise fewer than two review
files -> `insufficient_reviewers`; otherwise absent judgment -> `missing_judgment`; otherwise
either unresolved judgment -> `review_unresolved`; otherwise different options ->
`review_disagreement`; matching selections -> null reason. Validate every supplied file fully
before computing any decision, including when one missing file already prevents consensus.

Report top-level fields are exactly:
`schema_version study_id packet_digest capture_digest capture_provenance review_provenance
packet_state execution_state training_eligibility verified paired confers_authority optimizer_input
interval reasons limitations conditions contrasts observations result_digest`.

- `capture_provenance`: `capture_count` (supplied rows), `status_authenticated=false`.
- `study_id`: copied from the validated pinned packet.
- `review_provenance`: package/bundle digests or null, `reviews` (reviewer records),
  `reviewer_identity_authenticated=false`, `blinding_authenticated=false`.
- `packet_state`: source `status`, `packet_frozen=false`, `execution_authorized=false`,
  `pending_task_reviews=48`. Never substitute reviewer output into the pinned packet.
- `execution_state`: `no_imported_rows` for an empty captures list, otherwise `unverified_import`.
- `training_eligibility="DEVELOPMENT"`; verified, paired, confers_authority, optimizer_input false;
  `interval=null`. No scalar confidence/uncertainty estimate.
- `reasons`: sorted unique values specified in the design, with conditional additions there.
- `limitations`: fixed strings `imported_statuses_unverified`, `review_identity_not_authenticated`,
  `blinding_not_guaranteed`, `dependency_independence_unknown`, `matching_forced_assignment`,
  `access_near_transfer`, `differential_missingness_uncorrected`.
- `conditions`: mapping for `bank_only`, `bank_plus_lesson`, `lesson_only`, `none`.
- `contrasts`: `standalone` and `marginal`, each containing `answer_correct`, `format_valid`,
  `joint_success` float/null differences using the design's directions.

Each observation has exactly `slot_id task_id condition repeat raw_status missingness_origin
reason_utf8_base64 response_utf8_base64 selection_source selected_option answer_correct
format_valid joint_success resolution_reason review_binding`. Missingness origin is
`explicit`/`implicit` only for missing rows, else null. Implicit missing raw reason is
`capture_not_supplied`. Base64 preserves exact UTF-8; null remains null.
Selection source is canonical/review_consensus/unresolved for completed rows, else null.
Resolution reason is a consensus failure constant for unresolved rows, else null.
Review binding is the corresponding ReviewSet decision for a noncanonical completed row,
otherwise null. The four outcomes are null for every noncompleted row.

Each condition contains `expected completed failed refused truncated censored missing resolved
correct incorrect unresolved format_valid joint_success cross_tab rates semantic_coverage`.
Cross-tab has `format_valid`/`format_invalid`, each with `correct incorrect unresolved` counts.
The valid/unresolved cell is necessarily zero; keep it visible rather than omitting the cell.
Rates map the three metrics to `{numerator, denominator, value, withheld_reason}`. Numerator
is the known success count, denominator is completed, even when the value is withheld.
Withholding priority is `no_completed_outputs`, then semantic-only `unresolved_answers`, else null.
Semantic coverage is `{known_correct, unknown_slots, expected:96, lower, upper,
kind:"logical_coverage_bound"}`. Integer fields support independent reproduction of float values.

### Task 1: Bounded records and canonical selection

**Files:** Create records module, scoring test module, and synthetic fixture helper named above.

**Interfaces:**
- Consumes: `canonical_json(value: object) -> str`, `content_digest(value: object) -> str`,
  and existing `prepare_followup() -> JsonObject` in test fixtures only.
- Produces: `validate_json_value(value: object, *, max_bytes: int) -> None`,
  `decode_json_text(text: str, *, max_bytes: int) -> Any`,
  `read_json_object(path: Path, *, max_bytes: int) -> JsonObject`,
  `validate_followup_inputs(packet: object, captures: object) -> FollowupInputs`,
  `canonical_option(text: str) -> str | None`.
- Constants: `MAX_INPUT_BYTES=4*1024*1024`, `MAX_ARTIFACT_BYTES=8*1024*1024`,
  `MAX_RESPONSE_BYTES=8192`, `SUPPORTED_PACKET_DIGEST` as above and `CAPTURE_SCHEMA`
  equal to `knowledge-utility-followup-captures-v1`.
- Test helpers: `fresh_packet() -> JsonObject`, `capture_batch(packet, rows=()) -> JsonObject`,
  `row_for(packet, condition, index, *, status="completed", text=None, reason=None) -> JsonObject`,
  `case_for_slot(packet, slot_id) -> JsonObject`. `index` selects within a condition in schedule
  order; completed text is supplied explicitly. Every helper returns fresh mutable test data.

1. Write `test_followup_input_binding_and_missing_roster` and canonical parameterized tests:
   ```python
   validated = validate_followup_inputs(packet, capture_batch(packet))
   assert len(validated.slots) == 384
   assert len(validated.cases) == 48 and validated.rows == {}
   assert validated.packet_digest == SUPPORTED_PACKET_DIGEST
   assert canonical_option(' {"answer":"A"} \n') == "A"
   assert canonical_option('{"answer":"A","answer":"B"}') is None
   assert canonical_option('{"answer":"a"}') is None
   assert canonical_option('I select A.') is None
   ```
   Also cover canonical wrong option as a valid option, extra keys, fences, array, JSON constants,
   invalid JSON and empty text. None is semantic noncanonicality, not capture rejection.
2. Run `python -m pytest tests/test_knowledge_utility_followup_scoring.py -q`.
   Expected: collection/test failure from the absent records module/function; inspect the cause.
3. Implement the interfaces with exact JSON types, iterative bounded traversal and lexical depth
   screening before `json.loads`. Reject duplicate keys, nonfinite numbers, invalid Unicode,
   cycles and non-JSON Python objects; validate encoded canonical size without recursive string
   interpretation. Check raw read/text byte cap independently, including whitespace. Normalize
   decoding/scalar overflow/recursion failures to ValueError. No shared utility is changed.
   Value validation measures canonical UTF-8 bytes; file/text reads additionally measure raw
   bytes. Generated artifacts reserve one extra byte for the publisher's trailing newline.
4. Write rejection tests for changed-but-resealed packet keys/rubric, pilot schema, forged digest,
   duplicate/unknown slot, extra/missing fields, 385 rows, all invalid status/text/reason combinations,
   bool-as-int mistakes, >8192-byte multibyte response, >1024-byte reason, and excessive nesting.
   Verify failed/missing require null text, while refused/truncated may retain text.
   Add byte/depth/escaped-string tests: JSON-looking response content is preserved as one string;
   bracket characters inside strings do not consume structural depth; root depth32 passes and
   depth33 fails. A 5 MiB valid object fails at 4 MiB and passes at 8 MiB; >8 MiB fails.
5. Run the file again. Expected: all pass. Check input dictionaries remain equal to snapshots;
   modifying one returned snapshot must not alter original inputs or another call's result.
6. Run Ruff/Black/mypy on the new module and commit only its files with
   `feat: validate offline follow-up captures`.

### Task 2: Opaque review package and public export

**Files:** Create `scripts/review_knowledge_utility_followup.py` and its review tests.

**Interfaces:**
- Consumes: Task 1 `FollowupInputs`, `validate_followup_inputs`, `canonical_option`,
  `validate_json_value`, size constants and canonical digest helpers.
- Produces: `prepare_response_review(packet: object, captures: object) -> JsonObject`,
  `export_public_review(packet: object, captures: object, package: object) -> JsonObject`.
- Internal reusable validator: `_validate_review_package(inputs: FollowupInputs,
  package: object) -> JsonObject`. Public entry points always obtain inputs from Task 1.

1. Write `test_public_projection_and_repeated_responses`. Supply canonical, two identical
   noncanonical responses in different conditions, refused and truncated rows. Assertions:
   ```python
   package = prepare_response_review(packet, captures)
   bundle = export_public_review(packet, captures, package)
   assert set(bundle) == {"schema_version", "instructions", "cases", "bundle_digest"}
   assert len(bundle["cases"]) == 2
   ids = [case["review_id"] for case in bundle["cases"]]
   assert ids == sorted(ids) and len(set(ids)) == 2
   assert all(set(case) == {"review_id", "prompt", "options", "response_text"}
              for case in bundle["cases"])
   assert len({m["slot_id"] for m in package["private_mapping"]}) == 2
   ```
   Assert both raw response strings are exactly retained, with correct public prompt/options.
2. Run `python -m pytest tests/test_knowledge_utility_followup_review.py -q`.
   Expected: missing review module/function failure, not an unrelated dependency failure.
3. Implement prepare/export and package validator. Use `secrets.token_hex(16)` once per eligible
   slot; a collision raises ValueError rather than looping forever. Package has every eligible
   slot exactly once, no ineligible slots, exact public projection and exact mapping bindings.
   Recompute all nested/top-level digests and validate exact field sets and sorted ID order.
   Check final canonical serialization plus trailing newline <=8 MiB before returning output.
4. Add tests for missing/extra mapping entries, mismatched public text/options, duplicate ID,
   duplicate slot, non-hex ID, wrong order, stale response/task/rubric, injected public fields,
   changed instructions, and wrong self-digests. Include tampering with recomputed self-digests
   so equality-to-expected content is tested beyond hash checks. Reject assigning a noncanonical
   case to a canonical/refused/missing slot. Test a deliberate random-ID collision by patching
   the randomness source; assert ValueError. Test empty eligible set produces empty arrays.
5. Test content shaped like `{ "condition": "bank_only", "expected_option": "A" }`
   remains exact response text, not new public fields. Test generated output cap and input
   immutability. Run both new test files; expected all pass.
6. Run new-module Ruff/Black/mypy and commit `feat: prepare blinded follow-up response reviews`.

### Task 3: Review-file eligibility, binding and consensus

**Files:** Extend review module and review tests; extend synthetic fixture helper.

**Interfaces:**
- Consumes: Task 1 inputs and Task 2 `_validate_review_package`.
- Produces: `resolve_response_reviews(inputs: FollowupInputs, package: object | None,
  reviews: tuple[object, ...]) -> ReviewSet`. This internal integration entry expects inputs
  produced by Task 1; scoring revalidates caller objects before calling it.
- Test helper: `review_file(package, reviewer_id, selections: dict[str, str | None]) -> JsonObject`;
  map review IDs to letters/null; use all four true eligibility declarations and finalized true,
  with synthetic opaque evidence references. It never calls a model.

1. Write `test_consensus_requires_two_distinct_finalized_reviews` with one noncanonical response:
   ```python
   inputs = validate_followup_inputs(packet, captures)
   one = review_file(package, "reviewer-1", {review_id: "A"})
   two = review_file(package, "reviewer-2", {review_id: "A"})
   single = resolve_response_reviews(inputs, package, (one,))
   assert single.decisions[slot_id]["selected_option"] is None
   assert single.decisions[slot_id]["reason"] == "insufficient_reviewers"
   agreed = resolve_response_reviews(inputs, package, (one, two))
   assert agreed.decisions[slot_id]["selected_option"] == "A"
   assert agreed.decisions[slot_id]["reason"] is None
   ```
2. Run the review test file. Expected: failure for the unimplemented resolution interface.
3. Implement strict review schemas, ID regex, eligibility booleans exactly true, finalized exactly
   true, bounded references, selected/null status invariant and duplicate/unknown-ID checks.
   Evidence references must be nonblank strings; reject >8 entries per judgment or >384
   judgments per file, including when an artifact remains below its overall byte cap.
   Validate zero/one/two files and all supplied records before deriving decisions. A package
   without review files is valid; review files without a package or >2 files fail. Accept case-
   sensitive distinct reviewer IDs as unauthenticated claims only. Preserve exact input digests.
4. Parameterize: no package, no reviews, one review, absent judgment, unresolved judgment,
   disagreement and agreement. Assert the fixed reason priority. Add draft, false/integer/missing
   eligibility, same reviewer ID, malformed ID, unknown review ID, duplicate judgment, wrong bundle,
   invalid selected option/status, stale package and oversize reference cases; expected ValueError.
   Include blank/nonstring references, nine references and 385 judgments as rejection cases.
   Verify a malformed single file still rejects rather than being ignored as insufficient consensus.
5. Verify rejected artifacts do not mutate packet/package/reviews. Reversing reviewer-file order
   yields the same ReviewSet; changing a raw response invalidates its old review package.
   Run both new test files and changed-file Ruff/Black/mypy; expected all pass.
   Commit `feat: validate follow-up review consensus`.

### Task 4: Complete observations and separate metric report

**Files:** Create scoring module; extend scoring tests.

**Interfaces:**
- Consumes: Task 1 validated inputs/canonical selection and Task 3 ReviewSet.
- Produces: `score_followup(packet: object, captures: object, review_package: object | None = None,
  reviews: tuple[object, ...] = ()) -> JsonObject` with the report contract above.

1. Write `test_empty_capture_retains_every_slot_and_null_rates`:
   ```python
   result = score_followup(packet, capture_batch(packet))
   assert len(result["observations"]) == 384
   for condition in result["conditions"].values():
       assert condition["expected"] == condition["missing"] == 96
       assert condition["completed"] == 0
       assert condition["rates"]["answer_correct"]["value"] is None
       assert condition["semantic_coverage"]["lower"] == 0.0
       assert condition["semantic_coverage"]["upper"] == 1.0
   assert result["execution_state"] == "no_imported_rows"
   assert not result["verified"] and not result["confers_authority"]
   ```
2. Run the scoring test file. Expected: missing scorer/function failure. Implement strict entry
   validation, canonical selection, consensus joining and a deterministic 384-row projection.
   Build condition counters and all contrasts from observations, never caller-provided totals.
   Seal and bound the full report, including base64 expansion and trailing newline, at 8 MiB.
3. Write `test_mixed_outcomes_keep_semantic_rate_withheld`. For `none`, create nine rows:
   canonical correct; canonical wrong; noncanonical correct consensus; noncanonical wrong
   consensus; noncanonical unresolved; failed; refused; truncated; explicit missing. Other
   slots are omitted. Use the task key only to construct synthetic expected judgments, never
   in the review projection. Assert completed5, correct2, incorrect2, unresolved1, resolved4,
   failed1, censored2, missing88, format_valid2, joint_success1. Rates are format2/5, joint1/5,
   semantic null with numerator2/denominator5 and unresolved_answers. Coverage is [2/96,94/96].
   Cross-tab valid=(1,1,0), invalid=(1,1,1). All noncompleted outcomes are null even if text
   contains the correct option. Explicit/implicit missingness and exact raw base64 roundtrip.
4. Run the mixed test; inspect any failure and implement only missing report behavior.
   Add follow-up resolving the final answer correctly: semantic
   becomes3/5 while format2/5 and joint1/5 stay unchanged. Contradictory prose marked unresolved
   by reviewers must stay unresolved; no automatic inference is permitted.
5. Add all-canonical 384-slot fixture: correct counts none72, lesson_only84, bank_only80,
   bank_plus_lesson88, completed96 each. Assert semantic/joint standalone=12/96 and marginal=8/96
   with pytest.approx; format contrasts zero. Cross-check all counters against independent
   integer fixture totals. For any null component rate, the corresponding contrast is null.
6. Test exact report fields, digests, authority flags/reasons, reviewer provenance, packet state,
   input immutability, stable repeat calls and stable reversed reviewer-file input order. Never
   substitute the five proposed groups as independent clusters. Run both new files plus existing
   preparation/scoring tests; expected all pass. Commit `feat: report separate follow-up outcomes`.

### Task 5: Offline CLIs, documentation and final validation

**Files:** Add `main(argv: list[str] | None = None) -> None` to review and scoring modules;
extend their tests; update protocol and documentation-link test; create ADR0023 named above.

**Interfaces:** Consume Tasks 1–4 and existing `publish_json(path: Path, data: JsonObject) -> None`.
All commands and flags are exactly those in the design. Scorer `--review` is repeatable at most
twice; `--review-package` without reviews is allowed. No `--live` or credential/network flag.

1. Write CLI roundtrip tests for prepare -> export -> two synthetic finalized reviews -> score,
   and no-review score. Run before adding main functions; expected missing-entry-point failure.
2. Implement parsing, bounded reads (4 MiB packet/captures, 8 MiB other artifacts), pure function
   calls and exclusive output. Canonical serialized output plus newline must fit its output cap.
   Read/validation errors exit2 with `cannot process follow-up: invalid input`; publication errors
   exit2 with `cannot publish follow-up: file operation failed`. Standard argparse usage errors
   remain standard. Never interpolate exception bodies/raw content into error messages.
3. Add tests for existing destination, missing parent, unsupported hard links, competing writer,
   staged write failure, and cleanup failure after successful publication. Existing/competing
   file bytes remain intact; after cleanup failure a complete new output may remain with exit2.
   Test sentinel response/reason strings never appear in errors. Capture raw strings in artifact
   outputs only as the protocol specifies. Verify oversized files are rejected before unbounded reads.
4. Add offline tests that patch socket connections, provider calls and credential loaders to fail;
   all successful CLI workflows still pass. No evidence-reference string is opened or fetched.
   Reject --live, more than two --review files, missing required paths and draft reviews.
5. Document exact schemas, static errors, publication semantics, candidate pin, pending freeze,
   eligibility assertions, separate reviewer delivery, review finalization before condition joining,
   null outcomes/rates, missingness, coverage bounds, and all fixed reason/limitation strings.
   Mark the original proposed scoring section as implemented by these offline commands, without
   implying task freeze or live capture exists. ADR records this boundary and v1 compatibility.
   Add ADR to DOCUMENTS in test_documentation_links.py. No unrelated documentation edits.
6. Run targeted commands below, inspect every result, and fix failures before committing
   `feat: expose offline follow-up scoring workflow`. Run the full suite after all task changes.
   Do not repeatedly broaden tests once all required checks pass without a new failure/change.

## Verification commands and evidence

Use the existing environment; no dependency installation. On this workspace set:

```powershell
$env:PYTHONPATH="$PWD;$PWD/src;$PWD/modules;$PWD/reasoning-generalization-tracer/src"
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

Use the configured repository virtual environment's Python executable for `python` below.
Record logs outside tracked code; do not commit model
outputs, old review files or generated artifacts.

```text
python -m pytest -q tests/test_knowledge_utility_followup_scoring.py tests/test_knowledge_utility_followup_review.py tests/test_knowledge_utility_followup.py tests/test_knowledge_utility_pilot.py tests/test_knowledge_utility_scoring.py tests/test_knowledge_utility_capture.py tests/test_knowledge_utility_capture_transport.py tests/test_documentation_links.py
python -m ruff check .
python -m black --check --line-length 100 scripts/knowledge_utility_followup_records.py scripts/review_knowledge_utility_followup.py scripts/score_knowledge_utility_followup.py tests/followup_scoring_fixtures.py tests/test_knowledge_utility_followup_scoring.py tests/test_knowledge_utility_followup_review.py tests/test_documentation_links.py
python -m mypy --follow-imports=silent scripts/knowledge_utility_followup_records.py scripts/review_knowledge_utility_followup.py scripts/score_knowledge_utility_followup.py
git diff --check
python -m pytest -q
```

Expected: all commands exit0; retain full-suite pass/skip/warning counts and inspect deviations.
Scoped mypy is consistent with existing CI and preserves pre-existing transitive type debt;
do not call it repository-wide type success. No new tests should merely mirror implementation.

Generate one clearly labeled synthetic 384-slot capture, private package, public bundle,
two synthetic reviewer files, and report outside the repository. Include noncanonical outcomes
and compare integer counts and rate withholding against a separate calculation. This smoke run
does not authenticate execution, reviewers, or model performance. Preserve all predecessors.

## Plan self-review and handoff

Coverage: records/bounds/canonical rules -> Task1; projection and exact bindings -> Task2;
eligibility/consensus/finalization -> Task3; all metrics/provenance -> Task4; CLI/publication/docs
and compatibility -> Task5. The five review-focus cases each have an owning test step.
All consumed interfaces are defined above; no service integration or new dependency is needed.
The sole planning refinement is parameterized local JSON validation because shared helpers
cannot honor 8 MiB. Pilot behavior and shared helper defaults are unchanged.

After maintainer approval, implement natively in this dedicated branch, task by task, followed
by one fresh-context final review. Validate findings against current code before fixing them.
Commit/push/create a PR under standing authorization after required validation. Attach the PR;
do not merge, freeze the candidate, post review comments, or execute provider requests.
