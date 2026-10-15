# Follow-up Capture Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with the preserved native execution method. Steps use numbered instructions, not checkbox tracking, to respect AGENTS.md's prohibition on alternate issue trackers. Do not invoke `bd` or edit `.beads`.

**Goal:** Capture the exact registered 384-slot follow-up with separate execution approval and durable evidence, then export into the existing offline scorer.

**Architecture:** Two dedicated modules implement follow-up record validation/storage and capture orchestration/export/CLI. Reuse the existing public transport, normalization, money, and publication helpers; leave the pilot and scorers unchanged. Approval is a separately retained run declaration and never modifies the frozen packet.

**Tech Stack:** Python standard library, existing repository record/transport helpers, pytest, Ruff, Black, mypy; no new dependency.

**Spec:** [Approved capture design](2026-10-15-knowledge-utility-followup-capture-design.md).

**Status:** Written design approved on 2026-10-15. This plan awaits review; implementation and live execution have not been authorized by this document. Native implementation and one fresh final reviewer remain the established execution method.

## Global constraints

- Exact frozen packet: `c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1`; validate the production registration as well as this digest.
- Exactly 384 slots, in frozen order; at most 384 counting requests followed by at most 384 generations. No retries, resume, calibration, endpoint override, or model substitution.
- Profile: model `gpt-5.4-mini-2026-03-17`, reasoning effort `none`, temperature 1, output cap 256, tools empty, store false, service tier `default`, truncation `disabled`, stream false. Input cap 2,048 tokens.
- Preserve packet bytes, original evidence/nulls, all ten limitations, every actor message and answer key, scoring schemas/arithmetic, and the two-reviewer adjudication boundary.
- Generation budget is a positive decimal string with at most six fractional digits and 32 characters; require full-roster reservation and reject amounts above USD 5.00. No actual budget is approved here.
- Counting policy `no_charge` requires `accept_unknown_counting_cost=false`; `unverified_accepted` requires true and separate human acceptance. Never price counting usage as generation or claim unknown billed costs are zero.
- Before live mode, bind the clean tracked HEAD, normalized fresh run directory, manifest/profile, execution digest, UTC prices/compatibility assertions no more than 24 hours old, and explicit approval reference. No credential access until metadata is retained.
- Packet/manifest/preparation/retained record/capture caps: 4 MiB. Standalone prices/execution: 1 MiB. Provider body: 1 MiB. UTF-8, depth <=32, unique keys, finite numbers, exact field/types; canonical outputs include the publication newline in their cap.
- Use exclusive writes and hard-link output publication. Preserve partial files and existing destinations. Raw exception text and credentials never enter diagnostics.
- Exports remain DEVELOPMENT, unverified, unpaired, and ineligible for optimizer input or training authority; scorer execution authorization stays false.
- Follow AGENTS.md import sections, Python line length <=100, Black formatting, and repository quality/documentation precision gates. Preserve unrelated untracked history files.
- Tests use fake transport and synthetic execution assertions only. No test reads real credentials or calls a provider. Existing local artifacts are read-only for final offline checks.

## Review focus

1. Exact freshness boundaries and timezone offsets: accept exactly 24 hours, reject future and offset/naive assertions; historical export must still work days later. Task 2 tests these cases.
2. Windows aliases, junctions, and competing writers: directory aliases bind consistently, existing targets block live calls, and linked artifacts fail recovery. Tasks 2–4 include platform-specific cases and race tests.
3. File failure after dispatch: neither a failed result write nor a failed terminal write permits another call or discards the reservation. Task 3 injects these failures; Task 4 recovers from the surviving evidence.
4. Parse errors versus corrupt but parseable records: only a final truncated/syntactically malformed record can represent uncertainty; altered identities, duplicate keys, excessive depth/size, and accounting errors fail export. Task 4 tests the distinction.
5. Maximum retained output and aggregate size: bound each read and canonical output without truncating evidence, and process raw bodies one slot at a time. Tasks 1, 3–5 cover large messages, provider bodies, and exported text.

## Files and dependency order

| File | Responsibility / first task |
| --- | --- |
| `scripts/knowledge_utility_followup_capture_records.py` | Fixed manifest and preparation (1), execution validation (2), durable store (3) |
| `scripts/capture_knowledge_utility_followup.py` | Retention and live orchestration (3), offline recovery export (4), CLI (5) |
| `tests/followup_capture_fixtures.py` | Real frozen reconstruction, explicitly synthetic execution records, fake provider responses and transport (introduced as needed in 1–3) |
| `tests/test_knowledge_utility_followup_capture_records.py` | Manifest, execution bindings, decimal accounting and record-store tests (1–3) |
| `tests/test_knowledge_utility_followup_capture.py` | Live-state and interruption tests (3) |
| `tests/test_knowledge_utility_followup_capture_export.py` | Recovery and malformed-history tests (4) |
| `tests/test_knowledge_utility_followup_capture_cli.py` | CLI flags, offline purity, errors and publication (5) |
| `docs/experiments/knowledge_utility_followup.md` | Capture operator reference and draft/frozen wording correction (5) |
| `docs/adr/0025-knowledge-utility-followup-capture.md` | Capture decision and authority boundary (5) |
| `tests/test_documentation_links.py` | Include the new ADR in checked documentation (5) |

Tasks are sequential. Keep the current dedicated checkout/branch; do not create another worktree
or change unrelated files. Base is merged PR #799, commit `ba2cebb95339cd5494ef12c0585b919d1db8ce06`.
Read this plan and the spec before implementation. Each task ends with its focused checks and a
commit; native implementation ends with one fresh whole-branch review, not per-task agents.

## Commands and fixtures

Use the existing Python interpreter:
`C:/Users/evanh/Documents/Codex/2026-10-07/files-pasted-by-the-user-you/work/venv/Scripts/python.exe`.
In PowerShell set a task-specific alias variable `$followupPython` to that path. Set `PYTHONPATH`
to `$PWD;$PWD/src;$PWD/modules;$PWD/reasoning-generalization-tracer/src`, `HF_HUB_OFFLINE=1`,
and `TRANSFORMERS_OFFLINE=1`. Commands below assume repository cwd and that variable.

The test helper `approved_packet() -> JsonObject` reconstructs the frozen packet from
`prepare_followup()` and `tests/fixtures/knowledge_utility_followup/accepted_freeze_record.json`,
then asserts the exact registered digest. Do not monkeypatch the registry or modify the fixture.

Use `NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)` and `COMMIT = "a" * 40` in tests.
`price_record() -> JsonObject` uses synthetic rates I=`"0.75"`, O=`"4.5"`, the fixed profile,
NOW's UTC timestamp and `no_charge`. These numbers are test arithmetic, not current prices.
`execution_record(manifest, run_dir, *, budget="1.032192") -> JsonObject` supplies all exact
execution fields from the spec, synthetic provider-review/approval references, fixed caps/flags,
normalized directory, COMMIT and its content digest. No helper manufactures operational approval.

`provider_result(text: str = '{"answer":"A"}', *, status: str = "completed") -> TransportResult`
creates valid synthetic provider responses with the pinned model and bounded usage.
`FakeTransport(responder: Callable[[str, JsonObject, int], TransportResult] | None = None)` exposes
`post(endpoint, payload) -> TransportResult` and a `calls` list, deep-copying payloads. Its default
count result is object `response.input_tokens`, input_tokens 100, usage input 100/output 0;
default generation uses `provider_result()`. The index passed to responder is zero-based across
all calls. Fault-injection tests own their fake failures rather than adding production knobs.

## Task 1: Exact request projection and offline preparation

**Files:** Create the records module, fixture helper, and records test module listed above.

**Interfaces produced:**

```python
build_manifest(packet: object) -> JsonObject
prepare_capture(packet: object, prices: object | None = None) -> JsonObject
```

Reuse `validate_supported_packet`, `content_digest`, `canonical_json`, `validate_json_value`,
`parse_prices`, and `reservation_micro_usd`. The helper's returned objects are fresh snapshots.
Do not invoke pilot manifest construction. Pin the fixed profile locally with named constants.

1. Write `test_manifest_projects_all_384_slots` with these core assertions:

   ```python
   result = build_manifest(approved_packet())
   assert len(result["requests"]) == 384
   assert result["schema_version"] == "knowledge-utility-followup-requests-v1"
   assert result["manifest_digest"] == (
       "af85b29c1f52f70ba5a05a7f56aa11e6848fe8ed8f13d4aaf4c04df81bea74c2"
   )
   assert len((canonical_json(result) + "\n").encode()) == 700160
   ```

   For every index, assert equality to the source slot identity and exact profile plus actor
   messages; counting payload has only `model input reasoning tools`. Check request digests,
   no shared mutable objects, and rejection of draft/pilot/modified/resealed/other registered input.

2. Write `test_preparation_is_offline_and_labels_unknowns`: no prices yields null reservation
   estimate and `token_counts="unknown"`; synthetic prices yield R=2688 and roster=1032192
   micro-USD with generation-only scope. Unknown counting policy yields null counting cost;
   no-charge yields string `"0"`. Assert original packet/prices unchanged and output <=4 MiB
   including newline. Test zero/invalid rates and field/type substitutions.

3. Run `& $followupPython -m pytest tests/test_knowledge_utility_followup_capture_records.py -q`.
   Expected initially: missing implementation fails collection or assertions; resolve unrelated
   fixture/import failures before counting this as the red phase.

4. Implement the two functions in the records module using the exact spec projection and schemas.
   Enforce 384 ordered unique slots and two messages per slot after validating the pinned packet.
   Validate positive rates and timestamp syntax for estimates, without freshness requirements.

5. Run the same tests and existing registration tests; require all pass. Commit only task files:
   `feat: prepare exact follow-up capture requests offline`.

## Task 2: Execution bindings, freshness, and exact money

**Files:** Extend the records module and its tests; add synthetic execution helper to fixtures.

**Interfaces consumed:** Task 1's regenerated manifest and price estimate helpers.

**Interfaces produced:** Frozen dataclass `FollowupExecution` with `raw: JsonObject`,
`prices: PriceRecord`, `budget_micro_usd: int`, `reservation_micro_usd: int`, plus:

```python
validate_execution(
    raw: object, manifest: JsonObject, run_dir: Path, *,
    live_commit: str | None = None, now: datetime | None = None,
) -> FollowupExecution
```

Both optional arguments absent means historical validation. Both present means live validation;
reject supplying only one. Historical validation checks syntax, digests, directory binding,
profile, price/policy and budget; live validation also checks exact HEAD binding, timestamp
freshness relative to aware UTC `now`, and absence of the target directory. Neither mode writes
files, reads credentials, or calls the network. Return a deep copy of validated raw data.

1. Write `test_execution_full_roster_budget_and_binding`: valid synthetic record produces
   reservation 2688 and budget 1032192; accept budget `"5"`; reject `"1.032191"`, `"0"`,
   `"5.000001"`, negative/exponent/float/bool/oversized values, zero rates, unknown/extra/missing
   fields, and altered or stale execution digest. Rehash altered inputs when testing inner binding
   checks rather than merely testing the outer digest. Caps must be integer 384, never booleans.

2. Write parameterized `test_live_freshness_and_historical_export`:

   ```python
   # Apply the timestamp to both prices and provider_review, then reseal the execution.
   # At NOW: NOW and NOW - timedelta(hours=24) pass live validation.
   # NOW + timedelta(microseconds=1) and NOW - timedelta(hours=24, microseconds=1) fail.
   # Historical validation still accepts syntactically valid old/future UTC assertions.
   ```

   Reject naive/non-Z offsets, malformed or >64-byte strings, naive `now`, false/numeric provider
   flags, invalid official URLs, blank/NUL/oversized references, and profile type substitution.

3. Write `test_counting_policy_requires_matching_acceptance` for both valid combinations and
   every mismatch. Write `test_run_directory_alias_binding_and_reuse`: platform-normalized aliases
   bind, different directories/relative paths fail, and existing directories/files/dangling links
   fail live validation. Historical validation accepts the same retained directory binding.

4. Run the records tests red, then implement strict execution validation using Decimal, exact JSON
   comparisons, `content_digest`, and aware UTC datetime arithmetic. Check path existence with
   lstat semantics so dangling links cannot look like fresh destinations. Reuse public money and
   normalization helpers; do not silently clamp budgets to USD 5.00.

5. Run records tests plus existing pilot price/execution tests. Require green, Ruff and Black for
   changed Python files; commit `feat: bind follow-up execution approval and budget`.

## Task 3: Durable store, retained results, and one-shot execution

**Files:** Extend records module/tests; create runner and live tests; extend fake transport helper.

**Interfaces produced in records module:** `FollowupRunStore` with `root: Path`, `identity: JsonObject`,
classmethod `create(run_dir: Path, execution: FollowupExecution, manifest: JsonObject) -> FollowupRunStore`,
and `write_record(relative: str, data: JsonObject) -> None`. It owns names and attaches identity.
Use exclusive binary creation, canonical JSON plus newline, flush and fsync. Reject data that
overwrites identity, invalid paths/slots, and oversized records before opening the destination.

**Interfaces produced in runner:**

```python
retained_result(result: TransportResult, execution: FollowupExecution, *, generation: bool) -> JsonObject
restore_transport(raw: object) -> TransportResult
count_is_valid(result: TransportResult) -> bool
run_live(
    packet: object, execution: object, run_dir: Path,
    transport_factory: Callable[[], Transport],
) -> JsonObject
```

`run_live` always regenerates the manifest, invokes the public `implementation_commit()` check,
and validates raw execution with current aware UTC time before directory creation. Tests patch
that imported clock/commit dependency; the production CLI exposes no bypass. It then claims and
initializes the store before invoking the factory. The factory is invoked once, after validation.
`restore_transport` checks exact TransportResult fields, bounded base64/body, exact boolean/int
types, bounded nullable metadata, timestamps, and nonnegative finite elapsed seconds.

1. Write `test_single_owner_and_durable_intent_before_post`: two processes racing for the same
   directory yield one owner; an instrumented transport sees durable count/generation intents
   before each post. Existing/partial directories prevent factory access. Test filename bounds
   000 and 383 accepted, 384/traversal/alternate names rejected; intent bytes survive write failure.

2. Write `test_full_roster_counts_before_generations`:

   ```python
   summary = run_live(packet, raw_execution, run_dir, lambda: fake)
   assert [endpoint for endpoint, _ in fake.calls] == (
       ["responses/input_tokens"] * 384 + ["responses"] * 384
   )
   assert summary["state"] == "finished"
   assert summary["reserved_micro_usd"] == 1032192
   assert summary["total_billed_cost_usd"] is None
   ```

   Verify payload order, all record identities including execution digest, count usage retained
   with zero output and null costs, generation estimates distinct from reservations, no key text
   in records, source packet unchanged, and returned model/usage projections from retained bytes.

3. Write `test_count_failure_prevents_all_generations` at first/middle/last count for excessive
   input count, bool count, malformed body, non-200, redaction, transport error, and timeout.
   Assert no generation call. Write `test_generation_outcomes_control_progress`: refusal and
   truncation permit next slot; wrong model/usage, malformed output and transport failure stop.
   No completed response is graded by the runner and no request is retried.

4. Write `test_write_or_interrupt_failure_never_resends`, injecting failures at metadata setup,
   each intent write, post, result write, and terminal write. Include KeyboardInterrupt during
   count and generation. Assert no further calls; reservation remains for generation intent even
   if persistence fails; pending work gives uncertain terminal state when terminal is writable.
   A terminal-write exception propagates to CLI handling without deleting existing artifacts.
   For retained complete failure results, terminal is stopped rather than missing-evidence uncertain.

5. Write `test_retained_result_bounds_and_roundtrip`: exact roundtrip of valid transport, rejection
   of oversized/invalid base64 and nonfinite flags/elapsed fields, count usage never generation-priced,
   invalid provider JSON retained as bytes with null parsed provider, and output-size rejection
   before persistence. Normalize only complete unredacted transport with the existing public helper.

6. Run new live/store tests red, implement the spec state machine with separate pending-count and
   pending-generation flags, then run records/live tests plus pilot capture and transport tests.
   Require all pass. Commit `feat: retain bounded follow-up capture attempts without retries`.

## Task 4: Offline recovery from retained history

**Files:** Extend runner; create export tests. Use Task 3 fake runs for fault cases.

**Interfaces consumed:** Manifest/approval/store identity from Tasks 1–3 and retention/restore helpers.
**Interface produced:** `export_run(packet: object, run_dir: Path) -> JsonObject`.

The export returns captures-v1 only; it never writes to the run tree. Derive complete outcomes
from restored transport, and compare retained provider/usage/model/cost projections against fresh
`retained_result` output. Do not use saved parsed projections to decide outcomes.

1. Write `test_full_and_partial_exports_match_durable_history`: full fake run exports 384 rows in
   schedule order; completed output text is exact, refusals/truncations retain their status, missing
   generation intents yield missing rows, and final absent/syntax-invalid result yields failed
   `uncertain_attempt`. The scorer accepts exports and preserves DEVELOPMENT/false authority flags.
   Repeated export yields identical data and does not change any original file hash.

2. Write `test_history_rejects_gaps_conflicts_and_edited_projections`: parameterize missing early
   intents, results without intents, complete result after partial intent, later work after a failed
   or uncertain slot, any generation before all counts pass, wrong run/slot/request/execution
   identity, wrong schema/types, altered provider/usage/cost projections, and reservation changes.
   Assert ValueError; do not accept the error as a missing observation.

3. Write `test_partial_record_is_distinct_from_invalid_record`: a final file with truncated JSON
   or invalid UTF-8 can represent an interrupted write; duplicate keys, depth>32, excessive bytes,
   parseable nonobjects/field errors and accounting contradictions reject. Catch JSON syntax and
   UTF-8 decode errors narrowly before exact validation. Never treat generic ValueError/OSError as
   uncertainty. Absent/incomplete optional terminal is recoverable; conflicting identity is not.

4. Write `test_export_tree_and_historical_bindings`: reject unknown root/subdirectory entries,
   directory masquerading as record, symlinks, Windows reparse points/junctions, and path escape.
   Use lstat before reads; do not resolve a child link to test it. Add native platform cases where
   link creation is supported, with explicit skip reasons otherwise. Changed current HEAD and old
   price timestamps do not invalidate an internally consistent historical run. Filesystem read
   errors fail export. Test processing bounded records without collecting all raw bodies in memory.

5. Run export tests red, implement ordered prefix validation and recovery, then run the records,
   live, export, follow-up scoring and response-review tests. Require green and no new authority
   claims. Commit `feat: export follow-up captures from durable evidence`.

## Task 5: CLI, operator documentation, and integrated validation

**Files:** Extend runner with `main(argv: list[str] | None = None) -> None`; create CLI tests,
update protocol, add ADR 0025 and include it in `tests/test_documentation_links.py`.

**Interfaces consumed:** `prepare_capture`, `run_live`, and `export_run`; public credential
transport factory and exclusive `publish_json`. Use strict existing readers with the spec's
per-file caps. Validate output size including newline before publishing.

1. Write `test_default_prepare_and_export_do_not_access_credentials_or_network`: patch factory
   and credential readers to fail if invoked. Default preparation works without prices and retains
   unknown token counts; optional synthetic prices produce the exact labeled reservation estimate.
   Export writes the exact captures-v1 envelope and leaves original files unchanged.

2. Write `test_live_cli_validates_before_factory_and_returns_controlled_status`: all invalid
   bindings/modes and dirty tracked checkout fail before factory; an approved synthetic fake run
   completes with exit 0; every stopped/uncertain state exits 2. Exercise actual argparse entry
   via a subprocess for offline help/preparation/export; live orchestration uses injected fakes
   in-process and no operational approval or credentials.

3. Write `test_cli_rejects_wrong_flags_and_preserves_destinations`: required/irrelevant flags,
   mutually exclusive live/export, input bounds/depth/duplicates, missing parent, existing output,
   concurrent publication winner, unsupported hard links, post-publication cleanup failure, and
   max-size output. Preserve destinations. Assert exact two static error strings from the spec;
   no body, secret, exception content, or raw provider error leaks to stderr.

4. Implement CLI dispatch and update operator docs with exact flags/schemas/caps, generation-only
   accounting, stale-price behavior, partial-directory recovery, no-resume rule, and separate live
   approval. Correct protocol's draft-only import wording and earlier statement that no follow-up
   live command exists. Explain why scorer `execution_authorized=false` remains unchanged even
   for an exported run. ADR distinguishes tooling approval from live execution approval.

5. Run all new and affected tests in one invocation. From repo root:

   ```powershell
   $followupTests = @(Get-ChildItem tests/test_knowledge_utility*.py | ForEach-Object FullName)
   $followupPythonFiles = @(
       'scripts/knowledge_utility_followup_capture_records.py',
       'scripts/capture_knowledge_utility_followup.py',
       'tests/followup_capture_fixtures.py',
       'tests/test_documentation_links.py'
   ) + @(Get-ChildItem tests/test_knowledge_utility_followup_capture*.py | ForEach-Object FullName)
   & $followupPython -m pytest @followupTests tests/test_documentation_links.py -q
   & $followupPython -m ruff check @followupPythonFiles
   & $followupPython -m black --check --line-length 100 @followupPythonFiles
   & $followupPython -m mypy scripts/knowledge_utility_followup_capture_records.py scripts/capture_knowledge_utility_followup.py --follow-imports=silent
   git diff --check
   ```

   Require zero failures, clean static checks, and Python lines <=100. Inspect every
   skipped platform test. Preserve existing draft report byte goldens. Full-suite reruns are
   warranted by changes to shared code, unexplained integration failures, or reviewer concerns;
   otherwise finish after these affected checks pass rather than repeat unrelated tests.

6. Run offline preparation against the actual retained frozen packet in a new output location
   outside the repo. Verify the manifest digest above, unchanged source/evidence bytes, and no
   credential/factory access. Do not supply current prices, create approving execution input,
   make model/counting calls, or export synthetic observations as study outcomes.

7. Commit the CLI/docs and tests with `feat: expose bounded follow-up capture workflow`.
   Review the complete branch against `ba2cebb95339cd5494ef12c0585b919d1db8ce06` using CodeRabbit
   committed review and one fresh reviewer via the requesting-code-review skill. Give the reviewer
   the approved design/plan, exact base/head, validation evidence, and retained artifact bindings;
   require read-only review, no child reviewers, documentation findings and Declined to judge list.
   Fix valid findings minimally, validate affected behavior, and record any rejected finding's reason.

8. After checks/review pass, push and create/attach the PR under standing user authorization.
   Do not merge or run live. Summarize exact tests/reviews, deferred limitations, and that concrete
   execution approval is still required after fresh provider and price review.

## Plan self-review and handoff

The spec's projection, approval, pricing and freshness requirements map to Tasks 1–2; storage,
counting/generation, retention and interruption to Task 3; historical recovery to Task 4; CLI,
documentation, compatibility and delivery to Task 5. All five review-focus cases have named tests.
Signatures above are the contracts between tasks; no undefined production helper is assumed.
The stricter distinction between partial syntax and invalid structured records implements the
spec's explicit separation of uncertainty from conflicting identity/accounting evidence.

The reference manifest was independently projected from the retained frozen packet during planning:
384 requests, digest `af85b29c1f52f70ba5a05a7f56aa11e6848fe8ed8f13d4aaf4c04df81bea74c2`,
700,160 canonical bytes including newline. This was a read-only calculation, not product
implementation, token counting, or a provider call. No unit-test results are claimed for code
that does not exist yet.

Review this written plan before native implementation. The separate execution record, fresh
provider/price evidence, budget, credential selection and unknown counting-charge acceptance are
future operational inputs, not blanks to fill with inferred approvals during implementation.
