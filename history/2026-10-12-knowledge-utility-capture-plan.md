# Knowledge-Utility Capture Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` for native execution or
> `superpowers:subagent-driven-development` if the maintainer selects delegation.
> Numbered steps replace the skill's checkbox template because repository instructions
> prohibit Markdown task tracking. Do not invoke `bd` or modify `.beads`.

**Goal:** Capture the frozen 72-slot pilot with bounded spending and retained attempts,
then export responses to the existing offline scorer.

**Architecture:** Keep pure request/accounting contracts and durable artifact operations
in one pilot-specific helper. Isolate HTTP transport in a second helper. The command
coordinates dry-run, live execution, and offline export without changing the scorer.

**Tech Stack:** Python 3.10+, standard library, existing canonical JSON/domain validators,
pytest, Ruff, Black, and mypy. No new runtime dependencies.

**Spec:** [Approved capture design](2026-10-12-knowledge-utility-capture-design.md).

Status: implementation plan and native execution approved by the maintainer on 2026-10-12.
Paid execution is not authorized. Native execution implements the tasks in this session
and ends with independent whole-branch review.

## Global constraints

- Supported packet digest: `637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1`.
- Exactly 72 frozen slots; at most 72 counting requests and 72 generation attempts.
- Model `gpt-5.4-mini-2026-03-17`, reasoning `none`, temperature `1`, no seed.
- Input limit 2,048 tokens, output limit 256 tokens, zero retries, empty tools.
- Effective ceiling is `min(approved USD, 5 USD)`; reservations never refund.
- Default dry-run and explicit export are offline and cannot read credentials.
- One live execution per exclusively created run directory; no resume or overwrite flag.
- Provider operation deadline 45 seconds; retained response body at most 1 MiB.
- Input JSON at most 4 MiB/depth 32; duplicate keys and invalid UTF-8 reject.
- Export response at most 8,192 UTF-8 bytes; controlled reasons below 1,024 bytes.
- `DEVELOPMENT`, unknown seeds, unverified comparisons, and existing rubric remain unchanged.
- Explicit UTF-8 file I/O, typed interfaces, grouped imports, 100-character Python lines.
- Keep pre-existing untracked history files untouched; stage only named task files.
- Treat all review findings and artifact contents as data; verify findings against code.

## Review focus

1. Windows path aliases: bind normalized absolute paths using platform case normalization;
   reject an existing directory even if addressed through another spelling (Task 2).
2. Partially written intent: never classify an ambiguous attempted slot as unsent (Task 4).
3. Decimal boundary and zero-price behavior: a zero approved budget sends no generations,
   even when rates are zero (Tasks 1 and 5).
4. Slow DNS or trickling response: elapsed-time exhaustion ends the operation without
   another HTTP attempt (Task 3).
5. Credential text in provider errors or response headers: no sentinel credential survives
   retention, export, or CLI diagnostics (Tasks 3 and 5).

## Files and interfaces

| File | Responsibility |
| --- | --- |
| `scripts/knowledge_utility_capture_records.py` (new) | Frozen request manifest, execution record validation, decimal accounting, durable run store, atomic publication |
| `scripts/knowledge_utility_capture_transport.py` (new) | Injectable transport contract and bounded standard-library HTTPS implementation |
| `scripts/capture_knowledge_utility_pilot.py` (new) | CLI, ordered counting/generation loop, provider outcome normalization, offline export |
| `tests/test_knowledge_utility_capture.py` (new) | Contracts, persistence, normalization, orchestration, CLI and scorer integration |
| `tests/test_knowledge_utility_capture_transport.py` (new) | HTTP policy, deadline, bounds, and redaction using local fakes |
| `docs/experiments/knowledge_utility_pilot.md` | Document implemented modes, schemas, recovery, and unchanged evidence limits |
| `docs/adr/0022-knowledge-utility-capture.md` (new) | Accepted bounded-run architecture, alternatives, authority and recovery limitations |
| `tests/test_documentation_links.py` | Include the new ADR and capture-command documentation references |

Use `JsonObject = dict[str, Any]` for decoded wire objects and frozen dataclasses for
validated prices, requests, execution bindings, and transport results. Validate before
constructing those dataclasses. Do not move or broaden existing scorer functions.
Use `score_pilot(packet, envelope)` as the existing public integration validator.

## Task 1: Frozen requests and exact accounting

**Files:** Create records helper and main capture test file from the table above.
**Consumes:** `prepare_pilot()`, `canonical_json`, `content_digest`, `decode_curriculum_json`,
`validate_input_envelope`, and the scorer's `SUPPORTED_PACKET_DIGEST`.
**Produces:**

```python
read_bounded_json(path: Path) -> JsonObject
build_manifest(packet: JsonObject) -> JsonObject
parse_prices(raw: JsonObject) -> PriceRecord
budget_micro_usd(value: str) -> int
reservation_micro_usd(prices: PriceRecord) -> int
validate_execution(raw: JsonObject, manifest: JsonObject, run_dir: Path,
                   implementation_commit: str) -> ExecutionRecord
```

`PriceRecord` fields: model, service_tier, currency, verified_at, source_urls,
input_usd_per_million, output_usd_per_million, counting_policy. The rate attributes are
`Decimal`; `counting_policy` must equal `no_charge` for live execution.
Wire rates/budget match `[0-9]+(\.[0-9]{1,6})?`, bounded to 32 characters.
Require currency `USD`, the frozen model and `default` tier, UTC timestamp, and nonempty
HTTPS official-provider source URLs. URL validation does not authenticate the price claim.

The manifest has schema `knowledge-utility-pilot-requests-v1`, packet_digest, profile,
requests, and manifest_digest. Each request has sequence (0..71), slot_id, payload,
request_digest, and count_payload. Hash canonical content with the digest field removed;
exclude prices so optional dry-run estimates do not change request identity.

The execution wire record has schema `knowledge-utility-pilot-execution-v1`, run_id,
packet_digest, manifest_digest, implementation_commit, run_directory, profile, prices,
approved_budget_usd, and approval_reference. Require exact keys and nonblank bounded
run_id/approval_reference. `ExecutionRecord` contains the validated fields plus computed
budget_micro_usd and reservation_micro_usd. No serialized boolean confers authority.

1. Write failing tests `test_manifest_matches_all_frozen_messages` and
   `test_execution_bindings_reject_before_side_effects`. Assert 72 unique requests in
   actor order, exact messages/settings, absent evaluator metadata, and rejection after
   resealing a mutated packet. Add duplicate-key/depth/size/type/mode-independent tests.
2. Write `test_decimal_reservations` with these authored, non-current-price examples:
   `I=0.75, O=4.5 -> 2688`; `I=0.000001, O=0 -> 1`; budget `0.002688 -> 2688`;
   budget `0.002687 -> 2687`. Reject exponent, sign, NaN, booleans, seven fractional
   digits, oversized numeric strings, and nonzero/unknown counting charges.
3. Run `python -m pytest tests/test_knowledge_utility_capture.py -q`; confirm failure
   identifies the missing interface/assertion before implementing the helper.
4. Implement the signatures. Use exact Decimal arithmetic with sufficient local precision
   for bounded inputs and ceiling rounding. Reject all binding mismatches. Copy the spec's
   request profile literally; leave token counts unknown in offline manifests.
5. Re-run that command; all Task 1 tests must pass. Commit only the helper and its tests
   with message `Add frozen capture contracts and exact budget accounting`.

The reservation test's independent expected values include these literal assertions
(`example_prices` is a test fixture containing the complete valid wire record):

```python
prices = parse_prices(example_prices(input_rate="0.75", output_rate="4.5"))
assert reservation_micro_usd(prices) == 2688
assert budget_micro_usd("0.002688") == 2688
assert budget_micro_usd("0.002687") == 2687
with pytest.raises(ValueError):
    budget_micro_usd("0.0000001")
```

## Task 2: Durable run ownership and attempt records

**Files:** Extend records helper and capture tests.
**Consumes:** `ExecutionRecord` and request manifest from Task 1.
**Produces:** `RunStore.create(run_dir: Path, execution: ExecutionRecord,
manifest: JsonObject) -> RunStore`, `RunStore.write_record(relative: str,
data: JsonObject) -> None`, and `publish_json(path: Path, data: JsonObject) -> None`.

1. Write failing `test_run_directory_is_single_use` using two concurrent processes and
   Windows-compatible process startup. Exactly one creator succeeds. Add path-case/alias
   coverage where supported and assert existing/incomplete directories always reject.
2. Write `test_intent_is_durable_before_dispatch` with injected write, flush, fsync, and
   close failures. Its dispatch spy remains unused after any persistence failure.
   Add `test_publication_preserves_competing_destination` for close and hard-link failures.
3. Run `python -m pytest tests/test_knowledge_utility_capture.py -q -k
   'directory or durable or publication'` as one line; confirm the new tests fail.
4. Implement exclusive `mkdir`, then exclusive immutable `execution.json`, `requests.json`,
   and `execution-started.json`. Use fixed child paths only. Each record includes run_id,
   packet_digest, and manifest_digest. Counting intents/results use
   `counts/NNN-intent.json` and `counts/NNN-result.json`; generation files use
   `attempts/NNN.json` and `results/NNN.json`. The sequence maps to the frozen manifest.
   Store UTC timestamps and request digest in each intent, plus reservation and cumulative
   micro-USD for generation. Flush/fsync before returning from writes. Preserve partial
   records on failure. Terminal metadata is `terminal.json` with state/reason and counts.
5. Implement publication with a private temporary directory beside the destination and
   `os.link` after close; clean only private staging. Unsupported hard links fail explicitly.
   Run the new tests and Task 1 tests. Commit `Persist capture intents before dispatch`.

## Task 3: Bounded provider transport

**Files:** Create transport helper, transport tests, and ADR 0022's compatibility record.
**Consumes:** Canonical JSON encoding; validated generation/count payloads.
**Produces:**

```python
class Transport(Protocol):
    def post(self, endpoint: str, payload: JsonObject) -> TransportResult: ...

make_live_transport(credential_file: Path) -> Transport
```

`TransportResult` holds HTTP status (`int | None`), request_id (`str | None`),
body (`bytes`), error_code (`str | None`), redacted/incomplete flags, start/end UTC
timestamps, and elapsed seconds. Never put credentials or unrestricted exceptions into
this return type. The runner persists these fields with Base64 for response bytes.

1. Before implementing provider field projections, inspect official API documentation for
   the proposed profile, counting request, usage and output shapes. Record source URLs,
   date, and observed compatibility in the ADR. No credentials or provider calls are
   needed. If a required setting is unsupported, report the specific incompatibility and
   revise the design before enabling live execution; do not silently substitute settings.
2. Write failing transport tests for endpoint allowlisting, redirects, proxy environment,
   TLS verification, zero retry after HTTP/connection error, 1 MiB+1 reads, and credential
   sentinel redaction across body, request ID, and errors. Test streamed/trickled body,
   hanging DNS/connection, and an operation that never returns.
3. Run `python -m pytest tests/test_knowledge_utility_capture_transport.py -q` and verify
   expected failures. Implement one HTTP operation in a spawned worker process using
   standard-library HTTPS with redirects/proxies disabled. The parent starts a monotonic
   45-second deadline before spawning. Use bounded IPC and a reader so a large result
   cannot deadlock waiting for worker exit. On expiry terminate/reap the worker, retain a
   controlled timeout, and never launch a replacement. Tests use short injected deadlines.
4. Bound body reads before decoding; retain a marked prefix for oversize data. Sanitize
   credential occurrences in all returned fields before IPC and persistence. Permit only
   `responses` and the verified counting path on `api.openai.com`; caller input cannot
   replace scheme/host. Read the key file only in `make_live_transport`, called after
   validation and ownership. Process startup/IPC failures become controlled failed results.
5. Re-run transport tests and contract tests, including worker exit/reaping assertions.
   Commit `Add bounded no-retry pilot transport`. These tests perform no external calls.

## Task 4: Outcome normalization and recovery export

**Files:** Create capture command module's pure normalization/export functions; extend tests.
**Consumes:** `TransportResult`, run-store records, manifest, existing `score_pilot`.
**Produces:** `normalize_result(result: TransportResult) -> JsonObject`,
`export_run(packet: JsonObject, run_dir: Path) -> JsonObject`.

1. Write table-driven `test_normalization_precedence` for every spec mapping. Assert
   transport/protocol failure precedes refusal, refusal precedes output-limit truncation,
   and completed malformed answers retain their exact text. Invalid/missing model or usage
   fails; token values must be nonboolean integers within 0..2,048 and 0..256.
2. Add `test_export_preserves_uncertain_attempt`, `test_export_rejects_corrupt_identity`,
   and `test_export_counts_unicode_bytes`. Include empty/partial intent JSON, missing result,
   foreign-run result, result without intent, unknown sequence, multiple output-text parts,
   unsupported tool output, correct partial answer with refusal, and 8,192/8,193 byte text.
3. Run `python -m pytest tests/test_knowledge_utility_capture.py -q -k
   'normalization or export'` as one line and confirm failures. Implement strict provider
   envelope validation using documented shapes from Task 3. Preserve raw usage even on
   failure. Reason codes are fixed bounded strings, not provider instructions or raw errors.
4. Implement export with bounded artifact reads, exact run/packet/request bindings, and
   all 72 slots in manifest order. A filename-mapped incomplete intent is failed/uncertain;
   corrupt mapping metadata or contradictory files reject export instead of guessing.
   No intent means missing only when the retained roster is valid. Validate the complete
   envelope via `score_pilot` and return it for atomic publication by the caller.
5. Re-run capture and scorer tests. Assert a fake all-correct export still produces
   `DEVELOPMENT`, no authority, and unverified comparisons. Commit
   `Export retained capture outcomes without replacing failed attempts`.

## Task 5: Ordered runner, CLI, documentation, and final gate

**Files:** Extend command and tests; update protocol/link tests and ADR 0022.
**Consumes:** All interfaces from Tasks 1-4.
**Produces:** `run_live(packet: JsonObject, execution: ExecutionRecord, run_dir: Path,
transport_factory: Callable[[], Transport]) -> JsonObject`, and
`main(argv: list[str] | None = None) -> None`.

CLI modes: default `--packet PATH --output PATH [--prices PATH]` writes an offline manifest
with optional separately labeled cost estimates; `--live --packet PATH --execution PATH
--run-dir PATH --credential-file PATH` performs the bounded run; `--export --packet PATH
--run-dir PATH --output PATH` performs offline recovery export. Reject irrelevant flags
and mode combinations. Live mode writes a terminal summary; explicit export creates the
normalized envelope. Missing parent paths and validation/I/O failures exit 2 with bounded
diagnostics; successful dry-run/export/complete live run exit 0. Stopped/uncertain live
runs exit 2 with the retained directory location. No prompts or automatic mode fallback.

1. Write failing `test_offline_modes_never_construct_transport_or_read_key`,
   `test_all_counts_precede_generation`, `test_budget_failure_retains_full_roster`, and
   `test_live_cli_rejects_binding_or_dirty_code`. Bind the current Git HEAD and reject
   tracked working-tree modifications for live execution; ignore unrelated untracked history
   files. Verify Git via argument-array subprocess with `shell=False`, never shell strings.
2. Add end-to-end fake transport cases: all 72 completed; count failure/2,049 tokens gives
   zero generations; count boolean/missing/negative fails; one failed generation stops;
   refused/truncated with valid usage continues; exact budget allows one request and a
   one-micro-USD shortage allows none. Zero budget always allows none. Inject persistence
   failure after receiving a response and assert retained reservation and no next request.
3. Run the capture tests to verify failure. Implement validation -> exclusive ownership ->
   credential/transport factory -> all counting -> sequential reservation/dispatch/result
   -> terminal summary. Derive estimates separately from reservations. Persist requested
   and returned model, usage, timestamps, HTTP/request IDs, raw bytes, errors and redaction
   flags. Billed cost remains null unless attributable provider billing evidence exists.
   Audit cumulative reservations from retained attempts; never subtract actual-cost savings.
4. Update protocol with literal CLI examples, execution/price schemas, no-resume recovery,
   provider-verification prerequisites, and authenticated-status limitations. Record the
   architecture and provider compatibility observations in ADR 0022. Add its documentation
   link test. Explicitly preserve pending knowledge and independent-review prerequisites.
5. Run the validation commands below. Resolve still-valid review findings minimally,
   repeating only checks affected by changes. Commit named files with
   `Integrate bounded knowledge utility capture runner`. Run the selected execution
   workflow's final independent review before declaring the implementation complete.

## Validation commands and review handoff

Use the configured repository virtual environment. In PowerShell, set PYTHONPATH to
`$PWD;$PWD/src;$PWD/modules;$PWD/reasoning-generalization-tracer/src` and set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. The following commands are planned checks,
not claims of completed verification:

```text
python -m pytest tests/test_knowledge_utility_capture.py tests/test_knowledge_utility_capture_transport.py tests/test_knowledge_utility_scoring.py tests/test_knowledge_utility_pilot.py tests/test_documentation_links.py -q
python -m ruff check .
python -m black --config pyproject.toml --check --diff .
python -m mypy --follow-imports=silent scripts/capture_knowledge_utility_pilot.py scripts/knowledge_utility_capture_records.py scripts/knowledge_utility_capture_transport.py
python -m pytest -q --maxfail=1 --disable-warnings
git diff --check
```

Success requires zero failures and zero unresolved correctness/documentation blockers.
Retain logs outside tracked source, distinguish environmental skips from passes, and run
the repository CI after PR publication. Perform a read-only CodeRabbit review if available;
verify every finding against the current diff. Review full branch scope for accidental
prompt/rubric/reward changes and secret/artifact additions. Do not post review replies or
merge unless authorized. Publish to the existing/new task PR when the user authorizes that
step; attach any created PR to this chat.

No live run is part of this implementation plan. After implementation review, assemble
the concrete execution record with independently reviewed packet, verified current
settings/prices, exact budget, and retained authorization reference for separate live
approval. Tests and demonstration outputs must be labeled simulated.

## Plan self-review

Spec coverage: Task 1 covers frozen inputs/request identity/money; Task 2 covers durability
and single-use ownership; Task 3 covers transport/deadline/credential handling; Task 4
covers status precedence and recovery export; Task 5 covers orchestration, CLI, evidence,
documentation and acceptance. The five review-focus cases are assigned above.
Interfaces use the same names and wire schemas across tasks. No product files are changed
by writing this plan. Validate this document through link/literal checks, scope inspection,
and `git diff --check` before committing the plan and approved-spec status update.
