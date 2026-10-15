# Follow-up capture runner design

Status: Direction, written design, implementation plan, and native implementation approved on
2026-10-15. Implementation in progress; live execution remains separately gated.
The maintainer approved tooling implementation; credential access, provider requests, and spending
remain outside that approval. An eventual live run requires its own
concrete execution approval after implementation, review, and provider compatibility/price checks.

## Purpose and scope

Capture the accepted 48-task follow-up once in its fixed 384-slot order, retain inspectable
request/result evidence, and export responses into the existing offline scorer. The user approved
a dedicated runner reusing the existing transport, exact execution bindings, conservative cost
accounting, and no automatic retries. Success means reproducible capture bookkeeping, not a
positive knowledge-utility result or authenticated study.

The registered packet is `c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1`.
Its source is `718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`, candidate is
`e8a41152d1f647779a4facbfddc06c2defd61987377950bb345066df172561b9`, and decision is
`4c012dc2300e2780af357651d62273bf991eeddcd32015a11050d1c9d68cc767`.
PR #799 registered these bindings; merged baseline is
`ba2cebb95339cd5494ef12c0585b919d1db8ce06`.

Preserve that packet, all ten accepted limitations, the original first-pass null, original pilot
artifacts, answer keys, messages, condition order, scoring arithmetic, and response-adjudication
rules. No generation seed, authenticated independence, confidence interval, reward, optimizer
input, or training authority is introduced. No follow-up outcomes exist yet.

The frozen packet's `execution_authorized=false` and null approved budget remain historical
pre-execution facts. A separate execution record authorizes only a named run; it never changes
the packet or makes the scorer claim authenticated execution.

## Alternatives and component boundaries

Use a dedicated follow-up runner. A generic pilot/follow-up engine would expand compatibility
risk for a single additional study; manual provider calls would weaken durable intent tracking.

- Add `scripts/knowledge_utility_followup_capture_records.py` for manifest, execution validation,
  exact money checks, and a follow-up run store with slots 000 through 383.
- Add `scripts/capture_knowledge_utility_followup.py` for offline preparation, live orchestration,
  retained result construction, recovery export, and CLI parsing.
- Reuse public `Transport`, `TransportResult`, and `make_live_transport` from the existing
  `knowledge_utility_capture_transport` module unchanged. Its endpoint allowlist, HTTPS handling,
  secret redaction, disabled redirects/proxies, bounded bodies, and operation deadline remain intact.
- Reuse public `normalize_result` from `capture_knowledge_utility_pilot`: its fixed model and
  token limits match this packet. It classifies provider outcomes without grading answers.
  Do not call pilot `run_live`, `export_run`, `build_manifest`, or `RunStore` for the follow-up.
- Reuse public price parsing, decimal reservation calculation, timestamp, and exclusive JSON
  publication helpers where their contracts match. Do not import private pilot helpers or
  refactor the pilot to support the follow-up. Reuse its public `implementation_commit` check.
  Add focused follow-up retention/restore helpers.
- Use the existing follow-up strict JSON readers, registered packet validator, and scorer.
  No new dependency or user-selectable endpoint is needed.

Implementation updates the follow-up protocol and adds an ADR for capture. It also corrects the
older draft-only wording in the protocol's import section to agree with registered frozen support.
This design adds capture interfaces; existing capture and scoring schemas keep their meanings.

## Deterministic request manifest

Validate the complete frozen packet through the production registry and additionally require
the exact frozen digest above. Reject the supported draft, pilot packet, synthetic freezes,
and any other future registration. Require 384 actor requests matching the 384 schedule slots
in sequence, with unique slot IDs and exactly two public messages per request.

The generation profile is exactly:

```json
{"model":"gpt-5.4-mini-2026-03-17","reasoning":{"effort":"none"},"temperature":1,"max_output_tokens":256,"tools":[],"store":false,"service_tier":"default","truncation":"disabled","stream":false}
```

These are the repository's proposed settings, not a claim of current provider availability or
compatibility. Before live approval the operator verifies them against current official sources
and the intended account. An incompatible setting requires a new reviewed design and artifact;
the runner does not substitute a model, reduce effort, omit a field, or change a task.

For each slot, generation payload equals this profile plus `input=actor.messages`. No answer
keys, derivations, task-review records, condition labels, or evaluator annotations are added.
Counting payload projects exactly `model input reasoning tools` from that payload. The manifest
has exactly `schema_version packet_digest profile requests manifest_digest`, using schema
`knowledge-utility-followup-requests-v1`. Each request has exactly
`sequence slot_id payload request_digest count_payload`; sequence is an integer 0–383,
`request_digest=content_digest(payload)`, and manifest digest excludes only its own field.
No previous response, response ID, tool output, or conversation state enters a later request.

Offline preparation writes exactly `manifest reservation_estimate token_counts`.
`token_counts` is `"unknown"`; the estimate is null without a supplied price file. With prices,
the estimate contains `per_request_micro_usd roster_micro_usd budget_scope counting_policy
counting_cost_usd`, using the calculation below. Preparation never counts tokens remotely.

## Separate execution approval

The operator first presents the exact manifest, implementation commit, run directory, current
price/compatibility assertions, generation ceiling, and counting-charge policy to the maintainer.
Only an explicit approval of those inputs permits creating an approving execution record.
Neither the runner nor offline preparation synthesizes such approval. Earlier pilot budgets,
freeze approval, registry approval, design approval, and implementation approval do not supply it.

The execution object has exactly:

```text
schema_version run_id packet_digest manifest_digest implementation_commit run_directory
profile prices provider_review approved_budget_usd approved_count_requests
approved_generation_requests accept_unknown_counting_cost execution_authorized approval_reference
execution_digest
```

Schema is `knowledge-utility-followup-execution-v1`. `execution_authorized` is strictly true;
both approved request counts are strictly integer 384. `run_id` and `approval_reference` are
nonblank UTF-8 strings of at most 1,024 bytes, with no NUL. The approval reference is an opaque
citation to the actual human decision, not instructions or a resource to fetch. Packet digest,
manifest digest, and complete profile must match the regenerated manifest with exact JSON types.
`execution_digest` binds every other execution field.

`implementation_commit` is 40 lowercase hexadecimal characters matching the live checkout's
HEAD. Before live mode, reject tracked checkout changes, staged or unstaged, using the existing
implementation-commit check. The operator runs the checked-in module in the repository's configured
environment; untracked planning documents remain permitted. This check binds a reviewed revision;
it does not authenticate the host, Python environment, or import resolution. No CLI commit argument
or environment-variable override is provided by the runner.

`run_directory` is an absolute path, at most 4,096 UTF-8 bytes without NUL, normalized through
resolution and platform case normalization and equal to the selected CLI directory. Its parent
must already exist. The run directory itself must not exist, even if empty or left by a failed run.
Reject reuse before credential access or network activity. The operator must preserve old run
directories and examine uncertain dispatch before requesting a separately approved new run.

`prices` uses the existing exact PriceRecord fields and fixed model/default-tier/USD profile.
Rates are bounded unsigned decimal strings with at most six fractional digits, parsed with
Decimal; generation rates must be positive. `provider_review` has exactly
`verified_at source_urls model_available profile_compatible counting_charge_reference`.
The two flags are strictly true. `source_urls` is a list of 1–10 bounded HTTPS URLs on the same
official-domain allowlist as prices. `counting_charge_reference` is a nonblank string up to
1,024 UTF-8 bytes: evidence of no charge, or a statement that charges remain unknown.

Both price and compatibility timestamps are strings of at most 64 UTF-8 bytes containing valid
UTC `Z` instants. At live start they must
be neither future-dated nor older than 24 hours. Offline preparation/export check syntax but
do not impose current-time freshness on historical evidence. These are operator assertions;
URL syntax and timestamps do not authenticate availability, compatibility, or billing policy.
The runner never fetches the cited URLs. A failed provider preflight stops execution.

## Accounting and limits

Let input and output USD-per-million rates be I and O. Reserve
`R = ceil(2048 * I + 256 * O)` integer micro-USD for each generation attempt, with sufficient
Decimal precision and no floating-point arithmetic. Full-roster reservation is `384 * R`.
The approved budget must be a positive decimal USD string of at most 32 characters, with no
exponent or sign and at most six fractional digits, representable at micro-USD precision,
at least that full-roster reservation and at most USD 5.00. Reject outside this range before
directory creation, credentials, or counting calls. The USD 5.00 implementation ceiling follows
the pilot's conservative ceiling; it grants no spending permission. A higher ceiling requires
a reviewed change. Runtime reservation checks still run before every generation dispatch.

The budget scope is explicitly `generation_only`. Counting policy is either `no_charge` with
`accept_unknown_counting_cost=false`, or `unverified_accepted` with that flag true. The latter
requires explicit human acceptance of unknown counting charges for this run. No total-dollar
ceiling is claimed when counting charges are unknown. Limit counting to at most 384 requests;
no retries, additional calibration, compatibility probe, or model-list request is made by the runner.

Reserve the full R before creating each generation intent; never release a reservation after
failure, refusal, truncation, timeout, missing usage, or low reported usage. Usage-based generation
estimates are separate from reservations. For valid bounded usage, compute estimates with
uncached input and output rates. Billed cost remains null without billing evidence.

Retain counting usage if present, including input tokens and zero output tokens, but always set
`cost_scope="counting"`, `estimated_cost_usd=null`, and `billed_cost_usd=null` for counting results.
Never price counting usage using generation rates. Summary counting cost is `"0"` only for the
declared no-charge policy, otherwise null; label the policy as an assertion. Total billed cost
remains null for both policies. Current prices, availability, and billing evidence are not supplied
or approved by this design.

## Durable records and live state machine

After validating all inputs, claim the fresh directory with exclusive creation. Persist execution,
manifest, and started metadata before reading the explicitly selected credential file. No credential
search, environment-key fallback, key copying, credential logging, or credential hash is added.
Use the transport's existing credential-file validation. Offline commands do not access credentials.

The store permits only these names:

```text
execution.json requests.json execution-started.json terminal.json
counts/000-intent.json ... counts/383-intent.json
counts/000-result.json ... counts/383-result.json
attempts/000.json ... attempts/383.json
results/000.json ... results/383.json
```

Metadata wrappers contain the run identity `run_id packet_digest manifest_digest execution_digest`
and one payload: `execution`, `manifest`, or `utc` as applicable. Each slot record additionally
binds `sequence slot_id request_digest`. Writes use exclusive creation, flush, and fsync; existing
or partial files are never overwritten or removed for retry. This is process-interruption evidence,
not a guarantee of atomic multi-file state or storage durability through every hardware failure.

First count all 384 requests serially. Persist a count intent with `utc payload` before its call,
then persist the result. Accept only a complete, unredacted HTTP-200 provider object with no
transport error, `object="response.input_tokens"`, and an exact integer `input_tokens` in 0–2048.
A malformed, failed, oversized, timed-out, or excessive count stops the run before any generation.
The existing transport bounds each operation at 45 seconds plus bounded cleanup and never retries.

After all counts pass, generate serially in the frozen order. Before each dispatch, verify the
next reservation fits, then persist its intent with `utc attempt_number=1 reservation_micro_usd
cumulative_micro_usd`. Persist the retained result before proceeding. Completed responses,
declared refusals, and output-limit truncations continue to the next slot. A normalized failed
response stops further generations. A timeout or exception after intent creation is potentially
billed and never causes another dispatch of that slot.

Retained results have the slot/run identity plus exactly `transport provider usage requested_model
returned_model cost_scope estimated_cost_usd billed_cost_usd`. Transport fields match TransportResult,
with body base64-encoded. Preserve only transport-redacted provider bytes and parsed content;
invalid parsed content is null. Usage and returned model come from that retained provider object.
The exporter rederives these projections and estimates instead of trusting edited summaries.

On normal completion, write terminal `state="finished"`, `reason="roster_finished"`. Controlled
stops use state `stopped` and reason `setup_failed`, `count_failed`, `generation_failed`, or
`budget_exhausted`. Interrupted execution or file failure uses reason `interrupted_or_io_failure`
and state `uncertain` if a count or generation intent may have been dispatched without a complete
result; otherwise `stopped`. Track pending counting and generation separately. A terminal write
may itself fail: preserve all files and return exit 2, without assuming no calls occurred.

Terminal fields additionally include `utc count_requests generation_attempts reserved_micro_usd
budget_scope counting_policy counting_cost_usd total_billed_cost_usd`. Counts represent dispatch
attempts initiated by this process, not completed responses or authenticated provider accounting.
Recovery derives evidence from retained intents/results; it does not trust a terminal summary to
authorize further work. There is no resume or automatic new-run mode.

## Recovery export and existing scoring

Export is offline and read-only on the run directory. Rebuild the manifest from the pinned packet,
validate retained execution/manifest identities and exact schemas, and require the started record.
Compare historical implementation binding syntactically and internally, not against today's HEAD.
Validate all expected subdirectories and reject unexpected names, non-files, symlinks, or reparse
points within the run tree. Resolve only the explicit run directory; never follow embedded paths.

Complete count pairs must match the manifest payload and result projection. Count intents form a
prefix, each result requires its intent, and no count follows a failing or incomplete count. If any
generation intent exists, all 384 counting results must be present and passing. Generation intents
also form a prefix; results require intents, with no later attempt after a failed or uncertain slot.
Refused/truncated/completed results permit later attempts. Verify each complete intent's integer
reservation and cumulative amount, reserving R even for an incomplete final intent.

Export exactly 384 rows in schedule order using the existing captures-v1 envelope. A slot without
generation intent is `missing`; a final malformed/partial intent or absent/malformed result is
`failed` with controlled reason `uncertain_attempt`. A result with no intent or with an incomplete
intent is inconsistent and rejects the export. Parseable records with wrong identities, fields,
digests, types, or accounting also reject; they are not silently relabeled as missing. File I/O
errors fail export. An incomplete optional terminal record does not prevent recovery from valid
durable records, but a complete terminal with a conflicting identity is rejected.

Use existing response normalization for completed/refused/truncated/failed classifications and
the existing scorer to validate the export. Never rewrite provider text, infer refusals from prose,
extract a guessed option, or grade a censored/uncertain response as incorrect. Unsent reasons are
derived from the count/generation history (`count_failed`, `generation_failed`, or `not_attempted`),
not from unverified terminal text. Further semantic adjudication uses the existing two-reviewer
workflow. Report-v2 remains DEVELOPMENT, unverified, unpaired, with execution authorization,
optimizer input, and authority false. Raw capture evidence is retained separately; exports do not
extend the scoring envelope or make authenticated provenance claims.

## CLI and bounded I/O

```text
python -m scripts.capture_knowledge_utility_followup --packet FROZEN --output PREPARATION [--prices PRICES]
python -m scripts.capture_knowledge_utility_followup --live --packet FROZEN --execution APPROVAL --run-dir NEW_DIRECTORY --credential-file KEY_FILE
python -m scripts.capture_knowledge_utility_followup --export --packet FROZEN --run-dir EXISTING_DIRECTORY --output CAPTURES
```

Modes are mutually exclusive. Reject irrelevant flags. Default is offline preparation.
Live mode has no output, price, endpoint, model, limit, or retry override. Export accepts no
credential or execution override. The operator supplies the credential path locally at execution;
it is neither embedded in the manifest nor copied into approval or result artifacts.

Use strict UTF-8, unique JSON keys, finite numbers, depth at most 32, and exact field/type checks.
Packet, manifest, preparation, and each retained record are at most 4 MiB; standalone prices and
execution input files are at most 1 MiB. Body decoding retains the existing 1 MiB transport limit.
Export reads a fixed bounded roster, one record at a time; do not accumulate all raw provider bodies.
The run directory can consume multiple GiB at these per-record limits; disk exhaustion stops dispatch and
requires recovery inspection. Exported captures are at most 4 MiB, with existing 8,192-byte text
limits. Validate canonical output sizes including one byte for the publication newline.

Offline output uses existing exclusive hard-link publication into an existing parent directory.
If cleanup fails after publication, a complete destination can coexist with exit 2; inspect before
retrying with a new name. Never delete an existing artifact to make a retry work.

Exit 0 means successful preparation/export or a completed roster. Invalid arguments use argparse.
Other offline failures return exit 2 with `cannot process follow-up capture: invalid input or file operation`.
Live failures/stops return exit 2 with `follow-up capture stopped; preserve run directory and inspect intents`.
Do not print exception contents, response bodies, credentials, or raw provider errors. Local operators
inspect controlled retained records for details. An error exit never proves a request was unbilled.

## Verification and delivery

| Requirement | Verification |
| --- | --- |
| Exact admission and actor projection | Reject draft, pilot, synthetic/modified frozen input; assert all 384 payloads, order, and digests against the approved packet; ensure evaluator metadata never enters payloads. |
| Approval before side effects | Fake credential/transport hooks remain untouched on changed binding, stale/future timestamp, bad policy, dirty tracked checkout, inadequate/excessive budget, existing directory, or malformed record. |
| Money and counting | Decimal boundary cases and upward rounding; full-roster sufficiency; unknown-charge acceptance; counting usage retained with null estimates/bills; every attempt reserves once. |
| Live orchestration | Fake transports cover all 384 counts/generations, count failure, refusal, truncation, malformed usage/model, deadline, keyboard interrupt, failed intent/result/terminal writes, and no later dispatch after a stop. |
| Recovery | Test every interruption boundary, prefixes/gaps, unknown names, linked files, mismatched identities, altered projections/reservations, partial final records, absent terminal, and existing output preservation. |
| Compatibility and authority | Run pilot capture/transport tests, follow-up preparation/freeze/scoring/review/registration tests, draft goldens, CLI smoke tests, and documentation-link checks. Empty and synthetic exports remain descriptive only. |
| Credentials and network | No live calls in tests. Prove offline modes do not invoke credentials/transport; reuse transport redaction/redirect/deadline tests without weakening existing behavior. |

The implementation plan will map these checks to small native implementation steps and one fresh
final reviewer. Use the repository quality gate, TDD, Ruff, Black, appropriate type checks, and
CodeRabbit. Inspect the final diff for documentation contradictions and unintended pilot changes.
Full-suite validation follows if implementation changes or integration failures justify it; focused
checks must cover the complete capture and scoring boundary before claiming completion.

After review and merge, prepare concrete execution inputs using fresh official compatibility and
price evidence. Present them for separate human execution approval, including any unknown counting
charges and the selected local credential path. No current price, budget, credential, or live-run
approval is inferred from this design. If approved inputs cannot be satisfied, report the failed
precondition rather than changing the model, packet, budget, or accounting policy during a run.
