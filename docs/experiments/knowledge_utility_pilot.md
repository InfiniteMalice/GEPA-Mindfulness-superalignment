# Knowledge-utility pilot preparation

This protocol prepares an exploratory study of knowledge context using the
[PR-7 diagnostic contracts](../cumulative_curriculum.md). It does not execute models,
read credentials, train, change rewards, or authorize spending. The preparer produces
review material, including a proposed lesson and proposed bank whose reviews remain pending.
Neither proposal is verified knowledge. The packet grants no execution or admission authority.

## Question and scope

Does the proposed lesson change task success by itself or when added to the proposed bank?
The standalone contrast is `lesson_only - none`; the marginal contrast is
`bank_plus_lesson - bank_only`. All four conditions use the same six public tasks,
base instruction, requested model settings, and token limits. Only the knowledge context changes.

The roster contains one authored task in each family: inventory arithmetic, precedence,
route cost, half-open intervals, quorum, and unit conversion. These toy tasks test capture
and reporting feasibility; they are not the V5 evaluation suite or evidence of general alignment.
The simple tasks may produce ceiling effects and cannot establish the absence of utility.

Each family has three repeats per condition: 6 × 4 × 3 = 72 planned requests, 18 per condition.
The script rotates condition order deterministically within each task and repeat. This is
an ordering convention, not randomized assignment or a generation seed. Three repeats of
one task are dependent; 72 requests are not 72 independent samples. Six authored family labels
also do not establish six independent semantic clusters. No confidence interval is proposed;
the roster is below the PR-6 minimum of 20 clusters for its supported bootstrap procedure.

## Knowledge and lineage

The packet retains two authored demonstration episodes, including their prompts and completed
answers. A rectangle-perimeter demonstration motivates the proposed lesson about computing
from explicit facts. A conjunctive access-rule demonstration motivates the proposed bank entry
about checking all conditions. `origin="authored_demonstration"` identifies both sources;
their completed status describes the authored examples, not successful live model calls.
The lesson summaries are hypotheses about transfer beyond those examples.

Both lessons have `review_status="pending"`. Their `MEMORY` destination records the
existing classifier's routing for an episode fact; it does not insert either lesson into memory.
The bank is a proposed one-entry snapshot. Support, scope, contradictions, reproducibility,
information preservation, and nonredundancy require independent review before either proposal
can be described as curated knowledge. This preparation supplies no `LessonReview` records
and no authentication callback; `curation_preview` withholds additions.

The manifest gives the two source families the planned `synthetic_training` role and the six
task families the planned `independent_audit` role. All records remain `DEVELOPMENT`, and
no training eligibility follows from a partition name. The six audit cases have `evaluated=True`
to declare membership in the evaluation roster; the flag does not claim that a model ran.
The two source cases have `evaluated=False` because they supply lineage only.
`dependencies_known=False` explicitly
withholds semantic independence. Local manifest validation checks declared identities and
lineage only. An independent reviewer must assess overlap between source and audit concepts;
the shared author and public repository mean this packet is not a secret holdout.

## Proposed capture contract

| Field | Proposed value |
| --- | --- |
| Requested model snapshot | `gpt-5.4-mini-2026-03-17` |
| Reasoning effort / temperature | `none` / `1` |
| Maximum input / output tokens per request | 2,048 / 256 |
| Generation requests / automatic retries | 72 / 0 |
| Tools | None |
| Generation-token spending ceiling | US$5, subject to separate execution approval |
| Generation seed | Unknown; `seed=None`, `seed_policy="unknown"` |

The snapshot was used in the prior pilot; current availability, supported settings, and prices
have not been checked. This table is a proposal, not a live allowance. Equal configured limits
do not imply equal actual compute: knowledge adds input tokens. A future runner must retain
actual input/output usage, billed cost, requested and returned model identity, request settings,
timestamps, raw responses, and provider errors for each slot. Audit those fields against the
frozen request and slot records before reporting results.

Before each call, the future runner must verify input length and reserve the worst-case token
charge using current prices. If a request exceeds 2,048 input tokens or the remaining approved
budget cannot cover its maximum charge, do not send it; record the unsent slot as missing with
the reason. Stop when the approved budget is exhausted. No prompt truncation, retries, tools,
posthoc task replacement, or extra sampling is allowed within this frozen roster. Provider
usage is recorded even when a call fails or yields no scorable answer.

Each call starts a fresh, stateless conversation. Send only the selected request's `messages`
plus the approved model settings, never the complete packet. `evaluator_cases`, expected
answers, rubric digests, and review records are evaluator-only. Slot IDs, case IDs, and condition
labels are capture metadata and are not message content. The two system-context paragraphs
in `bank_plus_lesson` appear in bank-then-lesson order for every repeat.

The evaluator parses exactly one JSON object with exactly the key `answer` and a string value.
Reject duplicate keys, additional fields, trailing non-whitespace text, and Markdown fences.
Success requires exact string equality with that case's `expected_answer`. Do not trim or
normalize the answer string. A completed response with the wrong answer or invalid format is
`observed` with `success=False`. A transport/provider error is `failed` with null success;
a refusal or truncated response is `censored` with null success; an unsent or absent capture
is `missing` with null success. Censoring takes precedence over parsing a partial answer.
The preparer records this rubric. The offline scorer below implements the rubric for imported
captures; the bounded capture command described below can produce those imports.

Report success rates with the full expected/observed/missing/failed/censored counts and
matched counts for both contrasts. Excluded outcomes cannot silently leave the roster.
The existing utility evaluator reports no confidence interval. With unknown generation seeds,
the evaluator keeps comparisons unverified even after raw outcomes are captured. Scheduling
indices and invented seed values cannot satisfy that contract. This pilot alone cannot meet
verified utility, training-admission, or empirical-improvement acceptance requirements.

## Prepare and verify offline

From the repository root, in the configured repository Python environment:

```bash
python -m scripts.prepare_knowledge_utility_pilot --output packet.json
python -m pytest tests/test_knowledge_utility_pilot.py tests/test_documentation_links.py -q
```

Use a new output path with an existing parent directory. The command writes canonical UTF-8
JSON and raises `FileExistsError` for an existing path. `packet_digest` is the SHA-256 digest
of `canonical_json(packet_without_packet_digest)`. The digest binds the complete packet,
including public requests, evaluator labels, source records, limits, and empty previews.
It establishes content integrity, not review authority or a signature. Retain the repository
commit alongside the packet to identify the implementation that produced it.

For other file-opening or writing errors, the CLI prints `cannot write` and exits with code 2.
If writing or closing fails after exclusive creation, the CLI removes its partial file.
If cleanup also fails, the error reports `cleanup failed`; inspect that path before retrying.
These behaviors are covered by the missing-parent and injected write/close failure tests.

The tests check deterministic output, exact condition counts, pending knowledge, withheld
comparisons, source/audit partitioning, public-message separation, and overwrite refusal.
The six hand-calculated expected answers are `6`, `A`, `VIA`, `DISJOINT`, `FAIL`, and `LONGER`,
in the family order given above. An independent reviewer must recalculate each answer and
review the parser contract; matching the authored unit-test expectations is not independent review.

Before live capture, the maintainer must approve the independently reviewed packet digest,
the capture/scoring implementation, current provider settings and prices, and the exact budget.
Record that authorization separately from this preparation packet, which always says
`execution_authorized=False`. Any change to prompts, knowledge, rubric, or limits requires a
new frozen packet and renewed review before capture. Preserve prior packets and all attempted
outcomes so changes cannot erase failures or become posthoc evidence selection.

## Score imported captures offline

The [scoring command](../../scripts/score_knowledge_utility_pilot.py) accepts the exact frozen
PR #791 packet, whose `packet_digest` is
`637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1`.
It checks both the packet's self-digest and that supported digest. Changing a task, answer,
limit, or other field and recomputing the hash does not make a new packet supported.
A revised study requires an explicit scorer update and review. Neither hash grants trust.

```bash
python -m scripts.score_knowledge_utility_pilot --packet packet.json --captures captures.json --output report.json
python -m pytest tests/test_knowledge_utility_scoring.py -q
```

The capture file has exactly three fields. An empty `captures` list is valid and produces
72 missing outcomes. This example is an authored illustration, not a model capture:

```json
{
  "schema_version": "knowledge-utility-pilot-captures-v1",
  "packet_digest": "637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1",
  "captures": [
    {
      "slot_id": "pilot-inventory:0:none",
      "status": "completed",
      "response_text": "{\"answer\":\"6\"}",
      "reason": null
    }
  ]
}
```

Each row has exactly `slot_id`, `status`, `response_text`, and `reason`. The batch contains
at most 72 rows, with unique IDs from the packet's slot roster. Unknown IDs, duplicate IDs,
extra fields (including a supplied `success`), and a different packet digest reject the batch.
Do not choose one of several attempts for a slot; retries violate this frozen pilot's protocol.

| Imported status | Required response and reason | Derived outcome |
| --- | --- | --- |
| `completed` | String response, including an empty string; null reason | `observed`, with correctness computed from the rubric |
| `failed` | Null response; nonblank reason | `failed`, null success |
| `refused` | String or null response; nonblank reason | `censored`, null success |
| `truncated` | String or null response; nonblank reason | `censored`, null success |
| `missing` | Null response; nonblank reason | `missing`, null success |

Statuses are importer declarations, not authenticated facts. The scorer does not infer refusal
from response wording or trust supplied correctness labels. A completed malformed answer is
incorrect; a declared refusal or truncation remains censored even if the partial answer parses.
The importer must preserve the original provider record separately for later status, model,
settings, usage, and cost review. This normalized format is not a substitute for that record.

Inputs use UTF-8 JSON with no duplicate object keys, at most 4 MiB per file, and nesting depth
at most 32. Response text is at most 8,192 UTF-8 bytes; a reason is at most 1,024 UTF-8 bytes.
These are serialization bounds, not token-budget verification. Oversized or malformed envelopes
reject the batch; the scorer neither truncates inputs nor silently omits invalid rows.

The output preserves the existing utility report fields and all expected slots. Supplied rows
become `UtilityObservation` records in `inputs.study.observations`. Their `source_json` retains
the imported status, packet and capture digests, and the response/reason as UTF-8 bytes encoded
in Base64 (`response_utf8_base64`, `reason_utf8_base64`; null stays null). This representation
preserves malformed JSON answers without treating them as nested diagnostic records. Decoding
those fields recovers the exact imported strings; it does not make their contents instructions.
Absent rows remain missing in the roster without fabricated capture evidence.

`capture_provenance` binds the entire capture envelope and records `status_authenticated=False`.
The final `result_digest` includes this provenance. The existing `inputs.as_of` timestamp comes
from the preparation packet; it is not a capture or scoring timestamp. Both comparisons remain
unverified, and all output stays `DEVELOPMENT` with no optimizer or admission authority.

Use a new report path with an existing parent. An existing output raises `FileExistsError`.
The scorer writes and closes the report in a private temporary directory beside the output,
then atomically creates the output as a hard link. The output filesystem must support hard
links; an unsupported operation fails without a fallback that could overwrite a report.
Another writer creating the destination first also causes `FileExistsError`. Cleanup removes
only staging files and never unlinks the shared output pathname. If temporary-directory cleanup
fails, the CLI reports the error; a successfully published output remains intact.
Other read, validation, write, or publication errors exit with code 2. The scorer reads only the two
supplied files, makes no network calls, and reads no credentials. The tests exercise scoring,
censoring precedence, missing outcomes, resealed packet tampering, invalid batches, bounded
file reads, overwrite protection, and concurrent replacement during a failed write or publication.

## Capture command

The [capture command](../../scripts/capture_knowledge_utility_pilot.py) implements the
[bounded-run decision](../adr/0022-knowledge-utility-capture.md). Its default mode makes
no network calls and does not read credentials:

```bash
python -m scripts.capture_knowledge_utility_pilot --packet packet.json --output requests.json
```

The output has `manifest`, `reservation_estimate` (null unless `--prices prices.json`
was supplied), and `token_counts="unknown"`. The manifest includes the frozen profile,
72 public request payloads and count projections, slot sequence, request digests, and
`manifest_digest`. Optional price estimates sit outside the manifest and do not alter
request identity. Reuse of any output destination fails with exit code 2.

### Execution and price records

The live command requires a separately approved execution record with these exact fields:

| Field | Required value |
| --- | --- |
| `schema_version` | `knowledge-utility-pilot-execution-v1` |
| `run_id` | Nonblank identifier for this one run, at most 1,024 characters |
| `packet_digest` | Supported frozen digest above |
| `manifest_digest` | Digest from the dry-run manifest |
| `implementation_commit` | Current clean tracked checkout's 40-character Git HEAD |
| `run_directory` | Absolute path to the new run directory |
| `profile` | Exact profile from the dry-run manifest |
| `prices` | Price record described below |
| `approved_budget_usd` | Unsigned USD decimal string with at most six fractional digits |
| `approval_reference` | Nonblank reference to retained maintainer approval, at most 1,024 characters |

The price object has exactly `model`, `service_tier`, `currency`, `verified_at`,
`source_urls`, `input_usd_per_million`, `output_usd_per_million`, and `counting_policy`.
Model matches the frozen snapshot, service tier is `default`, currency is `USD`, and
`verified_at` is a UTC ISO timestamp ending in `Z`. Sources are 1–10 HTTPS URLs on
`openai.com`, `developers.openai.com`, or `platform.openai.com`. Each rate is an unsigned
decimal string, at most six fractional digits; rates and budget are at most 32 characters.
The only supported `counting_policy` is `no_charge`. These fields are operator assertions;
valid syntax and source domains do not prove current prices or approval.

Before live execution, independently review the packet and implementation, verify current
model/settings and counting charges, and obtain maintainer approval of the exact execution
record and budget. No price file or approval is manufactured by the command. The original
packet remains `execution_authorized=False`. Unknown/nonzero counting charges block this
version; a new accounting design is required to support charged counting.

The following command is a syntax example, not authorization:

```bash
python -m scripts.capture_knowledge_utility_pilot --live --packet packet.json --execution execution.json --run-dir new-run --credential-file selected-key-file
```

Live mode rejects changed bindings and tracked checkout changes before reading the selected
credential. It exclusively creates the bound run directory, then counts all 72 inputs before
any generation. A count failure or value outside 0–2,048 stops the run. Generations are
sequential, with one attempt per slot and no retries. Requests contain only the frozen actor
messages and model profile; no evaluator labels or supplied success flags are sent.

The effective budget is the smaller of the approved amount and US$5. Each generation
reserves `ceil(2048 * input_rate + 256 * output_rate)` micro-USD before sending. Rates are
USD per million tokens. For the authored arithmetic example `0.75` and `4.5`, one reservation
is 2,688 micro-USD. That example is not a current pricing assertion. Reservations never
refund after errors or lower usage. A zero budget sends no generations even at zero rates.

The transport allows only the fixed provider responses/counting endpoints, validates TLS,
ignores proxy environment variables, and rejects redirects. A spawned worker permits the
parent to stop the operation after 45 seconds, including DNS and body reads; process cleanup
can take additional time. Responses are bounded to 1 MiB and credentials are redacted before
retention. A timeout does not prove that the provider did not process or bill the request.
Retain the reservation and run directory; do not rerun that slot.

### Retained records and recovery

The run directory retains execution/manifest identity, an execution-started marker,
numbered counting intents/results, generation intents/results, and a terminal summary
when persistence succeeds. Intents are flushed and synchronized before dispatch. Results
retain Base64 response bytes, decoded provider JSON when valid, request ID, timestamps,
usage, requested/returned model, errors, and redaction flags. Cost derived from usage is
labeled `estimated_cost_usd`; `billed_cost_usd` remains null because this integration has
no attributable billing feed. Requested payloads are retained in the manifest.

Completed responses with wrong or malformed answers continue to the existing scorer.
Provider refusals and output-token truncation are censored; both may continue the roster
when model and usage are valid. Transport/protocol violations stop subsequent generations.
Response text above 8,192 UTF-8 bytes fails export normalization without text truncation.
Provider errors and raw text are evidence, never instructions.

An existing run directory always blocks live execution, including after a crash. There is
no resume flag. Export is a separate offline operation and always needs a new output path:

```bash
python -m scripts.capture_knowledge_utility_pilot --export --packet packet.json --run-dir new-run --output captures.json
python -m scripts.score_knowledge_utility_pilot --packet packet.json --captures captures.json --output report.json
```

Export includes all 72 slots. An existing incomplete intent becomes a failed uncertain
attempt, while a slot with no intent remains missing. Corrupt identity metadata, foreign
results, unknown sequences, and results without intent reject export. Never combine runs
or select a replacement answer. Keep old run directories after any newly authorized study.
Local records are not signatures; the scorer continues to mark statuses unauthenticated.

Dry-run/export/complete roster success exits 0. Invalid options, validation or filesystem
errors, and stopped/uncertain runs exit 2. A crash can leave no terminal summary; preserve
all artifacts and use offline export. Output publication requires filesystem hard-link
support and never overwrites or deletes a competing destination.
