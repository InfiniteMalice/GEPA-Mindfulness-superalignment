# Knowledge-utility pilot capture runner

Status: proposed written specification; conversational direction approved on 2026-10-12.
Written-spec review, implementation planning, and live execution remain separate stages.

## Purpose and boundaries

The [frozen pilot protocol](../docs/experiments/knowledge_utility_pilot.md) has an
offline preparer and [scorer](../scripts/score_knowledge_utility_pilot.py).
It needs a bounded runner that
collects provider responses without changing the experiment or hiding failed attempts.
The baseline is manual capture import. The intended improvement is reproducible request
construction, conservative spending accounting, and traceable capture export.

This specification adds no reward, penalty, optimizer input, curation authority, or
training eligibility. All evidence remains `DEVELOPMENT`. Unknown generation seeds and
unreviewed semantic dependencies continue to withhold verified utility comparisons.
The existing incorrect-answer rubric remains unchanged.

The scope is one command, focused tests, and protocol documentation. Proposed files are
`scripts/capture_knowledge_utility_pilot.py`, `tests/test_knowledge_utility_capture.py`,
and an update to `docs/experiments/knowledge_utility_pilot.md`. Small local helpers are
acceptable if they keep transport, accounting, and export independently testable.
No general provider framework, wrapper rewrite, new dependency, or automatic resumption
is included. Implementation will record the accepted architecture in an ADR.

## Selected approach

Adapt the earlier pilot's explicit HTTP transport and durable reservation approach into
the repository command. The earlier runner is a reference, not a dependency or source
of current prices and execution authority. Its action parser and 84-generation roster
do not apply to this pilot.

The existing OpenAI wrapper is a simulator; turning it into a live provider would expand
the change beyond this experiment. Manual capture assembly would retain the current
scorer interface but would leave request identity and spending controls outside the
tested runner. A focused command provides those controls with fewer affected interfaces.

## Frozen input and command modes

The runner supports only packet digest
`637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1`.
It validates the self-digest and the supported digest before creating a live run.
Resealing changed prompts, labels, limits, or knowledge does not make a packet supported.
The input decoder rejects duplicate keys, invalid UTF-8, depth above 32, and files above
4 MiB, consistent with the scorer. Input paths and provider text are data, never commands.

Default mode is offline dry-run. It validates the packet and writes a new local request
manifest containing all 72 slots, request payloads and digests, limits, and price-dependent
reservation estimates if a price record was supplied. It does not read credentials,
construct a live transport, count tokens remotely, or claim that token counts are known.

Live mode requires an explicit `--live` option, a reviewed execution record, and a new
run directory. The execution record binds the packet digest, request-manifest digest,
implementation commit, unique run identifier, absolute run directory, model settings,
price record, and approved USD ceiling. Missing or mismatched bindings stop before any
network request. A serialized approval field is only an operator assertion: the operator
must retain the maintainer's actual approval outside the packet. The command does not
authenticate people or create review authority.

An offline export mode reads an existing run's retained artifacts and produces the
scorer envelope at a new destination. It never resumes requests or accesses credentials.
Dry-run and export cannot be combined with live mode. CLI parsing rejects ambiguous modes.

## Request projection

The runner uses the packet's actor-request order, once per slot. Each generation starts
a stateless conversation. The request contains exactly the selected `messages` as its
input plus this reviewed transport profile:

| Setting | Proposed value |
| --- | --- |
| Endpoint | `https://api.openai.com/v1/responses` |
| Model | `gpt-5.4-mini-2026-03-17` |
| Reasoning effort | `none` |
| Temperature | `1` |
| Maximum output tokens | `256` |
| Tools | Empty list |
| Storage | Disabled |
| Service tier | `default` |
| Truncation | Disabled |
| Streaming | Disabled |
| Generation seed | Omitted; remains unknown |

The implementation must verify this profile against official provider documentation
before live approval. Unsupported settings stop preparation for live execution; they
must not be silently omitted or substituted. Availability and current prices are not
asserted by this specification.

Do not add a provider JSON-schema constraint, JSON mode, extra instruction, conversation
history, or previous-response reference. The frozen prompt already requests the `answer`
object. The scorer, not the transport, judges answer format and correctness.
Evaluator cases, expected answers, rubric records, slot IDs, condition labels, and review
metadata are not sent as message content or provider metadata. Bank-plus-lesson context
preserves the frozen bank-then-lesson order.

## Token and spending limits

Live execution permits at most 72 token-count requests and 72 generation attempts, with
zero automatic retries for either operation. Each slot has one possible generation
attempt. Generations are sequential; no background workers or SDK retry defaults apply.

Before any generation, the runner counts input tokens for every projected request using
the provider's documented counting endpoint. The count projection must include every
token-bearing input from the generation request. Save the exact counting request and
result. A count is an integer from 0 through 2,048; booleans are invalid.
A failed count, unsupported counting profile, invalid count, or count above
2,048 stops the run before generation. The runner never truncates or shortens messages.

The price record contains model and service tier, currency, verification timestamp,
official source URLs, input/output rates, and the counting endpoint's charging policy.
The operator verifies those terms immediately before approving the live run. This version
permits counting only when the verified policy has no charge; unknown or nonzero counting
charges block live execution until the accounting design is revised.

Use decimal rates and integer micro-USD reservations, never binary floating-point money.
For each generation reserve the ceiling, in micro-USD, of the cost of 2,048 uncached input
tokens plus 256 output tokens at the approved rates. Do not assume cache discounts.
The effective ceiling is the smaller of the approved budget and the packet's US$5 ceiling.
Budget and rate fields are unsigned decimal strings without exponent notation, with at
most six fractional digits. Rates express USD per million tokens; the budget expresses
USD. Reject negative, nonfinite, malformed, or over-precision inputs. A zero budget
permits no generation. With input and output rates I and O, the per-call reservation is
`ceil(2048 * I + 256 * O)` micro-USD. The budget converts exactly to integer micro-USD.

Before sending, the runner durably records the slot, attempt number 1, exact request digest,
reservation, cumulative reservation, and UTC timestamp. If the next reservation would exceed
the ceiling, record that slot and all remaining slots as unsent and stop. Failed, refused,
truncated, timed-out, and uncertain attempts retain their full reservations. This version
never refunds reservations or spends savings from lower observed usage.

Retain actual provider usage separately. Compute a usage-based cost estimate from verified
rates, label it as an estimate, and leave billed cost unknown unless the provider supplies
an attributable billing value. Do not substitute estimates for invoices. Unexpected model
identity, missing/invalid usage, or usage outside the configured bounds stops subsequent
requests and is recorded as a protocol violation. Preserve any returned usage even then.
The local reservation ceiling is a request policy, not a guarantee about provider billing.

## Run ownership, durability, and interruption

Before transport creation or credential access, acquire the execution record's run
directory through exclusive directory creation. An existing directory, including an
incomplete run, blocks execution. The runner writes the immutable execution record and
an execution-started marker before token counting. Changing the directory invalidates
the execution binding; the command has no override or automatic recovery flag.

Within the exclusively owned directory, use numbered, exclusive-create attempt records
and result records. Use sequence numbers as filenames, never unsanitized slot IDs. Flush
and synchronize each intent/reservation record before its corresponding network call.
Any write, close, or synchronization failure stops execution. A request whose result could
not be persisted remains uncertain, with its reservation retained. No exception handler
may delete a run, intent, reservation, or published result to permit another attempt.

Suggested retained layout is `execution.json`, `requests.json`, `counts/`, `attempts/`,
`results/`, and a terminal summary. Each artifact binds the run and packet identities.
The terminal summary distinguishes finished roster, stopped run, and uncertain attempt.
A crash may prevent a summary; offline export derives uncertainty from durable intent
records that lack complete results. An incomplete final JSON record is evidence of an
interruption, not permission to send another request. Fixed sequence-to-slot mappings
allow an incomplete attempt record to identify an uncertain slot. If corrupt or missing
run metadata prevents that mapping, export fails explicitly and preserves the run;
it cannot relabel an unknown attempt as unsent.

A second process targeting the same directory fails before any network call. The operator
must retain the directory and approval record; deleting or copying those records defeats
the local guard. This command is not a global authorization service. A subsequent live
run requires a new explicit authorization and must retain the original run as evidence;
it cannot supply replacement attempts for this frozen capture batch.

## Transport and retained evidence

Use a transport interface injectable in tests. The live implementation uses HTTPS with
certificate verification, a fixed provider host, no redirects, no environment-derived
proxies, and no automatic retries. Only the documented responses and counting endpoints
are allowed. Bound each operation with a 45-second total deadline and response reads to
1 MiB. Do not poll a pending response or follow provider-supplied URLs.

Read the API key from the explicitly selected local credential source only after live
validation and run ownership succeed. The key appears only in the authorization header.
Never retain that header, the credential file contents, or an unrestricted exception
string. Error records use controlled reason codes and bounded, redacted provider details.
Tests inject sentinel credentials and check every artifact and diagnostic for leakage.

Retain request payload/digest, start/end UTC timestamps, elapsed duration, requested and
returned model identity, HTTP status, provider request ID from an allowlisted header,
bounded response bytes, decoded response when valid, usage, reservation, and error state.
If a response exceeds the bound, retain the bounded prefix marked incomplete and stop.
If credential redaction modifies a response, record that redaction occurred; never claim
the stored response is byte-identical. Malformed JSON, unknown structures, incomplete
records, and provider text remain untrusted evidence.

## Deterministic capture export

Export `knowledge-utility-pilot-captures-v1` using the existing scorer's exact envelope
and row fields. Include all 72 slots in frozen order. Do not add success labels or mix
multiple runs. Keep provider evidence in the run directory because the scorer envelope
does not contain provider usage or billing fields.

Apply the following rules in order. Unsupported or malformed transport records stop live
execution; export preserves the reason instead of selecting a plausible answer.

| Evidence | Export status and content |
| --- | --- |
| No generation intent | `missing`, null response, controlled unsent reason |
| Intent without a complete result, transport error, invalid envelope, or protocol violation | `failed`, null response, controlled reason |
| Valid provider refusal content | `refused`, retained output text if present, refusal reason |
| Valid incomplete response due to the output-token limit | `truncated`, retained output text if present, truncation reason |
| Completed response containing exactly one output-text part | `completed`, exact text, null reason |
| Completed response with no output text and no refusal or unsupported output | `completed`, empty string, null reason |
| Any other status or unsupported output structure | `failed`, null response, controlled reason |

Refusal takes precedence over token-limit truncation. Neither requires parsing an answer.
Ordinary refusal wording inside completed output text remains a completed response for
the scorer to judge. Documented non-text reasoning items may coexist with one text part;
tool calls, multiple output-text parts, and unknown output item types are unsupported.
Valid completed/refused/truncated responses with valid usage permit the next slot;
failed or protocol-violating responses stop subsequent generations.

Response text above 8,192 UTF-8 bytes cannot be exported as completed or censored text:
retain the bounded provider evidence, export `failed` with an explicit size reason, and
stop subsequent calls. Export reasons are controlled strings below 1,024 UTF-8 bytes.
Never truncate text to make an answer fit. A malformed or incorrect answer within a
completed response is passed unchanged to the existing scorer and counts as incorrect.

Publish the normalized envelope only after validating the whole export with the existing
scorer contract. Use private staging and atomic no-overwrite publication, as the scorer
does. An existing destination or unsupported atomic publication fails without removing
another writer's file. Recovery export always targets a new path. Imported statuses stay
unauthenticated in the scorer; retained local evidence is not a signature.

## Verification and acceptance

Implementation acceptance requires the following offline evidence. This specification
does not claim that those future tests have run.

| Requirement group | Verification |
| --- | --- |
| Packet and modes | Tampered/resealed packet, malformed inputs, conflicting modes, and mismatched execution bindings reject before transport or credential access. Dry-run/export use a credential and network spy that fails if called. |
| Request projection | Capture fake-transport payloads for all 72 slots; compare exact messages/order/settings and assert evaluator metadata is absent. |
| Counting | Exercise 0, 2,048, 2,049, boolean, missing, malformed, and failed counts; all counts finish before the first generation; failures produce zero generations. |
| Accounting | Independently calculate decimal reservation examples, rounding boundaries, exact-budget and one-micro-USD-short cases; failure/uncertainty cannot release reservations. |
| Durability and ownership | Inject failures before/after intent synchronization and after response receipt; race two processes for one run directory; restart attempts produce zero additional calls. |
| Transport | Simulate redirects, timeout/deadline, oversized bodies, HTTP errors, model mismatch, missing/out-of-range usage, and credential-bearing diagnostics; assert bounded retention and no retries. |
| Export | Exercise every mapping above, multiple text parts, partial valid answers, Unicode byte limits, missing results, corrupt records, and competing destination creation. |
| End-to-end | Fake all 72 provider results, export, and run the existing scorer; verify the full roster, exact correctness outcomes, retained failures, and unverified comparisons. |
| Scope | Review the diff for unchanged prompts, rubric, rewards, training admission, curation, and existing simulator behavior. |
| Documentation | Check links, literal digests/settings, mode descriptions, accounting examples, and failure behavior against code and tests; resolve all precision blockers. |

Before requesting live approval, independently review the packet's authored answers,
proposed lesson/bank, rubric, and source/audit overlap. Retain the review results without
promoting pending knowledge or unknown seed policy. Verify current model availability,
settings, endpoint behavior, prices, and credential provisioning against official provider
documentation. Present the exact frozen execution record, limits, and cost calculation
for the maintainer's approval. No historical pilot allowance authorizes this run.

Written-spec approval permits preparation of the implementation plan. Implementation
approval and live-execution approval remain distinct. The eventual implementation plan
must include focused tests, repository checks, a diff review, and the protocol/ADR updates.

## Specification review record

This document was checked against the current pilot protocol and scorer at base commit
`084d93035e82c8abc09872314dc3758c78aadab0`. The review checked fixed roster/digest,
actor/evaluator separation, status precedence, size limits, no-overwrite publication,
spending uncertainty, and withheld authority. Provider compatibility and current prices
remain explicit preconditions for live approval, not verified facts in this document.
The documentation-only change is validated by literal-contract and local-link checks,
`git diff --check`, and a scope review. Product behavior is unchanged at this stage.
