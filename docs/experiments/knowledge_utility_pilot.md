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
The capture runner and executable scorer are future work; the preparer only records this rubric.

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
