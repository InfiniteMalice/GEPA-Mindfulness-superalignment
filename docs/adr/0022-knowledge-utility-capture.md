# ADR 0022: Bounded knowledge-utility pilot capture

Status: design and native implementation approved 2026-10-12; counting-policy amendment
and one frozen pilot run with a US$0.20 generation-only budget approved 2026-10-12.

## Context and decision

The [pilot protocol](../experiments/knowledge_utility_pilot.md) freezes 72 requests and
an offline scoring contract. Add a pilot-specific runner with separate record and HTTPS
helpers. The simulator wrapper remains a simulator. A general provider framework would
expand scope; manual imports alone cannot enforce attempted-request accounting.

Default mode is offline. Live mode binds one run directory, commit, request manifest,
price assertion and budget to a retained operator execution record. That record does not
authenticate approval. A live run reserves worst-case generation charges before dispatch, keeps
uncertain reservations, and never resumes or retries. Export preserves all expected slots.

The transport uses one spawned process per HTTP operation. A parent deadline covers DNS,
TLS, headers and response reads. The worker cannot follow redirects or use proxy environment
variables. Only redacted, bounded responses cross back to the parent. Terminating a worker
cannot cancel a provider operation already accepted; its reservation therefore remains.

## Provider compatibility observations

Documentation inspected on 2026-10-12, without credential access or provider requests:

- The [model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini) lists the
  frozen snapshot, Responses support, and reasoning effort `none`.
- The [count reference](https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count)
  documents `POST /v1/responses/input_tokens`, model/input/reasoning/tools parameters, and
  an integer `input_tokens` with object type `response.input_tokens`.
- The [response reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
  documents the generation profile fields and returned usage/output structures.
- The [token guide](https://developers.openai.com/api/docs/guides/token-counting) explains
  that output usage includes non-visible formatting tokens. All reported output tokens
  count toward the 256-token bound.

These observations do not establish account-specific availability or a no-charge counting
policy. No current price record or live authorization is created by implementation.
Before live execution, the operator must verify settings, supply current generation prices,
and retain the maintainer's approval. Counting charges require either a supported no-charge
assertion or explicit acceptance of unverified charges as described below.

## Counting-policy amendment (2026-10-12)

The maintainer explicitly accepted unknown counting charges and a US$0.20 limit on generation
costs only, rather than on the total bill. Support `counting_policy="unverified_accepted"`
alongside `no_charge`. Merely unknown or unsupported policy values still fail validation.
The operator must retain approval of this policy in the execution record's approval reference.
This is an operator assertion, not authentication or evidence that counting is free.

Generation reservations and the 72-request roster remain unchanged. Counting calls remain
sequential, at most one per slot, with no retries. Under `unverified_accepted`, counting costs
and the total billed cost remain unknown; no upper bound on the total bill is claimed.
Dry-run estimates and terminal summaries identify the budget as generation-only. Counting
responses never receive a generation-rate cost estimate, even if they contain usage fields.

## Consequences and validation

Local exclusive ownership prevents concurrent reuse of one run path. It cannot prevent
an operator deleting evidence or fabricating a new approval. Preserve all run directories.
File synchronization protects dispatched intents against ordinary process interruption;
the directory guard and retained records are not a distributed transaction with the provider.

Tests simulate provider behavior, deadlines, concurrent ownership, budget boundaries and
interrupted persistence. No test result from this runner grants curation, verified utility,
training admission, or live execution authority.
