# ADR 0025: Bounded follow-up capture and offline recovery

Status: Accepted for tooling implementation. Direction, design, plan, and native implementation
approved on 2026-10-15. This approval does not authorize credential access, provider requests,
or spending; a concrete live run requires separate execution approval.

## Context

[ADR 0024](0024-knowledge-utility-followup-freeze.md) registers one accepted frozen 48-task,
384-slot packet for offline scoring. The original pilot runner supports a different packet and
roster. Follow-up execution needs its own exact request identity, bounded approval record,
one-shot dispatch, and recovery without weakening the existing scorer's authority boundaries.

## Decision

Add a follow-up record module and capture runner. Default CLI execution prepares requests offline;
explicit live mode validates the frozen packet, manifest, clean tracked implementation commit,
absolute run directory, provider compatibility assertions, fresh prices, request caps and budget
before claiming a new run directory or reading the selected credential file. Exact schemas and
commands are defined in the [protocol](../experiments/knowledge_utility_followup.md#bounded-follow-up-capture).

Complete all 384 counts before generation. Dispatch serially, reserve worst-case generation
cost before each durable intent, and never retry or resume. Count failure prevents generation;
generation failure stops subsequent dispatch. Refusals and token-limit truncations are retained
and permit later slots. Fsync each exclusive record write and preserve partial artifacts on failure.

Offline export validates the retained ordered history and derives outcomes from bounded transport
bytes. It rejects conflicting complete records, altered projections, gaps, and linked artifacts.
An interrupted final record may become an uncertain observation. Export never mutates the source
run, dispatches a request, or treats terminal summaries as authority over durable evidence.

Reuse public pilot transport, normalization, exact-money, and exclusive publication helpers.
Keep pilot behavior, scoring arithmetic, packet content and scoring schemas unchanged.

## Consequences and verification

Reservations and usage estimates cover generation only. Counting charges are either declared
zero with supporting operator evidence or explicitly accepted as unknown. Retained usage does
not price counting requests at generation rates. Billed costs remain unknown without billing data.

Approval references and hashes bind declarations; they do not authenticate human approval,
provider compatibility, billing, reviewer independence, or study provenance. The operator verifies
the concrete approval before running live mode. Exported reports remain DEVELOPMENT, unverified,
unpaired, and ineligible to confer authority or optimizer input. Their execution-authorized flag
stays false because scoring does not authenticate the separately retained live declaration.

Tests cover exact projection and approval binding, freshness and money boundaries, durable intent
ordering, all 384 fake calls per phase, interruption, failed writes, strict recovery, CLI isolation,
exclusive publication, unchanged draft report goldens, and documentation links. Platform link tests
skip explicitly when the operating system denies link creation. No test requires real credentials
or provider calls. Operational execution remains a later separately approved action.
