# ADR 0008: bounded System-One proposals from epistemic state

Status: experimental; disabled by default. Date: 2026-10-01.

## Context

The legacy factuality router consumes confidence, risk and verification budget. PRs 1–6 add
action-bound epistemic state and evidence continuity. Routing needs access to that state
without converting confidence or speed into execution authority.

## Decision

Add an opt-in adapter beside the legacy router. Validate a complete chronological PEO prefix
ending at the proposed action. Reuse its latest compatible reconciliation and typed relational
evidence. Convert host thresholds and legacy constraints into a bounded action set before
calling a named/versioned backend. Validate the backend's proposal against the untouched
snapshot and return a serializable diagnostic plus the existing `RoutingDecision` shape.

Use the same frozen inputs for candidate comparisons. Report raw routing errors separately
from guarded outcomes and callback latency. Ship the existing router and uncertainty rules
as baselines. Hosts can supply JEV, CLM or classifier callbacks without replacing policy or
adding SDK dependencies to this package. There is no automatic backend promotion.

## Alternatives and consequences

Changing the default pipeline would affect existing evaluations; explicit opt-in preserves
those callers. Selecting JEV or CLM without a matched workload benchmark would exceed the
available evidence. A generic tool execution framework would duplicate runtime governance
and expand this stage beyond routing, so existing action enums and authority gates are reused.

The host remains responsible for source authenticity, relevance, complete input history and
callback timeouts. A conservative gate can cause additional verification or abstention.
Unknown estimator channels do not enable continuation. Synthetic comparisons test software
contracts; actual backend choice requires a held-out host workload benchmark.

Implementation, precedence, runnable checks and research limitations are in the
[System-One guide](../system_one_routing.md). The [research registry](../recommendations/RESEARCH_TRACEABILITY.md#ref-jev-mem)
links all six sources to REC-008. Acceptance tests cover chronology, metadata binding, expired
evidence, policy overrides, callback errors, and matched benchmark inputs. No case, reward,
event-schema, persistence or authority contract changes.
