# ADR 0018: Propose semantic inquiry within host constraints

Status: Accepted for experimental opt-in diagnostics.

## Context

Existing uncertainty routing identifies broad continuation/retrieval needs. The information-gain
overlay describes a question, while hypothesis history retains unresolved alternatives. None selects
among semantic exploration distances under explicit gain, stakes, reversibility and compute constraints.

## Decision

Add a pure selector over bounded host-supplied SEARCH/DEBATE/SPARK candidates. Reuse the existing
inquiry flag, InformationGainQuestion, observable provenance, epistemic context and non-TRAIN labels.
Keep world/model/monitor uncertainty separate. Use evidence gap and hypothesis diversity to permit
breadth, then apply host distance caps, stakes, candidate reversibility and integer compute bounds.
Unknown monitor diagnostics defer selection; high monitor uncertainty prioritizes monitor inquiry.
Rank eligible candidates deterministically by expected gain, cost, distance and ID.

## Consequences

This adds interpretable proposals with no runtime authority or reward change. Hosts supply calibrated
estimates and public text, retain evidence, and enforce execution budgets and authority. No callback,
generator, tool execution, storage or budget reservation is added. Thresholds and ranking are local
experimental choices; paper mechanisms motivate the design but do not establish its effectiveness.
Contract tests cover input boundaries and matched selection changes; model evaluation remains future work.

The [guide](../semantic_exploration.md) specifies the rules and host responsibilities.
