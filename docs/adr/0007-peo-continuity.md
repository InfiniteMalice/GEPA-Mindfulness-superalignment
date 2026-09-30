# ADR 0007: Bind public evidence-use observations to PEO chronology

Status: accepted for experimental explicit use; disabled by default.

## Context

Existing continuity checks distinguish retained and omitted commitments and verified updates.
Retention alone cannot establish appropriate influence on a prediction or executed action.

## Decision

Capture prospective host observations in the existing committed prediction and retrospective
observations in a verification-linked epistemic assessment after reconciliation. Bind both to
the original commitment and PR-5 evidence-use snapshot. Reuse PEO validation and continuity's
verified update rules. Preserve unknown stages and historical failures in the resulting audit.

## Alternatives and consequences

A second event stream would duplicate causal identities. Retrospective reconstruction of all
stages would permit later reports to overwrite prospective evidence. Scoring text presence or
internal-state geometry would not establish actual evidence use.

Hosts must instrument and independently validate the observations. Digests detect rebinding,
not dishonest declarations. The API reports diagnostics only; it cannot infer motive, grant
authority, or prove causality. Missing telemetry yields an unresolved result. Keeping an earlier
failure as the primary classification requires callers to inspect the included continuity result
for a later legitimate update. See the [contract and tests](../peo_continuity.md).
