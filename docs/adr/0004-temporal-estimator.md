# ADR 0004: opt-in scalar temporal uncertainty estimation

Status: accepted for experimental diagnostics, disabled by default.

## Context

PR-1 supplies typed epistemic records and PR-2 validates causal outcome bindings. Neither computes
a temporal estimate. The next program stage needs a small numerical estimator with explicit
model-mismatch behavior; fixed Q/R covariance alone cannot respond to residual magnitude.

## Decision

Add a separately imported scalar random-walk estimator using existing records and reconciliation.
Require explicit enablement, a declared independent-noise model, a matching source/provenance
contract and a scalar representation with covariance. Use Joseph covariance updates on inliers;
reject outlier mean assimilation, adapt bounded Q/R, and expose mismatch/regime/insufficient
statuses. Insufficient-model state latches until the host explicitly initializes a new instance.

Validate the complete causal event sequence and finite arithmetic before committing memory state.
Retain the accepted history prefix digest and used identities to prevent replay. Return a detached
estimate, event, configuration snapshot and numerical diagnostics. Do not add runtime hooks,
logging, reward changes, action selection or persistence authority.

## Alternatives

Full matrix/diagonal models add covariance and correspondence complexity before there is a
validated semantic representation. A scalar channel keeps units and numerical assumptions
inspectable. Fixed-noise filtering alone can remain numerically confident under model mismatch.
Automatically replacing confidence fusion would change deployed behavior before calibration.

## Consequences and evidence

Gating can reject genuine changes and adaptation can eventually admit mismatches. Status thresholds
are heuristics, not calibrated change-point probabilities. Scalar covariance does not capture
unknown correlated sources; PR-4 owns multi-source fusion. Full-history validation costs grow with
trajectory length; callers serialize calls and retain the returned configuration for replay.

The [guide](../temporal_estimator.md) documents equations, defaults, reproducible matched/shift
comparison and validation commands. The [research register](../recommendations/RESEARCH_TRACEABILITY.md#ref-kalman)
separates published results from repository hypotheses. Tests cover numerical correctness, causal
binding, atomic failure, replay and legacy compatibility. Held-out calibration remains future work.
