# PR-3: experimental scalar temporal estimator

## Scope and reconciliation

User requested the next stage after PR-2 merged. Base main: 55b5027. The two attached program
specifications authorize autonomous implementation of the earliest missing stage. Superpowers
and repo-quality-gate apply; no Beads commands or tracking files are used.

Current state: PR-1 provides diagnostic state/measurement/innovation/update records; PR-2 provides
causal reconciliation and immutable action-bound events. There is no temporal estimator. Existing
confidence fusion and all 17 canonical cases remain unchanged. PR-4 multi-source fusion is deferred.

## Design and task contract

Implement one explicitly configured scalar random-walk estimator, with identity observation model.
Use a one-dimensional DiagonalState and one existing EpistemicMeasurement per step. Reject unsupported
dimensions, missing variance, nonfinite values, Boolean numbers, representation/source mismatches
and replay. Enabling requires a declared independent-noise assumption and a named measurement
provenance contract. That declaration does not establish semantic linearity, Gaussianity or truth.

The state holds an immutable estimate plus adaptive process noise, measurement-noise scale and
consecutive outlier count. Default configuration is disabled; reconcile returns None without work.
The host explicitly calls reconcile with an earlier event prefix, measurement, existing outcome
binding, prediction/observation/verifier event IDs, new update/estimate/event IDs and timestamp.
The committed predicted number must equal the estimator's current scalar mean. Reuse PR-2 to
validate the complete resulting reconciliation before changing any in-memory state. Return the
estimate and reconciliation diagnostics/event; never write logs, persist state, choose actions,
grant authority or provide optimizer credit.

Accepted update: Pminus=P+Q, S=Pminus+R_effective, K=Pminus/S; use a convex mean update and Joseph
covariance Pplus=(1-K)^2*Pminus+K^2*R_effective. Require finite, nonnegative results. Gate on
abs(residual/sqrt(S)). Gate failure rejects mean assimilation, increases bounded Q and R scale,
and retains/increases covariance through the explicit Q inflation. Consecutive failures produce
MODEL_MISMATCH, REGIME_SHIFT_SUSPECTED and then latched INSUFFICIENT_MODEL. An ordinary inlier
resets the streak and decays adaptive noise toward its baseline before the next step. A latched
insufficient model requires a new explicitly initialized instance. Thresholds are heuristics.

Innovation normalization uses the pre-adaptation S tested by the gate. World and monitor
uncertainty use variance/(variance+configured_scale), as bounded diagnostics; model uncertainty
is null because a mismatch status is not a calibrated probability. Configuration is serialized
and hashed into producer provenance; result diagnostics retain effective noises, gain and count.

After each accepted reconciliation, retain a digest and length of its full event prefix. Subsequent
calls must retain that prefix and commit a new prediction after it in stream order and no earlier
by timestamp. Reject reused measurement/evidence/estimate identities and previously observed outcomes.
This guards accidental repeated assimilation; unknown physical correlations still need host review
and the later PR-4 fusion stage. Failed validation or arithmetic leaves every state field unchanged.

## Alternatives and boundaries

Choose scalar over diagonal/matrix machinery because current causal bindings are scalar and it
makes covariance assumptions inspectable. Choose explicit rejection/noise adaptation over silently
absorbing a large innovation. Fixed Q/R covariance does not react to residual magnitude by itself.
Do not add entropy, trajectory extraction, control selection or a model backend in this PR.

## Implementation and verification

Write failing deterministic tests before implementation. Cover disabled behavior, hand-computed
Joseph updates, fixed-noise residual invariance, outlier variance inflation, persistent shift and
insufficient-model status, noise recovery, temporal replay, evidence/source/context mismatches,
invalid numeric configuration, scalar shape limits, overflow and state rollback.

Add a reproducible synthetic matched/shift trajectory comparison and report before/after numerical
evidence without claiming semantic calibration. Update guide, ADR and REC-002 research traceability
for Kalman, GRUET, Dual-Frontier, DEEPO and existing C3-JEPA. Run focused and full tests, Ruff/Black,
mypy, wheel smoke outside checkout, and a fresh independent whole-branch review before opening PR-3.

## Review focus

Check finite arithmetic and covariance positivity, pre/post-adaptation innovation meaning,
causal history and atomic state changes, measurement/evidence replay, current/historical prior IDs,
confidence normalization versus statistical claims, source-correlation assumptions, default-off
behavior, and that estimator results grant no action, persistence or reward authority.

## Execution evidence

Results and review decisions will be recorded after running the checks.
