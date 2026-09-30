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

Initial RED: the new estimator test module failed to import before implementation. After
implementation, 45 tests passed; expanded causal/numeric/snapshot checks raised this to 60.
The five-step synthetic comparison is recorded in docs/temporal_estimator.md with its command.

Fresh whole-branch review: review_pr3 inspected 55b5027..5e71412 and relevant dependencies/specs.
Two reproduced findings entered one fix pass:

- Missing explicit verifier binding allowed TOOL_RESULT assimilation despite a failed listed
  verifier. The two cases in test_estimator_requires_explicit_verifier_binding_before_assimilation
  failed before the fix. Require a non-null binding and reuse PR-2's successful-verification checks.
- Saturated Q growth could underflow to zero. The public-API test
  test_saturated_process_noise_cannot_underflow_to_zero_on_outlier failed before the fix.
  Return the cap directly on saturation instead of multiplying an underflowed quotient.

Final: Ruling: regrade noise-cap underflow from Minor to Important because a valid configuration
could produce zero covariance on an outlier despite positive Q/floor/cap. Include it in the same
fix pass. If the numerical boundary is wrong, the estimator can silently report false confidence.

Both regressions passed after the fix; the focused state/reconciliation/estimator/research suite
passed 270 tests. No second reviewer was dispatched and no review minors remain deferred.

Final: Ruling: calibration, semantic model suitability and physical source independence remain
explicit host-reviewed assumptions. Synthetic validation cannot establish these properties.
If the assumptions are wrong, finite covariance can still be misleading.

Final: Ruling: correlated-source fusion and conflicting-verifier hypothesis handling remain later
stages. PR-3 requires successful verification of its named channel; it does not arbitrate every
other verifier. If conflicting evidence is ignored by the host, estimates may be overconfident.

Final: Ruling: a replacement prior after latched insufficiency is an explicit host initialization
boundary. This module does not authenticate or calibrate that prior. A bad replacement prior can
restart the same model failure.

Final: Ruling: full-suite and installed-wheel completion are the implementer's verification duty;
the reviewer inspected code and focused reproductions. Results below provide the release evidence;
an unverified artifact could otherwise fail outside the checkout.

Documentation precision: BLOCK at the estimator guide's verified-observation claim was resolved
by the binding requirement and an explicit missing/failed-verification error contract. The RED/GREEN
tests above verify that correction. No unresolved documentation BLOCK or WARN remains.

Final: fixed verifier binding and noise-cap underflow — named regressions above RED→GREEN;
full suite 3494 passed, 18 skipped, 16 warnings in 198.12 seconds (Python 3.12.14, offline CPU,
OMP/MKL/OpenBLAS threads set to one). Command: python -m pytest -q --maxfail=1 --disable-warnings.

Final verification:

- Estimator suite: 63 passed; module coverage 94% (201 statements, 12 unexecuted validation branches).
- Focused state/reconciliation/estimator/research suite: 270 passed.
- Repository Ruff: all checks passed. Black: 572 files unchanged.
- Targeted mypy: estimator and registry passed; the CI-selected nine modules and isolated logging
  schema check also passed.
- Source distribution and wheel built successfully. Freshly reinstalled wheel outside checkout
  passed an analytical update, full causal validation, replay rejection, 30-reference registry
  load and bundled estimator-guide smoke check.
- git diff --check passed. No canonical cases, confidence/reward paths or authority modules changed.

Validation sequencing note: building an sdist creates a temporary source tree inside the checkout.
Overlapping test/formatter discovery saw that tree. Final repository checks ran after the build
completed. The pre-fix full-suite run was stopped; the result above is from the corrected code.

No PR-4 implementation or paper-performance claim is included. No Beads commands/files were used.
