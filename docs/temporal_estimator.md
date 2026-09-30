# Experimental scalar temporal estimator (PR-3)

## Scope and assumptions

`gepa_mindfulness.verification.temporal_estimator.ScalarTemporalEstimator` estimates one changing
scalar from causally verified numeric observations. It is disabled by default and has no runtime
hook. Callers explicitly enable and call it. Confidence fusion, routing, rewards, training,
authority and the 17 canonical cases retain their existing behavior.

The model is a scalar random walk with an identity observation function. Its committed prediction
must equal the previous mean. The host supplies variance in squared representation units and
declares independent process/measurement errors through `independent_noise=True`. This declaration
is an assumption, not a verification result. Arbitrary semantic state need not be linear or
Gaussian. Repeated or correlated judgments do not become independent by changing their IDs.
[PR-4](scalar_fusion.md) adds separate multi-source fusion; this estimator consumes one configured channel.

The constructor requires an available one-dimensional `DiagonalState` with covariance. Scalar
nonnegative covariance is symmetric and positive semidefinite by construction. Full matrices,
multiple dimensions, missing covariance, Boolean numbers and nonfinite values are rejected.
Measurement variance must be strictly positive; prior and process variances may be zero.

## Numerical update

Let `m,P` be the prior mean and variance, `Q` process variance per observation step, `y` the
observation, and `R` its supplied variance multiplied by the current adaptive noise scale:

```text
Pminus = P + Q
S = Pminus + R
z = (y - m) / sqrt(S)
K = Pminus / S
mplus = (1 - K)*m + K*y
Pplus = (1 - K)^2*Pminus + K^2*R
```

An accepted observation uses the Joseph covariance form in the last line to avoid subtracting
nearly equal variances. See the [JPL derivation, section III.A, equation 14](https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A-orig.pdf).
With fixed Q/R, posterior covariance is independent of residual magnitude. A large residual alone
does not increase it.

If `abs(z) > innovation_gate`, the estimator rejects mean assimilation. It grows Q (from at least
`process_variance_floor`) and the R multiplier by their configured factors, caps each at its
configured maximum, and sets `Pplus = P + adapted_Q`. The result records gain zero, the effective
Q/R used for this update, and `assimilated=False`. The measurement remains in the reconciliation
because it was used for gating and noise adaptation. Innovation normalization retains the
**pre-adaptation S**, which explains the gate decision.

Consecutive gate failures set `MODEL_MISMATCH`, then `REGIME_SHIFT_SUSPECTED` at its threshold,
then `INSUFFICIENT_MODEL` at its threshold. These are heuristic statuses, not formal change-point
tests. Increasing noise may admit a later observation before either threshold is reached. An
inlier resets the streak and decays adaptive noise toward baseline for the next step.
`INSUFFICIENT_MODEL` latches: subsequent observations cannot update the mean. The host must review
the assumptions and explicitly construct a new estimator to resume assimilation.

World uncertainty is `Pplus/(Pplus + uncertainty_scale)`; monitor uncertainty uses effective R
with the same normalization. Both are bounded diagnostics in `[0,1]`, computed without overflowing
the sum. They are not calibrated probabilities. Model uncertainty remains `None`; a mismatch
label is not converted into a probability. Any nonfinite intermediate aborts the whole step.

## Configuration and calling contract

```python
from gepa_mindfulness.verification.temporal_estimator import (
    ScalarEstimatorConfig,
    ScalarTemporalEstimator,
)

# initial_estimate is an existing EpistemicStateEstimate with scalar covariance.
estimator = ScalarTemporalEstimator(
    initial_estimate,
    ScalarEstimatorConfig(
        enabled=True,
        independent_noise=True,
        measurement_provenance_ref="sensor-v1",
    ),
)
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `False` | Disabled `reconcile()` returns `None` before inspecting inputs. |
| `independent_noise` | `False` | Must be true when enabled; host declares the numerical assumption. |
| `measurement_source` | `ConfidenceSource.TOOL_RESULT` | Every measurement must match this source. |
| `measurement_provenance_ref` | `scalar-monitor-v1` | Required measurement provenance contract. |
| `process_variance` | `0.1` | Baseline Q per observation step; no wall-clock scaling. |
| `process_variance_floor` | `0.01` | Minimum Q before outlier growth. |
| `max_process_variance` | `1000000` | Cap on adaptive Q, not posterior P. |
| `process_noise_growth` | `4` | Q growth factor on rejection. |
| `measurement_noise_growth` | `4` | R multiplier growth factor on rejection. |
| `max_measurement_noise_scale` | `100` | Cap on R multiplier. |
| `noise_decay` | `2` | Divide adaptive noise after an inlier, down to baseline. |
| `innovation_gate` | `3` | Maximum accepted absolute normalized innovation. |
| `regime_shift_after` | `3` | Consecutive rejection threshold, integer at least two. |
| `insufficient_model_after` | `6` | Larger threshold that latches mean rejection. |
| `uncertainty_scale` | `1` | Positive variance scale for bounded diagnostics. |

Configuration serializes with `to_dict()` and restores with `from_dict()`; deserialization requires
the exact field set. Floats must be finite; factors/decay and maximum R scale must be at least one.
Q may be zero; the other float fields must be positive. Maximum Q must cover baseline Q and its
floor. Counts must be built-in integers within the JSON-safe range, excluding booleans.

Call `reconcile(events, *, measurement, binding, prediction_event_id, observation_event_id,
verification_event_ids, update_id, estimate_id, event_id, timestamp, **metadata)`. The measurement
and `OutcomeMeasurementBinding` use the [existing reconciliation contracts](epistemic_state.md#causal-reconciliation-pr-2).
The call validates prediction → proposal → execution → observation → verifier ancestry and the
returned reconciliation using `validate_action_bound_sequence()`. Metadata is forwarded to the
existing event adapter. Every binding must name a verifier event that successfully verifies the
bound observation. Missing or unsuccessful verification raises `ValueError` without changing
estimator state. The measurement must match the initial context, representation and
dimension, as well as the configured source/provenance. Fresh measurement, estimate and evidence
IDs are required; put reusable calibration references in provenance rather than observation evidence.

`EstimatorResult` returns the reconciliation, immutable event envelope, configuration snapshot,
assimilation flag, gain, effective process/measurement variances and rejection count. Its `estimate`
and `innovation` properties expose existing records. The configuration's canonical JSON SHA-256
appears in update-method/provenance and innovation-normalization references. Retain the configuration
snapshot alongside events to support audit/replay; a hash alone cannot reconstruct configuration.

After success the host must include the returned event and its complete preceding history in the
next call. The estimator checks that prefix's digest and requires a fresh prediction after it,
with a timestamp no earlier than the previous reconciliation. Invalid ancestry, changed history,
duplicate identities and arithmetic errors raise `ValueError` before any internal state changes.
Returned snapshots do not share the estimator's state. Calls must be sequential; this instance is
not thread-safe. Full-history validation/hashing and retained evidence grow with trajectory length.

The estimator mutates only its own memory. It returns diagnostics, writes no logs or files, and
grants no action, persistence or optimizer authority. The host authenticates producers and reviews
calibration, independence, units and any decision about restarting a latched estimator.

## Synthetic comparison and validation

Reproduce the comparison from the repository root:

```text
python -m pytest tests/test_temporal_estimator.py -k synthetic -s -q
```

Both runs start at mean 0, variance 1, Q=1 and measurement variance 1. Observations are
`[0,0,1000,1000,1000]`. The baseline gate is 1000000 (all these observations accepted); the gated
run uses 3, with regime/insufficient thresholds 2/3. Other values use configuration defaults.

| Step | Observation | Baseline mean | Baseline variance | Gated mean | Gated variance | Gated status |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.666667 | 0 | 0.666667 | NONE |
| 1 | 0 | 0 | 0.625000 | 0 | 0.625000 | NONE |
| 2 | 1000 | 619.047619 | 0.619048 | 0 | 4.625000 | MODEL_MISMATCH |
| 3 | 1000 | 854.545455 | 0.618182 | 0 | 20.625000 | REGIME_SHIFT_SUSPECTED |
| 4 | 1000 | 944.444444 | 0.618056 | 0 | 84.625000 | INSUFFICIENT_MODEL |

This demonstrates covariance behavior and explicit rejection under a deliberately unsupported
shift. It does not show improved tracking accuracy: rejection prevents tracking a genuine jump.
False rejections, delayed recovery and incorrect independence assumptions remain possible.
Compare calibration, false confidence and tracking error on held-out temporal outcomes before
any routing integration. Existing confidence heuristics remain the deployed baseline.

Run `python -m pytest tests/test_temporal_estimator.py tests/test_epistemic_reconciliation.py
tests/test_epistemic_state.py tests/test_research_traceability.py -q` as one command. Tests cover
analytical updates, fixed-noise covariance, gating/recovery/latching, overflow rollback, causal
replay, source/context/evidence mismatches and detached snapshots. Full-suite and installed-wheel
checks cover compatibility and distribution resources.

## Research status

[Kalman](recommendations/RESEARCH_TRACEABILITY.md#ref-kalman) supplies the mathematical starting
point. [GRUET](recommendations/RESEARCH_TRACEABILITY.md#ref-gruet) motivates trajectory uncertainty
telemetry; this implementation extracts no reasoning graphs or private chain of thought.
[Dual-Frontier](recommendations/RESEARCH_TRACEABILITY.md#ref-dual-frontier) motivates separating
model error from decision validity. [DEEPO](recommendations/RESEARCH_TRACEABILITY.md#ref-deepo)
motivates keeping confident errors visible; no entropy-based training is implemented.
[C3-JEPA](recommendations/RESEARCH_TRACEABILITY.md#ref-c3-jepa) motivates explicit correspondence,
without establishing semantic linearity. Published results, repository inferences and untested
hypotheses are separated in the register. No paper's agent-performance experiment is reproduced.
See [ADR 0004](adr/0004-temporal-estimator.md). [PR-4 scalar fusion](scalar_fusion.md) is available as a separate diagnostic API; automatic composition remains deferred.
