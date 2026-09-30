# Scalar source fusion and frozen Round-0 judgments (PR-4)

## Scope

`fuse_scalar_measurements()` combines 1–16 `EpistemicMeasurement` records for the same scalar
target. Callers explicitly invoke this diagnostic API. Importing it installs no runtime behavior;
existing confidence heuristics, routing, rewards, authority and the 17 canonical cases are unchanged.

Each measurement retains the existing `ConfidenceSource`, evidence, provenance and correlation
group. A source label, a different group name, or a judgment made before discussion does not
establish statistically independent errors. The host must establish common units/target, valid
marginal error variances and any declared covariance. Arbitrary semantic judgments need not meet
these assumptions. A finite variance is not a truth certificate.

## Fusion modes

For source values `x_i`, marginal variances `P_i`, and normalized nonnegative weights `a_i` summing
to one, the implementation supports the existing `CorrelationTreatment` vocabulary:

| Mode | Mean | Variance | Required declaration |
| --- | --- | --- | --- |
| `COVARIANCE_INTERSECTION` (default) | `sum(a_i*x_i/P_i) / sum(a_i/P_i)` | `1 / sum(a_i/P_i)` | Positive marginal variances; cross-correlation may be unknown. |
| `KNOWN_COVARIANCE` | `sum(a_i*x_i)` | `a^T C a` | Full ordered source-error covariance matrix and nonempty provenance. |
| `CONSERVATIVE_BOUND` | `sum(a_i*x_i)` | `max(P_i)` across all supplied sources | Positive marginal variance bounds. |
| `UNRESOLVED_CORRELATION` | Unavailable | Unavailable | Retains inputs without computing an estimate; marginal variance may be missing. |

Covariance Intersection (CI) mixes information with weights summing to one, so repeated identical
estimates do not receive the variance reduction of independent observations. The mathematical
bound assumes valid input error bounds. It does not repair biased, miscalibrated or semantically
incompatible inputs. The conservative bound follows from the maximum marginal bound for a convex
combination, including under unknown cross-correlation. Neither mode detects malicious sources.

Known-covariance weights are declared by the caller, uniform by default; they are **not optimized**.
This avoids matrix inversion and supports singular positive semidefinite (PSD) covariance, such
as perfectly correlated copies. Explicit negative correlation is supported; perfect anticorrelation
can produce zero variance under that declared model. Such a declaration requires host review.

The matrix must be finite, exactly symmetric, and match the input order and marginal variances.
Exact rational LDL decomposition rejects indefinite matrices, including invalid zero pivots.
There is no jitter or tolerance that turns an invalid matrix into an accepted one. Rounded empirical
matrices may therefore need a separately reviewed covariance-estimation procedure before use.

If two sources share an evidence ID or a non-null correlation group, known zero cross-covariance
is rejected. The same rule applies to every pair marked `peer_exposed=True`. Explicit nonzero
known covariance remains usable. This is a conservative input policy: shared provenance need not
mathematically imply nonzero error covariance, but this API requires the host to retain the
reported dependence. Supplying covariance/provenance in another mode raises rather than ignoring it.

## Calling and audit contract

```python
from gepa_mindfulness.verification.epistemic_state import CorrelationTreatment
from gepa_mindfulness.verification.scalar_fusion import fuse_scalar_measurements

# measurements contains existing, compatible EpistemicMeasurement records.
unknown = fuse_scalar_measurements(measurements, estimate_id="fusion-unknown")
known = fuse_scalar_measurements(
    measurements,
    estimate_id="fusion-known",
    mode=CorrelationTreatment.KNOWN_COVARIANCE,
    covariance=((1.0, 0.5), (0.5, 1.0)),  # Example for two unit-variance sources.
    covariance_provenance=("covariance-calibration-v1",),
    weights=(1.0, 1.0),
)
```

The full keyword interface is `estimate_id`, `mode`, `weights=None`, `covariance=None`,
`covariance_provenance=()`, `peer_exposed=False`, `uncertainty_scale=1.0`. All numerical inputs
reject booleans, strings and nonfinite values. Weights must be nonnegative, match the source count
and have positive total. Numeric modes require positive marginal variances; missing covariance
never becomes identity covariance. Empty batches, repeated measurement IDs, unavailable values,
mixed contexts, representations or dimensions raise `ValueError`.

Standard-library rational arithmetic avoids intermediate overflow and preserves exact PSD checks
for these small batches. Converted variances round upward when necessary to preserve the bound.
A positive variance or coefficient that cannot be represented as a positive float raises
`ValueError`, instead of silently becoming zero. Means and normalized uncertainty retain normal
floating-point rounding. The 16-source limit keeps exact matrix validation bounded in dimension.

`ScalarFusionResult` contains the estimate and detached input measurements, correlation treatment,
original `weight_inputs`, normalized `weights`, actual `mean_weights`, covariance/provenance,
peer-exposure flag and uncertainty scale. CI mixing weights differ from its mean coefficients.
Reported coefficients are floats; retain `weight_inputs` and input records for exact replay.
`to_dict()` exports detached JSON; it is an audit output, not an authenticated import format.

The estimate uses a scalar `DiagonalState`; world uncertainty is the bounded diagnostic
`variance / (variance + uncertainty_scale)`. Model and monitor uncertainty remain `None`.
Unresolved mode returns `Availability.UNAVAILABLE` with null state and uncertainty, not zeros.
All input evidence and provenance survive, including inputs assigned zero weight. The numerical
function checks contracts, not evidence truth or causal history; use the panel below for verified
Round-0 collection.

The result is a source-fusion estimate, without an additional temporal prior. It does not fabricate
a raw observation or an `UncertaintyUpdateRecord`. Passing its mean into PR-3 as though it were a
raw verified observation would require a new explicit statistical and causal contract. This stage
does not compose the two algorithms or claim independence from a temporal prior.

## Verify before discussion

`VerifiedJudgmentPanel(panel_id, participant_ids)` fixes a unique cohort of 1–16 participants.
The host must withhold peer judgments while collecting Round-0. The collector cannot detect
external communication, authenticate participants or make their model errors independent.

1. For each participant, the host calls `add_judgment(participant_id, events,
   reconciliation_event_id=..., measurement_id=...)` with a complete existing causal event history.
   The collector runs `validate_action_bound_sequence()` and requires an explicit successful
   verifier binding for the named measurement. Numeric observations, evidence, chronology and
   verifier provenance use PR-2's existing checks. A source label alone is insufficient.
2. After every participant submits, the host calls `open_discussion(released_at)` with an aware
   RFC3339 timestamp no earlier than any submitted reconciliation. This seals the collector and
   returns a detached `FrozenRoundZero` packet in declared cohort order. The packet includes
   participant IDs, measurements, reconciliation event IDs, panel ID and release timestamp.
3. The host may now expose that packet to peers. `fuse_round_zero(estimate_id=..., **options)` uses
   the retained original snapshots, so later consensus cannot overwrite prior dissent.
4. To compare later judgments, call `fuse_post_discussion(measurements, estimate_id=..., **options)`
   with one fresh measurement ID per participant in cohort order and the same context/target.
   The method forces `peer_exposed=True`; callers cannot override it. Its numeric declarations
   are not newly verified observations. Known zero cross-covariance fails; CI remains the default.

Incomplete cohorts, duplicate participants/measurements, failed or unbound verification, and
conflicting reuse of an event ID all fail before storing a judgment. A failed release leaves the
collector open. After release, additions and another release fail. Returned snapshots cannot
change retained judgments. Calls must be sequential; the collector is not thread-safe and has no
persistence, transport, voting, action-selection or reward behavior. Retained event hashes grow
with the histories supplied; the host retains full events for later audit.

## Numerical comparison and validation

For five equal-valued sources with unit marginal variance and uniform weights:

| Declaration | Fused variance | Normalized world uncertainty (scale 1) |
| --- | --- | --- |
| Known independent errors (`C = I`) | 0.2 | 1/6 |
| Known perfectly correlated errors (every `C_ij = 1`) | 1 | 0.5 |
| Unknown correlation (CI) | 1 | 0.5 |
| Unknown correlation (conservative bound) | 1 | 0.5 |

This comparison exposes the cost of a false independence assumption. It demonstrates arithmetic,
not improved LLM calibration or paper benchmark replication. CI may lose useful independent
information; uniform known-covariance weights may be inefficient. Incorrect marginal bounds,
biased inputs, undeclared peer exposure or false covariance declarations can invalidate the result.

From the repository root:

```text
python -m pytest tests/test_scalar_fusion.py tests/test_judgment_panel.py tests/test_temporal_estimator.py tests/test_epistemic_reconciliation.py tests/test_research_traceability.py -q
```

Analytical tests check the table, heterogeneous weights, PSD/singular/indefinite covariance,
extreme scales, source preservation and conservative defaults. Panel tests cover complete and
failed collection, immutable dissent, chronology and post-discussion dependence. Host review of
calibration, common-target semantics and real peer isolation remains necessary before deployment.

## Research status

[Julier and Uhlmann](recommendations/RESEARCH_TRACEABILITY.md#ref-ci) motivate conservative fusion
under unknown cross-correlation. The [author-coauthored 2025 treatment](https://discovery.ucl.ac.uk/id/eprint/10217482/)
also describes arbitrary-count and equal-weight CI; no optimized mixture-weight algorithm is adopted.
[Unanimity Without Persuasion](recommendations/RESEARCH_TRACEABILITY.md#ref-unanimity) motivates
preserving and verifying judgments before exposure. [Weakly Supervised Quantum Error Mitigation](recommendations/RESEARCH_TRACEABILITY.md#ref-wsqem)
is only a structural analogy for aggregating weak signals; quantum results do not validate LLM
fusion. [GRUET](recommendations/RESEARCH_TRACEABILITY.md#ref-gruet) motivates diagnostic trajectory
uncertainty, without a private-chain-of-thought reward. Published findings and repository inferences
remain separate in the register. See [ADR 0005](adr/0005-correlation-fusion.md). PR-5 is deferred.
