# Temporal epistemic records and causal reconciliation

## Status and scope

`gepa_mindfulness.verification.epistemic_state` implements PR-1 of the temporal
Predict → Execute → Observe → Reconcile program. These are experimental diagnostic records,
constructed explicitly by callers. Importing the module installs no runtime behavior. Existing
confidence fusion, routing, rewards, training and authority remain unchanged.

The record module does not resolve evidence or event references, authenticate producers, verify
truth, estimate state, authorize an action or persist an update. PR-2 adds explicit causal
validation through `epistemic_reconciliation` and the existing action-bound sequence validator.
An available number and an `EXTERNAL_VERIFIER` source label do not establish verified evidence.

## Contracts

| Type | Meaning and validation |
| --- | --- |
| `EpistemicContext` | `run_id`, nullable `repeat_id`, and existing `EvaluatedSystemVersion`. Rejects blank identity and Boolean, negative or non-JSON-safe repeat IDs. |
| `EpistemicStateEstimate` | Estimate ID, context, estimator version, three optional uncertainty values, optional numerical state, optional action/prediction IDs, availability, evidence and provenance. |
| `EpistemicMeasurement` | Measurement ID, context, existing `ConfidenceSource`, representation ID, target dimension, scalar value, optional uncertainty/variance, correlation group, availability, evidence and provenance. |
| `InnovationRecord` | Predicted and actual scalar measurements, prediction/observation IDs, context, evidence, provenance and declared mismatch status. The signed residual is `actual - predicted`. |
| `UncertaintyUpdateRecord` | Complete prior/posterior snapshots, nonempty unique available measurements, update method, correlation treatment, model mismatch, unresolved hypothesis IDs, evidence and provenance. |

All four records require nonempty producer provenance. Available state/measurement records and
innovations also require typed `EvidenceReference` values. References retain their source kinds,
including non-observable kinds when used for diagnostics. Update evidence contains the union of
prior, posterior and used measurement references; provenance strings identify producer artifacts
or contracts. Neither field authenticates its named source.

The constructor copies arrays to tuples and snapshots nested records. `to_dict()` returns detached
JSON and revalidates the record. `from_dict()` requires exactly the serialized fields, including
`schema_version="epistemic-state-v1"`; unknown fields, missing fields and unsupported versions
raise `ValueError`. Callers use typed enums in constructors and exact enum strings in JSON.

### Uncertainty and covariance

`world_uncertainty`, `model_uncertainty` and `monitor_uncertainty` remain separate normalized
diagnostics in `[0,1]`. The estimator version identifies the producer's normalization contract.
These numbers are not automatically probabilities or estimation variances. A missing component
is `None`; the module never derives a scalar routing confidence.

`Availability.UNAVAILABLE` requires null numerical fields. Available estimates need at least one
uncertainty value or a numerical state; available measurements need a value. Unknown uncertainty
or variance can remain null on an available measurement. Zero is a valid measured value or
variance and never represents missing telemetry.

`DiagonalState` names a representation that defines units and semantics. It has unique ordered
dimension names, finite scalar values, and an optional equal-length tuple of finite nonnegative
variances in squared units. Such a diagonal covariance is symmetric and positive semidefinite by
construction, including zero entries. Nested arrays/full matrices fail validation; the module
does not drop off-diagonal values. Null variances mean unavailable covariance.

All numeric fields reject booleans, strings, nonfinite values and integers too large for finite
float conversion. Derived innovations also reject overflow. Normalized innovation is signed
`residual / sqrt(innovation_variance)`. It exists only with a positive finite variance and an
explicit `normalization_basis` reference naming the producer's statistical justification.
The module checks that both fields exist, not that the justification is mathematically valid.
This diagnostic is not a probability, squared innovation statistic or deception classification.

### Identity and update consistency

`EpistemicContext.validate_event(event)` checks run, repeat, model and harness identity against an
existing `EventEnvelope`. It does not resolve prediction/action/observation IDs or validate order.
Callers must run `validate_action_bound_sequence(events)` with the reconciliation and its causal
inputs before treating an update as causally reconciled. Constructing records or envelopes alone
does not establish valid ancestry.

An update rejects mismatched contexts and reused prior/posterior estimate IDs. When numerical
states exist, their representations and ordered dimensions must agree, and each measurement must
name that representation and one of those dimensions. Without numerical state, target dimensions
remain declared diagnostic labels; this stage does not validate their semantics. Unavailable
measurements cannot appear in `measurements` (which records measurements used in the update).

Correlation treatment defaults to `UNRESOLVED_CORRELATION`. `KNOWN_COVARIANCE`,
`COVARIANCE_INTERSECTION` and `CONSERVATIVE_BOUND` are producer-declared method labels only.
No label executes or certifies fusion. A null correlation group means unknown dependence;
different group names also do not establish independence. [PR-4 scalar fusion](scalar_fusion.md) validates numerical covariance contracts; the host must
review the physical covariance/correlation assumptions.

## Example

```python
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import EpistemicContext, EpistemicStateEstimate
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

estimate = EpistemicStateEstimate(
    estimate_id="estimate-1",
    context=EpistemicContext("run-1", None, EvaluatedSystemVersion("model-1", "harness-1")),
    estimator_version="declared-normalization-v1",
    world_uncertainty=0.6,
    model_uncertainty=None,
    monitor_uncertainty=0.2,
    evidence_refs=(EvidenceReference("observation-log", EvidenceSourceKind.EXTERNAL_RECORD),),
    provenance=("producer-config-v1",),
)
assert EpistemicStateEstimate.from_dict(estimate.to_dict()) == estimate
```

## Causal reconciliation (PR-2)

`EpistemicReconciliation` contains an `UncertaintyUpdateRecord`, prediction and observation
event IDs, nonempty verifier event IDs, and exactly one `OutcomeMeasurementBinding` per used
measurement. Each binding carries an `InnovationRecord`, prediction/observation JSON paths,
and an optional verifier event ID. Paths contain exact string object keys or nonnegative integer
array indices; an empty path selects a scalar root. Both selected values must be finite numbers.
Booleans, null, numeric strings and missing telemetry fail validation.

`make_epistemic_reconciliation_event(record, **metadata)` emits `epistemic_reconciliation` in the
existing stream. The payload uses `epistemic-reconciliation-v1`; nested PR-1 records retain their
schema. Serialization is exact and detached, and the envelope deep-freezes the payload. The adapter
derives run/repeat/model/harness, action, evidence and parent IDs and rejects conflicting overrides.
The caller supplies an event ID and timestamp or uses the existing envelope defaults.

The sequence validator requires these conditions:

- All inputs occur earlier in the stream. Parents are exactly prediction, observation, then the
  declared verifier events, in that order. Each verifier binds the same observation.
- Observation ancestry resolves through execution and proposal to the selected prediction.
  Posterior action/prediction IDs and innovation prediction/observation IDs match that chain.
  All records and events share run, repeat, model and harness identity.
- Timestamps are timezone-aware RFC3339. Prediction strictly predates execution; proposal lies
  between them; observation follows execution; each verifier lies between observation and
  reconciliation. Equal timestamps are allowed except prediction versus execution. Both stream
  order and timestamps must pass; this additional timestamp rule applies only to reconciled chains.
- Innovation numbers equal the selected prediction and observation values; measurement value
  equals the latter. Innovation and measurement retain identical typed evidence sets, and their
  reference IDs occur in the observation. Prior evidence IDs occur in the committed prediction.
  All update evidence IDs occur in prediction, observation or verifier inputs.
- Optional prior action/prediction IDs resolve to the same evaluation context and agree when both
  are supplied. They may name the current action/prediction for an action-conditioned prior.
  Historical predictions or executed actions must occur before the current prediction in stream
  order, with timestamps no later than that prediction. Null IDs allow an initial unbound prior.
- Each update ID is unique. Reconciliation is append-only; it cannot use `superseded_by`.

### External verification boundary

An `EXTERNAL_VERIFIER` measurement requires a verifier event binding. Any binding that names a
verifier, regardless of source label, must retain that verifier's `verifier_refs` in measurement
provenance and use observable evidence. Legacy `VerificationResult` requires `verified=True` and
keeps its existing observation-level meaning. A relational result requires both
`claimed_outcome_supported=True` and `provenance_intact=True`, with every typed measurement
reference bound to each of those fields. Local execution findings alone cannot certify outcomes.

A diagnostic without a verifier binding may coexist with a failed verification; it cannot use
the external-verifier source label. This code validates recorded relationships, not producer
authentication, evidence truth, selected-path units or the update algorithm. Legacy string
reference IDs do not independently establish source kinds. The host must authenticate producers,
preserve captured source kinds, and review the measurement contract before relying on results.
Prior numbers remain producer declarations; committing prior evidence does not commit those numbers.

### Residual scope

| Residual | Meaning | Current support |
| --- | --- | --- |
| World | Predicted outcome versus observed outcome | Explicit numeric paths; `actual - predicted`. |
| Action | Intended/proposed action versus executed action | The existing action schema requires exact proposal/execution equality and rejects rewrites. No independently observed action representation exists for a numeric residual. |
| Report | Verified action/outcome versus a later public report | Deferred until a typed later-report event and its causal binding exist. |

Unavailable residual categories are not reported as zero. No residual establishes motive,
deception, reward eligibility, execution authority or persistence authority. Mismatch labels and
normalization assumptions remain producer declarations. PR-2 adds no runtime producer; the
[PR-3 scalar estimator](temporal_estimator.md) is separately imported and explicitly enabled.

Legacy assessment parents remain verifier events. V5 validates reconciliation cell identity, but
its scored outcome still accepts only `{"passed": bool}`; numeric telemetry belongs to an additional
action trajectory in that cell. Adding a valid diagnostic trajectory does not change optimizer scores.

## Research, compatibility and next stage

[REF-KALMAN](recommendations/RESEARCH_TRACEABILITY.md#ref-kalman) separates the mathematical result,
repository inference, hypothesis, experiment and implementation status. Existing WMLLM/DWM
summaries motivate prospective prediction and distinct outcome records. No paper's behavioral
results are reproduced by these contract tests.

Existing event JSON and the canonical 17 cases require no migration. The research registry now
permits a null `arxiv_id` for a DOI-backed source and requires its exact `https://doi.org/<doi>`
canonical URL. Existing arXiv entries retain their prior fields and URL validation.

PR-2 now binds reconciliation to prediction, execution, observation and verifier ancestry.
The [research register](recommendations/RESEARCH_TRACEABILITY.md#ref-fta) distinguishes published
findings from repository hypotheses for FTA, PINNForge, C3-JEPA and AI Neuroscientist. Synthetic
contract tests do not reproduce their experiments. PR-3 adds an experimental, disabled-by-default
[scalar temporal estimator](temporal_estimator.md); PR-4 adds [scalar correlation-aware fusion](scalar_fusion.md). See [ADR 0003](adr/0003-epistemic-reconciliation.md).

## Verification

PR-5 adds [qualitative evidence and memory eligibility](evidence_memory.md) around existing
measurements. Numeric availability remains separate from evidence status and permission.

From the repository root, run:

```text
python -m pytest tests/test_epistemic_state.py tests/test_epistemic_reconciliation.py tests/test_action_bound_events.py tests/test_action_bound_event_sequence.py tests/test_epistemic_process_rewards.py tests/test_research_traceability.py -q
```

`test_epistemic_state.py` covers the validation rules above, exact round trips, detached snapshots,
identity mismatches and rejection by the existing optimizer component contract. Existing event
and reward tests provide compatibility coverage. Registry tests check DOI metadata and reciprocal
research links. Deployment assumptions such as evidence authentication and normalization validity
require host review; the experimental PR-3 estimator has no deployed runtime integration. Reconciliation tests
cover missing/forward inputs, chronology, residual spoofing, evidence laundering, immutability,
schema round trips and V5 score compatibility.
