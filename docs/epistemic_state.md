# Temporal epistemic record contracts

## Status and scope

`gepa_mindfulness.verification.epistemic_state` implements PR-1 of the temporal
Predict → Execute → Observe → Reconcile program. These are experimental diagnostic records,
constructed explicitly by callers. Importing the module installs no runtime behavior. Existing
events, confidence fusion, routing, rewards, training and authority remain unchanged.

The module does not resolve evidence or event references, authenticate producers, verify truth,
estimate state, reconcile an event sequence, authorize an action or persist an update.
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
PR-2 must perform that work through the existing action-bound event sequence before treating an
update as causally reconciled. A constructed PR-1 record is not a valid runtime reconciliation.

An update rejects mismatched contexts and reused prior/posterior estimate IDs. When numerical
states exist, their representations and ordered dimensions must agree, and each measurement must
name that representation and one of those dimensions. Without numerical state, target dimensions
remain declared diagnostic labels; this stage does not validate their semantics. Unavailable
measurements cannot appear in `measurements` (which records measurements used in the update).

Correlation treatment defaults to `UNRESOLVED_CORRELATION`. `KNOWN_COVARIANCE`,
`COVARIANCE_INTERSECTION` and `CONSERVATIVE_BOUND` are producer-declared method labels only.
No label executes or certifies fusion. A null correlation group means unknown dependence;
different group names also do not establish independence. PR-4 must verify a selected algorithm's
covariance/correlation assumptions before consuming these records for fusion.

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

## Research, compatibility and next stage

[REF-KALMAN](recommendations/RESEARCH_TRACEABILITY.md#ref-kalman) separates the mathematical result,
repository inference, hypothesis, experiment and implementation status. Existing WMLLM/DWM
summaries motivate prospective prediction and distinct outcome records. No paper's behavioral
results are reproduced by these contract tests.

Existing event JSON and the canonical 17 cases require no migration. The research registry now
permits a null `arxiv_id` for a DOI-backed source and requires its exact `https://doi.org/<doi>`
canonical URL. Existing arXiv entries retain their prior fields and URL validation.

PR-2 depends on these types and must bind reconciliation to existing prediction, execution,
observation and verifier ancestry. It must validate causal ordering and verifier evidence before
introducing a reconciliation event adapter. PR-3 can then implement an explicitly gated estimator.

## Verification

From the repository root, run:

```text
python -m pytest tests/test_epistemic_state.py tests/test_action_bound_events.py tests/test_action_bound_event_sequence.py tests/test_epistemic_process_rewards.py tests/test_research_traceability.py -q
```

`test_epistemic_state.py` covers the validation rules above, exact round trips, detached snapshots,
identity mismatches and rejection by the existing optimizer component contract. Existing event
and reward tests provide compatibility coverage. Registry tests check DOI metadata and reciprocal
research links. Deployment assumptions such as evidence authentication and normalization validity
require host review; this record-only PR supplies no deployed estimator to qualify.
