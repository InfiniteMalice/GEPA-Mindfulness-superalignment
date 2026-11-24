# System-One routing (experimental, opt-in)

The optional [structured-decision extension](decision_routing.md) wraps this same guard
with public-context/readout binding, raw calibration/error diagnostics and separate
recorded-depth research. It adds no execution or verifier authority.

`gepa_mindfulness.factuality_observability.epistemic_routing` feeds recorded uncertainty into
bounded route proposals. The existing `choose_routing_action()` and default pipeline are
unchanged. No model service, network client, model weights or new dependency is installed.

## Calling the adapter

The host supplies a `RoutingRequest` with a tuple of 1–10,000 `EventEnvelope` objects, a
`decision_event_id`, the existing `RoutingContext`, and a `context_changed` boolean. The
chronological tuple ends at that ID's `action_proposed` event and includes its prediction
commitment and all causal ancestors. The decision requires a conversation ID and checkpoint
step. Inputs are validated by `validate_action_bound_sequence()` before routing.

```python
from gepa_mindfulness.factuality_observability.epistemic_routing import (
    RoutingPolicy, RoutingRequest, route_epistemic,
)

# events and context come from the host's existing action-bound logger and factuality router.
request = RoutingRequest(
    events=tuple(events), decision_event_id=events[-1].event_id,
    context=context, context_changed=False,
)
assessment = route_epistemic(request, policy=RoutingPolicy(enabled=True))
public_diagnostic = assessment.to_dict()
route_proposal = assessment.routing_decision()
```

With the default `enabled=False`, `route_epistemic()` returns `None` and never calls the
backend. Invalid enabled inputs raise `ValueError`. Missing evidence produces a conservative
route. `RoutingAssessment` records the input digest, decision/reconciliation IDs, backend
name/version, raw proposal, selected route, override, failure category and callback latency.
`to_dict()` returns a detached JSON-compatible dictionary; the host decides whether to log it.
This version adds no structured event type or automatic persistence.

## Evidence and precedence

The adapter uses the latest earlier reconciliation in the same run, repeat, conversation,
model/harness, case, stripe and seed. Causal ancestors must match that unit. Evidence outside
its declared validity window (inclusive `valid_from`, exclusive `valid_until`),
superseded reconciliation, later unreconciled verification, unavailable state, and changed
context prevent continuation. The default maximum state age is 300 seconds, measured against
the proposal timestamp. The host supplies the complete prefix and truthful timestamps.

The first applicable row determines the required route:

| Condition | Required proposal |
| --- | --- |
| Verification budget exhausted | Abstain if viable; otherwise escalate |
| Representation-sensitive or mandatory verification | External verification |
| Domain risk at/above host threshold, or irreversible proposed action | External review |
| Missing, stale, expired, superseded, foreign, unavailable or changed-context state | External verification |
| Monitor uncertainty unknown or at/above threshold | Independent monitor verification |
| Model uncertainty unknown or at/above threshold | Verification with reduced autonomy |
| Aggregate update or any binding has a mismatch status other than `NONE` | Decompose and verify; reconsider the hypothesis |
| World uncertainty unknown or at/above threshold | Retrieve more evidence |
| Insufficient typed verification | External verification |
| Complete low uncertainty and typed verification | Preserve the legacy router's decision |

Uncertainty defaults to a 0.6 threshold and domain risk to 0.8. These are host policy values
over diagnostics, not calibrated probabilities. The policy accepts finite thresholds in [0,1]
and maximum state ages in [0,86400] seconds. Unknown channels stay unknown.

For the last row, every reconciliation verifier must supply a typed relational result with
task fit, satisfied dependencies, intact provenance, valid authorization scope and supported
outcome. Contradiction status must be `none` and repeated-failed-route must be false. Each
affirmative finding's evidence must cover the posterior state's observable evidence references.
A generic verifier's `verified=True` cannot satisfy this continuation check.

The selected reconciliation must bind the latest observation of its action and cite every
verifier of that observation in the supplied prefix. A later verifier of any observation of
that action also blocks reuse. Both legacy and typed verifiers resolve through their validated
observation parent, so an absent envelope `action_id` cannot hide new evidence. Earlier
observations may remain in history when a newer observation has its own reconciliation.

These findings concern the earlier action. The host checks relevance to the new action and
sets `context_changed=True` when the evidence no longer applies. The host also authenticates
verifiers and evidence issuers. Event validation checks recorded relationships; it cannot
authenticate their real-world source or establish the next action's outcome.

## Backend and authority boundary

`RoutingBackend(name, version, propose)` wraps a host callback. The callback receives a
detached, frozen `RoutingFeatures` snapshot and returns one existing `RecommendedAction` enum
member. The candidate vocabulary contains eight actions. The guard accepts the required
route, escalation, or viable abstention. If the required route is `ACCEPT`, the backend may
choose another action in the vocabulary, subject to abstention viability.

An out-of-policy enum proposal falls back to the required route. Exceptions and malformed
results recommend escalation and retain a category, without copying exception text into the
diagnostic. The guard uses a separate snapshot so callback mutation cannot rewrite its
inputs. Callbacks are trusted Python code: this interface does not sandbox code or cancel a
hung callback. The host enforces service timeouts and resource limits in its adapter.

`ACCEPT` proposes continuation. The host executor still applies runtime governance, capability
grants, irreversible-action approval and evidence policy. The adapter cannot issue an
authorization token, execute a tool, alter a skill, award reward or promote a backend. When
the reason requests independent verification or reduced autonomy, the host routes to an
independent verifier or limits the executor; a route label does not perform either action.

## Candidate comparison

`evaluation.system_one_benchmark.benchmark_routing(cases, backends, policy=...)` accepts
1–10,000 unique `RoutingCase` objects and 1–32 unique backend name/version pairs. Each case
contains a request and an explicit tuple of acceptable actions. These fixture IDs are separate
from the unchanged canonical 17-case manifest. All inputs and labels are prepared before any
callback; each callback receives the same feature snapshot once per case.

The report retains every raw proposal and guarded result. Per backend it counts raw/guarded
correct routes, disallowed proposals and failures, with mean/median/maximum callback latency.
Latency includes any callback cold start and excludes causal validation. There is no combined
alignment score or automatic winner selection. A successful guard must not conceal a poor
backend's raw proposal quality.

Two executable baselines are supplied: `LEGACY_BACKEND` uses the existing router's action;
`UNCERTAINTY_BACKEND` uses the deterministic policy rules. A host classifier, JEV or CLM adapter
uses the same `RoutingBackend` contract. The CLM project exposes state/action ranking and typed
choices; this repository supplies no SDK-specific integration or pretrained-model results.

Run the reproducible synthetic contract comparison from a development checkout:

```console
python -m pytest tests/test_epistemic_routing.py tests/test_system_one_benchmark.py -q
python -c "import json; from tests.test_system_one_benchmark import *; print(json.dumps(benchmark_routing(benchmark_cases(), (LEGACY_BACKEND, UNCERTAINTY_BACKEND), policy=ENABLED), indent=2))"
```

On 2026-10-01, the 15 authored fixtures produced:

| Backend | Raw correct | Disallowed proposals | Guarded correct | Failures |
| --- | ---: | ---: | ---: | ---: |
| Legacy router v1 | 5/15 | 10 | 15/15 | 0 |
| Uncertainty rules v1 | 15/15 | 0 | 15/15 | 0 |

These fixtures check the designed policy, including unknown channels, mismatch, generic
verification, budget and high risk. The rules implement that policy, so this is a software
contract result, not evidence that the rules outperform learned backends on real tasks. No
JEV/CLM endpoint or host classifier was configured. Before choosing one, the host should run
held-out workload cases with the same evidence, budgets and candidate set, retaining model
version, hardware/service configuration, failures and cold/warm latency conditions.

## Research traceability and limits

Six primary sources informed this adapter. The [research registry](recommendations/RESEARCH_TRACEABILITY.md#ref-jev-mem)
records exact metadata, source mechanisms, repository hypotheses and limitations, linked to
REC-008. Jev-Mem motivates separating fast control from deliberation; CLM motivates a common
state/action interface; Toollery motivates a bounded candidate vocabulary; SEEK motivates
separate routing evaluation. LADDER's entity-triggered diffusion retrieval is different from
these uncertainty thresholds. The controlled cybersecurity topology study motivates retaining
hypothesis reconsideration as an option; this adapter implements no reasoning graph.

Skill banks, tool registries, memory ranking and learning/promotion remain outside this stage.
The tests verify fail-closed routing and audit consistency. They do not establish alignment,
causal evidence influence, calibrated uncertainty, source-paper reproduction or model quality.

## MDAC typed procedures

The optional `typed_routing` argument to `run_decision_workflow` consumes an evaluator-independent, recorded prediction for a frozen routing task. Deterministic tasks use the existing routing guard. Enumerable tasks select from their declared actions. Generative work goes to stronger reasoning and verification. An unknown task or explicit unknown choice escalates to human review. The target may concern clarification, epistemic uncertainty, semantic inconsistency, jailbreak suspicion, authorization status, diagnostic intervention, escalation, or reasoning depth. Authorization status is a classification target. Only the existing authority owner can issue an executable grant.

The hypothesis is that calibrated routine choices can be resolved with a smaller decision model without increasing downstream error. Li et al. compile enumerable procedural steps and calibrate a confidence threshold against a slow model. This adaptation fits a separate cutoff for each checkpoint and target using independently labeled or execution-verified cases. Teacher consensus cannot supply a ground-truth label. The procedure selects the broadest confidence prefix meeting the caller's accuracy and minimum-sample requirements, including ties together. The cutoff describes empirical selective accuracy on the calibration sample. Held-out reliability and actual compute savings require new measurements. [Thinking Fast, Anywhere, §3.3](https://arxiv.org/html/2611.23751)

Use the default-off configuration as a starting point. Supply action-bound events, existing host authentication, authority, clock and handler mappings through the same workflow documented above. A typical enumerable step looks like this. [Example configuration](../configs/mdac_typed_routing.example.json)

```python
from evaluation.improvement_records import record_digest
from evaluation.system_one_benchmark import benchmark_typed_calibration
from gepa_mindfulness.factuality_observability.decision_workflow import run_decision_workflow
from gepa_mindfulness.factuality_observability.epistemic_routing import prepare_routing
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.factuality_observability.typed_routing import (
    FastPrediction, RoutingTask, TypedRoutingInput, calibrate_threshold,
)

features = prepare_routing(request, enabled_policy)
task = RoutingTask(
    case_id, features.input_digest, "clarification", "enumerable",
    (Action.CLARIFY, Action.ACCEPT), high_consequence=False,
)
cutoff = calibrate_threshold(calibration_labels, target_accuracy=0.95, minimum_examples=20)
quality = benchmark_typed_calibration(cutoff, held_out_labels, enabled=True)
prediction = FastPrediction(record_digest(task), model_version, choice, confidence)
result = run_decision_workflow(
    request, typed_routing=TypedRoutingInput(task, prediction, cutoff),
    config=enabled_workflow, policy=enabled_policy, public=public_context,
    authenticate=authenticate, authorize=authorize, clock=clock, handlers=handlers,
)
```

A `CalibrationLabel` records its case, checkpoint, target, proposed choice and confidence, expected route, evaluator and evidence reference. Accepted label sources are `independent_label` and `verified_execution`. Hosts authenticate the source, its independence, capture lineage and execution evidence. Keep calibration examples disjoint from held-out scenarios and model inputs. The library checks case identity reuse. An unseen case family requires host-managed partitioning.

A missing cutoff, insufficient support, low confidence or unknown prediction causes deferral. A high-consequence decision requests independent verification or human review regardless of confidence. The host authenticates the exact typed input and consumes a fresh exact-scope grant before any handler runs. No migration is needed for ordinary workflow callers. The CPU tests use scripted outcomes. A model service may supply a bound public prediction from its pinned checkpoint through the same interface.

To score a recorded model, set `MDAC_TYPED_MODEL_LABELS` to a JSON object containing `calibration` and `held_out` arrays of serialized `CalibrationLabel` records, along with `minimum_examples` and `target_accuracy`. Run `python -m pytest -q tests/test_mdac_typed_routing.py -k optional_model`. Held-out Brier score, calibration error, coverage, unknown count and selective accuracy are available through `benchmark_typed_calibration`.
