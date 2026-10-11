# Verified causal reward adapter

`CausalRewardAdapter` is an explicitly invoked wrapper around `RewardPipeline`. It is
disabled by default (`enabled=False`, `weight=0.0`, `max_adjustment=0.0`). No trainer or
CLI invokes the adapter automatically. The existing 17 cases, fallback Case 0, provider
formulas and training eligibility rules are unchanged.

## Correctness and causal accounting

The baseline is the actual pipeline total, including any configured integrity overlay.
An existing incorrect-answer deduction remains intact. The additional causal deduction
addresses independently verified unjustified sensitivity or a missed required update.
A correctness-only deduction does not establish overlap with that distinct causal failure.

The host reviews the actual semantics of every active reward layer. An `already_counted`
layer means that layer already penalized **the same causal failure**, including aliases
or renamed components. Any such layer suppresses the entire additional deduction.
An `unknown` layer or missing required layer withholds optimizer access. Provider and
configuration identifiers are declarations until the host authenticates them.

A new deduction requires complete Boolean observations, `single_factor_verified=True`,
`before_correct=True`, `after_correct=False`, and `change_justified=False`. It then requires
`action_changed=True` for an irrelevant intervention, or `required_update=True` and
`update_satisfied=False` for a relevant intervention. The deduction is
`min(weight, max_adjustment)`, never a positive bonus. Correct targets, acceptable alternative
actions, justified updates, semantically compound interventions, consistently wrong controls
and unknown observations receive no causal deduction. No-op arithmetic preserves signed zero.

## API and result states

Import the adapter and decision from `gepa_mindfulness.training.causal_reward`; import the
five input records, `request_digest`, `reward_record_digest`, and `validate_input_envelope`
from `gepa_mindfulness.training.causal_reward_records`.

Construct `CausalRewardAdapter(pipeline, *, policy, authenticate_verdict=None, authorize_use=None)`.
Call `score(target_request, *, observation=None, context=None, verdict=None, accounting=None)`.
The existing observation is forwarded to the pipeline. Only the target is scored, once per call.
The comparison arm is retained evidence, not another provider call.

| Condition | Returned value | Optimizer access |
| --- | --- | --- |
| Disabled or zero weight | Same exact `RewardResult` returned by the pipeline | Existing host gates; no new accessor |
| Invalid enabled structure or binding | `ValueError` | No decision |
| Missing/disputed/unauthenticated evidence, rejected admission or use, unknown accounting | Baseline-only `CausalRewardDecision`, `disposition="withheld"` | `optimizer_reward()` raises |
| Authenticated severe violation or unknown severe state | Baseline-only decision retaining severe state | Accessor raises regardless of positive total |
| Already counted or non-penalizable, otherwise admitted and severe-cleared | `disposition="unchanged"` | Accessor returns baseline |
| Fully admitted eligible failure, not counted | `disposition="applied"` | Accessor returns baseline minus bounded deduction |

Enabled decisions expose `base_result`, `baseline_total`, `deduction`, `diagnostic_total`,
`disposition`, `reasons`, `severe_status`, and canonical JSON `audit`. Severe status is
`verified_violation`, `verified_clear`, or `unknown`. An authenticated severe verdict remains
visible even when source admission fails or accounting is missing. A claimed severe flag is
not verified until semantic authentication succeeds and the verdict is resolved.

Use `optimizer_reward()` explicitly. Decisions have no `.total`, numeric conversion or
`from_dict`. `to_dict()` exports a DEVELOPMENT audit report with no live authorization;
pickle restoration is rejected. Reading a diagnostic scalar manually bypasses the supported
API and is not authorized by the report. Each score performs fresh checks; repeated calls do
not accumulate deductions. Authorization covers the immediate scoring transaction. To consume
stored results later, rescore original inputs. Revocation after a completed transaction is a
host lifecycle responsibility.

## Exact input records

All constructors are keyword-only and all records are frozen. Each record has `to_dict()`
and `from_dict(data)`. Wire objects require every listed field plus `schema_version`; unknown
or missing fields fail. The schema tags are `causal-reward-policy-v1`,
`causal-reward-context-v1`, `causal-reward-verdict-v1`, `causal-reward-accounting-v1`, and
`causal-reward-use-request-v1`. Digests are lowercase SHA-256 of complete canonical JSON;
digests bind content but do not authenticate it.

| Record | Required fields apart from schema tag |
| --- | --- |
| Policy | `policy_id`, `policy_version`, `provider_id`, `provider_version`, `provider_config_digest`, `enabled`, `weight`, `max_adjustment`, `experiment_refs` |
| Context | `before`, `target`, `before_cell`, `target_cell`, `before_case_mapping`, `target_case_mapping`, `before_family_id`, `target_family_id`, `before_lineage`, `target_lineage`, `before_eligibility`, `target_eligibility`, `before_factors`, `target_factors`, `change`, `seed_policy`, `sources`, `target_event_refs` |
| Verdict | `context_digest`, `policy_digest`, `evaluator`, `status`, `human_required`, `relevance`, `single_factor_verified`, `before_correct`, `after_correct`, `action_changed`, `change_justified`, `required_update`, `update_satisfied`, `severe_violation`, `reason`, `evidence_refs`, `provenance` |
| Accounting | `context_digest`, `verdict_digest`, `policy_digest`, `baseline_total`, `provider_id`, `provider_version`, `provider_config_digest`, `layers`, `evidence_refs` |
| Use request | `context`, `verdict`, `accounting`, `policy`, `baseline_total`, `integrity_snapshot`, `event_key` |

Tuple fields encode as JSON arrays. A request contains exactly `trajectory` and
`observable_references`; trajectory serialization must round-trip exactly through the existing
`Trajectory` API. A cell contains `case_id`, `case_version`, `stripe_id`, `subtype`, `repeat_id`,
`seed`, `model_version`, and `harness_version`, validated against the existing registries.
A case mapping pairs the literal trajectory case ID with that cell's canonical manifest key.
Null legacy IDs and fallback Case 0 cannot establish an enabled canonical comparison.

Both arm trajectories have distinct IDs and share model, policy, backend name/version,
repeat and harness identities. Their seeds and models match their own cells. `seed_policy`
is `shared` (seeds equal) or `per_arm` (actual seeds retained). Factor maps have identical
keys with exactly the one literal difference specified by `FactorChange`. Factor values are
canonical JSON strings. A host independently checks semantic single-factor control.

Constructor `sources` pairs an `EvidenceReference` with complete JSON object text. On the wire,
each pair contains the reference object and decoded source object. The source index covers
both requests' allowed references with unique IDs. Target event references belong to the target
request and include observable action or output evidence. Verdict evidence includes these target
references even on the trusted-evaluator route. Accounting and verdict references resolve through
this index; before-only evidence cannot stand in for the target event. Context request properties
and serialized dictionaries return detached copies.

Verdict status is `verified`, `unresolved`, or `disputed`; relevance is `irrelevant`, `relevant`,
or `unknown`. All eight facts are exact Boolean or null. `human_required=True` leaves the verdict
unresolved. Contradictions rejected: irrelevant with a required update; unchanged action with a
justified change; no required update with an unsatisfied update.

`RewardProvenance.component_name` is fixed to `verified_causal_penalty`. Observable provenance
carries evidence references and no evaluator; trusted-evaluator provenance carries the same
evaluator as the verdict and no provenance references. Both routes still require target event
evidence. Private reasoning, latent state, attention and cache references are not causal support.
Exact existing trajectory fields, including trace-reference strings, remain bound for identity;
they do not become reward evidence. No callback exception text is exported.

Accounting layer names are unique; statuses are `not_counted`, `already_counted`, or `unknown`.
The required roster includes `provider` and, with an active integrity overlay, `reward_integrity`.
Host authorization verifies all provider-internal layers and any additional names. Incorrect
baseline, digest or provider bindings raise `ValueError`; incomplete semantics withhold.
The use request includes the actual complete integrity breakdown, including provenance,
weights and self-serving-justification diagnostics. Constructor `integrity_snapshot` is JSON
object text or null; the wire field is a decoded object or null.

Bounds: two arms, one target per call, 64 factor keys per arm, 256 source entries, 256 unique
lineage IDs per arm, 32 accounting layers and 16 external-record experiment references.
Record identity strings are limited to 256 characters, verdict reasons to 4,096 characters.
The complete input envelope is at most 4 MiB UTF-8 with depth at most 32 (root depth one).
Cycles, unsupported values, nonfinite numbers and Boolean numeric configuration fail.
Inputs are rejected rather than truncated. Positive weight requires a positive cap and
nonempty external-record experiment references, even if the adapter is disabled.

## Host responsibilities and live-use gate

`authenticate_verdict` receives a detached dictionary containing full `context`, `policy`, and
`verdict`. The host resolves current evidence and evaluator permissions, independently checks
correctness, acceptability, relevance, semantic single-factor control and severe status, and
returns built-in `True` only after authenticating those exact claims. A different evaluator ID
alone does not establish independence from the producing model or optimizer.

`authorize_use` receives the full use-request wire object. The host authenticates current
policy/experiment permission, admitted source-family/lineage completeness, the actual provider
configuration, all active layers' semantics and current source/evaluator permissions. The host
resolves alternate IDs for the same underlying event to prevent duplicate causal charging.
The local event key covers the complete target request and sorted target event references;
comparison labels, metric names and reasons do not multiply a charge.

Both callbacks receive fresh mutable copies. Mutation, exceptions, missing callbacks or returns
other than built-in `True` reject that gate. The adapter retains original facts. A missing
required record prevents invoking a callback with an incomplete payload. Structurally valid
semantic authentication runs even if a separate admission gate fails.

When an accounting record is present, the adapter checks its baseline's canonical JSON encoding
against the actual pipeline total before invoking either callback. Numerically equal but
differently encoded values, such as `0.0` and `-0.0` or `1` and `1.0`, raise `ValueError` for an
accounting baseline mismatch. Hosts must retain the pipeline total's scalar type and signed zero
when constructing accounting records.

Both arms and each source need explicit TRAIN declarations. The existing recursive admission
checker also inspects complete retained request/source data: nested DEVELOPMENT, REGRESSION,
HIDDEN_EVAL, failed V5 records and incomplete generated-data review remain excluded. An outer
TRAIN label never overrides a nested restriction. Local validation cannot detect omitted
provenance; the host must resolve complete origin records before authorizing use. Diagnostic
`CausalPair` records and pilot captures are not relabeled as training examples.

The approved implementation and authored fixtures do not activate a live penalty. Real use
requires independently authenticated causal evidence, admitted lineage, exact causal accounting
and separately authorized experiment settings. The pilot has not established these conditions.
PR-7 implementation does not depend on an active PR-5 penalty; training admission and empirical
improvement/deployment claims retain their separate gates. See [ADR 0021](adr/0021-causal-alignment-diagnostic-extension.md)
and the existing [epistemic process reward contract](epistemic_process_rewards.md).

## Reason order and verification

Applicable reasons appear in this fixed order: `missing_context`, `missing_verdict`,
`unresolved_verdict`, `authentication_rejected`, `admission_rejected`, `use_not_authorized`,
`severe_violation`, `severe_unknown`, `accounting_unknown`, `already_counted`, `not_attributable`,
`target_correct`, `adjustment_applied`. The audit retains separate gate outcomes and source
admission details. An unchanged numeric baseline does not imply optimizer eligibility.

`tests/test_causal_reward_records.py` checks exact schemas, bounds and snapshots;
`tests/test_causal_reward_admission.py` checks recursive admission, attribution and event identity;
`tests/test_causal_reward.py` checks callbacks, mutation, no-op compatibility and severe access;
`tests/test_causal_reward_integration.py` checks both provenance routes, real reward composition,
revocation, host alias/origin resolution and the example below. Production activation additionally
requires host integration review of independent verification, exhaustive source/configuration
resolution, revocation, budgets and semantic duplicate detection. Those responsibilities cannot
be demonstrated by the fixture callbacks.

## Executable offline example

This example uses authored TRAIN-origin objects and explicitly named fixture callbacks. It makes
no network calls, writes no training data, and supplies no independent live-use evidence. Copy
the block into Python with the repository installed. It prints deterministic full audit JSON for
adjusted, unchanged, withheld and severe decisions, plus default passthrough identity.

```python
import json
from dataclasses import replace

from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import FactorChange, canonical_json
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance, TrustedEvaluatorContract, VerificationRoute,
)
from gepa_mindfulness.training.causal_reward import CausalRewardAdapter
from gepa_mindfulness.training.causal_reward_records import (
    CausalRewardAccounting, CausalRewardContext, CausalRewardPolicy,
    CausalRewardVerdict, reward_record_digest,
)
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.trajectory import Trajectory


class FixtureProvider:
    def score(self, request):
        return 0.8


class RecordingFixturePipeline(RewardPipeline):
    def score(self, request, *, observation=None):
        self.last = super().score(request, observation=observation)
        return self.last


refs = tuple(EvidenceReference(arm, EvidenceSourceKind.OBSERVABLE_OUTPUT)
             for arm in ("before", "target"))
requests = tuple(RewardRequest(Trajectory(
    trajectory_id=arm, case_id="authored-case", prompt=f"{arm} public prompt",
    response="correct" if arm == "before" else "wrong", backend_name="fixture",
    backend_version="1", model_identifier="fixture-model", policy_version="1", seed=11,
    evidence_references=(ref,), sampling_parameters={"training_eligibility": "TRAIN"},
), (ref,)) for arm, ref in zip(("before", "target"), refs))
cell = V5EvaluationCell(1, "17case-v5", "NONE", None, 0, 11, "fixture-model", "fixture-harness")
key = next(case.key for case in load_case_manifest().cases if case.id == 1)
context = CausalRewardContext(
    before=requests[0], target=requests[1], before_cell=cell, target_cell=cell,
    before_case_mapping=("authored-case", key), target_case_mapping=("authored-case", key),
    before_family_id="authored-family", target_family_id="authored-family",
    before_lineage=("authored-before",), target_lineage=("authored-target",),
    before_eligibility="TRAIN", target_eligibility="TRAIN",
    before_factors=(("pressure", "false"),), target_factors=(("pressure", "true"),),
    change=FactorChange("pressure", "false", "true"), seed_policy="shared",
    sources=tuple((ref, canonical_json({"training_eligibility": "TRAIN"})) for ref in refs),
    target_event_refs=(refs[1],),
)
disabled_policy = CausalRewardPolicy(
    policy_id="fixture-only", policy_version="1", provider_id="fixture",
    provider_version="1", provider_config_digest="a" * 64,
)
pipeline = RecordingFixturePipeline(FixtureProvider())
unchanged_base = CausalRewardAdapter(pipeline, policy=disabled_policy).score(context.target)
output = {"disabled": {"same_object": unchanged_base is pipeline.last, "total": unchanged_base.total}}
policy = replace(disabled_policy, enabled=True, weight=0.25, max_adjustment=0.1,
                 experiment_refs=(EvidenceReference("fixture-experiment", EvidenceSourceKind.EXTERNAL_RECORD),))
evaluator = TrustedEvaluatorContract("fixture-reviewer", "1", "authored-test-only")
verdict = CausalRewardVerdict(
    context_digest=reward_record_digest(context), policy_digest=reward_record_digest(policy),
    evaluator=evaluator, status="verified", human_required=False, relevance="irrelevant",
    single_factor_verified=True, before_correct=True, after_correct=False, action_changed=True,
    change_justified=False, required_update=False, update_satisfied=True, severe_violation=False,
    reason="Authored unjustified flip; not empirical evidence", evidence_refs=(refs[1],),
    provenance=RewardProvenance("verified_causal_penalty", "fixture comparison",
                                VerificationRoute.TRUSTED_EVALUATOR, evaluator=evaluator),
)


def fixture_authenticate(payload):
    # This allowlist is software-test scaffolding, not independent semantic authentication.
    return payload["context"] == context.to_dict() and payload["policy"] == policy.to_dict()


def fixture_authorize(payload):
    # Real hosts resolve complete lineage, active config, causal layer semantics and permissions.
    return payload["baseline_total"] == 0.8 and payload["integrity_snapshot"] is None


adapter = CausalRewardAdapter(pipeline, policy=policy,
    authenticate_verdict=fixture_authenticate, authorize_use=fixture_authorize)
for name, current in (
    ("adjusted", verdict), ("unchanged", replace(verdict, after_correct=True)),
    ("severe", replace(verdict, severe_violation=True)),
):
    accounting = CausalRewardAccounting(
        context_digest=reward_record_digest(context), policy_digest=reward_record_digest(policy),
        verdict_digest=reward_record_digest(current), baseline_total=0.8,
        provider_id=policy.provider_id, provider_version=policy.provider_version,
        provider_config_digest=policy.provider_config_digest,
        layers=(("provider", "not_counted"),), evidence_refs=(refs[1],),
    )
    decision = adapter.score(context.target, context=context, verdict=current, accounting=accounting)
    output[name] = decision.to_dict()
    if name == "severe":
        try:
            decision.optimizer_reward()
        except ValueError:
            output["severe_accessor_blocked"] = True
    else:
        decision.optimizer_reward()
output["withheld"] = adapter.score(context.target, context=context, verdict=verdict).to_dict()
print(json.dumps(output, sort_keys=True))
```
