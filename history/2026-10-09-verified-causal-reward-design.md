# PR-5: independently verified causal reward adapter

Date: 2026-10-09.
Status: Written design approved on 2026-10-09; the maintainer explicitly corrected implementation and activation gates on 2026-10-10. The existing approved implementation plan/native method may proceed with the default-disabled adapter and authored tests. Live deductions and training use still require independently authenticated evidence and separate activation authorization.
Inspected baseline: PR-4 merge `b6a7de7c339c3b24b8a95e585e7f0b403bb8a8ff`, tree-equivalent to local `35c50c5e7b4ab8fb28cd19d755f7765ade0b46ac`.

## 1. Intent and acceptance boundary

The maintainer requested a separately reviewable, optional causal reward adjustment after the four diagnostic PRs. An independently verified inappropriate decision under a controlled intervention may receive a bounded negative adjustment. Existing rewards remain the starting point.

The selected design is a separately invoked adapter around `RewardPipeline`. Existing providers, the integrity overlay, canonical identities, training admission and action authorization retain their contracts. No existing trainer or command-line default invokes the adapter.

The design succeeds when an enabled call can explain every applied or withheld adjustment using exact authenticated inputs. Disabled calls return the existing pipeline result directly. Correct behavior, acceptable alternatives, appropriate updates and unverified judgments receive no causal deduction. A severe violation cannot be cleared by a positive scalar.

This is architectural work under the brainstorming skill. The written design and subsequent implementation plan/native method were approved. The 2026-10-10 correction removes the live-failure-example prerequisite for software implementation; none of these approvals supplies independent evidence or activates an experiment.

## 2. Reuse and module boundaries

| Existing contract | Use in PR-5 |
| --- | --- |
| `RewardPipeline`, `RewardResult`, `RewardRequest`, `Trajectory` | Call the existing pipeline once for the target, preserve its result, and use its request and trajectory as the reward boundary. |
| `RewardProvenance`, `VerificationRoute`, `TrustedEvaluatorContract` | Preserve the exclusive observable-evidence and trusted-evaluator routes. |
| `EvidenceReference` and its observable source-kind predicate | Limit supporting references to observable records already permitted by the requests. |
| `V5EvaluationCell` | Validate each arm's canonical case, stripe, subtype, repeat, seed, model and harness coordinates. |
| `FactorChange` and canonical JSON/content digest helpers | Record the single intervention and bind complete detached payloads. |
| `require_training_eligible` | Retain nested source restrictions, generated-data review and failed-V5 rejection. |
| PR-1 through PR-4 diagnostics | Supply design controls and evaluation evidence; never convert their reports into reward authority or training examples. |

Proposed new production files:

- `gepa_mindfulness/training/causal_reward_records.py`: exact wire records, bounded validation and digests.
- `gepa_mindfulness/training/causal_reward.py`: optional composition, authentication and withholding decisions.

Proposed supporting files: focused records/adapter/integration tests, `docs/causal_reward_adapter.md`, an ADR 0021 update recording approvals actually received, and documentation packaging/link coverage if the new guide requires it.

Do not modify the canonical manifest, stripe taxonomy, existing reward formulas, `CausalPair` eligibility, `require_training_eligible` rules, provider interfaces or training CLI defaults. A change to one of those contracts is outside this design.

## 3. Training-origin evidence and admission

The new interface consumes a host-authored review of two existing training-origin `RewardRequest` objects. It is not a new general-purpose causal evaluation engine. The host supplies the complete retained source provenance for both requests and all evidence referenced by the review.

Each arm has a validated `V5EvaluationCell`. The host's explicit case mapping connects `Trajectory.case_id` to the cell's manifest identity; the adapter checks that mapping against the manifest. A missing or inconsistent mapping prevents enabled reward use. The two cells may have different canonical cases when the intervention legitimately changes the case. Both cells retain their correct identities.

The host supplies explicit `TRAIN` admission declarations for both arms and each source-provenance entry. Missing declarations are ineligible for this new interface even where a legacy loader accepts untagged data. The adapter invokes the existing recursive eligibility check on both serialized requests and all source records. Non-TRAIN content, malformed provenance and failed V5 outcomes withhold optimizer access without changing their labels.

An incorrect action observed in a training-origin rollout is distinct from an existing failed V5 evaluation record. The adapter does not transform the latter into the former. When the source is a failed V5 record or diagnostic capture, the host must retain that lineage and the adapter rejects admission. When the host cannot establish the origin, admission is unknown and reward use is withheld.

The host independently resolves source identities and attests that provenance is complete. Local schema checks cannot detect a host that deliberately omits a source. The guide must state this trust boundary and require a test that a retained nested restriction defeats an outer TRAIN label.

No admissible training-origin failure dataset has been established in this chat. If no such inputs are available under current policy, the adapter has no eligible penalty examples. This design does not weaken admission to manufacture an executable demonstration.

## 4. Public API and compatibility

Proposed interface:

```python
adapter = CausalRewardAdapter(
    pipeline,
    policy=policy,
    authenticate_verdict=authenticate_verdict,
    authorize_use=authorize_use,
)
result = adapter.score(
    target_request,
    observation=existing_integrity_observation,
    context=reward_context,
    verdict=verdict,
    accounting=accounting,
)
```

`pipeline` is the existing `RewardPipeline`. `observation` is forwarded through its existing API. The comparison arm is contextual evidence; the adapter never calls the provider to score that arm.

When `policy.enabled` is false or its weight is zero, the adapter calls `pipeline.score(target_request, observation=observation)` once and returns that same `RewardResult` object. It does not inspect optional causal inputs or invoke either causal callback. Existing provider errors propagate as before. Configuration itself is validated at construction.

When enabled with nonzero weight, the adapter returns a `CausalRewardDecision`. This new result deliberately has no `.total`, `__float__`, or implicit optimizer conversion. Existing provider readers therefore cannot silently consume a diagnostic total. Callers explicitly use `optimizer_reward()` after the required checks. No existing trainer is modified to do so in PR-5.

The enabled decision retains the original `RewardResult`, an immutable snapshot of its scalar `total`, the nonnegative causal deduction, the diagnostic total, the disposition and reasons, the severe-event state, and exact audit bindings. `diagnostic_total` equals the baseline total when an adjustment is withheld. That numeric equality does not override the decision's optimizer-admission state.

`optimizer_reward()` returns the diagnostic total only for a live, fully admitted decision with independently cleared severe-event status. Otherwise it raises `ValueError` with a stable reason. Exported reports are audit-only and cannot be deserialized into an optimizer-authorized decision. To consume a stored report, an application must rescore the original inputs and perform fresh checks. There is no cached approval restore path.

The accessor limits accidental misuse by supported integrations. It cannot prevent arbitrary application code from reading a diagnostic field and bypassing the API. Host integration review remains mandatory before optimizer activation.

## 5. Records and exact bindings

Every new input record is immutable after construction, has an exact versioned wire schema, and rejects unknown fields. Snapshot mutable nested structures into canonical JSON-backed values. Reject Boolean numeric inputs, nonfinite numbers, duplicate IDs, non-observable evidence and malformed digests.

| Record | Required content |
| --- | --- |
| `CausalRewardPolicy` | Policy ID/version; `enabled=False`; finite nonnegative `weight=0.0`; finite nonnegative `max_adjustment=0.0`; effective baseline-provider/configuration identity; reviewed experiment authorization references. A positive weight requires a positive maximum and nonempty authorization references. |
| `CausalRewardContext` | Before and target request snapshots; per-arm V5 cells and explicit case mappings; source-family and lineage identities; per-arm factor mappings; exactly one `FactorChange`; explicit shared or independently recorded seed policy; complete source records keyed by observable evidence reference; explicit per-source/arm admission declarations. |
| `CausalRewardVerdict` | Context digest; policy digest; evaluator; status; `human_required`; nullable exact-Boolean facts listed below; relevance; bounded reason; observable evidence references; `RewardProvenance` whose component is `verified_causal_penalty`. |
| `CausalRewardAccounting` | Context, verdict and policy digests; actual baseline scalar; effective provider/configuration identity; exhaustive named baseline/overlay layers; per-layer overlap disposition; source references establishing the accounting. |
| `CausalRewardUseRequest` | Complete detached context, verdict, accounting and policy; actual baseline scalar; actual integrity breakdown if present; computed event key. This is the payload presented to the host authorization callback. |
| `CausalRewardDecision` | Base result plus numeric snapshot, applied deduction, diagnostic total, disposition, ordered withholding reasons, severe status, audit digests, provenance, evidence references and a non-serialized live admission state. |

Verdict facts are `single_factor_verified`, `before_correct`, `after_correct`, `action_changed`, `change_justified`, `required_update`, `update_satisfied`, and `severe_violation`. Each is `True`, `False`, or `None`. Status is `verified`, `unresolved`, or `disputed`; relevance is `irrelevant`, `relevant`, or `unknown`. A status string does not authenticate the record.

Both factor maps have identical keys. Their literal differences must equal exactly the declared `FactorChange`. The independent verifier additionally confirms that the complete observable prompts and conditions actually differ only as declared. A literal factor map alone cannot prove semantic single-variable control.

The before and target requests need different trajectory IDs. Both requests record their actual prompt, response and relevant observable action/outcome references. Both cells and trajectories must agree on model identity, recorded seed and case mapping. The pair retains the same model, policy version, backend and harness; repeat IDs match. A shared seed policy requires equal seeds. Missing model/policy/harness or seed data cannot establish an enabled penalty. Independent seeds are recorded rather than silently treated as shared.

Request digests cover every field of `Trajectory.to_dict()` and the exact allowed observable references. No reward, sampling, policy or provenance field is dropped to simplify a join. Both arm payloads, cell identities, factors, admission declarations and all source records contribute to the context digest. Matching text or a matching user-provided event label is insufficient.

Source-reference IDs are unique within the context, with one source kind and one retained source record each. Every reference used by the verdict or accounting must resolve through this index and appear in the corresponding request's allowed observable references. A target penalty requires observable evidence of the target action/outcome even on the trusted-evaluator provenance route. The comparison request may supply comparison-only evidence, but does not expand the target request's evidence boundary.

The existing `RewardProvenance` routes stay exclusive: observable provenance carries references and no evaluator; trusted-evaluator provenance carries the evaluator and no references in that provenance object. The latter route's separate action/outcome evidence still exists in the verdict and context. When the route contains an evaluator, it must equal the verdict's evaluator.

Host policy/experiment authorization references are control-plane records authenticated by `authorize_use`. They are not factual action evidence and do not become trajectory evidence or a workaround for source eligibility.

Bounds for the first schema: one target decision per call, two arms, at most 64 factor keys per arm, 256 source-reference entries, 32 accounting layers, and 16 experiment authorization references. IDs are at most 256 characters and reasons at most 4,096 characters. The complete new canonical input envelope is at most 4 MiB with JSON nesting at most 32. Existing trajectory restrictions still apply. An oversized input raises a validation error; it is never truncated.

## 6. Authentication and use authorization

Authentication has two explicit responsibilities:

1. `authenticate_verdict` receives a detached payload containing the full context, policy and verdict. It resolves the evaluator's identity and independently checks the exact facts against current authorized evidence. It returns the built-in value `True` only for acceptance.
2. `authorize_use` receives the detached `CausalRewardUseRequest`. It checks current policy/experiment permission, complete training-origin lineage, actual baseline-provider configuration and exhaustive overlap accounting. It also confirms the current source/evaluator permissions needed to use the evidence for the specified target.

The adapter compares callback payloads with their pre-call snapshots. Mutation, an exception, missing callback or any result other than built-in `True` withholds the adjustment. A callback receives no authority to edit the retained request or reward decision. There are no process-wide trusted verdict caches.

The adapter does not infer evaluator independence from a different string ID. Host integration review must establish separation from the producing model and optimizer, exact record authentication, source resolution, and revocation behavior. Fixture allowlists demonstrate software behavior only.

A changed target, model, policy version, source, factor, verdict, baseline scalar or accounting layer requires a different binding and fresh authorization. Repeated scoring is a fresh evaluation of the same inputs, not an accumulated debit. Use authorization applies only to the immediate host scoring transaction; persistent delayed consumption requires rescoring. Source revocation after a completed transaction remains a host lifecycle responsibility.

## 7. Causal deduction and accounting

Compute `deduction = min(weight, max_adjustment) * failure`, where `failure` is either 0 or 1. No positive causal bonus exists. The effective baseline scalar is `RewardPipeline.score(...).total`, including an existing configured integrity overlay. The diagnostic total is the baseline minus the deduction. If the deduction is zero, assign the baseline scalar directly rather than recomputing it; preserve exact numeric equality, including signed zero. Reject a nonfinite arithmetic result.

An eligible penalty requires complete observations, authenticated single-factor control, `before_correct=True` and `after_correct=False`. The conservative baseline-correctness requirement avoids attributing an existing error to the intervention. Then require one of these two predicates:

| Relevance | Additional predicate | Failure |
| --- | --- | --- |
| Irrelevant | `action_changed=True` and `change_justified=False` | 1 |
| Relevant | `required_update=True`, `update_satisfied=False`, and `change_justified=False` | 1 |
| Any other combination | Correct target, unknown fact, no required update, justified change, or unattributed error | 0; report the specific disposition |

Reject contradictory facts when `relevance=irrelevant` and `required_update=True`, when `action_changed=False` and `change_justified=True`, or when `required_update=False` and `update_satisfied=False`. An unknown Boolean is not a contradiction. Correctness and update satisfaction remain independent: satisfying one required update does not prove the entire target action correct. A verified `change_justified=True` always exempts the action change from a causal deduction, including on the relevant-intervention path. Do not derive semantic action change from string inequality. For independently verified correct targets, no penalty is possible even if a diagnostic label says failure.

`CausalRewardAccounting` identifies every active reward layer, including the provider and any integrity overlay. Each layer declares `not_counted`, `already_counted`, or `unknown` for the independently verified causal failure on the exact event, not merely for target-answer incorrectness. The host authenticates completeness against the effective configuration. Only all-`not_counted` accounting permits a new deduction. Any already-counted or unknown layer withholds the entire additional penalty; no inferred fractional subtraction is attempted.

Compute the event key from the target request digest and the canonical sorted identities of the target action/outcome evidence. Do not include arbitrary pair labels, metric names, display reasons or verifier aliases. One call handles one event, so multiple causal explanations cannot multiply its deduction. The host's accounting review must also detect semantically identical events represented by alternate external source IDs; digest equality alone cannot solve that problem.

The existing incorrect-answer penalty measures target correctness. The additional causal deduction measures independently verified unjustified sensitivity or a missed required update. Penalizing an incorrect answer alone does not count the distinct causal failure: both deductions may coexist only when every other authorization and admission condition passes. A provider or overlay already penalizing that same causal failure counts as overlap and withholds the additional deduction. The host must review each active layer's actual semantics; a low score, renamed component or shared event ID cannot determine overlap by itself. Unknown causal overlap remains withheld. The causal component is fixed as `verified_causal_penalty`; renaming the same causal failure cannot create a second charge.

## 8. Severe events and decision states

An authenticated `severe_violation=True` creates a blocked result independently of weight arithmetic and overlap. The optimizer accessor raises even when the baseline and diagnostic total are positive. A severe event is not assigned an invented scalar severity weight. Preserve the event and evidence in the audit result.

If severe status is unknown or the verdict is unauthenticated, an enabled result cannot expose an optimizer reward through the new accessor. The diagnostic numeric value remains the baseline. This is a restriction on the new enabled API; existing disabled pipeline behavior remains unchanged.

Evaluate semantic verification, admission, accounting and severe status as separate dimensions. A known severe event remains visible when accounting is missing or source admission fails. Do not short-circuit a structurally valid authenticated verdict merely because a different gate withholds the deduction.

| Condition | Numeric result | New optimizer accessor |
| --- | --- | --- |
| Disabled or zero weight | Return the existing `RewardResult` directly | Not introduced on this path; existing host gates apply |
| Structurally invalid enabled input | Raise `ValueError` | No result |
| Missing, disputed or unauthenticated verdict | Preserve baseline | Raise |
| Source/admission/policy authorization unavailable or rejected | Preserve baseline | Raise |
| Authenticated severe violation | Preserve baseline; retain severe audit event | Raise |
| Accounting unknown | Preserve baseline | Raise |
| Already counted, otherwise fully admitted and severe-cleared | Preserve baseline | Return baseline |
| Verified non-penalizable behavior, fully admitted and severe-cleared | Preserve baseline | Return baseline |
| Verified eligible failure, not counted, fully admitted and severe-cleared | Subtract bounded deduction | Return adjusted scalar |

Use stable reason codes, in this order when several apply: `missing_context`, `missing_verdict`, `unresolved_verdict`, `authentication_rejected`, `admission_rejected`, `use_not_authorized`, `severe_violation`, `severe_unknown`, `accounting_unknown`, `already_counted`, `not_attributable`, `target_correct`, `adjustment_applied`. Include all applicable reasons; a display reason never changes the predicate. Unsupported compound attribution maps to `not_attributable` and cannot carry an applied deduction.

## 9. Scoring sequence and side effects

1. Validate and snapshot policy at adapter construction. The policy is not mutable between calls.
2. On a disabled or zero-weight call, use the direct existing-pipeline path described above.
3. On an enabled call, detach and validate the target request and any supplied causal records. Missing optional records are withholding states. Malformed supplied records raise errors. No callback runs before structural validation.
4. Call the existing pipeline exactly once for the detached target and forwarded existing observation. Snapshot its finite total and active integrity breakdown. Verify that the target snapshot was not changed by provider execution. Provider exceptions or mutation do not produce a causal reward result.
5. Check exact bindings and existing admission rules. Authenticate any structurally valid supplied verdict even when admission has failed, so a verified severe event remains visible.
6. Validate accounting against the actual baseline result. Invoke use authorization for complete structurally valid inputs; retain its outcome independently from semantic facts.
7. Determine severe, overlap and causal predicates. Apply a deduction only if every required condition is established. Return a decision holding the baseline result and the independent audit dimensions.

The adapter performs no network requests, file writes, source fetching, generated-code execution, model calls, persistent event storage or trainer updates. Host callbacks own any external verification and must be budgeted separately by the application. Inputs, callbacks and exceptions must not cause private reasoning to be included in exported evidence. Existing trace-reference fields may be retained for exact request identity but cannot be used as causal reward evidence.

## 10. Separate implementation, activation and empirical-evidence gates

The maintainer corrected the earlier gate on 2026-10-10: an admissible live causal failure is not a prerequisite for implementing and testing the default-disabled adapter. The existing approved plan and native execution method govern software work. Authored fixtures may exercise applied, withheld and severe-blocked paths, but must be labeled software-contract tests rather than empirical causal evidence.

Before a real nonzero deduction or optimizer use, the host must authenticate the exact verdict, complete admitted training-origin lineage, provider/overlay accounting of the distinct causal failure, and policy authorization. The experiment packet identifies the independent evaluator and method, actual model/harness versions, planned controls, all attempted/missing/disputed observations, per-arm correctness and adjudicated causal predicates. Include appropriate updates, acceptable alternatives, consistently wrong controls and unavailable/revoked evidence. Held-out diagnostic records remain excluded from training.

The completed pilot establishes captured observations, including one correct-to-incorrect stochastic flip. It does not establish independently verified unjustified sensitivity. Existing default reward deductions establish incorrect-answer accounting, not automatic accounting of that distinct causal failure. The pilot remains inadmissible for a live additional deduction because causal verification, source admission and exact use authorization are not established; no captured result is relabeled.

No nonzero deployment weight is approved. A real experiment needs a separately reviewed weight, cap, host integration and activation record. The approved accounting distinction is not evidence that the pilot qualifies or that causal reward training improves behavior.

PR-7 design and offline curation may proceed through their normal design/plan approvals. PR-7 implementation follows prerequisite software work; it does not require an active PR-5 penalty or discovery of a qualifying live failure. Curriculum admission, live training, withheld-scenario usefulness and deployment remain separate checks. PR-6 supplies the independent evaluation reporting boundary; a fixture result supplies none of those empirical claims.

**Current disposition:** software implementation is no longer blocked by the missing live failure packet. The additional pilot penalty stays withheld, and live experiment/training activation is not authorized by this correction.

## 11. Validation matrix

| Contract | Verification |
| --- | --- |
| Default and zero-weight compatibility | Exact `RewardResult` object identity, scalar signed-zero preservation, provider once, causal callbacks zero, ignored malformed optional causal inputs. |
| Structural bindings | Round-trip exact schemas; unknown keys; duplicate IDs; nonfinite/Boolean numerics; nesting/size limits; per-arm V5 changes; shared and independent seed rules. |
| Authentication | False/truthy/non-Boolean results, exceptions, callback mutation, changed sources, changed baseline scalar, swapped target, policy/evaluator replay and repeated rescoring. |
| Eligibility | Explicit training-origin success case; missing declaration; nested non-TRAIN; failed V5 source; stripped-origin host rejection; generated-source review requirements. |
| Causal predicates | Both failure modes; correct/justified updates; acceptable alternatives; wording-only differences; consistently wrong baseline; unknown relevance/correctness; compound changes. |
| Accounting | Incorrect-answer penalty alone permits a separately verified causal deduction; a provider/overlay already counting the same causal failure withholds it; missing layers, unknown overlap, aliases and repeated scoring cannot add duplicate deductions. |
| Severe events | Positive base plus known severe event blocks accessor; unknown severe blocks enabled accessor; known severe remains visible despite rejected admission or missing accounting. |
| Provenance and evidence scope | Both exclusive routes; mismatched evaluator/component; private reasoning; foreign references; target outcome evidence missing; altered reference source kind. |
| Integration | Existing reward, V5, causal diagnostics, training eligibility and CLI regressions; no new adapter call sites in current trainers. |
| Publication | Full suite, Ruff, Black, scoped mypy, source-line limit, wheel imports, executable offline guide, package/source parity and diff review. |

Baseline evidence from the preceding policy review: **480 passed, 6 warnings in 12.53 seconds** in `work/pr5-policy-baseline.log`. This is a historical result, not a test run of this proposed implementation. No production implementation exists in this stage, so the matrix defines future acceptance checks rather than claiming them passed.

The tests do not need a model service. Fixture authentication is explicitly labeled authored. A successful fixture cannot supply the live activation gate or a production trust anchor.

## 12. Scope, alternatives and review disposition

The chosen adapter keeps existing entry points stable and exposes the new risk at an explicit call site. Extending `RewardPipeline.score()` directly would make the new gate easier to invoke accidentally and enlarge compatibility testing. Accepting PR-1 failure labels as penalties would bypass correctness, eligibility and authentication. Relaxing failed-V5 admission would change an existing training contract. Those alternatives are outside the selected design.

Self-review applied the repository documentation precision gate. The design explicitly distinguishes unchanged diagnostic numerics from optimizer admission, defines overlap independently of score magnitude, binds authorization to actual baseline output, and separates implementation readiness from live activation evidence. Every operative requirement is covered by the validation matrix or host experiment/integration review. Remaining external evidence requirements are stated as readiness conditions, not hidden implementation decisions.

The approved design includes the training-origin schema and conservative `before_correct=True` attribution rule. The approved implementation plan now treats independent empirical evidence as a live-use prerequisite, not a prerequisite for Task 1. This correction authorizes no live penalty, training run, paid experiment or deployment.
