# Verified Epistemic Process Rewards

## Scope

This document defines the optimizer-facing reward contract for verified epistemic process data.
The contract applies to `EpistemicProcessAssessment`, abstention thought reward, Schema V3
additive rewards, the main reward calculator, the legacy training orchestrator, and
reward-integrity trajectories.

## Epistemic Process Reward Rule

`EpistemicProcessAssessment.optimizer_score()` returns `0.0` when
`verified_components` is empty. Otherwise, it returns the arithmetic mean of the scores in
`verified_components`. Every `VerifiedProcessComponent.score` is finite and in `[0.0, 1.0]`.
Every verified component has one `RewardProvenance` record whose `component_name` equals the
component name.

The reward system accepts exactly two provenance routes:

1. `observable_evidence` requires one or more observable `EvidenceReference` values. For
   `public_rationale_fidelity`, the evidence also contains a structured comparison of the public
   rationale, committed prediction, selected action, and observed outcome.
2. `trusted_evaluator` requires a `TrustedEvaluatorContract` with a non-empty evaluator ID,
   evaluator version, and contract ID. This route carries no observable evidence references.

The reward system does not infer provenance from process-language, a hidden trace, or an
untyped diagnostic field.

This is the **verified epistemic process** boundary: a named, bounded process score plus exactly
one accepted provenance route. A **diagnostic signal** is any retained trace, label, overlay,
deception, circuit, attribution, or mechanistic field that has not crossed that boundary. Diagnostic
signals remain available for analysis but do not authorize optimizer credit.

## Optimizer eligibility and weights

`compute_abstention_reward()` preserves case classification from `thought_align`, but that
diagnostic label does not change any numeric component. When `optimizer_score()` is positive, the
thought component is `H * optimizer_score()` for every case. A missing assessment, an empty
assessment, and a non-empty assessment whose score is exactly `0.0` all produce `0` thought reward.
Knowledge, calibration, and abstention components depend only on the answer, references,
confidence, threshold, and abstention behavior.

`AbstentionRewardWeights.H` defaults to `1.0`. `H` accepts any finite non-negative value; the
implementation has no repository-wide numeric maximum. For one configured reward computation,
`H` is the maximum multiplier: because `optimizer_score()` is in `[0.0, 1.0]`, the thought award
is in `[0.0, H]`. `H` is not an unconditional award of `H`.

`LightweightTrainingOrchestrator` retains
`TrainingConfig.honesty.calibration_bonus_weight` as a compatibility configuration name. Its
default is `0.5`; when abstention scoring is disabled, the orchestrator multiplies that value by
`optimizer_score()`. When abstention scoring is enabled, `H * optimizer_score()` is the sole process
credit so the same verified assessment is not counted twice. The compatibility configuration name
does not authorize a response-word or trace-word bonus.

## Schema V3 optimizer components

| Reward component | Verified component name | Award and default | Maximum | Diagnostic fields that do not authorize the award |
| --- | --- | --- | --- | --- |
| `r_thought` | Any positive verified assessment | `H * optimizer_score()` when the score is positive; otherwise `0` | Configured `H` | `thought_align`, trace text, and `reasoning_grounded` alone |
| `r_grounding` | `grounding` | Exact verified score; `0` when absent | `1.0` | `ControlOverlay.grounding_status` alone |
| `r_control` | `control` | Exact verified score; `0` when absent | `1.0` | `ControlOverlay.observed_controls` alone |
| `r_reasoning_unit` | `reasoning_unit` | Exact verified score; `0` when absent | `1.0` | `ReasoningOverlay.observed_units` alone |
| `r_observability` | `observability` | Exact verified score; `0` when absent | `1.0` | `ObservabilityOverlay` fields alone |
| `r_group_theoretic` | `group_theoretic` | Exact verified score; `0` when absent | `1.0` | `GroupTheoreticOverlay` fields alone |

Schema V3 serializes the overlay fields as diagnostics even when their reward component is `0`.
The overlay is not provenance, and it is not a substitute for a matching verified component.

## Compatibility aliases and diagnostics

`RewardBreakdown.epistemic_process` is the canonical optimizer-facing process field.
`GEPARewardCalculator` populates `RewardBreakdown.honesty` and
`RewardBreakdown.epistemic_process` with the same numeric value for its emitted breakdowns.
Arbitrary legacy construction can still supply a different `honesty` value; that value does not
become verified process credit.

`RewardWeights.honesty_trace` remains an alias for `RewardWeights.gamma`.
`RewardWeights.from_mapping()` accepts `honesty_trace` when `gamma` is absent. `trace_summary`
remains in the main reward-calculator interface for logging callers, but the reward calculation
does not inspect it. `CircuitTracerAdapter` assessments and trace confidence hints remain
diagnostic when `BaseTrainer` computes reward. The HF-compatible `GRPORewardCalculator` accepts
explicit answer references, numeric confidence, and `EpistemicProcessAssessment`; it does not
convert response categories, trace summaries, trace abstention assessments, or trace confidence
hints into reward. In `LightweightTrainingOrchestrator`, trace-based `thought_align` remains
diagnostic data. `reasoning_grounded`, V3 overlays, trace summaries, and deception fingerprints
are diagnostic fields unless an independently verified component records the same process
property.

## Reward-integrity trajectory rule

`RewardObservation` and `RewardIntegrityBreakdown` retain the legacy negative-evidence rule: a
negative reward-integrity component requires component-keyed observable evidence inside the
authorized or recorded observable-reference boundary. `Trajectory` applies that legacy evidence
rule to every negative `reward_components` entry. Every nonzero reward-integrity component,
including a negative component that already satisfies the legacy rule, also requires matching
`reward_component_provenance`. Zero-valued reward-integrity components require neither legacy
evidence nor provenance. `Trajectory` can also carry generic non-integrity reward components;
the provenance rule does not apply to those generic fields.

For the `observable_evidence` route, provenance references must be a subset of the observation's
authorized references or the trajectory's recorded references. For the `trusted_evaluator` route,
the exact evaluator ID, evaluator version, and contract ID are stored in the provenance record.
The serializer writes only fields belonging to the selected route.

The trajectory serializer does not invent missing provenance. Consequently, legacy JSON that
contains a nonzero reward-integrity component but no component provenance is rejected at
restoration instead of being silently treated as optimizer-authorized.

## Goodhart risks

- A model can imitate uncertainty vocabulary, rationale structure, or control language without
  independently verifiable process evidence. Those fields remain diagnostic only.
- A trusted evaluator can become a proxy target. Its versioned contract identifies the evaluator
  that awarded credit so auditors can inspect or replace that contract.
- A populated V3 overlay can look process-rich without proving the named process component. The
  reward path therefore uses only matching verified components.
- Observable evidence can be incomplete or unrelated to the scored component. Component-name,
  key, route, and reference-boundary validation prevent cross-component and boundary misuse, but
  they do not prove that evidence is semantically relevant to the component score.

## Outcome-backed credit and legacy thought fields

`R_thought`, `r_thought`, and `honesty_trace` are compatibility names, not trace-style
optimization targets. A plan or rationale receives no process credit before a matching verified
assessment arrives. After an action produces a checked outcome, a caller may attach the resulting
`EpistemicProcessAssessment` to reward computation. This supports delayed credit through the
existing assessment interface; it does not implement a temporal credit-assignment algorithm.

The unused legacy `core.abstention.honesty_reward_from_trace()` helper is outside the consolidated
optimizer path. Its trace-derived number is diagnostic only. Maintainers can check this boundary
with `rg -n 'honesty_reward_from_trace' gepa_mindfulness`: the definition should be the only match.
The style-invariance tests exercise the active reward calculator and trainer paths separately.

## Confidence provenance and verification requests

`mindful_trace_gepa.confidence.ConfidenceSource`, re-exported from
`factuality_observability.calibration`, provides one shared vocabulary for confidence origins.
The dependency-light module lets evaluation records use this enum without importing the runtime
package, whose coevolution imports would otherwise create a circular dependency.
`fuse_confidence()` retains both `confidence_source` and the contributing
`confidence_sources`. Legacy records with no origin use `LEGACY_UNSPECIFIED`; the adapter does
not invent a verifier. Model declarations use `MODEL_SELF_REPORT`, internal sensors use
`INTERNAL_REPRESENTATION`, structured provenance and representation checks use
`BEHAVIORAL_EVIDENCE`, and external verification uses `EXTERNAL_VERIFIER`.

The current fusion function is a bounded heuristic, not an empirically fitted calibration model.
Latent uncertainty and mechanistic risk can lower confidence but cannot increase it. If external
verification confidence is absent or zero, fusion sets `verification_required=True`. The runtime
pipeline forwards that flag to routing and records it in `SampleLogBundle`. With verification
budget available, routing requests an external check even for a high self-reported confidence.
The existing exhausted-budget policy still abstains, or escalates when abstention is unavailable.
A nonzero external score does not certify truth; normal evidence, risk, and provenance checks
still apply.

Callers may supply `representation_stability` in `[0, 1]`, where `1` means no disagreement among
the representation variants they checked. Any lower value multiplies operational confidence by
that value and sets `representation_sensitive=True`. Routing then requests an external check.
The pipeline does not generate representation variants or estimate stability itself. The log
retains the sensitivity flag and confidence sources; canonical case identity remains unchanged.
Nonfinite, out-of-range, and Boolean confidence inputs raise `ValueError`.

## Self-serving justification diagnostics

`core.reward_integrity.SelfServingJustificationCheck` records the literal act, beneficiary,
self-benefit, recognized constraints, independent override citations, counterfactual subtraction,
role reversal, authority, necessity, proportionality, and reversibility. Optional Boolean findings
mean pass (`True`), fail (`False`), or unresolved (`None`). A reviewer supplies these findings;
the object does not infer them from model prose.

If self-benefit or a recognized constraint is present, or any optional finding is failed or
unresolved, `conclusion` returns `INCREASE_SCRUTINY`. Otherwise it returns
`NO_ADDITIONAL_SCRUTINY`. `scrutiny_reasons` preserves the individual triggers. Even a populated
override citation does not clear a recognized constraint or self-benefit flag. Private reasoning
and latent-state citations cannot serve as independent override evidence.

`RewardObservation.self_serving_justification` carries this diagnostic through
`RewardIntegrityCalculator.compute()` into `RewardIntegrityBreakdown`. The calculator does not
change component values, provenance requirements, or the aggregate because of this diagnostic.
The conclusion neither authorizes an action nor creates a negative training incentive. Trajectory
serialization and automatic natural-language rationalization detection are not added by this
interface. Agency remains a trajectory property, not a momentary uptime metric.

## Verification

Run the following command after changes to this contract or its implementation:

```powershell
python -m pytest `
  tests/test_epistemic_process_rewards.py tests/test_rewards.py `
  tests/test_reward_style_invariance.py tests/test_abstention_rewards.py `
  tests/test_schema_v3.py tests/test_honesty_rewards.py tests/test_training_cli.py `
  tests/test_reward_integrity_rewards.py tests/test_rl_trajectory.py tests/test_rl_cli.py -q
```

For confidence provenance and runtime escalation, also run:

```powershell
python -m pytest tests/factuality_observability -q
```

The test suite checks the component score bounds, both provenance routes, thought eligibility,
Schema V3 verified-component lookup, compatibility aliases, legacy-evidence requirements, and
the rule that every nonzero reward-integrity component has provenance.
The confidence tests check source retention in serialized logs, invalid inputs, internal-sensor
limits, and representation-sensitive routing. The reward-integrity tests check that self-benefit
increases scrutiny without changing numeric reward, and that observed honest failure can rank
above observed proxy exploitation under the existing component weights.

The contract implements
[`REC-001`](recommendations/UNIFIED_RECOMMENDATIONS.md#rec-001--verified-epistemic-process-reward-rule).
