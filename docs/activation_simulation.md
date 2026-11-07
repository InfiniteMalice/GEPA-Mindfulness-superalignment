# Activation simulation and downstream fidelity

Maturity: **Experimental**, opt-in software. Reports and control records are **Diagnostic**,
`DEVELOPMENT`, and excluded from optimization. Predictive, causal, semantic and alignment claims
remain separate and **Unqualified**. This extends the [native research protocol](mechanistic_research.md)
and Gao's activation-simulation challenge; it does not resolve the Cammarata/Gao disagreement.

## Typed workflow

Use an existing `FrozenNativeModel`, its exact `HookSpec`, committed scalar explanations and
the original generation batches. `prepare_simulation_controls(commits, generation_batches,
seed=..., logit_positions=..., probes=..., enabled=True)` freezes means/population descriptive
standard deviations, matched-complexity predictors, random-unit choices, label permutation,
activation/output positions and optional answer-token mappings before locked evaluation.
The original generation data remain in `SimulationControls`, alongside digests and timestamps.
Timestamp text alone does not establish temporal custody or human/actor isolation.

`simulate_activations(backend, hook, commits, locked_inputs, controls=..., logit_positions=...,
enabled=True)` requires exact committed locked inputs, model/computation, scalar components,
hook and positions. Groups contain up to 16 scalar components at the same hook; vector head
predictors and cross-layer groups are unsupported. All selected activation and logit positions
must be unmasked in every example. Locked sequences must share a length within declared model
capacity. Unsupported paths fail rather than silently moving an intervention.

The host executor reconstructs controls from original generation evidence and replays every
generation batch on the frozen copy. Altered generation values, derived statistics or derived
component assignments fail. Answer-token probe declarations are supplied by the host and are
not authenticated by reconstruction; changing them proposes a new control record/digest. The host
must compare that digest with its protected preregistration to establish custody. It then captures
baseline activations and logits in one actual forward pass. The explanation predictor receives
only closed public token/position features, never the actual held-out activation. The frozen policy
is `causal_prefix`: positive token/relation offsets are rejected, including for matched-complexity
baselines. At position p no predictor may use the gold token at p+1. PR 4's broader retrospective
offset hypotheses remain available for discovery, but future-token hypotheses cannot enter this
causal-decoder simulation. Proposed values
are cast to the actual activation dtype/device; requested and applied values are both retained.
Each condition runs a further real forward pass with exact tensor substitution at the declared
input/output hook. Temporary hooks are removed on failure; model state/computation is checked
before and after. Baseline activation must match across runs. Source weights, mode and RNG are
preserved; CUDA numerical qualification remains unrun.

`replace_selected_activations` is an inert tensor utility: `[B,T,D]` original plus exactly shaped
`[B,P,C]` replacement, identical floating dtype/device, unique in-range positions/components, and
finite values. It clones the original. There is no implicit broadcast, position shift, cast or
device transfer. The executor preserves other tuple members at tuple-valued output hooks.

`downstream_kl(original_logits, substituted_logits)` computes
`KL(P_original || P_substituted)` using float64 log-softmax at each matched position. It rejects
incompatible or nonfinite data and numerical breakdown. Only negative floating roundoff within
1e-12 is corrected to zero. KL is computed from actual logits, not inferred from activation error.
That tolerance is a numerical policy, not a certified mathematical error bound.

## Matched conditions

Every report includes each condition or an explicit unavailable result with a reason:

| Condition | Construction |
|---|---|
| explanation | Frozen public-context predictor applied to its declared components |
| identity | Original values reinserted at the exact same selection; numerical fidelity control |
| ablation | Selected activation values replaced by zero |
| mean | Original generation mean for each component |
| random_predictor | Seeded Gaussian values with generation mean/descriptive scale; no held-out fitting |
| shuffled_labels | Frozen predictors assigned to different group components via a derangement; unavailable for singleton groups |
| magnitude_matched_random_direction | Seeded directions with the explanation intervention's per-position L2 magnitude, constructed in float64 then rounded to activation dtype |
| random_neurons | Same predictors applied to distinct preselected random components at the same hook and matched group size |
| matched_complexity | Generation-only ridge predictor with the same feature count/kinds and randomized token/relation/position features |
| cross_context | Original selected values from another matched input, preserving positions; unavailable for one input |

Single-dimensional random directions are degenerate controls; they cannot substitute for grouped
or path experiments. A zero sampled direction uses a deterministic unit-vector fallback. Random
units need not have matched semantics or activation variance, so this control cannot establish
method superiority. Matched-complexity baselines are limited to the same closed feature class.
Large replacements can leave the model's normal activation manifold; effects alone do not show
that the manipulated component naturally computes the proposed explanation.

## Measurements and original evidence

Reports retain the frozen controls, original generation batches, predictor commitments, locked
inputs, model identity, exact hook, original evaluated logits and every condition's evaluated
logits. Per-position records contain actual KL, next-token choices, confidences, shifted
teacher-forced NLL and multiclass Brier scores. Gold token at logit position p is input token
p+1; if it is unavailable/masked, NLL and correctness remain null. Last-position KL is valid
without inventing a next-token label. Output positions must be frozen in the control record.

Original logits and changed logits allow independent recomputation of KL and NLL. Raw activation
rows retain actual/requested/applied values, components and positions. Summaries preserve mean KL,
NLL change and its denominator, applied activation MAE, top-token changes, ten-bin ECE with
per-bin counts, input-distribution/OOD groups, and weak/negative/near-zero regime coverage.
Weak threshold is the frozen predictor activation threshold; near-zero is its explicit 0.01
fraction. These are declared prototype settings, not universally calibrated scientific cutoffs.

Optional `LogitChoiceProbe(probe_id, example_id, position, choices, expected_label=None)` binds
host-declared `ChoiceToken(label, token_id)` mappings before evaluation. Choice changes and
optional correctness are computed from actual mapped logits. Host validation is required to
establish that a token mapping represents the task's behavior and that expected labels are true.
The evaluator does not infer semantic honesty, refusal legitimacy or deceptive intent from logits.

The executor records actual forward counts: generation replay, baseline and measured conditions.
It caps stored distribution values before execution (default 1M, configurable up to 2M).
Larger evidence requests fail clearly; they require a smaller declared evaluation or later external
artifact-storage integration. Original relevant evidence is retained rather than replaced by
uncheckable summary scores. Record serialization uses existing bounded exact diagnostic contracts.

## Interpretation and governance

An incorrect constant can produce tiny single-neuron KL because the unit has little downstream
influence or redundant paths compensate. Low KL cannot by itself establish an explained neuron.
Compare activation errors, control effects, grouped behavior, semantic task outcomes and later
causal/path experiments. Per-distribution differences on tiny examples are descriptive; no
confidence interval, calibrated uncertainty or superiority claim is inferred from one seed.

These reports remain `PREDICTIVE_HYPOTHESIS`, with measurement authentication explicitly
`NOT_PERFORMED`. Deterministic replay verifies arithmetic consistency against the supplied frozen
model; independent host custody/provenance and accepted-store registration are separate duties.
No optimizer reward/penalty, parameter update, runtime permission, constitutional override or
canonical V5 modification is introduced. Subsequent causal registration and alignment-specific
studies must retain nulls, contradictions, controls and original records.
