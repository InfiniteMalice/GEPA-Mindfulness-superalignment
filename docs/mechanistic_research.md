# Falsifiable native-component research

Maturity: **Experimental** opt-in software. Measurements, hypotheses, graphs and reports are
**Diagnostic**, `DEVELOPMENT`, and excluded from optimization. Predictive validity, causal
validity and alignment improvement are separate, **Unqualified** empirical claims.

## Competing hypotheses and standards of evidence

[Cammarata's *Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) motivates
careful inspection of native signed activations, writers/readers and recursive contextual motifs.
[Gao's *Neural Networks Are Not Already Clean*](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)
challenges whether plausible labels explain full activation functions, weak activations and
computation in superposition. These motivate competing hypotheses, not universal conclusions.
The [pinned primary-source audit](phase2_research_sources.md) preserves their disagreement.

The research target is predictive and causal validity, not a cleanliness score. A neuron may
represent a relation, prediction, transition, conditional conjunction, multiple mechanisms, or
something unknown. Its activation token need not name its concept. A coherent label fitting
selected examples is a hypothesis; prediction error or a failed intervention can falsify it.
Abstract moral values are not presumed to have single-neuron representations.

## Exact native hooks

`FrozenNativeModel(model, model_id=..., revision=..., enabled=True)` copies an exact supported
Hugging Face `GPT2LMHeadModel` or `LlamaForCausalLM`, puts the copy in evaluation mode and freezes
parameters. It does not change the original model's mode, gradients or weights. The default
50M-parameter copy budget may be raised explicitly, up to 1B; copying and hashing can be costly.
Dense single-device CPU/CUDA models are supported structurally; only CPU tiny-model paths are
software-qualified here. Quantized, meta, sharded, already hooked, subclassed and other architecture
backends fail explicitly. Internal block/MLP/attention/norm/activation classes must match their
declared configuration; unsupported substitutions and instance-forward overrides are rejected.
Only eager and SDPA attention implementations are supported. No model/tokenizer is downloaded
and no external service is called.

`hook(layer=..., kind=HookKind...)` returns an exact `HookSpec`. Capture rejects altered module
paths, layers, hook sides, transformations or geometry. Model tensors (including nonpersistent
buffers), serialized configuration, explicit attention implementation, module classes and ordinary
scalar forward settings are hashed before and after capture. This covers approved library
semantics; it is not a proof about arbitrary Python monkeypatching or the entire execution
environment. Pre-existing hooks are rejected. Temporary hooks are removed even on forward errors.

| Kind | GPT-2 tensor | Llama tensor |
|---|---|---|
| `MLP_PRE` | `c_fc` output before activation | `gate_proj` output before activation |
| `MLP_POST` | `c_proj` input after the actual activation | `act_fn` output: activated gate, not gated product |
| `MLP_UP` | Unsupported | `up_proj` output |
| `MLP_PRODUCT` | Unsupported | `down_proj` input: `act_fn(gate_proj(x)) * up_proj(x)` |
| `ATTENTION_HEAD` | Concatenated head values before `attn.c_proj` | Concatenated query-head values before `self_attn.o_proj`, including GQA layouts |
| `MLP_RESIDUAL` | MLP output before residual addition | MLP output before residual addition |
| `ATTENTION_RESIDUAL` | Attention output tuple's first tensor | Attention output tuple's first tensor |
| `RESIDUAL_STREAM` | Block input before pre-normalization | Block input before pre-normalization |

Measurements retain actual layer, tensor path, hook location, transformation, normalization
description, structural dependencies, physical model/config digest, supplied revision, library
versions, component indices, token positions, token IDs, attention masks, dtype and device.
Attention heads remain vectors with exact head geometry; no scalar reduction is invented.
Padding is excluded from measured positions. Inputs are bounded to eight sequences of 128 tokens,
the architecture's declared context capacity, 64 selected components and 2,048 observations.
Unsupported shapes, vocab indices, dtypes, masks and nonfinite values fail clearly.

`component_weights` exposes actual local MLP writers/readers and their row/column orientation.
GPT-2 Conv1D and Llama Linear weight layouts differ. Llama's two branch writers provide MLP
context; they are not both direct inputs to every gate/up hook. Normalization, bias, gating and
additional residual paths must also be considered. These local weights and declared dependency
paths are structural context, not exhaustive circuits or intervention evidence.

`measurement_graph` reuses existing `AttributionGraph`/networkx comparison facilities for one
example with scalar components. It creates observed nodes, zero attribution scores and no edges.
Raw measurements remain in metadata. Vectors and multi-example graphs require a separate declared
analysis and are rejected rather than silently averaged or pooled. Neither this graph nor
existing gradient-times-activation graphs establishes causal influence.

## Discovery sampling and explanations

`sample_signed_regimes(batch, discovery_ids=..., seed=..., per_regime=...,
thresholds=SamplingThresholds(...), enabled=True)` records absolute thresholds and seeded samples
from positive/negative high, medium and weak regimes, near-zero, and exact-zero/nonactivating
regimes. Negative/weak regimes are not discarded. Missing regimes remain explicitly missing;
zeros do not substitute for missing negative evidence. Sampling currently requires one scalar
component and exactly the declared discovery examples. Thresholds require calibration by
architecture, layer, scale and activation transformation; there is no universal threshold.

Use `SplitManifest` to commit separate generation, development and locked example references.
References bind exact public inputs and a second identity-independent token/mask digest. Duplicate
IDs or exact contexts across splits are rejected. This does not detect semantic paraphrase leakage:
the host must separately group related prompts and keep locked inputs/outcomes inaccessible to
discovery and fitting. Source labels and supplied records do not authenticate this isolation.

`ExplanationHypothesis` preserves description, mechanism category, competing explanations,
falsifiable predictions, a bounded predictor and generation IDs. Categories include arbitrary
mixture, shared feature, related context, hierarchical generalization, conditional conjunction,
multiple mechanisms and unknown. No single label is mandatory.

`generate_context_hypotheses` provides a deliberately limited deterministic discovery interface:
lexical offset associations fitted using **all** generation observations, with an explicit unknown
constant baseline. It does not claim semantic understanding or implement an autonomous scientist.
Researchers can supply richer hypotheses through the same typed interface. Upstream/downstream
inspection and local context belong in their reasoning; plausible text never upgrades evidence.

`generate_near_misses` perturbs declared feature offsets using the generation vocabulary, retaining
public contexts and predicted changes. It excludes exact split duplicates and bounds proposals
to 32. Actual activation is explicitly **NOT_MEASURED** until the host separately reserves and
measures a new example. A proposed near miss is not a verified negative example. Prediction-neutral
perturbations may also be useful controls. No held-out outcome is read to construct a predictor.

## Frozen prediction and falsification

`ActivationPredictor` is a closed, serialized function of permitted `NativeInput` token IDs,
relative offsets, token-equality relations and positions. It cannot access an activation tensor,
logit, external file, arbitrary callback or target label at prediction time. Rule predictors and
deterministic ridge fits support at most 16 features and coefficients bounded by 1e6. Ridge fitting
uses generation data only, with positive bounded regularization and numerical breakdown checks.
Its limited function class is a baseline, not a general model of native computation.

`freeze_explanation(hypothesis, splits, generation_batch, clock=host_clock, max_mae=...,
activation_threshold=..., enabled=True)` records a detached digest commitment including model,
hook, scalar component, split references, predictor, tolerances, fixed evaluation token positions
(the generation batch's selected positions) and aware host timestamp.
The host must register it immutably **before** disclosing locked inputs/outcomes. A serialized
timestamp or self-authored manifest does not prove that order. Development tests may motivate
revision; revising after locked outcomes requires a new hypothesis and a new locked split.

`evaluate_explanation(commit, locked_batch, enabled=True)` requires all committed locked inputs,
the same physical model/hook/component, exact token identities, and the committed position set.
Every locked example must contribute measurements; an all-masked selected region cannot count
as evaluated coverage. Per-example variable position protocols are not implemented. It predicts from public
context alone, then compares actual supplied measurements. Reports retain each prediction,
actual activation, input/measurement digest, signed error opportunities, MAE/MSE, false positives,
false negatives, sign errors, weak/negative coverage and per-distribution metrics. A positive
activation decision is `value >= activation_threshold`; negative-pole errors are additionally
visible through signed predictions/sign errors and negative counts. The exact `ood` group has a
separate MAE and an ID-to-OOD degradation. Numerical overflow fails rather than emitting infinity.

A frozen tolerance can be met or the predictions can be falsified. Either outcome remains
`PREDICTIVE_HYPOTHESIS`: measurement authentication is explicitly **NOT_PERFORMED** by this
arithmetic evaluator. The host must independently verify provenance and temporal custody before
admitting evidence. Null and failed hypotheses must be retained; no language-model review can
make these records authoritative.

## Subsequent experimental stages and governance

The full research loop is Observe → Hypothesize → Predict → Intervene → Measure → Replicate →
Revise. This stage implements native observation, bounded hypothesis/predictor construction and
locked prediction measurement. Activation substitution/downstream KL, null/random/mean/ablation
controls, grouped-neuron interactions, path interventions and independent causal registration
belong to the following stages and are explicitly **NOT_RUN** in these reports. Low single-neuron
impact or low activation error cannot establish that a circuit is explained.

Discovery, prediction, intervention execution, independent verification, skeptical review and
evidence registration must remain distinct responsibilities. Deterministic tools may perform
them; identical models given different role names are not independent verification. Only trusted
host authority may admit results through existing protected evidence/governance mechanisms.
Diagnostic data cannot become hidden-thought penalties, inferred deception punishments, optimizer
credit, runtime grants, or a constitutional override. Parameter surgery and training require
separate candidate models, authorization and protected regression evaluation.

Alignment studies should first measure actual honesty/abstention, clarification/resumption,
semantic laundering, dual-path behavior, value conflicts and CVA authorization/evidence changes
on unchanged V5 cells. A refusal correlation does not establish ethical understanding or deceptive
intent. Proper circuits must distinguish justified from unnecessary refusal and change predictions
when valid evidence or authorization changes. This stage does not claim those studies have run.
