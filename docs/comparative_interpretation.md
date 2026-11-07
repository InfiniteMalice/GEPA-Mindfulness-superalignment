# Comparative native and decomposed interpretation

Maturity: **Experimental**, opt-in CPU software; records/reports remain **Diagnostic**,
DEVELOPMENT, and excluded from optimization. Learned-dictionary quality, semantic explanation,
trained-model causal understanding, method superiority and alignment improvement remain **Unqualified**.
This extends [native discovery](mechanistic_research.md), [activation simulation](activation_simulation.md)
and [causal evidence](causal_mechanistic_evidence.md), preserving the competing primary-source
standards in the [source audit](phase2_research_sources.md).

## Four matched method families

`ComparisonProtocol` retains native, native_context, sparse_autoencoder and other_decomposition
arms. Optional dictionaries may be absent, with explicit reasons; those arms remain UNAVAILABLE.
At least one method must be available, otherwise protocol validation fails before capture.
Each arm shares the exact physical model/hook, split manifest, activation/output positions,
selected-feature count, public-predictor feature count and forward/evidence budget. The protocol
allows up to eight arms, eight inputs per split, 64 scalar native coordinates, 64 dictionary
features, eight selected features and 16 closed public features/positions. Native head-vector
comparisons, opaque decomposition implementations, unsupported nonlinearities and non-CPU devices
fail explicitly. These are bounded research-prototype limits, not full-scale SAE qualification.

Native arms use identity coordinates. Native_context is a caller-proposed contextual hypothesis
with declared circuit-context references and permitted public features. Use actual native/attention
measurements and component writers/readers during generation-side investigation; retain those
artifacts with the references. The API does not resolve a pointer, verify semantic interpretation
or establish that an explanation actually used the referenced context. The reproduction artifact
supplies actual generation attention measurements and local weights, while preserving this limit.
Comparing different feature information is a declared experimental contrast; equal feature count
alone does not match semantic information or discovery effort.

`FeatureDictionary` contains exact finite encoder/decoder matrices, biases, nonlinearity, identity,
model/hook binding, origin claim and observable evidence references. For activation vector a:
`z = nonlinearity(a @ encoder + encoder_bias)` and `reconstruction = z @ decoder + decoder_bias`.
Supported nonlinearities are linear and ReLU. The SAE arm supports fixed ReLU dictionaries;
training, top-k/gated SAE variants and arbitrary executable modules are not implemented here.
External learned weights may be supplied through the typed contract, but their training/origin
claims remain unverified. Constructed ReLU controls are explicitly not learned SAEs. No pickle,
remote code, decomposition-training loop or optimizer signal is accepted.

## Typed preparation and locked evaluation

`native_dictionary(backend, hook, enabled=True)` creates an actual identity map bound to the
verified architecture hook. `encode_decode(dictionary, activations)` is an inert finite tensor
utility with explicit float64 computation, not a training or verification-authority API.

`ComparisonArm` freezes method, dictionary, selected features, public TokenFeature objects,
hypothesis, activation-error tolerance, signed EffectPrediction and context/availability provenance.
`ComparisonBudget` matches selection/predictor counts and reserves all nine possible condition
forwards per arm. A smaller budget is rejected before experiment execution. `ComparisonProtocol`
reuses SplitManifest and LogitContrast. Future-token features, changed models, incompatible hooks,
masked/out-of-range positions and unsupported tensor geometry fail before intervention.

`prepare_comparison(backend, protocol, generation_inputs, clock=..., enabled=True)` captures actual
full-hook generation values on the frozen copy and fits bounded ridge ActivationPredictor records.
The predictor receives only closed current/past public token/position features. It never reads the
held-out target activation. Generation means/population descriptive scales, matched-complexity
predictors, random feature selections and group-label permutations are frozen in ArmFit records.
ComparisonCommit retains the original generation batch, protocol, fits and timestamp.

`evaluate_comparison(backend, commit, locked_inputs, enabled=True)` re-captures generation data
and recomputes every fit/control. Altered sources or apparently coherent changed controls are
rejected by actual model replay. Exact locked identities/content must match the split manifest.
Host custody, timing and locked-data isolation remain external obligations; record timestamps and
round trips alone do not establish them. Reports remain PREDICTIVE_HYPOTHESIS with authentication
NOT_PERFORMED even after successful numerical replay.

Dictionary coordinates remain separate from native-neuron measurement identities. Matrices and
ridge fitting use float64; proposed native values are explicitly cast to the actual hook dtype
before the exact existing replacement kernel. Source model, copied model state, mode and RNG
remain unchanged. Hooks are removed through the existing frozen-forward implementation.

## Separate reconstruction, prediction and intervention fidelity

For selected dictionary features, substitution retains the original reconstruction residual:
`a_new = a + (z_new_selected - z_selected) @ decoder_selected`.
This equals the modified decoded vector plus the original residual, including nonzero decoder bias.
It does not pretend that the dictionary reconstructs the entire activation. Full reconstruction
is tested separately. A zero decoder can leave outputs unchanged while explaining nothing; tests
retain that failure rather than accepting zero KL as evidence of understanding.

| Condition | Actual construction |
|---|---|
| identity | Reinsert exact original native values; require exact logits |
| reconstruction | Insert the entire decoded activation, including decoder bias |
| explanation | Replace selected features with frozen public-context predictions, retaining residual |
| ablation | Zero selected encoded features, retaining other features and residual |
| mean | Replace with original generation feature means |
| random_predictor | Seeded Gaussian predictions using original generation means/scales |
| matched_complexity | Generation-only predictors with matched feature count/kinds and randomized parameters |
| random_features | Ablate a disjoint feature group with the same selected-feature count |
| shuffled_labels | Derange predicted group labels; unavailable for singleton groups |

Unavailable controls have reasons rather than fabricated values. A missing disjoint group prevents
specific causal support. Equal feature counts do not guarantee matched perturbation magnitude,
variance, decoder density, informativeness or semantics; every actual native L2 change is retained.

Per-position original/changed logits support actual `KL(P_original || P_changed)`, teacher-forced
next-token NLL/Brier, top-token changes and calibration from the existing simulation tools.
Distribution/OOD groups retain per-example sources. Reconstruction MSE/MAE, actual/predicted feature
values, feature MAE and frozen tolerance results are separate metrics. Generation-std-normalized
MAE helps expose arbitrary dictionary units; zero-scale features stay explicitly unnormalized.
Neither normalization nor a passing numeric tolerance establishes semantic explanation.

The signed ablation prediction is compared with the actual target-minus-reference logit change
and random-feature control, per input: MET, FALSIFIED or NOT_SPECIFIC. These are bounded functional
intervention results, not independently admitted causal evidence or proof of ethical intent.
All causal evidence remains UNRESOLVED. Redundancy, off-manifold effects and untested paths remain
possible. Behavioral label meaning requires separate host validation and the existing alignment
study/evidence-registration boundaries.

Reports retain negative/weak/near-zero native regimes, whether a negative feature pole is meaningful,
dictionary size, selected decoder nonzeros, predictor complexity and descriptive feature sparsity.
Thresholds are declared prototype settings, not cross-architecture cleanliness calibrations.
Unexplained residuals and semantic behavior remain explicit rather than replaced by a cleanliness score.

## Cost, reproduction and limitations

Per-arm CPU wall time measures locked dictionary/prediction/intervention execution, including
existing frozen-state checks. Shared generation replay/refitting and baseline forward counts
are reported separately. Original discovery, decomposition training and original fit time are
not reconstructed from those timings. The probe separately meters preparation and context capture.
Generic identity-matrix execution has framework overhead; these timings are not an optimized
comparison of native inspection tools. Different dictionary sizes have different compute costs
under the same forward cap. No transformer time-space theorem follows.

The [recorded experiment](experiments/phase2_pr8_validation.md) preserves all method arms, raw
measurements, seeds, nulls, rejected predictions, controls, source hashes and actual costs. Its
random tiny GPT-2/Llama networks, artificial token contexts and constructed dictionaries do not
resolve the Cammarata/Gao disagreement or qualify a trained model's alignment. No method is declared
superior from a single benchmark, no circuit becomes a reward target, and no parameter update or
runtime/promotion authority is introduced. Existing inference paths require none of these optional APIs.
