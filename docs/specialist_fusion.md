# TF-OPD numerical research protocol

Maturity: **Experimental** bounded CPU kernels and **Diagnostic** proposals. Disabled by default.
Full checkpoint fusion, native teacher generation/activation hooks, model retention and research
effectiveness remain **Unsupported** and unmeasured. No model is loaded, updated or deployed.

This optional increment implements the linear-layer equations in [TF-OPD](https://arxiv.org/abs/2610.22236).
It reuses existing diagnostic serialization, `SystemConfig`, observable evidence references,
`validate_input_envelope` and the unchanged [training eligibility guard](research_credit_accounting.md).
It adds an explicit `fusion` extra with NumPy; base-package imports and default-off calls need
no NumPy. Install the numerical experiment with `pip install -e '.[fusion,dev]'`.

## Host data and source binding

`FusionSource` binds the common parent checkpoint, specialist checkpoint, architecture, layer,
model/harness system configuration, retained rollout digest and observable provenance references.
Its bounded metadata includes explicit TRAIN eligibility, runtime, selection-policy and evaluator
versions plus sampling configuration. An ungraded cohort can explicitly use a `not_run` evaluator
version. Cohort labels are declarations and never verification results.

The host must authenticate checkpoints, actual common initialization, collection identity and
independently admitted training data through existing mechanisms. Matching strings, hashes and
TRAIN labels do not establish those facts. Nested DEVELOPMENT/REGRESSION/HIDDEN_EVAL or generated
data without the existing required review are rejected before activation processing. The guard
cannot detect stripped provenance, which is prohibited. No admission policy is created here.

`collect_input_statistics(source, inputs, config=...)` accepts a bounded dense token-by-feature
matrix, computes its input Gram and divides by that source's actual token count. The host must
supply the teacher's own kept rollout activations, including prompt and answer positions, and
pin tokenizer, source dataset, model, collector and evaluator versions in retained provenance.
This library does not install hooks or prove that supplied arrays came from those rollouts.
FP32 accumulation follows the primary method; configurable FP64 accumulation is a numerical
ablation. `LayerStatistics` retains the input digest, count, precision and complete source.

`screen_rollout_text(text, finished=..., enabled=True)` exposes a separate diagnostic screen.
It records a digest and compression statistics, never the text or a correctness judgment.
The last 20000 characters are encoded as UTF-8; compressed-byte ratio below 5 percent is tested
with integer arithmetic. UTF-8 byte normalization and zlib-wrapped DEFLATE are explicit local
choices. The source describes a character tail without publishing an authoritative codec
implementation. A screen flag is not verified unproductive repetition, a DeLoop penalty,
proof of a good recovery, training admission or permission to act. Preserve raw rollouts,
independently graded and ungraded cohorts, legitimate repetition and selection failure counts.

## Candidate computation

The disabled example is `configs/specialist_fusion.example.json`, consumed explicitly with
`TFOPDConfig.from_dict`; no loader or model path is executed. `TFOPDConfig()` is disabled. Both numerical entry points return None before inspecting source
arrays or importing NumPy. Explicit enabled configuration is required. Prototype limits are
8 specialists, matrix dimensions 64, 4096 activation rows, 32 references per source, 16384-byte
source metadata, 1 MiB serialized statistics and 4 MiB candidate output. These are original CPU
controls, not claimed upstream hyperparameters. No large-model runtime is configured implicitly.

`fuse_linear_layer(base, specialists, statistics, domain_weights=..., config=...)` requires
matching declared parent/architecture/layer identities, unique specialist checkpoints, matching
matrix shapes, positive finite weights and symmetric PSD input Grams. It shrinks off-diagonal
Gram entries with alpha (default 0.9), applies domain weights and solves the delta equation in
FP64. Default candidate storage is explicitly FP32-rounded; FP64 storage is an ablation.
The returned immutable numeric tuples are candidate values, not a model checkpoint.

The solve rejects nonfinite, singular or excessively ill-conditioned systems, without silently
adding a ridge, using a pseudo-inverse, repairing a Gram, truncating a layer or substituting a
baseline. Underflow in solve arithmetic or scaled teacher-energy diagnostics also rejects the
candidate instead of returning a baseline or labeling positive energy as zero. This conservative
range limit can reject inputs with a mathematically valid solution. PSD checks are conservative:
negative eigenvalues from finite-precision roundoff can
cause rejection. Native BF16/FP16 tensors, nonlinear tensor task arithmetic, tied output heads,
automatic layer grouping, LoRA integration and full checkpoint IO are not implemented. Layer-role authentication
and compatibility remain host responsibilities.

Diagnostics retain source/statistics/weight digests, configuration, NumPy version, condition,
solve residual, solve/stored update magnitudes, input/accumulation/storage precision and storage
quantization error. Teacher update shares are reported separately for the FP64 delta solve and the actual rounded
candidate. They use scaled logarithmic energies so tiny nonzero updates retain meaningful ratios. True computed zero reference energy makes the ratio None;
unrepresentable positive energies/ratios have an explicit missing status. Raw logarithmic
energies remain visible. The source's base-relative large-delta warning is advisory, not proof
that specialists share a base or that alignment is retained.

All source/statistics/candidate records and screen outputs remain DEVELOPMENT. Candidates carry
`optimizer_eligible=False` and `authority_granted=False`; the shared guard rejects them as direct
optimizer evidence. No live parameter, optimizer, source array, evidence store, protected test
or accepted baseline is mutated. Numerical teacher fit supplies no correctness or alignment credit.

## Source and adaptation boundary

[REF-TF-OPD](recommendations/RESEARCH_TRACEABILITY.md#ref-tf-opd): Bangji Yang, Jiajun Fan,
Zhaoyu Zhu, Hongbo Ma and Ge Liu, *Training-Free On-Policy Distillation*, arXiv:2610.22236v1,
dated 2026-10-15. Sections 3.1–3.5 and Appendix B checked 2026-10-31. Paper license: arXiv
non-exclusive distribution license. No authoritative method-code repository/license was
established from the primary paper; no upstream code is copied or imported.

The [authored method note](recommendations/source_artifacts/tfopd-2610.22236v1-method.txt)
and its digest in the [integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md)
identify retained local documentation, not publisher HTML/PDF bytes. The numerical equation
subset is implemented; native rollout generation, learned specialists, actual transformer
activation collection and reported model-level results are not reproduced.

## Reproducibility and protected evaluation

Use the existing [independent improvement audit](reliable_improvement_evaluation.md) and
[research reproducibility contracts](research_ablation_suite.md) for external comparisons.
This arm does not append conditions to the fixed Studies A–F or canonical 17-case manifest.
Hold task/corpus distributions and relevant compute budgets fixed; pin common initialization,
specialist checkpoints, precision, selected training rollouts, independent graders, held-out
lineage, model/harness versions and seeds. Record actual collection/solve/answering costs.

Compare base, specialists, task arithmetic and fusion; separately compare ungraded versus
independently graded training rollouts, loop-screen choices, Gram shrinkage and storage/evaluation
precision. Use matched evaluation precision to distinguish fusion from a format change.
Evaluate all 17 cases, calibration, justified/false abstention, honest completion, semantic-intent
robustness, constitutional behavior, coding/reasoning capability, interference and KL drift.
Retain failed candidates and protected regressions. Only existing governed admission may accept
a complete externally evaluated model; mathematical fit never authorizes promotion or spending.
A protected regression blocks promotion regardless of teacher fit or domain weighting.

Deterministic CPU tests establish software arithmetic and boundary behavior only. No model-level
training, API experiment or empirical capability/alignment improvement has been established.
