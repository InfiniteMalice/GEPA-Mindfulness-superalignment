# TF-OPD numerical research implementation plan

Goal: implement a bounded, opt-in CPU research kernel for rollout-derived input moments and
TF-OPD linear-layer fusion, preserving existing host admission and protected evaluation.
Authority: user master sections16.1/22/23 and standing ready-PR/push/review/merge instructions.
Use approved quality-gate fallback: spec/design, implementation, tests, review, evidence.
No bd/.beads. Repository has no usable local Git metadata; use an isolated verified snapshot
and the GitHub plugin. Baseline main 682e8af561c0f7cf203117c60d47e9e8a1f2d1dd,
tree 0742729a5e7c002c23bf90a1c1dcd5738d32a56f (merged PR813); local 18filemanifest checked.

Design: a candidate numerical result is never reward, certification, a checkpoint, deployment
or an authorized model update. Reuse ArtifactDiagnosticRecord/DiagnosticRecord serialization,
SystemConfig, public EvidenceReferences, validate_input_envelope and require_training_eligible.
Sources require explicit TRAIN metadata and retain model/harness/checkpoint/architecture/layer,
rollout/selection/runtime/sampling/evaluator versions and references. These bindings are declared
host provenance, not authentication. The host must supply independently admitted source data.
Reject held-out/restricted nested provenance before processing activations. All emitted records
are DEVELOPMENT, no optimizer or promotion authority. No changes to existing eligibility policy.

Approaches: a full model merger needs unconfigured checkpoint/model/runtime and operator budgets.
A bounded NumPy kernel provides actual Gram/solve/quantization behavior locally and avoids a new
hand-written matrix solver. Add only an optional fusion extra; base dependencies/imports stay
light. Native model loading/hooks, tied-head handling, nonlinear tensor task arithmetic and
full checkpoint fusion remain Unsupported. No empirical or retention/alignment claims.

Interfaces: TFOPDConfig(enabled=False, alpha=0.9, accumulation_dtype="float32",
storage_dtype="float32", condition_limit=1e12) with strict versioned serialization;
FusionSource(base_checkpoint_digest,specialist_checkpoint_digest,architecture_id,layer_id,
system,rollout_digest,source_metadata_json,evidence_refs); LayerStatistics(source,token_count,
gram,input_digest,accumulation_dtype); FusionCandidate with immutable float32/64-rounded weights,
complete source/statistics/weight digests and decomposed diagnostics. All use existing diagnostic
records. screen_rollout_text(text,finished,enabled=False) emits only compression diagnostics;
collect_input_statistics(source,inputs,config=TFOPDConfig()) and
fuse_linear_layer(base,specialists,statistics,domain_weights=None,config=TFOPDConfig())
return None before inputs/NumPy when disabled. NumPy is lazily imported only for numerical work.

Algorithm: X.T@X divided by actual token count, configured FP32/FP64 accumulation; each source
normalizes its own token count. Weight off-diagonal-shrunk Grams by positive domain weights,
solve the paper's delta equation in FP64 and explicitly round candidate storage to FP32/FP64.
Reject malformed/nonfinite/shape-incompatible, asymmetric/non-PSD and ill-conditioned/singular
systems; no silent ridge, inverse, pseudo-inverse, precision fallback or partial model update.
Report solve residual, condition, input/output precision, quantization error, update magnitude,
per-teacher unreproduced update share (None for zero denominator) and base-relative delta warning.
Compression uses UTF-8 bytes for the last 20000 characters; integer ratio below 5 percent. This is
an explicit screen adaptation, not verified action-loop detection or a behavioral penalty.

Hard prototype bounds: 8 specialists, matrix dimensions 64, 4096 activation rows, 1 MiB input record
and 4 MiB output. Counts and complete outputs bounded before large conversion/decomposition.
These are original CPU controls, not paper hyperparameters. Source refs max 32; metadata max 16384
UTF-8 bytes. No file IO, external API, model import/download, execution or store changes.

Tasks: first add failing tests for records/default-off, statistics normalization, shrinkage and
closed-form solutions (identical Grams weighted mean, orthogonal support, oblique analytical
solution), precision and singularity/overflow rejection, alias immutability and source/training
boundaries. Then implement training/specialist_fusion_records.py and specialist_fusion.py.
Use baseline/RED/GREEN evidence. Add protocol, disabled config and authored source note; update
existing source/recommendation matrix/reader/registry together and package linked resources.
Run related eligibility/provenance, citation/doc consistency and numerical suites, Python 3.10/
Black/100-column/import checks, real wheel and isolated default-off/resource smoke. One fresh
independent whole-branch reviewer is required by executing-plans; no implementer delegation.
Publish a ready PR through GitHub plugin, babysit CI/reviews, fix only verified issues, then merge.

Review focus: false provenance must not gain authority; held-outs cannot enter activation
statistics; singular/low-information directions cannot silently become a successful merge;
FP32 rounding and undefined diagnostic ratios must be visible; no helper mutates live tensors,
source data, protected Constitution/cases/manifests, verifiers or governance policy.
Pre-flight: both numerical functions consume the same immutable source/statistics records;
source field validation stays independent of lazy NumPy and default-off short-circuiting.
Ruling: explicit optional NumPy is the only new dependency, to obtain tested linear algebra;
if unavailable the enabled numerical path fails visibly and base imports/default-off still work.

Baseline: 116 existing source/registry/eligibility checks passed in 8.36s. New module absence
observed RED; initial numerical suite 23 passed. Two tiny-energy regressions observed RED
(undefined shares caused by binary64 squared-energy underflow), fixed with normalized log
energies and explicit unrepresentable statuses. Related numerical/source-credit tests: 95 passed.
Source/doc/registry suite: 188 passed in 10.55s. An additional stored-fit regression observed RED
because an exact solve could lose its update in FP32 storage without a per-teacher stored-fit
metric. Add separate solve/stored shares; no claim that a good numerical fit establishes alignment.
Ruling: optional NumPy extra plus explicit CI selection provides numerical coverage while base
wheel smoke checks lazy default-off behavior. No mandatory dependency or model runtime added.

The stored-fit diagnostic regression now passes: the FP64 solve and rounded candidate have
separate per-teacher shares. Related eight-module suite: 189 passed in 10.87s. A further RED
showed a caller could label FP64 Gram values as FP32; shared standard-library FP32 representability
validation now protects both statistics and stored candidate serialization without NumPy import.

Final independent whole-branch review: two Important numerical findings, no Critical or Minor
findings. Both reproduced RED: RHS multiplication could silently underflow into a baseline;
scaled energy could mislabel positive supported energy as zero when an unsupported large delta
component dominated normalization. Fixed by rejecting NumPy underflow throughout the solve and
teacher diagnostics, with no hidden rescaling, ridge or alternate answer. Both regressions
GREEN; eight-module suite 192 passed in 10.59s. Conservative rejection may exclude a mathematical
solution; documented explicitly. No second reviewer or implementer delegation.

Final: Ruling: full-model retention/alignment/runtime effectiveness remains Unsupported and
unmeasured, as declared. Shipping a diagnostic CPU kernel does not establish these properties;
the cost if that scope is misread is an unjustified model promotion, forbidden by existing gates.
Final: Ruling: host checkpoint/activation/admission authenticity remains external responsibility;
declared metadata is not a certificate. The cost if the host fails its duty is invalid research
input, never permission to alter a protected baseline.
Final: Ruling: the independent reviewer checked mathematics against the retained authored note,
not publisher content. Implementer primary-source verification preceded implementation; preserve
that distinction. The cost of a source mismatch is unsupported fidelity, requiring correction
before claiming reproduction. No full reproduction or empirical transfer is claimed.

Final wheel SHA256 ed1d10e61f034597326e946bce98b77a39ad57ad9cb01ce7e9e8c22d4b111375:
36 required resources and 3 changed runtime files byte-equal source. Isolated outside-checkout
imports, default-off functions, no NumPy import, registry and guide-resource smoke passed.
Local full Torch suite/Ruff unavailable; final remote CI must pass before merge.
