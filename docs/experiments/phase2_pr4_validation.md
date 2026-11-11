# Phase 2 PR 4: native inspection and falsifiable prediction

> Publication note: the original report below, including its tables and repeated-run claims,
> describes the historical source snapshot indexed in the
> [snapshot index](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/b5d888b24227e8358d244e33ce2faa9fd32d5b4c/docs/experiments/phase2_manifest_snapshots.json).
> Its whole-sequence splits are **unqualified for causal-prefix held-out claims**.
> The current script uses the corrected `causal-prefix-disjoint-v2` dataset; its output is a
> new measurement, not reproduction of the historical table. The separately retained
> [corrected PR 4 probe](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/b5d888b24227e8358d244e33ce2faa9fd32d5b4c/docs/experiments/phase2_prefix_reviewed_pr4_probe.json.gz)
> and [qualified rerun report](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/b5d888b24227e8358d244e33ce2faa9fd32d5b4c/docs/experiments/phase2_prefix_reviewed_validation.md)
> record that protocol. Original tables, raw evidence and full original report bytes remain
> available at their original pinned revisions; this banner changes no historical measurement.

Maturity: **Experimental** software and **Diagnostic** records. Scientific predictive validity,
causal validity, trained-model semantics and alignment improvement remain **Unqualified**.
Base: local PR 3 commit `e83b871`. No existing wire format or inference default is migrated.

## Implemented interfaces and inventory

New source: `interpret/native_records.py`, `interpret/native_neurons.py`,
`interpret/explanation_tests.py`, under `gepa_mindfulness`.
New tests: `tests/test_native_neurons.py`, `tests/test_explanation_tests.py`.
New guide: `docs/mechanistic_research.md`.
New reproducibility script/artifact: `docs/experiments/phase2_pr4_native_probe.py` and its
compressed JSON output. Validation logs, per-test XML and an artifact manifest accompany them.
Modified: documentation index, package-data declaration, wheel-resource test, and one misleading
`AttributionEdge` docstring (attribution alone does not establish causal influence).

The [typed API and research protocol](../mechanistic_research.md) describes model copies,
architecture-specific hooks, exact measurements, signed-regime sampling, upstream/downstream
weights, existing graph joins, split manifests, closed rule/ridge predictors, hypothesis generation,
near-miss proposals, frozen commitments and locked falsification. Only the exact supported GPT-2
and Llama model classes/internal layouts with eager/SDPA attention are supported; other backends
fail explicitly. Ordinary inference does not depend on these optional modules.

The software preserves competing interpretations, unknown regimes, original inputs/measurements,
versions/digests and negative/null results. It introduces no hidden-thought reward, circuit penalty,
training update, execution grant, accepted-store admission or canonical V5 category.

## Test-first development and review disposition

Initial native and explanation tests failed because their modules were absent. Subsequent
red/green boundaries covered exact real hook values, Llama gating/product distinctions,
Conv1D/Linear reader orientation, source-model preservation, hook cleanup, unsupported hooks,
masked positions, malformed dtype/context provenance, resource bounds, all signed sampling
regimes, exact split leakage, closed predictor inputs, generation-only fitting, locked identities,
null sign predictions and numeric metric overflow. The synthetic-amplitude tests are explicitly
software fixtures; the architecture tests compare actual tensor hooks.

Fresh read-only review independently passed the initial 26 focused tests and raised two important
findings. Both were reproduced before fixing:

1. Locked measurement positions were not committed, permitting favorable post-outcome selection;
   all-masked examples could appear present without measurements. The commitment now binds fixed
   evaluation positions, and split evaluation requires per-example coverage. Test cases demonstrate
   that previously favorable position filtering and empty masked coverage are rejected.
2. State tensors/serialized config did not fully identify computation: a replaced activation and
   an eager/SDPA switch could evade the old identity. Construction now checks supported internal
   module classes against declared activation/configuration. Identity hashing includes approved
   module semantics, ordinary scalar forward settings, attention implementation and nonpersistent
   buffers. Unsupported substitutions, instance-forward overrides and pre-existing hooks fail;
   runtime changes invalidate the frozen copy. Arbitrary Python/class monkeypatching remains
   outside the backend contract, not silently covered by a security claim.

Final focused tests: **31 passed**, 0.91 seconds. No deferred minor findings were reported.
The final fix pass used reproducing tests and the complete protected regression suite, rather
than substituting a second opinion for tests.

Reviewer topics outside this stage are explicit scope rulings: measurement authenticity/temporal
custody remain trusted-host obligations; CUDA/large-model performance is unqualified; trained
semantics/alignment claims are unmeasured; semantic paraphrase leakage requires host grouping;
intervention fidelity/downstream KL await later stages; arbitrary hostile Python is unsupported;
packaging/full-suite and the reproduction artifact are independently checked below. Their cost
if misapplied is invalid scientific or deployment inference, so reports/guide preserve these limits.

## Actual random-model software probe

Reproduce from the source checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m docs.experiments.phase2_pr4_native_probe --enabled > native-probe.json
```

The script constructs random tiny GPT-2/Llama models (seed 17, two layers, width 16, MLP width 32),
selects three layer-zero components per family with the same seeded sampling, commits all four
candidate predictors before capturing locked activations, and evaluates them on fixed new token
contexts. GPT-2 postactivation and Llama gated-product units are different transformations and
are reported separately. Generation-derived scale fractions define the declared sampling/metric
thresholds; these are prototype settings, not calibrated universal scientific standards.

| Family / component | MAE range across four candidates | Constant baseline MAE | Frozen criteria met |
|---|---:|---:|---:|
| GPT-2 / 26 | 0.027383–0.028929 | 0.028516 | 0 / 4 |
| GPT-2 / 19 | 0.013483–0.021471 | 0.018880 | 0 / 4 |
| GPT-2 / 23 | 0.028541–0.030034 | 0.029651 | 0 / 4 |
| Llama / 26 | 0.000936–0.001139 | 0.001008 | 0 / 4 |
| Llama / 19 | 0.001282–0.001361 | 0.001294 | 0 / 4 |
| Llama / 23 | 0.002275–0.002486 | 0.002396 | 0 / 4 |

All **24 prediction hypotheses were falsified** against their declared MAE criteria. All candidates,
constant baselines, per-token predictions/actual values, signed errors, OOD groups, original
measurements, configurations and source hashes are retained in `phase2_pr4_native_probe.json.gz`.
Missing high/negative/near-zero/nonactivating regimes remain explicit rather than filled with
invented examples. A second execution reproduced all model identities and numerical metrics;
timestamps and commitment digests legitimately change on a new run.

These results check the pipeline on untrained random networks and artificial token contexts.
They neither adjudicate Cammarata versus Gao nor measure language-model alignment. No causal
intervention, downstream KL, trained semantic mechanism, statistical superiority, training effect
or improved capability is established. Measurement authentication remains `NOT_PERFORMED` and
all reports remain `PREDICTIVE_HYPOTHESIS`, including hypothetical numerical successes.

## Quality gate

The maintainer-approved unavailable-`repo-quality-gate` fallback applies: stage specification,
design/ADR, test-first implementation, lint/format/type checks, fresh review and recorded evidence.
No beads automation was invoked. Both CI mypy commands and all four new source/probe files pass
type checking. Repository-wide Ruff passes; Black checks **837 Python files**. Wheel-resource
verification passes with the three modules and guide included.

Complete regression: **7,174 passed, 11 skipped, 18 warnings, 162 subtests passed**, 429.84 seconds,
exit 0. Optional DSPy/DAPO/native endpoint/Windows/CUDA/Mojo skips are unchanged and unqualified.
Only a local type-table annotation was corrected during the run; its function bytecode, constants
and names were checked unchanged, and the final typing check passes. The new standalone probe
was separately executed twice. Full stdout, compressed per-test JUnit and hashes are retained.

## Proposed PR description and migration

Add default-off native measurement for exact supported GPT-2/Llama layouts on frozen copies,
including signed/gated activation regimes, head/residual values, local writer/reader inspection
and observation-only existing-graph joins. Add disjoint input manifests, bounded closed predictors,
competing hypotheses, near-miss proposals and preregistered locked prediction/falsification metrics.

Preserve diagnostic-only authority and all original data. Validate position coverage and actual
computation identity, package the API guide, and retain honest random-model null/falsification
results. No migration, optimizer signal, parameter change, scientific causal claim or ordinary
inference requirement is introduced. Activation substitution, KL and causal experiments follow
in separate stages.
