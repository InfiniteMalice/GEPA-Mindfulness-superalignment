# Phase 2 PR 8: comparative interpretation

Maturity: **Experimental** software and **Diagnostic** evidence. Learned-SAE quality, semantic
explanation, trained-model causal validity, method superiority and alignment improvement remain
**Unqualified**. Base: local PR 7 commit `83ca0e2`. No legacy schema, inference default, training
objective, model parameter, canonical case or runtime authority is changed.

## Inventory and typed API

New source: `gepa_mindfulness/interpret/comparative_records.py`, `comparative.py`.
New tests: `tests/test_comparative_interpretation.py`. New [API/protocol](../comparative_interpretation.md),
`phase2_pr8_comparison_probe.py`, compressed raw results, stdout/JUnit and source/hash manifest
accompany this report. Documentation index, package data and wheel-resource expectations are
extended. The refinement is `history/2026-11-05-phase2-comparative-design.md`.

FeatureDictionary, ComparisonArm/Budget/Protocol, ArmFit and ComparisonCommit retain exact bounded
versioned contracts. Four method families share model/hook, data/splits/positions, selected count,
public-feature complexity and forward cap. Native coordinates are actual identity maps. Optional
SAE/other dictionaries use fixed finite weights; absent methods remain UNAVAILABLE. Contextual
hypotheses retain circuit references, but no reference alone proves explanatory use. Actual
source attention measurements and component weights are retained by the probe.

Preparation captures generation activations and freezes closed causal-prefix ridge predictors and
controls. Locked execution replays generation and recomputes fits, then performs real frozen-hook
forwards using the existing tensor/KL kernels. Full reconstruction, residual-preserving selected
feature substitution, activation prediction and downstream effects remain distinct. Reports retain
raw logits/activations/features, KL/NLL/calibration/OOD, signed ablation predictions, complexity,
regime coverage and actual costs. All causal evidence remains UNRESOLVED; no signal becomes reward.

## Test-first and review disposition

Initial tests failed on the absent module. The wheel test failed on the absent guide before package
integration. The normalization test failed on the missing generation-scale metric and then passed;
zero generation scales remain explicitly unnormalized. Existing and added checks cover real tiny
GPT-2/Llama forwards, scalar KL/NLL recomputation, source/mode/RNG preservation, exact identity,
fixed linear/ReLU dictionaries, grouped shuffled controls, unavailable methods, generation/control
replay, future-token rejection, mismatched model/budgets/hooks and malformed geometry.

A fresh read-only source review found no Critical or Important issue, with two Minor observations.
The reviewer did not independently execute tests. Both observations were addressed under these
explicit scope rulings; none are deferred:

1. All-unavailable configurations were accepted until capture produced an empty-fit failure.
   A reproducer failed before the fix. Protocol validation now rejects them before capture, while
   unavailable optional arms remain in otherwise executable comparisons. Ruling: early typed
   failure is part of the requested unsupported/unrun boundary. The cost is requiring at least
   one executable method instead of constructing a misleading empty measurement run.
2. Existing fixtures did not independently check a dense nonorthogonal decoder with nonzero
   encoder/decoder biases or the reported signed ablation outcomes. A new Python-scalar fixture
   reconstructs residuals, applied values, decoded activations and contrast/control outcomes from
   retained data; it passes without changing the intervention kernel. Ruling: the user explicitly
   requires checkable mathematical effects, so this coverage closes an evidentiary gap. Its cost
   is an additional bounded CPU regression fixture, not a broader causal claim.

Final focused validation: **15 passed**, 7.30 seconds. The zero-decoder fixture gives zero explanation
KL despite nonzero reconstruction/prediction error and a falsified prediction criterion. A 10x
feature scaling changes raw MAE by 10x while preserving generation-normalized MAE. Neither tiny KL,
zero decoder impact, arbitrary feature units nor a numeric tolerance is treated as understanding.

## Actual software comparison

Reproduce from the checkout with CPU extras and no downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m docs.experiments.phase2_pr8_comparison_probe --enabled > comparison.json
```

Model seed 71 initializes random tiny GPT-2/Llama networks; control seed 79 selects units and
controls. Each family uses one- and two-feature groups with all methods available, plus a
singleton condition with the optional other method unavailable. These six conditions reuse models,
inputs and seeds and are dependent software observations. Context records contain actual generation
attention outputs and local writers/readers; contextual predictors remain predeclared competitors,
without proof that those observations supplied a semantic explanation.

The ReLU dictionary is a constructed identity/ReLU control, not a trained SAE. The other dictionary
is a constructed reversed basis. Native/contextual/reversed maps reconstruct exactly in these
fixtures; ReLU reconstruction MSE is `3.622776e-4` for GPT-2 and `7.713637e-7` for Llama.

| Family, singleton | Method | Feature MAE | Explanation mean KL | Reconstruction MSE |
|---|---|---:|---:|---:|
| GPT-2 | Native | 0.015265 | 5.387315e-8 | 0 |
| GPT-2 | Contextual | 0.021197 | 4.665400e-8 | 0 |
| GPT-2 | Constructed ReLU | 0.005939 | 4.494569e-9 | 3.622776e-4 |
| GPT-2 | Reversed basis | 0.025657 | 1.774649e-7 | 0 |
| Llama | Native | 0.000593 | 1.970327e-10 | 0 |
| Llama | Contextual | 0.000682 | 2.297643e-10 | 0 |
| Llama | Constructed ReLU | 0.000566 | 1.495928e-10 | 7.713637e-7 |
| Llama | Reversed basis | 0.000614 | 9.165207e-10 | 0 |

All **24 planned method arms** are retained: 22 measured and two unavailable. There are 184 measured
conditions, 1,104 position-level output records and 204 actual batch forwards including preparation,
shared baseline/replay and generation attention context. All 66 signed ablation predictions remain:
50 FALSIFIED, eight NOT_SPECIFIC and eight MET. These are numeric functional controls on random
networks, not verified semantic mechanisms. The small KL values and different activation scales
cannot support method superiority. Frozen raw tolerances are prototype settings, not calibrated
cross-architecture explanation thresholds. All nulls, missing controls and original logits remain.

Observed per-arm locked CPU times are about 20-26 ms in this run, including framework/frozen-state
checks. They exclude original discovery and decomposition training. Actual preparation times,
shared replay/refitting, dictionary sizes and forward counts remain in the raw artifact. Repeated
conditions are not independent samples or a mathematical time-space result. No learned dictionary,
trained alignment task, parameter update, human-intent inference or accepted-store admission ran.

## Quality gate and proposed PR description

The unavailable named quality-gate skill uses the maintainer-approved fallback: task spec, refined
design, test-first implementation, formatting/lint/types, fresh review, fixes and recorded gate.
Beads automation was not invoked. Current read-only workspace edits/tests use explicitly scoped
escalation approvals; no alternate write route bypasses those restrictions. Full regression,
original stdout/compressed JUnit, raw probe results and hashes are retained in the adjacent manifest.

Protected full regression: **7,280 passed, 11 skipped, 18 warnings, 162 subtests passed**,
567.69 seconds, exit 0. Repository-wide Ruff and Black (859 Python files), both existing CI mypy
commands, and explicit type checks of the two modules plus probe pass. The isolated wheel-resource
test passes. Independent scalar checks verify all source/context/report digests, all 1,104 KL/NLL
output rows, identity preservation, 66 signed effects/control outcomes and actual forward counts.

Proposed description: Add default-off matched native/contextual and fixed-dictionary comparison
contracts and actual frozen-model evaluation. Replay generation-derived predictors/controls,
measure reconstruction separately from residual-preserving simulation, retain actual downstream
KL/NLL, calibration/OOD, signed ablation effects, complexity and costs. Preserve optional unavailable
methods, nulls, contradictions and diagnostic-only status. Include dense/bias/scale controls,
real tiny-architecture tests and reproducible random-model data. No migration, inference-default
change, reward route or superiority/alignment claim is introduced.
