# Phase 2 PR 5: substitution and downstream simulation

Maturity: **Experimental** software; all results **Diagnostic**. Model alignment, semantic
explanation, causal mechanism admission and method superiority remain **Unqualified**.
Base: local PR 4 commit `5e2b686`. Existing inference, graph, training and authority contracts
remain backward-compatible; no migration is required.

## API, trust boundary and inventory

New source: `gepa_mindfulness/interpret/activation_simulation.py` and
`activation_simulation_records.py`.
New tests: `tests/test_activation_simulation.py`.
New guide: `docs/activation_simulation.md`.
New reproduction script: `docs/experiments/phase2_pr5_simulation_probe.py`, with compressed
raw JSON output, validation artifacts and a source/hash manifest.
Modified: `docs/README.md`, package data, and wheel-resource expectations.

The [API guide](../activation_simulation.md) documents exact tensor replacement, real
`KL(P_original || P_substituted)`, shifted teacher-forced NLL, calibration, answer-token probes,
regime/OOD measurements, same-hook groups and explicit unavailable controls. Control records
freeze original generation sources, construction seed, predictors, mappings and evaluation
positions. Original generation measurements are replayed on the frozen actual model before
locked simulation. Only closed public-context predictors are used; no model content becomes
executable code or verification authority.

Identity, zero ablation, generation mean, random predictor, shuffled labels, magnitude-matched
directions, random components, matched-complexity and cross-context conditions are all retained
or explicitly unavailable. Exact original/changed logits permit independent score recomputation.
No score, circuit observation or low-KL result becomes optimizer input or an accepted-store grant.

## Test-first and independent review

Initial tests failed because the simulation module was absent. Red/green checks cover actual KL
direction against a known scalar calculation, exact replacement shape/dtype/device/positions,
frozen model/source/RNG preservation, stored-logit NLL/KL and ECE recomputation, control tampering,
generation provenance, singleton degeneracy, weak/negative regime counts, behavior-token
mappings, evidence budgets and exception cleanup. A forced zero random direction initially
failed magnitude matching; the deterministic unit-vector fallback now preserves the control.
An inaccurate constant with tiny downstream KL remains a hypothesis, not an explained neuron.

Fresh read-only review passed the initial **14 simulation tests** and additionally obtained zero
identity KL across 12 supported scalar hook/family combinations. Coherently forged generation
records plus rewritten commitment digests were rejected by actual model replay. No critical or
important defect was reported under the reviewer's initial diagnostic scope. Three minor
observations were regraded by the implementer against the explicit trust/position/fidelity
requirements and addressed in the final fix pass:

- Exact index types now fail before any forward call. Bool/float positions previously compared
  equal to integer positions and failed incidentally after entering execution. The reproducing
  cases failed before the fix and now pass. Ruling: strict pre-execution position validation is
  part of the requested typed intervention boundary; its cost is rejecting malformed aliases.
- Probe declarations are host-supplied. Documentation now distinguishes reconstructed statistics
  and component assignments from external probe custody. Ruling: the guide must not imply that
  reconstruction authenticates a changed host answer mapping; host protected preregistration
  remains necessary, with invalid scientific inference the cost of ignoring it.
- Positive token offsets could expose gold token p+1 while predicting activation at p. A real
  causal-model reproducer shows identical earlier activation under a future-token change while
  the retrospective predictor changes. PR 5 now rejects positive token/relation offsets and
  records `causal_prefix`; matched baselines also use past/current offsets only. Ruling: causal
  decoder simulation requires permitted-prefix features. The cost is that PR 4 retrospective
  hypotheses remain discovery-only and cannot use this new simulation path. PR 4's API/evidence
  is preserved, not rewritten.

Final focused tests: **48 passed**, 4.05 seconds (17 new simulation plus 31 native/prediction).
No deferred minor findings remain. Temporal custody/actor isolation, trained-model semantic or
alignment improvement, CUDA qualification, vector-head predictors and cross-layer paths remain
outside this stage. Those limitations are explicit rather than inferred from test success.
Packaging/full regression and the separate reproduction artifact are validated by the parent.

## Actual random-model results

Reproduce from the checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m docs.experiments.phase2_pr5_simulation_probe --enabled > simulation.json
```

Seed 53 initializes tiny two-layer GPT-2/Llama models and preselects three layer-zero components.
The first generated past-token hypothesis and constant baseline are tested as singleton and
three-component groups on new locked artificial token contexts. No candidate/control is selected
after inspecting locked results. These are software probes on random networks, not trained
language-model studies. GPT-2 postactivation and Llama gated-product scales differ.

| Family | Group | Predictor | Mean KL | Applied activation MAE | Mean shifted NLL change |
|---|---:|---|---:|---:|---:|
| GPT-2 | 1 | Past-token candidate | 1.884738e-7 | 0.027257 | +5.648500e-5 |
| GPT-2 | 1 | Constant | 1.263968e-7 | 0.023822 | -1.131181e-4 |
| GPT-2 | 3 | Past-token candidates | 7.231994e-7 | 0.026680 | -8.286183e-5 |
| GPT-2 | 3 | Constants | 6.393027e-7 | 0.024979 | -1.678489e-4 |
| Llama | 1 | Past-token candidate | 3.906445e-8 | 0.002136 | -6.739086e-5 |
| Llama | 1 | Constant | 3.904546e-8 | 0.002077 | -5.130982e-5 |
| Llama | 3 | Past-token candidates | 6.150278e-8 | 0.002134 | -7.365681e-5 |
| Llama | 3 | Constants | 5.655108e-8 | 0.002010 | -5.253562e-5 |

All eight identity conditions have zero KL. Across the eight studies, 76 measured conditions
retain **684** matched output-position records. Four singleton shuffled-label controls are
explicitly unavailable. Original and substituted logits were checked using a separate Python
scalar log-sum-exp implementation: every KL and shifted NLL agrees within 1e-12. A repeated
execution reproduces all numerical summaries and model identities; timestamp/digest changes
on new commitments are expected. All six source hashes match the retained final artifact.

Tiny KL accompanies substantial activation error, and simple baselines often equal or outperform
the candidate. These results do not establish a clean neuron, causal explanation, calibrated
uncertainty, training improvement, semantic task competence or alignment improvement. Original
controls/logits/activations, source configurations, IDs, seeds, nulls and raw measurements are
retained in `phase2_pr5_simulation_probe.json.gz`. Temporal authentication remains `NOT_PERFORMED`;
all evidence statuses remain `PREDICTIVE_HYPOTHESIS`. No model parameters changed.

## Quality gate and proposed PR description

The maintainer-approved fallback for unavailable `repo-quality-gate` applies: stage specification,
design, test-first implementation, formatting/lint/type checks, fresh review and this evidence
summary. Beads automation was not invoked. Ruff passes repository-wide; Black checks 841 Python
files. Both existing CI mypy commands and explicit checks of the two modules/reproduction script
pass. Wheel-resource verification passes after the guide was added to package data.

Protected full regression: **7,191 passed, 11 skipped, 18 warnings, 162 subtests passed**,
430.91 seconds, exit 0. The same optional backend/hardware skips remain unqualified. Full stdout,
compressed per-test JUnit, original simulation evidence and source/artifact hashes are retained.

Proposed description: Add default-off frozen activation simulation with actual downstream
KL/NLL, retained logits, matched diagnostic controls and same-hook neuron groups. Require exact
tensor/hook/input/position compatibility and causal-prefix predictors, replay original generation
sources, and preserve model/RNG state. Include optional answer-token probes, calibration/OOD
measurements, typed records, adversarial tests and honest random-model reproducibility artifacts.
No migration, inference-default change, parameter update, optimizer reward or causal/semantic
evidence admission is introduced.
