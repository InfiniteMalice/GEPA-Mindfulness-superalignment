# Phase 2 PR 6: causal circuit verification

Maturity: **Experimental** software and **Diagnostic** evidence. Trained-model causal explanation,
semantic understanding, deceptive intent and alignment improvement remain **Unqualified**.
Base: local PR 5 commit `428678a`. Ordinary inference and all legacy wire formats remain compatible;
no migration is required. Research APIs default off.

## Inventory and trust boundary

New modules under `gepa_mindfulness/interpret/`: `native_causal_records.py`,
`causal_interventions.py`, `mechanistic_evidence.py`, `mechanistic_graphs.py`.
New tests: `tests/test_native_causal_interventions.py`, `tests/test_mechanistic_evidence.py`,
`tests/test_mechanistic_graphs.py`. New [typed API/protocol](../causal_mechanistic_evidence.md),
`phase2_pr6_causal_probe.py`, raw compressed JSON, test logs/XML and source/hash manifest accompany
this report. Documentation index, package data and wheel-resource expectations are extended.
The stage's refinement is `history/2026-11-05-phase2-causal-design.md`.

Seven intervention modes perform actual copied-model forwards at factory-verified hooks:
patching, literal replacement, ablation, bidirectional component/residual swaps, targeted
restoration, functional path patching and controlled mediation. Every run retains logits,
applied activations, effect direction/magnitude, controls, model/input/site identities and existing
prediction/action-bound events. Parameter updates, optimizer rewards, inferred-intent penalties,
runtime grants and canonical-case changes are absent.

Raw measurements remain unresolved. Independent host authentication plus actual replay is needed
for supported findings; reviewer endorsement cannot supply it. Registry admission rechecks sources
and consumes the existing one-use WRITE grant bound to exact before/after snapshots. Host-pinned
graph annotations preserve original attribution weights and edge uncertainty. Accepted registries
remain fixed diagnostic, DEVELOPMENT records and cannot become training inputs.

## Test-first implementation and review rulings

Initial tests failed on missing causal/evidence/graph modules. The wheel test failed on the missing
new guide before package data was extended. Existing event serialization initially retained tuple
references; canonical JSON conversion restored the strict JSON contract. An accidental collision
with the existing synthetic `test_causal_interventions.py` filename was corrected immediately:
the legacy file was restored byte-for-byte from HEAD and its **68 tests passed**. New native tests
use a distinct filename. No legacy test deletion or relaxed assertion is part of the change.

Fresh read-only review passed the initial 38 tests and reported two Important scope defects:

1. A future intervention position or changed masked token could inflate replication counts despite
   an identical causal prefix. Both reproducing cases failed before the fix. Replication now hashes
   only the measured output's prefix and normalizes masked token values. Ruling: input naming and
   causally irrelevant changes cannot establish independent replication. The cost is rejecting
   nominally different examples that provide no new causal-context evidence.
2. A recipient patch finding could annotate its donor graph, including a donor with zero measured
   necessity effect. The reproducing case failed before the fix. Graph support now binds the primary
   context by mode; reverse swaps need separate primary evidence. Ruling: retaining both contrasts
   does not establish both directions of the preregistered hypothesis. The cost is more narrowly
   scoped annotations rather than transferring unsupported causal claims.

The review's Minor observation about mode-specific test strength was regraded against the explicit
intervention-fidelity requirement and addressed. Additional real GPT-2/Llama assertions compare
actual donor values with replacements, reverse swaps, head-coordinate expansion, complete residual
swaps, changed-mediator path values, restoration recovery arithmetic, and hook cleanup after an
invalid execution order. No new implementation was needed for these checks. No deferred findings
remain. One fresh review plus red/green fixes is used; an additional opinion is not substituted for
the protected regression suite.

Further boundary tests reject coherent forged logit traces, rewritten scores/events/levels, missing
or mutable authentication, self-review, stale snapshot scopes, missing/replayed grants and default-off
calls. Renamed, future-only and masked-token replications cannot upgrade evidence. Falsified/null
results remain available, and a later contradiction cannot be hidden behind an old positive graph
binding. Actual locked prediction replay supports only HELD_OUT_SUPPORTED, never causal evidence.

## Actual software experiment

Reproduce without network access or model downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m docs.experiments.phase2_pr6_causal_probe --enabled > causal-probe.json
```

Seed 71 initializes tiny two-layer GPT-2 and Llama models (width 8, MLP width 16, vocabulary 16).
The protocol predeclares all seven modes and retains every result. A separate GPT-2 control zeros
other weights, restores normalization gains, and sets one explicit MLP writer/reader with distinct
embedding/readout axes. Two predeclared prefixes exercise that constructed path; an opposite-sign
prediction provides a falsification control. This is deliberately engineered software evidence,
not a discovered semantic or ethical mechanism in a trained model.

| Condition | Runs | MET | FALSIFIED | NOT_SPECIFIC |
|---|---:|---:|---:|---:|
| Random GPT-2, seven modes | 7 | 1 | 4 | 2 |
| Random Llama, seven modes | 7 | 0 | 6 | 1 |
| Constructed positive prefixes | 2 | 2 | 0 | 0 |
| Constructed opposite prediction | 1 | 0 | 1 | 0 |

All **17 runs, 139 measured conditions and 6,672 logit values** are retained. Each recorded contrast
is independently recomputed from the stored logits. Constructed patch effects are
`5.178619384765625` and `5.237773776054382`; random-component controls are zero. The opposite-sign
prediction is falsified despite the identical first effect. A synthetic host callback and actual
replay produce replicated/contradicted software findings. That callback is explicitly a fixture;
the artifact does not establish real organizational independence or admit evidence to a registry.
Random-model passing tolerances do not explain a neuron or validate alignment. Controlled mediation
assumptions, off-manifold effects, redundancy and untested routes remain explicit limitations.

## Quality gate and proposed PR description

The unavailable named quality-gate skill uses the maintainer-approved fallback: spec, refined
design, test-first implementation, lint/format/types, fresh review, fix validation and recorded gate.
Beads automation was not invoked. Final suite results and source/artifact hashes are recorded in
`phase2_pr6_manifest.json`; full stdout and compressed per-test JUnit remain adjacent to this report.
Optional hardware/endpoints remain unqualified rather than silently covered by toy probes.

Final focused validation: **48 passed**, 7.27 seconds. Full protected regression:
**7,239 passed, 11 skipped, 18 warnings, 162 subtests passed**, 442.71 seconds, exit 0.
Repository-wide Ruff and Black (849 Python files), both existing CI mypy commands, and explicit
type checks of all four new modules plus the reproduction script pass. The wheel-resource test
passes in the full suite. Probe contrasts, all condition effects, primary/control/direction effects
and identity preservation are independently recomputed from all retained logits.

Proposed description: Add default-off actual native activation patching, replacement, ablation,
swaps, restoration, path and controlled mediation experiments on frozen copies. Bind predictions
and outcomes through existing action events, replay raw measurements under host verification,
retain contradictions/nulls, and register diagnostic findings only through exact scoped one-use
WRITE authority. Add context/site-bound graph overlays without changing attribution weights,
independent held-out prediction checking, adversarial tests and reproducible random/constructed
model results. No inference-default change, migration, parameter update, training signal, intent
claim or alignment-improvement claim is introduced.
