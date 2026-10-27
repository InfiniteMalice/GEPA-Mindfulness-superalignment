# Optional Subspace Tuning experiment

Maturity: **Experimental**. The standalone CPU adapter and authored matrix probe are separate
from the canonical RL engine. Tensor execution requires the existing `rl` extra. CUDA, AMP,
distributed training, PEFT/LoRA integration and canonical checkpoint integration are **Unsupported**
for this adapter. No Mindfulness retention or capability improvement has been established.

## Source and adaptation boundary

[Chavan et al., arXiv:2610.22293v1](https://arxiv.org/html/2610.22293v1), submitted 2026-10-15,
defines SST in sections 2.1–2.3 and Appendix B.8 Algorithm 1. Primary metadata and these method
sections were checked on 2026-10-27. The paper is CC BY 4.0. An authoritative upstream code
repository was not identified; no upstream code was copied. The stable bibliography entry is
[REF-SST](recommendations/RESEARCH_TRACEABILITY.md#ref-sst).

For each selected weight matrix, protect `k = floor(rank_fraction * min(rows, columns))` leading
left/right singular directions of its reference matrix. Let the native optimizer update its
ordinary moments using the **full gradient**. Measure its actual parameter change, including
weight decay, and replace that change with

```text
Delta_projected = (I - U_k U_k.T) Delta (I - V_k V_k.T)
W_next = W_before + Delta_projected
```

This implements the paper's applied-step formulation; projecting gradients alone is insufficient
under Adam's coordinate-wise scaling. The implementation uses exact reduced SVD in float64,
including for refresh. The paper's efficient Gram-based implementation and approximate SVD are
not reproduced. Reconstruction residual is logged as a numerical diagnostic, not as a behavioral
retention score. Degenerate singular values can make the basis nonunique.

Default references remain fixed. Optional refresh re-anchors after every configured number of
successful steps. Reports identify the basis step used and whether refresh occurred. Refresh
protects successive reference subspaces and weakens guarantees relative to the initial reference.

Repository adaptations: explicit target selection, CPU float32/float64 restriction, conservative
freeze fallback, snapshot budgets, native state rollback and DEVELOPMENT diagnostics. Rank 0.99
is the source's reported common setting, not a validated Mindfulness hyperparameter. The default
element limits are engineering budgets: 1,048,576 per decomposed target and 4,194,304 across native
optimizer parameters. Snapshotting all native parameters and optimizer moments is intentionally
expensive; this stage is for bounded experiments, not large-model throughput.

## Explicit host use

```python
from gepa_mindfulness.training.subspace_tuning import SSTConfig, create_sst_adapter

adapter = create_sst_adapter(
    optimizer,
    {"layer.weight": model.layer.weight},
    reference_id="operator-declared-checkpoint-and-model-version",
    config=SSTConfig(enabled=True, rank_fraction=0.99),
)
# The existing host must admit data, compute an eligible loss and run backward first.
report = adapter.step()  # replaces optimizer.step(); do not call both
optimizer.zero_grad(set_to_none=True)
```

The factory returns `None` by default, without importing PyTorch or inspecting the optimizer.
A host choosing this adapter must branch explicitly between it and the ordinary optimizer step.
The canonical configuration, CLI, engine and backend never activate it automatically.

Only exact native `torch.optim.SGD`, `Adam` and `AdamW` are implemented. Named targets must be
unique trainable parameters owned by that optimizer. All optimizer parameters must have dense,
finite, contiguous, nonaliased CPU float32/float64 storage within the snapshot budget. Sparse
gradients, closures, fused, capturable and differentiable optimization are unsupported. Do not
attach external optimizer hooks or mutate targets/groups while the adapter is active. This is a
trusted-host numerical library, not a sandbox against arbitrary Python callers.

Unselected parameters receive their ordinary native updates and are counted in reports. An
unsupported target shape or unavailable SVD raises before the first step by default. An explicit
`fallback="freeze"` suppresses that target's entire parameter update and logs why; it never
silently applies an unprotected update. Invalid/nonfinite parameters or gradients still fail.
If projection, diagnostics or refresh fails after the native step, all optimizer parameters and
native optimizer state are restored; the exception propagates and adapter counters stay unchanged.
External hook side effects and gradients are not rolled back. Run single-threaded with host-owned
parameters; concurrent mutation is unsupported.

Per-target diagnostics include native/applied update norms, SVD reconstruction residual, protected
rank, freeze reason and left/right leakage measured from the **actual stored** post-cast update.
Float32 rounding can cause nonzero leakage even when the double-precision projection is correct.
Rank-zero targets keep their native stored result exactly; fully protected or frozen targets
restore their original stored value exactly, avoiding cancellation or amplified projector roundoff.
No leakage threshold is silently interpreted as alignment. Reports retain source/reference identity
and optimizer name, and remain DEVELOPMENT data with `authority_granted: false`.

## Checkpoint and recovery

`adapter.state_dict()` returns versioned configuration, reference identity, counters, cloned
reference matrices and explicit frozen reasons. Restore it with `load_state_dict` only alongside
the matching model weights and native optimizer state, under the host's existing checkpoint
integrity controls. Invalid versions, names, shapes, dtypes, reference identity, configuration or
refresh counters fail before adapter mutation. Reference tensors are detached from caller-owned
checkpoint containers. The SVD is recomputed from the saved anchor on restore; exact bitwise
reproduction across BLAS/PyTorch versions or degenerate SVDs is not guaranteed. Record those
runtime versions. Restore does not authenticate a checkpoint or authorize model promotion.

The canonical engine does not serialize this adapter. Resuming through that engine without a
separate, matching adapter checkpoint is unsupported. Roll back to the declared model, native
optimizer and adapter checkpoint together after an interrupted external run.

## Bounded diagnostic and reproducibility

```python
from gepa_mindfulness.training.sst_probe import run_sst_probe
report = run_sst_probe(seeds=(0, 1, 2), steps=20, optimizer_name="AdamW", enabled=True)
```

This runs three paired conditions on identical authored 2x2 matrices: ordinary optimization,
half-rank SST and full-rank SST. It reports every seed's target-fit MSE, protected-input drift,
row-distribution KL and final spectral diagnostics. Fixed local generators preserve global RNG
state. The step/seed limits bound work; no API, download or model checkpoint is involved.
The fixture and analytic evaluator are versioned. These are numerical diagnostics, not held-out
behavioral tasks, confidence intervals or evidence of moral alignment. Full-rank protection can
block learning and makes that tradeoff visible.

Run `python -m pytest tests/test_subspace_tuning.py tests/test_sst_probe.py`. Real tensor tests
check native optimizer state, projection after weight decay, rank extremes, reference refresh,
checkpoint continuation, malformed inputs, rollback and bounded diagnostics. Without PyTorch,
those tests skip explicitly; passing configuration checks does not qualify the numerical path.

## LoCD + SST and alignment-first research protocol

The [distillation adapters](distillation_adapters.md) keep LoCD continuation loss separate from
ActionOPD semantic-decision comparison. For a host-authorized experiment, compute an admitted
LoCD loss and backward pass before the adapter step. This does not add a new loss or bypass
provenance/credit accounting. PEFT factor projection would constrain adapter coordinates rather
than the effective full-weight update, so it is not presented as compatible SST. Full-weight
CPU experiments are the implemented scope; merged-weight LoRA enforcement needs separate work.

The following protocols are **not executed** by this PR. Compare coherent model → alignment →
reasoning → coding; coherent model → reasoning → coding → alignment; alignment → added skills
with SST; and the same added skills without SST. Use declared identical starting checkpoints,
data versions, evaluator versions, compute budgets and multiple paired seeds. Separately compare
LoCD alone with LoCD + SST, using matched localized corrections. Sweep no refresh versus explicitly
declared refresh and check whether protection blocks necessary corrections.

At each stage use independently held-out tests under existing evaluation epochs and admission
rules: all 17 canonical cases with `CASE × STRIPE × REPEAT`, calibration, justified and false
abstention, semantic-intent robustness, honest completion, non-deception/non-coercion, general
skills, KL drift and protected-case regressions. Keep raw per-case/per-seed outcomes and paired
differences with justified uncertainty estimates; do not optimize the held-out suite. Use
independent model/harness versions for the four-cell coevolution comparison when either changes.
Missing behavioral evidence stays missing; do not replace it with spectral or toy metrics.
Spectral preservation cannot certify constitutional values, and no improvement is promoted while
a protected requirement regresses.
