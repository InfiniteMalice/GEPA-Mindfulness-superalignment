# Global Response Normalization (GRN)

Global Response Normalization (GRN) can be toggled for confidence aggregation, PPO
policy logits, and deception probes. GRN is disabled everywhere by default to
avoid changing existing behaviour.

## Layer behaviour

The shared `GlobalResponseNorm` layer supports 2D or 3D inputs and normalises
values along a configurable dimension before applying a learnable (or fixed)
scale/shift. The default configuration keeps parameters frozen and uses
`dim=-1` with `eps=1e-6`.

```python
from mindful_trace_gepa.train.grn import GlobalResponseNorm

grn = GlobalResponseNorm(dim=-1, eps=1e-6, learnable=False)
output = grn(inputs)
```

## Confidence aggregation

`src/mindful_trace_gepa/scoring/aggregate.py` accepts a `confidence_grn`
section. When enabled, GEPA dimension confidences are normalised before the
0.75 abstention cutoff is applied. Confidence values are clamped to `[0,1]`
after GRN.

```yaml
confidence_grn:
  enabled: true
  dim: -1
  eps: 1.0e-6
  learnable: false
```

## PPO policy head

`gepa_mindfulness/training/ppo_trainer.py` can apply GRN to policy logits before
computing token probabilities. Configure this via `policy_grn` in
`configs/ppo/ppo_default.yaml` (disabled by default).

## Deception probes

Probe activations can be normalised with GRN prior to per-token and pooled
scores. Add an `activation_grn` block to the probe config (default disabled and
non-learnable) to activate this path. CLI runs automatically forward the block
from `configs/deception/probes_linear.yaml`.
