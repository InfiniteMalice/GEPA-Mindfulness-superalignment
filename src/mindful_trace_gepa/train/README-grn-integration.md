# GRN Integration Notes

This repository already ships a minimal Global Response Normalization (GRN) utility in
`src/mindful_trace_gepa/train/grn.py`. The module defines:

- `GRNSettings`: dataclass carrying `enabled`, `dim`, `eps`, and `learnable` knobs.
- `GlobalResponseNorm`: the normalization module that wraps `torch.nn.Module` when torch is
  available. It supports 2D and 3D tensors with a residual connection.
- `build_grn(settings)`: helper that parses a `GRNSettings` instance or mapping and returns a
  configured module when `enabled=True`, otherwise `None`.

## Usage patterns

- Confidence heads: pass a `GRNSettings(enabled=True, dim=-1)` into `build_grn` and apply the
  resulting module to logits or scalar confidence vectors prior to computing calibration losses.
- Probe logits: reuse `build_grn` with `dim=-1` to stabilise deception probe outputs before taking
  penalties.
- Population fitness: apply `build_grn` to a `(batch, 1)` fitness tensor to smooth extremes. Keep
  this optional to preserve phase-0 attribution behaviour when disabled.

The new EGGROLL + MDT trainer imports `build_grn` directly and wires feature-specific flags to
avoid silent behavioural changes when GRN is disabled.
