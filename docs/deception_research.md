# Deception Research Integration

## Overview

Mindful Trace GEPA now combines three complementary layers of deception analysis:

1. **White-box linear probes** extract hidden activations from open weights models (Hugging Face and vLLM backends) and apply a simple logistic classifier to highlight steps that resemble deceptive behaviour. Layer selection and pooling (mean/CLS/last token) are fully configurable via YAML, and probe outputs are written as JSON so the viewer can render per-step heatmaps.
2. **Multimodal deception evaluation** loads common corpora (RLTD, OpSpam, MU3D) through text-first adapters. The loaders stream JSONL splits, normalise labels, and expose a unified iterator that yields `{id, text, label, meta}` dictionaries. Optional hooks surface audio/video metadata when `with_mm=True`, but the default configuration remains CPU-only for CI.
3. **Paired chains and GEPA integrity** keep the trace-level deception fingerprint. The existing baseline compares honest vs deceptive rollouts, while the new probe and dataset metrics feed into the summary aggregator for a holistic flag.

## Running the probe

The default configuration lives at `configs/deception/probes_linear.yaml`:

- `model_layer_indices`: layers to extract.
- `pooling`: aggregation strategy (`mean` by default).
- `threshold`: compute a decision boundary via fixed false-positive rate.
- `output`: control per-token/step output and destination.

Execute the CLI:

```bash
gepa deception probes --trace runs/trace.jsonl --model dummy --probe artifacts/dummy.pt \
  --config configs/deception/probes_linear.yaml --out runs/deception_probe.json
```

The command tolerates missing activations and closed APIs by falling back to deterministic synthetic features so CPU smoke tests succeed offline.

## Multimodal evaluation

Text-only loaders are exposed via `mindful_trace_gepa.data.mm_deception`:

```python
from mindful_trace_gepa.data.mm_deception import (
    iter_text_examples,
    load_mu3d_text_only,
    load_opspam_text_only,
    load_rltd_text_only,
)
```

Each loader reads `train/validation/test` JSONL splits and returns plain Python lists. The evaluation notebook (`notebooks/eval_deception_acl2025.ipynb`) demonstrates a baseline sweep that computes accuracy, F1, and AUROC on tiny synthetic samples, then writes `runs/mm_eval.json` for downstream aggregation.

Configure dataset paths via `configs/data/mm_deception.yaml`:

```yaml
rltd_path: "datasets/mm/RLTD/"
mu3d_path: "datasets/mm/MU3D/"
opspam_path: "datasets/mm/OpSpam/"
text_only: true
max_samples: 256
```

## Aggregating evidence

`mindful_trace_gepa.deception.score.summarize_deception_sources` merges:

- Paired-chain divergence scores.
- Linear probe flags and metrics.
- Multimodal evaluation summaries.

Invoke the CLI to produce `runs/deception_summary.json`:

```bash
gepa deception summary --out runs/deception_summary.json
```

The summary records per-source status, reasons, and a final boolean flag. The trace viewer automatically loads `deception_probe.json`, `deception_summary.json`, and `mm_eval.json` if they exist, rendering a deception badge plus a toggleable heat strip under the token confidence chart.

## Limitations & safety

- The probe is a lightweight linear classifier. High scores indicate similarity to deceptive traces but must not be treated as definitive proof.
- Multimodal hooks surface filenames only. Audio/video ingestion is intentionally disabled in CI to prevent accidental data exfiltration.
- Viewer overlays run fully offline—no external CDNs—and gracefully degrade when artifacts are absent.
- Downstream tooling should avoid exposing deceptive chain content to end users; use the summary outputs for internal triage only.

## References

- Goldowsky-Dill, N., Chughtai, B., Heimersheim, S., & Hobbhahn, M. (2025). *Detecting strategic deception with linear probes* (ICML 2025 poster).
- Miah, M. M. M., Anika, A., Shi, X., & Huang, R. (2025). Hidden in plain sight: Evaluation of the deception detection capabilities of LLMs in multimodal settings. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 31013–31034). Association for Computational Linguistics.
- arXiv. (2025). *arXiv:2508.06361* [Preprint].
