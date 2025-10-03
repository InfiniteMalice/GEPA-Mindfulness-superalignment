# GEPA Mindfulness Superalignment

This repository implements a full training pipeline that integrates GEPA-based
interpretability, paraconsistent imperative logic, and traceable honesty via
Anthropic-style thought tracing. The project targets Python 3.10+ and depends on
`torch`, `transformers`, `trl`, `pydantic`, `jinja2`, `pyyaml`, and
`self-tracing` (optional).

## Repository Structure

- `gepa_mindfulness/core` – GEPA contemplative principles, paraconsistent
  imperative modeling, abstention, reward shaping, adversarial probes, and
  self-tracing integration.
- `gepa_mindfulness/training` – Configuration models, PPO training orchestrator,
  and CLI tooling.
- `gepa_mindfulness/adapters` – Interfaces for policy backends and trace-to-
  checkpoint conversion.
- `gepa_mindfulness/configs` – YAML presets that expose the reward weight
  parameters (α, β, γ, δ) and model selections.
- `gepa_mindfulness/examples` – Runnable CPU and vLLM demonstrations.
- `scripts` – Shell helpers for running the complete pipeline end-to-end.
- `gepa_datasets` – Curated JSONL corpora for ethical QA, OOD stress tests,
  anti-scheming probes, abstention calibration, and thought-trace templates.

Each folder ships with its own README for quick orientation.

The dataset bundle ships in this repository as both the expanded
`gepa_datasets/` directory and the original `gepa_datasets.zip` archive so the
files can be consumed directly or redistributed in packaged form.

## Key Features

1. **GEPA Scoring** – Implements four contemplative principles (Mindfulness,
   Empathy, Perspective, Agency) and aggregates them alongside three alignment
   imperatives (Reduce Suffering, Increase Prosperity, Increase Knowledge) using
   paraconsistent logic to tolerate conflicting objectives.
2. **Self-Tracing** – Wraps the [Self-Tracing](https://github.com/recursivelabsai/Self-Tracing)
   library to capture framing, evidence, tensions, decisions, and reflections
   for every rollout. These traces produce GEPA checkpoints and honesty reward
   signals.
3. **Confidence-Aware Abstention** – Forces the policy to respond with “I don’t
   know” whenever confidence drops below 0.75 and computes a separate honesty
   reward emphasizing mindfulness and emptiness signals.
4. **Reward Shaping** – Combines task success, GEPA scores, honesty rewards, and
   hallucination penalties via configurable weights (α, β, γ, δ) defined in YAML
   configs.
5. **Adversarial Evaluation** – Provides OOD prompts inspired by
   [scheming-behavior research](https://arxiv.org/abs/2509.15541) to test for
   covert misalignment during rollout.
6. **Runnable Examples** – Includes a CPU-only demo (<10 minutes) and a vLLM
   integration example, plus a shell script for orchestrating the full pipeline.
7. **DSPy Declarative Modules** – Optional, policy-guarded DSPy-style pipelines
   that emit GEPA checkpoints and integrate with the trace viewer.
8. **Offline Trace Viewer** – Token- and checkpoint-level visualisation with
   deception overlays for honest vs deceptive chain inspection.
9. **Paired Chains Baseline** – Controlled honest/deceptive emitters and
   detectors to seed early deception analysis prior to reward tuning.
10. **Unsloth/PEFT Fine-Tuning** – Ready-to-run notebooks for Phi-3 Mini and
    Llama-3 8B that wire GEPA abstention, PPO reward blending, and offline
    trace/report generation into the LoRA training workflow.

## Getting Started

1. Install dependencies:

   ```bash
   pip install torch transformers trl pydantic jinja2 pyyaml self-tracing requests
   ```

2. Run the CPU example:

   ```bash
   python gepa_mindfulness/examples/cpu_demo/run_cpu_demo.py
   ```

3. Execute the full pipeline script:

   ```bash
   ./scripts/run_full_pipeline.sh
   ```

4. For vLLM integration, ensure a vLLM server is running and adjust
   `gepa_mindfulness/configs/vllm.yaml` as needed before executing
   `python gepa_mindfulness/examples/vllm_demo/run_vllm_demo.py`.

## DSPy Declarative Pipelines

DSPy-style modules live under `src/mindful_trace_gepa/dspy_modules`. They are
disabled by policy until explicitly enabled in `policies/dspy.yml`. To execute
the pipeline with GEPA checkpoint logging:

```bash
gepa dspy run --input examples/self_tracing_sample.jsonl --trace runs/trace.jsonl
```

To export the guarded prompt manifest:

```bash
gepa dspy compile --out dspy_artifacts/ --enable-optim
```

## Trace Viewer

The offline viewer bundles traces, tokens, and deception metadata into a single
HTML file that can be opened locally without external dependencies:

```bash
gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl --out report_view.html
```

The viewer assets are located in `src/mindful_trace_gepa/viewer/` and are served
without external CDNs.

## Paired Chains Baseline

Baseline datasets for honest/deceptive paired chains live under
`datasets/paired_chains/`. Generate chains and detector outputs via:

```bash
gepa paired run --data datasets/paired_chains/data.jsonl --out runs/paired/ --context research
```

Inspect a specific scenario with the split-view trace viewer:

```bash
gepa paired view safety_lab_001 --base runs/paired/ --out runs/paired/safety_lab_001_view.html
```

## Fine-Tuning with Unsloth/PEFT

Model presets live under `configs/models/`, the PPO reward weights under
`configs/ppo/`, and a standalone policy reference under
`configs/policies/policy.yml`. The notebooks in `notebooks/` combine these
presets with local dataset subsets so CPU smoke tests complete in a few
minutes. Launch either notebook to reproduce the LoRA workflow:

```bash
jupyter notebook notebooks/ft_phi3_mini_unsloth_gepa.ipynb
jupyter notebook notebooks/ft_llama3_8b_unsloth_gepa.ipynb
```

Each run writes `runs/tokens.jsonl`, `runs/trace.jsonl`, `runs/summary.json`,
and renders `report.html` plus `report_view.html` via the final cell:

```bash
!gepa score --trace runs/trace.jsonl --policy policies/default_cw4.yml --out report.html
!gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl --out report_view.html
```

## License

This project is provided for research and alignment experimentation. Review the
individual model licenses for any deployed checkpoints.
