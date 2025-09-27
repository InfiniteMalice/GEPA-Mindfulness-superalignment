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

Each folder ships with its own README for quick orientation.

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

## License

This project is provided for research and alignment experimentation. Review the
individual model licenses for any deployed checkpoints.
