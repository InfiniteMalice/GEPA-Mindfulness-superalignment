# Examples

Two runnable examples demonstrate the training pipeline:

- `cpu_demo` – A minimal end-to-end run that uses lightweight Hugging Face
  models. Execute `python run_cpu_demo.py` (or
  `python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo`) to observe PPO
  reward shaping and JSONL rollout logging under `training_logs/`.
- `vllm_demo` – Illustrates how to target a vLLM deployment by pointing the
  configuration to an engine endpoint defined in `configs/vllm.yaml`. Requires
  an active vLLM server and the `requests` dependency.

Each sub-folder contains its own README and runnable script.

## Quick CLI demos

Produce a fresh trace, summary, and viewer bundle (requires the `[dspy]` extra):

```bash
gepa dspy run --input examples/self_tracing_sample.jsonl --trace runs/demo_trace.jsonl
gepa score --trace runs/demo_trace.jsonl --out runs/demo_trace_summary.html
gepa view --trace runs/demo_trace.jsonl --tokens runs/tokens.jsonl --out runs/demo_trace_view.html
```

To explore paired honest/deceptive chains using the DSPy pipeline:

```bash
gepa paired run --data datasets/paired_chains/data.jsonl --out runs/paired/ --context safety_critical
```
