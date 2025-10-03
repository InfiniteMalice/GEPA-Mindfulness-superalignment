# Examples

Two runnable examples demonstrate the training pipeline:

- `cpu_demo` – A minimal end-to-end run that uses small models and completes in
  under 10 minutes on CPU hardware. Execute `python run_cpu_demo.py` from the
  folder to observe PPO reward shaping and trace logging.
- `vllm_demo` – Illustrates how to target a vLLM deployment by pointing the
  configuration to an engine endpoint. Requires an active vLLM server.

Each sub-folder contains its own README and runnable script.

## Quick CLI demos

Viewer demo from an existing sample trace:

```bash
gepa score --trace examples/sample_trace.jsonl --policy policies/default_cw4.yml --out report.html
gepa view --trace examples/sample_trace.jsonl --tokens runs/tokens.jsonl --out report_view.html
```

Paired chains baseline run:

```bash
gepa paired run --data datasets/paired_chains/data.jsonl --out runs/paired/ --context safety_critical
```
