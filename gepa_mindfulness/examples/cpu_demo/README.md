# CPU Demo

Run a short PPO training loop with lightweight models on CPU-only hardware.

```bash
python run_cpu_demo.py
```

The script loads the shared `configs/default.yaml` configuration, invokes the
training CLI with the bundled `prompts.txt` dataset, and performs two PPO
rollouts. Output is written to the directory requested via `--log-dir` (defaults
to `training_logs/` in the repository root) including:

- `rollouts.jsonl` – serialized prompt/response/reward records.
- `training.log` – CLI log output.
- Any adversarial evaluation summaries emitted by the orchestrator.
