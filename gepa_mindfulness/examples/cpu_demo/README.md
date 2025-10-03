# CPU Demo

Run a short PPO training loop with lightweight models on CPU-only hardware.

```bash
python run_cpu_demo.py
```

The script loads the shared `configs/default.yaml` configuration, runs two
rollouts, and prints the resulting rewards, trace summaries, and contradiction
reports.
