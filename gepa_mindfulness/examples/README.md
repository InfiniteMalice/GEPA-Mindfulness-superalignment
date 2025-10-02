# Examples

Two runnable examples demonstrate the training pipeline:

- `cpu_demo` – A minimal end-to-end run that uses small models and completes in
  under 10 minutes on CPU hardware. Execute `python run_cpu_demo.py` from the
  folder to observe PPO reward shaping and trace logging.
- `vllm_demo` – Illustrates how to target a vLLM deployment by pointing the
  configuration to an engine endpoint. Requires an active vLLM server.

Each sub-folder contains its own README and runnable script.
