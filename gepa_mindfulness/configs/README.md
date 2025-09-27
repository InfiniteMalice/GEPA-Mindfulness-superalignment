# Configuration Files

YAML configuration files define training hyper-parameters, reward weights, and
model selections. Two presets are provided:

- `default.yaml` – CPU-friendly settings for the minimal demo.
- `vllm.yaml` – Example configuration for integrating with an external vLLM
  engine.

Users can copy these files to craft new experiments and invoke the CLI with the
`--config` argument.
