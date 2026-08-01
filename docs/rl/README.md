# Portable PyTorch RL

The canonical `gepa rl` command runs PPO or GRPO with the repository's strict
reward-integrity pair format, local checkpoints, and structured JSONL logs. The default backend
uses PyTorch and Hugging Face Transformers on CPU or CUDA. It never downloads a model: the model
identifier must resolve from a local directory or the existing Hugging Face cache.

## Install the runtime

Use Python 3.10 or newer. Install the runtime extra for execution or the development extra for
execution plus repository checks:

```bash
python -m pip install -e '.[rl]'
python -m pip install -e '.[rl-dev]'
```

The `rl` extra installs bounded versions of PyTorch, Transformers, and PEFT. It does not install
TRL, Datasets, Accelerate, a CUDA-specific wheel, or a platform SDK. The package manager selects
the PyTorch build from its configured package index.

## Prepare an offline run

Create a JSONL dataset with one complete authored pair per line. Training and evaluation reject
plain-text prompts because the reward pipeline needs the chosen/rejected evidence and component
labels. The schema is closed; unknown or missing fields fail before model construction.

```json
{"record_id":"example-1:grounded_over_proxy","source_case_id":"example-1","source_case_version":"1.0","source_path":"authored/example.jsonl","source_line":1,"source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pair_rule":"grounded_over_proxy","prompt":"Answer with calibrated care.","chosen":"A grounded answer.","rejected":"An unsupported answer.","chosen_class":"grounded_success","rejected_class":"proxy_exploitation","chosen_reward_components":{"objective_fidelity":0.5,"feedback_integrity":0.5,"skill_transfer":0.5,"reality_contact":0.5,"exploit_disclosure":0.5,"long_horizon_agency":0.5,"benign_creativity":0.5,"repair_quality":0.5},"rejected_reward_components":{"objective_fidelity":-0.5,"feedback_integrity":-0.5,"skill_transfer":-0.5,"reality_contact":-0.5,"exploit_disclosure":-0.5,"long_horizon_agency":-0.5,"benign_creativity":-0.5,"repair_quality":-0.5},"diagnostics":{"central":"authored","supporting":[]},"schema_version":"reward-integrity-rl-pairs-v1"}
```

Save the record as `data/rl/pairs.jsonl`. Then save this PPO configuration as
`run.cpu.ppo.yaml`. Replace `/absolute/path/to/local-model` with a local Transformers model
directory or an identifier that is already present in the local cache.

```yaml
runtime:
  backend: pytorch
  device: cpu
policy:
  model_name: /absolute/path/to/local-model
  max_new_tokens: 32
algorithm:
  name: ppo
  learning_rate: 1.0e-5
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 10
  clip_range: 0.2
  value_coef: 0.1
reward:
  weights:
    alpha: 0.3
    beta: 0.3
    gamma: 0.2
    delta: 0.2
dataset:
  train_path: data/rl/pairs.jsonl
  format: jsonl
checkpoint:
  output_dir: runs/rl_cpu_ppo/checkpoints
  save_steps: 1
logging:
  log_dir: runs/rl_cpu_ppo/logs
  level: INFO
seed: 42
```

For GRPO, copy the file to `run.cpu.grpo.yaml`, change `algorithm.name` to `grpo`, and add these
keys under `algorithm`:

```yaml
  group_size: 4
  kl_coef: 0.05
  group_normalization_epsilon: 1.0e-8
  zero_variance_policy: skip
```

`zero_variance_policy: skip` omits a response group when every response receives the same reward.
Use a dataset and model that can produce reward variation, or a GRPO invocation can finish without
an optimizer step.

## Check capabilities before loading a model

Run the model-free doctor for each configuration:

```bash
gepa rl doctor --config run.cpu.ppo.yaml
gepa rl doctor --config run.cpu.grpo.yaml
```

The doctor prints one `AVAILABLE` or `UNAVAILABLE` line for every capability required by the
selected algorithm. Exit status `0` means all required capabilities are available. Exit status
`2` means at least one capability is unavailable.

## Train PPO and GRPO on CPU

Force the supporting libraries into offline mode, then give each invocation a relative optimizer
step budget:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gepa rl train --config run.cpu.ppo.yaml --max-steps 1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gepa rl train --config run.cpu.grpo.yaml --max-steps 1
```

Each command prints one JSON result. A completed one-step update reports `"global_step": 1`,
`"policy_parameters_updated": true`, distinct before/after policy checksums, and the checkpoint and
log locations. The broader `parameters_updated` field covers the trainable policy plus value head;
use the policy-specific fields as model-weight evidence.

> **Memory warning:** If a full-weight model and its frozen reference copy exceed available CPU
> RAM, the operating system can terminate the process before a checkpoint is written. Check the
> model's memory requirement before training and start with a small local model.

## Resume exactly from a checkpoint

Select the checkpoint directory printed by the prior command. `--max-steps` is relative to the
restored global step.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gepa rl resume \
  --config run.cpu.ppo.yaml \
  --checkpoint runs/rl_cpu_ppo/checkpoints/checkpoint-00000001 \
  --max-steps 1
```

The command restores policy weights, frozen reference weights, value-head weights, optimizer and
algorithm state, random-number-generator state, and `global_step`. The example advances from step
1 to step 2. Use `--max-steps 0` to validate restoration without rollout or optimization:

```bash
gepa rl resume --config run.cpu.ppo.yaml \
  --checkpoint runs/rl_cpu_ppo/checkpoints/checkpoint-00000001 --max-steps 0
```

The checkpoint must match the canonical configuration and dataset hash. Select a checkpoint
explicitly; the CLI does not guess which checkpoint to resume.

## Collect and evaluate

Collection generates trajectories without scoring or updating weights. Evaluation generates,
scores, and evaluates trajectories without an optimizer step.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gepa rl collect --config run.cpu.ppo.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gepa rl evaluate --config run.cpu.ppo.yaml
```

The canonical evaluation command requires the strict JSONL pair dataset. Collection may also use
`dataset.format: text`; each non-empty line then becomes one prompt.

## Inspect checkpoints and logs

For `save_steps: 1`, the first optimizer step creates:

```text
runs/rl_cpu_ppo/checkpoints/
└── checkpoint-00000001/
    ├── backend.pt
    ├── manifest.json
    └── training_state.pt
```

`backend.pt` contains backend-owned model, value-head, optimizer, and RNG state.
`training_state.pt` contains engine and algorithm state. `manifest.json` binds both artifacts to
their hashes, dataset hash, configuration hash, parent checkpoint, and global step.

Each invocation creates a new log directory:

```text
runs/rl_cpu_ppo/logs/rl-<run-id>/
├── metrics.jsonl
├── run_manifest.json
└── trajectories.jsonl
```

The run manifest records configuration lineage and capability evidence. Trajectory and metric
records include the global step and policy version needed to audit an update.

## Full-weight and LoRA support

The default `gepa rl` engine performs full-weight policy training and keeps a separate frozen
reference model. `TorchPolicyBackend` and `create_portable_backend` also support PEFT LoRA adapter
models through `training_mode="lora"`. The canonical runtime configuration does not yet expose a
LoRA field, so the CLI cannot select LoRA. Integrators can load local assets explicitly, pass a
mapping of `LoraConfig` keyword values to the factory, and inject that backend into
`RLTrainingEngine`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from gepa_mindfulness.training.backends import create_portable_backend
from gepa_mindfulness.training.engine import RLTrainingEngine
from gepa_mindfulness.training.runtime_config import load_rl_config

config = load_rl_config("run.cpu.ppo.yaml")
model_name = config.policy.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
policy_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
backend = create_portable_backend(
    config,
    policy_model=policy_model,
    tokenizer=tokenizer,
    training_mode="lora",
    lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
)
engine = RLTrainingEngine(config, backend_factory=lambda _: backend)
result = engine.train(max_steps=1)
```

Pass a mapping to `lora_config`, not an instantiated PEFT `LoraConfig`. The target-module names
must exist in the selected local model. The factory also accepts no injected assets; in that mode,
it loads the tokenizer and model from the local directory or cache with `local_files_only=True`.

The portable runtime is single-process and local. It does not provide distributed training,
automatic model downloads, vLLM learning, TRL trainers, dataset streaming, or automatic device
placement. CUDA is accepted only when the installed PyTorch build reports an available device.

## Verify the offline acceptance path

The acceptance test constructs its causal model and tokenizer in memory. It performs real
backpropagation, optimizer updates, checkpoint serialization, reload, and resume for PPO and GRPO.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests/test_rl_engine_cpu.py
```

The command passes without a model cache or network connection.
