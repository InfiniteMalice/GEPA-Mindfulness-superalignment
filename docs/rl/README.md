# Portable PyTorch RL

The `gepa rl` command is the sole canonical path that updates model weights. It runs PPO or GRPO
with the repository's strict reward-integrity pair format, local checkpoints, and structured JSONL
logs. The default backend
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

## Configure a single NVIDIA GPU

**Hardware verification status: not run on this CPU-only host.** The commands in this section are
operator procedures, not evidence that this checkout completed a CUDA optimizer update. Run the
marked acceptance lane at the end of this section on the target GPU before assigning a production
label to that machine and PyTorch build.

### Select and verify the CUDA-enabled PyTorch wheel

Use the official [PyTorch local-install selector](https://pytorch.org/get-started/locally/) to
choose the command for the target operating system and NVIDIA driver. The ordinary project
dependencies intentionally contain no CUDA wheel URL or `+cu` package pin. Install the selected
PyTorch wheel first; then install this repository. For example, the official PyTorch 2.9.1 CUDA
12.8 index uses these commands:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[rl]'
```

Do not use the example CUDA 12.8 index when the target driver requires a different selector
choice. After installation, run this model-free check:

```bash
python - <<'PY'
import torch

print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("PyTorch cannot use cuda:0")
print(torch.cuda.get_device_name(0))
PY
```

Exit status `0`, a non-null CUDA runtime, and the selected GPU name prove only that PyTorch can see
`cuda:0`. They do not prove a policy update or checkpoint round trip.

### Prepare a local model and single-GPU configuration

From the repository root, copy the strict template and replace its placeholder with an absolute
local Transformers model directory. This command also changes `checkpoint.save_steps` to `1`, so
the one-step training command below produces `checkpoint-00000001`.

```bash
cp configs/rl/cuda_single_gpu.yaml run.cuda.ppo.yaml
export MODEL_DIR=/absolute/path/to/local-transformers-model
python - <<'PY'
import os
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"]).expanduser().resolve(strict=True)
if not model_dir.is_absolute() or not (model_dir / "config.json").is_file():
    raise SystemExit("MODEL_DIR must be absolute and contain config.json")
if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
    raise SystemExit("MODEL_DIR must contain Safetensors or PyTorch model weights")
AutoConfig.from_pretrained(model_dir, local_files_only=True)
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
path = Path("run.cuda.ppo.yaml")
config = yaml.safe_load(path.read_text(encoding="utf-8"))
config["policy"]["model_name"] = str(model_dir)
config["checkpoint"]["save_steps"] = 1
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(path.resolve())
PY
```

Keep `runtime.device: cuda:0`. Choose `runtime.precision: fp32` first. Use `fp16` or `bf16` only
after the doctor and marked acceptance lane report support on the selected GPU.

### Check, train, and resume

Keep model loading offline. The doctor exits before model construction when CUDA, the device
index, or the selected precision is unsupported.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
gepa rl doctor --config run.cuda.ppo.yaml
gepa rl train --config run.cuda.ppo.yaml --max-steps 1
```

Exit status `0` from the doctor means its required capability report is fully supported. A
successful training result must report `"global_step": 1`,
`"policy_parameters_updated": true`, different policy-only before/after checksums, and
`checkpoint-00000001`. Resume from that explicit checkpoint without another rollout:

```bash
gepa rl resume --config run.cuda.ppo.yaml \
  --checkpoint runs/rl_cuda_single_gpu/checkpoint-00000001 \
  --max-steps 0
```

The resume result must report `"global_step": 1`, zero new trajectories, identical restored
policy checksums, and `"policy_parameters_updated": false`.

### Diagnose CUDA out-of-memory failures

When a CUDA allocation fails, `CudaOutOfMemoryError` captures same-process allocator statistics
before the failing process exits. Record the complete error. The error includes the operation,
device, precision, batch size, maximum new tokens, accumulation count, `allocator_stats=`, and
`allocator_summary=`. If a CUDA diagnostic query also fails, the same error includes
`allocator_diagnostic_errors=` and preserves the original OOM as the cause.

Use `nvidia-smi` separately to identify memory held by other GPU processes:

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
```

Stop an unrelated process only when you own it. Otherwise, select a smaller local model, reduce
`policy.max_new_tokens`, or reduce `algorithm.batch_size` when it is greater than `1`. Increasing
`algorithm.gradient_accumulation_steps` can preserve an effective batch after reducing the
per-step batch. The runtime reports the OOM and exits; it does not modify the configuration or
retry with weaker settings automatically. Re-run the doctor and the one-step command after every
configuration change.

### Run the hardware acceptance lane

```bash
python -m pytest --strict-markers -m cuda tests/test_rl_cuda.py -q -rs
```

The lane tests FP32 and probes FP16 and BF16 separately. An unsupported mixed precision case skips
with its device-specific reason. Each mixed-precision case requires the standalone CUDA probe and
the shared engine's model-logit and value-head outputs to use the requested dtype. On a supported
precision, the test also requires a policy-only parameter change, frozen reference state, CUDA
residency, canonical checkpoint artifacts, exact fresh-backend restoration, and restored global
step. Treat a skipped precision as unsupported evidence, not a pass.

### Launch the canonical two-GPU DDP configuration

**Distributed hardware verification status: not run on this CPU-only host.** Run the marked CUDA
acceptance lane on the target host before treating this command as hardware-qualified evidence.

Copy the shipped Distributed Data Parallel (DDP) template and replace its local-model placeholder:

```bash
cp configs/rl/cuda_ddp.yaml run.cuda.ddp.yaml
export MODEL_DIR=/absolute/path/to/local-transformers-model
python - <<'PY'
import os
from pathlib import Path

import yaml

model_dir = Path(os.environ["MODEL_DIR"]).expanduser().resolve(strict=True)
path = Path("run.cuda.ddp.yaml")
config = yaml.safe_load(path.read_text(encoding="utf-8"))
config["policy"]["model_name"] = str(model_dir)
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
```

From the repository root, launch exactly two local workers with the canonical CLI module:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --standalone --nnodes=1 --nproc-per-node=2 -m mindful_trace_gepa rl train --config run.cuda.ddp.yaml --max-steps 1
```

`torchrun` supplies `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. Each worker
resolves the strict topology in `run.cuda.ddp.yaml`, activates its configured local CUDA device,
and joins the NCCL process group through `env://`. When no process group exists, the canonical
engine initializes the group and destroys the owned group after success or failure. When an
embedding process initializes the group before engine entry, the engine validates that external
group and does not destroy it.

### Collect with the experimental llama.cpp Vulkan actor

**Native integration status: not run on this host.** The automated contract uses a local mock
server. Run the opt-in native test and the commands below on the target host before treating the
lane as native-runtime evidence. This backend is experimental and supports inference and
trajectory collection only. It does not support training, resume, evaluation, backward passes,
optimizer steps, value heads, or full-weight updates.

Build llama.cpp with Vulkan enabled, then inspect the executable and Vulkan loader without starting
a service:

```bash
cmake -B build -DGGML_VULKAN=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j
./build/bin/llama-server --version
vulkaninfo --summary
```

Verify that `llama-server --version` identifies the expected local build. Verify that
`vulkaninfo --summary` exits with status `0` and identifies the intended Vulkan device. A reachable
server proves only local llama.cpp inference; server reachability does not prove Vulkan execution.

Start one loopback-only server in a separate terminal. The alias must match `policy.model_name` in
`configs/rl/llama_cpp_vulkan_collect.yaml`:

```bash
export GGUF_MODEL=/absolute/path/to/model.gguf
./build/bin/llama-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model "$GGUF_MODEL" \
  --alias LOCAL_GGUF_MODEL_ID
```

In the repository terminal, inspect the executable, endpoint, model metadata, and Vulkan evidence:

```bash
gepa rl doctor \
  --backend llama-cpp-vulkan \
  --endpoint http://127.0.0.1:8080
```

The doctor reports each capability with its evidence. The doctor exits with status `2` because the
inference-only backend intentionally reports training capabilities as unsupported. Treat
`supports_vulkan: UNKNOWN` as missing Vulkan proof even when `supports_generation` is available.

Collect trajectories through the common engine and JSONL logger:

```bash
gepa rl collect \
  --config configs/rl/llama_cpp_vulkan_collect.yaml \
  --backend llama-cpp-vulkan \
  --endpoint http://127.0.0.1:8080 \
  --dataset data/synthetic/reward_integrity/rl_pairs_v1.jsonl \
  --output runs/llama_cpp_vulkan/logs
```

The command prints one JSON result. Read its `log_directory`, then verify the common manifest and
trajectory schema:

```bash
export RUN_DIR=/absolute/path/from-the-log_directory-field
python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"]).resolve(strict=True)
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
record = json.loads((run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0])
trajectory = record["trajectory"]
assert manifest["backend"] == "llama_cpp_vulkan"
assert record["actor_backend"] == "llama_cpp_vulkan"
for field in (
    "prompt_token_ids",
    "response_token_ids",
    "old_log_probs",
    "reference_log_probs",
    "value_predictions",
    "reward_total",
    "advantage",
    "return",
):
    assert trajectory[field] is None
print(manifest["device_capabilities"]["capabilities"]["supports_vulkan"])
PY
```

The nullable fields remain JSON `null` unless llama-server returns validated token or probability
evidence. The run manifest records the collection command's independently detected preflight
Vulkan state. Collection does not copy a prior doctor report or promote an unknown Vulkan state to
supported.

### Train with the experimental hybrid Mojo/Vulkan actor

This path is experimental hybrid training, not production readiness and not pure Mojo training.
The external coordinator performs generation only. A local PyTorch LoRA learner performs GRPO
evaluation, backward, optimizer step, checkpointing, and learner-native adapter export.

Copy `configs/rl/hybrid_vulkan_grpo.yaml`, replace the local Transformers model path, and bootstrap
`hybrid.adapter_store` with a verified current adapter through `LocalAdapterPublisher`. Supply the
coordinator executable and any actor endpoint at invocation time; do not add endpoints or secrets
to the configuration:

```bash
gepa rl train \
  --config run.hybrid.yaml \
  --backend mojo-vulkan-llamacpp \
  --learner pytorch \
  --coordinator-command /absolute/path/to/rl-coordinator \
  --actor-endpoint http://127.0.0.1:8080 \
  --max-steps 1
```

The command executes the supplied coordinator argv directly without a shell. `--learner mojo`, a
missing learner selection, a missing current manifest, or a learner without adapter-only export
evidence fails before actor startup. The actor request is bound to the current manifest's model,
adapter identifier, SHA-256, and canonical policy version. Default staleness policy rejects lagged
actor trajectories before reward scoring or learner evaluation. `down_weight` is an explicit
alternative and applies its logged weight once to observable rewards.

After each successful PyTorch optimizer step, the engine writes a checkpoint, exports only the
learner's LoRA trainables, and atomically publishes exactly the next policy version. The result and
logs prove publication; they do not claim the running actor loaded that adapter. The artifact is a
learner-native PyTorch LoRA state dict, not GGUF. GGUF conversion, llama.cpp deployment, and actor
reload are external operator steps and receive no success claim from this command.

This CPU-only host validates the mocked coordinator and tiny local PyTorch update. It does not
provide native Mojo, Vulkan-device, llama.cpp conversion, or post-publication actor-load evidence.

### Dependency version policy

The `dev` and `rl-dev` extras require `pytest>=8.0,<10`. Pytest 8 and 9 are the declared supported
test-runner majors. Pytest 10 remains excluded until the repository test suite qualifies that major.
The runtime-oriented `all` extra does not install pytest or other development tools.

PyTorch, Transformers, and PEFT retain upper bounds because the canonical engine consumes their
version-sensitive model, tensor, and adapter interfaces. Dependencies with only a lower bound
remain lower-bound-only because this repository has no evidence of an incompatible later release.
Maintainers should add an upper bound when a reproducible compatibility failure identifies the
first incompatible version, rather than adding a speculative cap.

## Prepare an offline run

Create a JSONL dataset with one complete authored pair per line. Training and evaluation reject
plain-text prompts because the reward pipeline needs the chosen/rejected evidence and component
labels. The schema is closed; unknown or missing fields fail before model construction.

```json
{"record_id":"example-1:grounded_over_proxy","source_case_id":"example-1","source_case_version":"1.0","source_path":"authored/example.jsonl","source_line":1,"source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pair_rule":"grounded_over_proxy","prompt":"Answer with calibrated care.","chosen":"A grounded answer.","rejected":"An unsupported answer.","chosen_class":"grounded_success","rejected_class":"proxy_exploitation","chosen_reward_components":{"objective_fidelity":0.5,"feedback_integrity":0.5,"skill_transfer":0.5,"reality_contact":0.5,"exploit_disclosure":0.5,"long_horizon_agency":0.5,"benign_creativity":0.5,"repair_quality":0.5},"rejected_reward_components":{"objective_fidelity":-0.5,"feedback_integrity":-0.5,"skill_transfer":-0.5,"reality_contact":-0.5,"exploit_disclosure":-0.5,"long_horizon_agency":-0.5,"benign_creativity":-0.5,"repair_quality":-0.5},"diagnostics":{"central":"authored","supporting":[]},"schema_version":"reward-integrity-rl-pairs-v1"}
```

Save the record as `data/rl/pairs.jsonl`. Then save this PPO configuration as
`run.cpu.ppo.yaml`. Replace `/absolute/path/to/local-transformers-model` with the absolute path to
a local Transformers model directory.

```yaml
runtime:
  backend: pytorch
  device: cpu
policy:
  model_name: /absolute/path/to/local-transformers-model
  max_new_tokens: 32
algorithm:
  name: ppo
  learning_rate: 1.0e-5
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 10
  clip_range: 0.2
  value_coef: 0.1
  max_grad_norm: 1.0
reward:
  weights:
    alpha: 0.3
    beta: 0.3
    gamma: 0.2
    delta: 0.2
  overlay_weight: 0.0
  integrity_overlay_enabled: false
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

The shipped `configs/rl/pytorch_cpu_ppo.yaml` and `configs/rl/pytorch_cpu_grpo.yaml` presets already
select the bundled `data/synthetic/reward_integrity/rl_pairs_v1.jsonl` strict pair dataset. Both
presets intentionally retain the local-model path template. Run a preset from the repository root
so its dataset path resolves, or replace `dataset.train_path` with an absolute path. Do not run
either preset with the unchanged model-path template.

Before model construction, set and validate the replacement directory:

```bash
export MODEL_DIR=/absolute/path/to/local-transformers-model
python - <<'PY'
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"]).expanduser()
if not model_dir.is_absolute():
    raise SystemExit("MODEL_DIR must be an absolute path")
model_dir = model_dir.resolve(strict=True)
if not (model_dir / "config.json").is_file():
    raise SystemExit("MODEL_DIR must contain config.json")
if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
    raise SystemExit("MODEL_DIR must contain Safetensors or PyTorch model weights")
AutoConfig.from_pretrained(model_dir, local_files_only=True)
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
print(model_dir)
PY
```

Exit status `0` and the printed absolute directory confirm that the path, model configuration,
weights, and tokenizer assets are locally readable. Replace `policy.model_name` in the copied YAML
with that printed directory. `gepa rl doctor` checks execution capabilities but does not perform
this model-artifact validation.

For GRPO, copy the file to `run.cpu.grpo.yaml`, change `algorithm.name` to `grpo`, and add these
keys under `policy`:

```yaml
  do_sample: true
  temperature: 0.7
  top_p: 0.9
```

Then add these keys under `algorithm`:

```yaml
  group_size: 4
  kl_coef: 0.05
  group_normalization_epsilon: 1.0e-8
  zero_variance_policy: skip
```

`zero_variance_policy: skip` omits a response group when every response receives the same reward.
Use a dataset and model that can produce reward variation, or a GRPO invocation can finish without
an optimizer step.

The reward-integrity overlay is disabled by default. To enable it, set
`reward.integrity_overlay_enabled: true` and set `reward.overlay_weight` to a finite positive value.
The GEPA `beta` weight continues to control only base GEPA alignment. Logs preserve
`gepa_alignment`, record the overlay as `reward_integrity_aggregate`, and retain all eight authored
integrity components. `pre_clip_gradient_norm` records the trainable gradient norm before clipping;
`algorithm.max_grad_norm: null` disables clipping.

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

The portable CPU runtime is single-process and local. The CUDA runtime additionally supports the
strict DDP and full-state FSDP topology in `RuntimeConfig.distributed`. The runtime does not provide
automatic model downloads, vLLM learning, TRL trainers, dataset streaming, automatic device
placement, or sharded-optimizer checkpoint restore. CUDA is accepted only when the installed
PyTorch build reports an available device.

## Verify the offline acceptance path

The acceptance test constructs its causal model and tokenizer in memory. It performs real
backpropagation, optimizer updates, checkpoint serialization, reload, and resume for PPO and GRPO.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests/test_rl_engine_cpu.py
```

The command passes without a model cache or network connection.
