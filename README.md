# GEPA Mindfulness Superalignment

This repository implements a full training pipeline that integrates GEPA-based
interpretability, paraconsistent imperative logic, and traceable honesty via
the Circuit Tracer thought logging system. The project targets Python 3.10+ and
depends on `torch`, `transformers`, `trl`, `pydantic`, `jinja2`, `pyyaml`, and
`requests`. Optional Circuit Tracer support is provided by the
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) project.
Public packages for that integration are not currently available, so the
feature remains disabled unless you have been granted access to the upstream
distribution.

## Repository Structure

- `gepa_mindfulness/core` – GEPA contemplative principles, paraconsistent
  imperative modeling, abstention, reward shaping, adversarial probes, and
  Circuit Tracer integration.
- `gepa_mindfulness/training` – Configuration models, PPO training orchestrator,
  and CLI tooling.
- `gepa_mindfulness/adapters` – Interfaces for policy backends and Circuit
  Tracer-to-checkpoint conversion.
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
2. **Circuit Tracing** – Wraps the
   [Circuit Tracer](https://github.com/safety-research/circuit-tracer)
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
9. **Unsloth/PEFT Fine-Tuning** – Ready-to-run notebooks for Phi-3 Mini and
   Llama-3 8B that wire GEPA abstention, PPO reward blending, and offline
   trace/report generation into the LoRA training workflow.

## Cloning the repository

If you have Git access, the easiest way to obtain the project with the complete
file structure is to clone it directly from the public remote. Replace the
placeholder owner with the actual namespace if it differs:

```bash
git clone https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment.git
cd GEPA-Mindfulness-superalignment
```

After cloning you should see folders such as `gepa_mindfulness/`,
`examples/`, `notebooks/`, and `scripts/` when running `ls`. If you receive the
project as a ZIP archive instead, unzip it and then optionally follow the
[Working from a ZIP download](#working-from-a-zip-download) steps below to add
Git metadata locally.

> **Line ending note:** The repository includes a `.gitattributes` file that
> normalizes checked-in source files to use LF newlines so tools such as Black
> behave the same on Linux, macOS, and Windows. If you have globally enabled
> automatic CRLF conversion, you can keep it; Git will translate to LF on
> commit. Developers configuring Git for the first time may prefer:
>
> ```bash
> git config --global core.autocrlf input  # recommended on Unix-likes
> # or
> git config --global core.autocrlf true   # recommended on Windows
> ```
>
> Those settings match the defaults used by many Python projects and prevent
> spurious formatting churn in CI.

## Working from a ZIP download

Some users obtain the project as a plain folder (for example, by downloading a
ZIP file from a teammate or a file share). You can still track local changes
with Git even if the original `.git/` metadata is missing:

```bash
cd /path/to/GEPA-Mindfulness-superalignment
git init
git add .
git commit -m "Initial snapshot"
# Optional: connect to the public upstream
git remote add origin https://github.com/<owner>/GEPA-Mindfulness-superalignment.git
git fetch origin
```

From there you can create branches, commit changes, and (if you configured a
remote) push your work upstream. If you later clone the official repository
with `git clone`, the Git metadata will already be included and these steps are
not necessary.

## Getting Started

Follow the steps below from the repository root (`pwd` prints
`.../GEPA-Mindfulness-superalignment`). Each step builds on the previous one so
the CLI and DSPy helpers work out of the box.

### 1. Create and activate a virtual environment

```bash
sudo apt update
sudo apt install python3.12-venv  # once per machine; pick the venv package matching your Python minor version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The `EXTERNALLY-MANAGED` guard in recent Debian / WSL images blocks direct
`pip` usage from the system interpreter. Activating the virtual environment
ensures subsequent installs remain isolated. Use `deactivate` when you are done
working to exit the environment.

### 2. Install the core dependencies

```bash
pip install torch transformers trl pydantic jinja2 pyyaml requests
```

If you have private access to the optional Circuit Tracer wheel, install it now
with the URL provided by the maintainer. The remainder of the project—including
the DSPy helpers—runs without Circuit Tracer if you skip this step.

### 3. Install project extras (CLI + DSPy)

```bash
pip install -e .[dspy]
```

The editable install registers the `gepa` console script and pulls the optional
`dspy-ai` dependency required for `gepa dspy compile`. If you prefer not to
install DSPy support, use `pip install -e .` instead and skip the compilation
features.

### 4. Add notebook tooling (optional but recommended)

```bash
pip install notebook
```

Once installed, the `jupyter notebook ...` commands below will be available in
the activated environment.

### 5. Run the CPU example

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

A convenience wrapper is also available via `python examples/run_cpu_demo.py`.
If you see a “file not found” error, confirm that you are inside the project
directory and that the `gepa_mindfulness/examples/cpu_demo/` folder exists.

### 6. Execute the full pipeline script

```bash
./scripts/run_full_pipeline.sh
```

### 7. Launch the vLLM integration (optional)

Ensure a vLLM server is running and adjust
`gepa_mindfulness/configs/vllm.yaml` as needed before executing:

```bash
python gepa_mindfulness/examples/vllm_demo/run_vllm_demo.py
```

## Deception Detection: Monitoring Only, Reward Honesty

Classical reinforcement learning wisdom warns against **penalising deception**:
doing so simply trains better liars. GEPA therefore emphasises *positive
reinforcement for transparency* while treating deception detection as a
monitoring signal for offline circuit surgery.

### Reward Structure

* `reward_weights.gamma` (honesty) carries the highest emphasis in
  `configs/training/phi3_dual_path.yml` and
  `configs/training/phi3_dual_path_corrected.yml`.
* `_compute_reward` combines task success, GEPA alignment, and honesty bonuses
  while **ignoring deception penalties**. Detected deception only triggers
  logging and fingerprint storage.
* `_honesty_reward` incentivises calibrated confidence, explicit uncertainty
  markers, and “I don’t know” statements when confidence is low.

### Monitoring Workflow

1. **Train with honesty rewards** – run the standard PPO/dual-path training
   loop. Any detected deception is logged to
   `runs/deception_fingerprints/fingerprints.jsonl` without affecting rewards.
2. **Analyze fingerprints** – use `scripts/analyze_deception_fingerprints.py`
   to summarise deception hotspots and propose circuit ablation targets.
3. **Manual ablation** – review the suggested targets, then run
   `scripts/ablate_deception_circuits.py` offline to suppress suspicious
   circuits. This step is intentionally manual.
4. **Validate** – compare original vs ablated checkpoints with
   `scripts/validate_ablation.py` to confirm deception dropped without hurting
   task accuracy.
5. **Monitor in real time** – `scripts/deception_dashboard.py` provides a
   console dashboard that surfaces deception rates, domain breakdowns, circuit
   activations, and alerts.

This approach rewards honesty, preserves transparency, and shifts punitive
action to a deliberate, offline circuit-ablation review instead of the training
loop itself.

## Dual-Path Circuit Tracing for Deception Detection

GEPA now supports **single-prompt dual-path reasoning** to expose deception at
the circuit level.

### How It Works

1. **One Prompt, Two Paths**: A single instruction asks the model to explore both
   cautious and confident approaches in one forward pass.
2. **Circuit Capture**: When available, Anthropic-style circuit tracing records
   neuron activations for each path so we can compare activations directly.
3. **Divergence Detection**: We examine which circuits fire for each path and
   look for confidence inversions, risk suppression, and reward seeking.
4. **Deception Signals**: If the careful path highlights uncertainty but the
   confident path fires reward circuits—and the model recommends the confident
   path—we flag deception.

### Usage

```bash
# Run dual-path evaluation
gepa dspy contrastive-run \
  --data datasets/dual_path/data.jsonl \
  --out runs/contrastive/ \
  --context safety_critical

# View circuit activations
gepa view dual-path safety_001 --base runs/contrastive/
```

### Dataset

See `datasets/dual_path/` for domain-diverse prompts (medical, financial,
safety, technical) that never mention deception explicitly. The model believes
it's performing thorough reasoning; we inspect which circuits fire.

## Running the CLI Locally

Follow the steps below from the repository root (`pwd` prints
`.../GEPA-Mindfulness-superalignment`). Each step builds on the previous one so
the CLI and DSPy helpers work out of the box.

### 1. Create and activate a virtual environment

```bash
sudo apt update
sudo apt install python3.12-venv  # once per machine; pick the venv package matching your Python minor version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The `EXTERNALLY-MANAGED` guard in recent Debian / WSL images blocks direct
`pip` usage from the system interpreter. Activating the virtual environment
ensures subsequent installs remain isolated. Use `deactivate` when you are done
working to exit the environment.

### 2. Install the core dependencies

```bash
pip install torch transformers trl pydantic jinja2 pyyaml requests
```

If you have private access to the optional Circuit Tracer wheel, install it now
with the URL provided by the maintainer. The remainder of the project—including
the DSPy helpers—runs without Circuit Tracer if you skip this step.

### 3. Install project extras (CLI + DSPy)

```bash
pip install -e .[dspy]
```

The editable install registers the `gepa` console script and pulls the optional
`dspy-ai` dependency required for `gepa dspy compile`. If you prefer not to
install DSPy support, use `pip install -e .` instead and skip the compilation
features.

### 4. Add notebook tooling (optional but recommended)

```bash
pip install notebook
```

Once installed, the `jupyter notebook ...` commands below will be available in
the activated environment.

### 5. Run the CPU example

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

A convenience wrapper is also available via `python examples/run_cpu_demo.py`.
If you see a “file not found” error, confirm that you are inside the project
directory and that the `gepa_mindfulness/examples/cpu_demo/` folder exists.

### 6. Execute the full pipeline script

```bash
./scripts/run_full_pipeline.sh
```

### 7. Launch the vLLM integration (optional)

Ensure a vLLM server is running and adjust
`gepa_mindfulness/configs/vllm.yaml` as needed before executing:

```bash
python gepa_mindfulness/examples/vllm_demo/run_vllm_demo.py
```

### Working from a ZIP download

```bash
# Option 1: install the CLI (adds `gepa` to ~/.local/bin when the venv is active)
pip install -e .[dspy]  # enables `gepa dspy compile`

# Option 2: install without DSPy extras
pip install -e .

# Option 3: call the module without installing
PYTHONPATH=src python -m mindful_trace_gepa <subcommand> [...]
```

If your host Python is PEP 668 "externally managed", make sure you perform the
editable install from the virtual environment's interpreter (e.g.
`.venv/bin/python -m pip install -e .[dspy]`). The wrapper script in the
repository root also works without installation when the `PYTHONPATH` is set:

```bash
PYTHONPATH=src ./gepa <subcommand>
```

Both approaches ensure the `mindful_trace_gepa` package resolves even when the
project has not been installed into the system Python.

## Running the CLI Locally

Most examples below assume the `gepa` command is available on your `PATH`.
When developing from a fresh clone you can either install the package in
editable mode or invoke the module directly without installation:

```bash
# Option 1: install the CLI (adds `gepa` to ~/.local/bin when the venv is active)
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Option 2: call the module without installing
python -m mindful_trace_gepa <subcommand> [...]
```

If you prefer not to activate the virtual environment every time, you can also
run `./gepa <subcommand>` from the repository root — the wrapper forwards to the
same CLI entry point.

## DSPy Declarative Pipelines

DSPy-style modules live under `src/mindful_trace_gepa/dspy_modules`. They are
disabled by policy until explicitly enabled in `policies/dspy.yml`. To execute
the pipeline with GEPA checkpoint logging:

```bash
gepa dspy run --input examples/self_tracing_sample.jsonl --trace runs/trace.jsonl

``gepa`` resolves relative paths against the project tree automatically, so the
example works even if you run it from ``src/mindful_trace_gepa/dspy_modules`` or
another sub-directory.
```

To export the guarded prompt manifest:

```bash
gepa dspy compile --out dspy_artifacts/ --enable-optim
```

The compilation command requires the optional `dspy-ai` dependency from the
`pip install -e .[dspy]` step above. If the package is missing the CLI will
explain how to install it.

## Trace Viewer

The offline viewer bundles traces, tokens, and deception metadata into a single
HTML file that can be opened locally without external dependencies:

```bash
gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl \
         --out report_view.html --page-size 200 --max-points 5000
```

The viewer assets are located in `src/mindful_trace_gepa/viewer/` and are served
without external CDNs.

## Deception Research Integration

Mindful Trace GEPA now ships white-box and dataset-level deception tooling:

- [Deception research guide](docs/deception_research.md) summarising probes, datasets, and safety notes.
- [Linear probe configuration](configs/deception/probes_linear.yaml) for CLI runs.
- [ACL 2025 evaluation notebook](notebooks/eval_deception_acl2025.ipynb) covering text-only baselines.

Execute the probe pipeline via:

```bash
gepa deception probes --trace runs/trace.jsonl --model dummy --probe artifacts/dummy.pt   --config configs/deception/probes_linear.yaml --out runs/deception_probe.json
```

Combine probe and multimodal evaluation into a single report:

```bash
gepa deception summary --out runs/deception_summary.json
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

Run these commands from the repository root (the directory that contains this
`README.md` file); Jupyter resolves notebook paths relative to the current
working directory. If you launch the server from another folder (for example,
`/mnt/c/Users/<name>` on WSL) the notebook path will not be found. Use `pwd` to
confirm you are inside the cloned repository before invoking `jupyter`.

To launch the notebooks or PPO trainer across multiple GPUs:

```bash
accelerate launch notebooks/ft_phi3_mini_unsloth_gepa.ipynb
accelerate launch notebooks/ft_llama3_8b_unsloth_gepa.ipynb
deepspeed --num_gpus=8 trainer/ppo_gepa.py --config configs/ppo/ppo_default.yaml
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
