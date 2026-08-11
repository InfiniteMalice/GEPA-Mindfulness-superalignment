# GEPA Mindfulness Superalignment User Manual

Document version: 1.0  
Date: 2026-08-11  
Software version: 0.1.0  
Source commit: `cde19d7bfbcea4df86cdb6d40be3e9c293b380b5`  
Source repository: <https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment>  
Applicable Python version: 3.10 or later

## 1. About this manual

This manual tells you how to install and use GEPA Mindfulness Superalignment.
It covers the command-line tools, the local web workbench, datasets, reports,
and reinforcement learning (RL).

This manual uses ASD-STE100-style controlled English. It uses short sentences,
active voice, and consistent terms. No independent organization certified this
manual against the proprietary ASD-STE100 dictionary.

The manual describes the source at the commit that is shown above. Later source
changes can change commands, options, defaults, or results.

### 1.1 Intended users

Use this manual if you are one of these users:

- A researcher who evaluates model traces.
- A data author who creates alignment datasets.
- An operator who runs local PPO or GRPO training.
- A reviewer who uses the ontology workbench.
- A developer who integrates the public Python modules.

### 1.2 Scope

This manual covers the supported and experimental user surfaces in the repository.
It does not give deployment approval. It does not certify a model as aligned.
It does not replace a safety policy, a legal requirement, or a system instruction.

## 2. Safety and data notices

Read these notices before you run the software.

> CAUTION: A trace can contain private prompts, model answers, scores, and metadata.
> The offline viewer puts this data in one HTML file. Protect that file as you
> protect the source trace.

> WARNING: The `--response` option in the dual-path evaluator can load a Python
> file. A Python file can execute arbitrary code. Use only a file that you trust.

> CAUTION: RL training can use much memory. A full policy and a frozen reference
> model can exist at the same time. The operating system can stop the process
> before it writes a checkpoint.

> CAUTION: The repository does not download a model for canonical RL. Use a local
> Transformers model or an existing local cache entry. Verify the model path before
> you start training.

> CAUTION: A generated ontology bundle is noncanonical. It does not change the
> canonical ontology. It does not prove that a model learned or deployed a behavior.

> CAUTION: Deception scores are research signals. Synthetic activations, heuristic
> scores, and simulated judge results are not measured evidence from a real model.

## 3. System description

GEPA Mindfulness Superalignment is a research toolkit. It combines these functions:

- Trace generation and storage.
- GEPA score aggregation.
- Offline trace review.
- Deception research utilities.
- Synthetic dataset tools.
- Cognitive Pairwise Training (CPT) data construction.
- Socratic Self-Refine (SSR) evaluation.
- Portable local RL with PPO or GRPO.
- A governed ontology workbench.

GEPA means Genetic-Pareto. In this repository, GEPA also names the alignment
checkpoints and score surfaces in the trace tools.

PPO means Proximal Policy Optimization. GRPO means Group Relative Policy
Optimization. DSPy is the optional declarative pipeline dependency.

### 3.1 Main data flow

```text
Input JSONL
    |
    v
DSPy pipeline or contrastive baseline
    |
    +--> trace JSONL
    +--> token JSONL
    +--> deception JSON
    +--> summary JSON
            |
            +--> score report
            +--> tiered score JSON
            +--> offline HTML viewer
```

RL uses a separate data flow:

```text
Strict chosen/rejected JSONL + local model + YAML configuration
    |
    v
gepa rl doctor
    |
    v
gepa rl train, collect, evaluate, or resume
    |
    +--> JSON result on standard output
    +--> structured JSONL logs
    +--> checkpoints for training runs
```

### 3.2 Maturity and support limits

| Function | Maturity | Operational limit |
| --- | --- | --- |
| Basic trace score and viewer | Supported lightweight tools | Input files must use the expected JSON or JSONL structure. |
| CPU PPO and GRPO through `gepa rl` | Supported | Use a local Transformers model. |
| CUDA and distributed PyTorch | Implemented, but hardware is not qualified by the repository host | Run acceptance tests on the target hardware. |
| DSPy run and compile | Optional | Install the `dspy` extra. Optimization is off by default. |
| CPU demo | Compatibility simulator | It does not prove a Transformers weight update. |
| vLLM demo | Compatibility inference example | It does not use the canonical RL engine. |
| llama.cpp with Vulkan | Experimental actor | It supports collection only. It does not train or resume. |
| Mojo coordinator with PyTorch learner | Experimental hybrid | Use an operator-supplied coordinator and `--learner pytorch`. |
| Pure Mojo learner | Unsupported | `--learner mojo` stops with an error. |
| Tier-1 judge in the current source | Simulated wrapper | It does not call a remote judge service. |
| Ontology workbench | Local governed review surface | The canonical ontology stays read-only. |

## 4. Terms and abbreviations

| Term | Meaning |
| --- | --- |
| Artifact | A file that a command creates. |
| Canonical | The authoritative path, format, or source in this repository. |
| Checkpoint | A saved RL state that contains model, optimizer, and run data. |
| CPT | Cognitive Pairwise Training. |
| CUDA | The NVIDIA graphics processor compute platform. |
| DDP | Distributed Data Parallel. |
| DSPy | An optional declarative language-model pipeline package. |
| GEPA | The repository alignment and trace framework. |
| GGUF | A model file format that llama.cpp uses. |
| GRN | Global Response Normalization. |
| GRPO | Group Relative Policy Optimization. |
| JSON | JavaScript Object Notation. |
| JSONL | JSON Lines. Each non-empty line contains one JSON object. |
| LoRA | Low-Rank Adaptation. |
| Manifest | A file that identifies artifacts, hashes, and lineage. |
| PPO | Proximal Policy Optimization. |
| RL | Reinforcement learning. |
| SSR | Socratic Self-Refine. |
| Trace | An ordered record of pipeline events. |
| YAML | A human-readable configuration format. |

## 5. Installation

### 5.1 Prerequisites

Install these items:

- Git.
- Python 3.10 or later.
- `pip` for the selected Python installation.
- Node.js 22.13.0 or later, only for the ontology workbench.

Use a source checkout for repository scripts, presets, notebooks, and the
experimental hybrid path. A wheel does not contain all repository assets.

### 5.2 Get the source

Run these commands:

```bash
git clone https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment.git
cd GEPA-Mindfulness-superalignment
```

### 5.3 Create a virtual environment

On Linux or macOS, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is disabled, use the environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

### 5.4 Select an installation

Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

Then install one package set.

| Requirement | Command |
| --- | --- |
| Basic CLI, score, viewer, and contrastive baseline | `python -m pip install -e .` |
| DSPy pipeline and compiler | `python -m pip install -e '.[dspy]'` |
| Canonical RL runtime | `python -m pip install -e '.[rl]'` |
| RL runtime and development checks | `python -m pip install -e '.[rl-dev]'` |
| Interpretability plots and NetworkX | `python -m pip install -e '.[interpret]'` |
| PDF support | `python -m pip install -e '.[pdf]'` |
| vLLM integration | `python -m pip install -e '.[vllm]'` |
| All optional runtime packages | `python -m pip install -e '.[all]'` |

The `all` extra does not install the test tools. Use `dev` or `rl-dev` for tests.

### 5.5 Verify the basic installation

Run:

```bash
gepa --help
```

If the shell cannot find `gepa`, run:

```bash
python -m mindful_trace_gepa --help
```

The command must show these command groups:

```text
dspy, cpt-build, ssr-run, view, score, deception,
score-auto, judge, clf, rl
```

### 5.6 Install from a wheel

Build and install a wheel when you do not need repository-only assets:

```bash
python -m pip install build
python -m build
python -m pip install dist/gepa_mindfulness-*.whl
gepa --help
```

The hybrid YAML file and Mojo source are not in the wheel.

## 6. Quick start with no model

This procedure gives a result without a local model and without DSPy.

### 6.1 Run the contrastive baseline

From the repository root, run:

```bash
mkdir -p runs/contrastive
gepa dspy contrastive-run \
  --data datasets/dual_path/data.jsonl \
  --out runs/contrastive \
  --context general
```

On Windows PowerShell, run:

```powershell
New-Item -ItemType Directory -Force runs\contrastive | Out-Null
gepa dspy contrastive-run `
  --data datasets\dual_path\data.jsonl `
  --out runs\contrastive `
  --context general
```

The command creates these principal artifacts:

- `runs/contrastive/summary.json`
- `runs/contrastive/fingerprints.jsonl`
- One response text file for each record.
- One deception JSON file for each record.

Open `summary.json`. Verify that `counts.dataset_records` is greater than zero.

### 6.2 Score the supplied sample trace

Create the output directory. Then run the score command:

```bash
mkdir -p runs/quickstart
gepa score \
  --trace examples/sample_trace.jsonl \
  --out runs/quickstart/score.html
```

Open `runs/quickstart/score.html` in a browser. Verify that the report shows
three events.

### 6.3 Build an offline viewer

The viewer accepts an empty or missing token file. Token charts are empty in
that condition.

```bash
gepa view \
  --trace examples/sample_trace.jsonl \
  --tokens runs/quickstart/tokens.jsonl \
  --out runs/quickstart/viewer.html
```

Open `runs/quickstart/viewer.html`. Verify that the timeline shows the sample
trace stages.

## 7. Generate a DSPy trace

Install the `dspy` extra before you use this procedure.

### 7.1 Prepare the input

Create a JSONL file. Each row must contain `inquiry` or `prompt`.
The `context` field is optional.

Example:

```json
{"inquiry":"How can a team preserve review quality during a fast release?","context":"engineering"}
```

### 7.2 Run the pipeline

Run:

```bash
mkdir -p runs/dspy
gepa dspy run \
  --input examples/self_tracing_sample.jsonl \
  --trace runs/dspy/trace.jsonl
```

The command reads `policies/dspy.yml` by default. The policy disables
optimization by default.

### 7.3 Inspect the output

The command writes these files in the trace directory:

| File | Content |
| --- | --- |
| `trace.jsonl` | GEPA checkpoint events. |
| `tokens.jsonl` | Token records and confidence values. |
| `summary.json` | Model label, policy version, thresholds, and shard location. |
| `deception.json` | Deception results for each input row. |
| `manifest.json` | Shard metadata, only when sharding starts. |

The token data is synthetic unless a wrapper supplies measured token data.
Do not label synthetic confidence as measured model confidence.

### 7.4 Enable optional trace fields

Use these options only when you need their data:

```bash
gepa dspy run \
  --input input.jsonl \
  --trace runs/trace.jsonl \
  --dual-path \
  --enable-value-decomp \
  --enable-dvgr \
  --use-grn-value-decomp \
  --long-context
```

`--dual-path` selects the dual-path chain. It requires the optional DSPy
dependency. `--enable-value-decomp` adds a value-decomposition event.
`--enable-dvgr` adds the deep-value generalization metric. `--use-grn-value-decomp`
applies GRN to value projection.

### 7.5 Control token records and shards

The default values are:

- Log probabilities: on.
- Top alternatives per token: 3.
- Token sample interval: 16.
- Events per shard: 10,000.

Use `--no-with-logprobs` to disable log-probability records. Use
`--shard-threshold 1` for a small sharding test.

### 7.6 Compile guarded prompts

Run:

```bash
gepa dspy compile \
  --out dspy_artifacts \
  --config configs/policies/dspy.yml
```

Add `--dataset DATASET.jsonl` to supply optimization examples. Add
`--enable-optim` to permit prompt augmentation.

> CAUTION: Inspect compiled prompts before use. The configuration requires human
> approval. A compiled prompt is not an approved production policy.

The compiler rejects configured forbidden phrases. It also preserves the GEPA
invariants that the compiler checks.

## 8. Use the offline viewer

### 8.1 Build a standard viewer

Run:

```bash
gepa view \
  --trace runs/dspy/trace.jsonl \
  --tokens runs/dspy/tokens.jsonl \
  --out runs/dspy/viewer.html
```

The result is one self-contained HTML file. You can open it without a server.

### 8.2 Add deception and dual-path data

Run:

```bash
gepa view \
  --trace runs/dspy/trace.jsonl \
  --tokens runs/dspy/tokens.jsonl \
  --deception runs/deception_probe.json \
  --dual-path runs/dspy/deception.json \
  --out runs/dspy/viewer.html
```

If you omit the optional paths, the builder searches the trace directory and
`runs/` for known artifact names.

The hidden `--paired` option is a compatibility alias for `--dual-path`.
Use `--dual-path` in new commands.

The viewer searches for these files:

- `deception_probe.json`
- `deception_summary.json`
- `mm_eval.json`
- `deception.json`
- `scores.json`

### 8.3 Control viewer size

Use `--page-size` to limit the embedded trace events. The default is 200.
Use `--max-points` to limit embedded token events. The default is 5,000.

Example:

```bash
gepa view \
  --trace runs/trace.jsonl \
  --tokens runs/tokens.jsonl \
  --page-size 500 \
  --max-points 10000 \
  --out runs/viewer.html
```

### 8.4 Use a shard manifest

Use `--manifest PATH` when the manifest is not next to the trace.
If you omit the option, the viewer looks for `manifest.json` next to the trace.

## 9. Score traces

The repository has two score commands. They serve different purposes.

### 9.1 Build a simple HTML score report

Use `gepa score` to calculate means for existing `principle_scores` and
`imperative_scores` fields.

```bash
mkdir -p runs/report
gepa score \
  --trace runs/dspy/trace.jsonl \
  --policy policies/default_cw4.yml \
  --out runs/report/score.html
```

The report shows event count, principle means, imperative means, and policy data.
The command does not infer missing principle scores.

Use `--stream` or `--no-stream` to control event streaming. Streaming is on by
default. `--no-stream` loads all events before the calculation.

### 9.2 Score a sharded trace

Run:

```bash
gepa score \
  --trace runs/dspy/trace.jsonl \
  --sharded \
  --manifest runs/dspy/manifest.json \
  --out runs/report/sharded-score.html
```

Use `--zstd` when the trace shards use Zstandard compression. Install the
`zstandard` package when compressed input requires it.

### 9.3 Run tiered wisdom scoring

Use `gepa score-auto` to score mindfulness, compassion, integrity, and prudence.
Tier 0 uses deterministic heuristics.

```bash
gepa score-auto \
  --trace runs/dspy/trace.jsonl \
  --config configs/scoring.yml \
  --out runs/dspy/scores.json
```

Add `--no-print` to suppress the JSON copy on standard output.

Add `--judge` to include the Tier-1 judge. In the applicable source version,
the judge uses a local simulated wrapper. It does not send a request to a remote
model service.

```bash
gepa score-auto \
  --trace runs/dspy/trace.jsonl \
  --config configs/scoring.yml \
  --judge \
  --out runs/dspy/scores.json
```

### 9.4 Train and use the Tier-2 classifier

Prepare a labeled JSONL file. Then run:

```bash
gepa clf train \
  --labels datasets/labels/gold.jsonl \
  --config configs/classifier/default.yml \
  --out artifacts/classifier
```

The command writes classifier artifacts and `metrics.json`.

When you use `--classifier`, always supply the configuration and artifact paths:

```bash
gepa score-auto \
  --trace runs/dspy/trace.jsonl \
  --config configs/scoring.yml \
  --classifier \
  --classifier-config configs/classifier/default.yml \
  --classifier-artifacts artifacts/classifier \
  --out runs/dspy/scores.json
```

The current argument parser does not give a usable path default when the
`--classifier` switch is present. The two explicit path options avoid that issue.

### 9.5 Export low-confidence dimensions

Run:

```bash
gepa clf triage \
  --scores runs/dspy/scores.json \
  --threshold 0.6 \
  --out datasets/labels/triage.jsonl
```

The output contains each dimension below the threshold.

### 9.6 Run the judge command directly

Use `--mock` for a deterministic local response:

```bash
gepa judge \
  --trace runs/dspy/trace.jsonl \
  --out runs/dspy/judge.json \
  --mock
```

The direct judge command writes one Tier-1 JSON artifact.

## 10. Run deception research utilities

### 10.1 Run a linear probe

Prepare these inputs:

- A trace JSONL file.
- A probe weight file.
- A probe YAML configuration.
- A model identifier for provenance.

Then run:

```bash
gepa deception probes \
  --trace runs/trace.jsonl \
  --model local-model-id \
  --probe artifacts/probe.json \
  --config configs/deception/probes_linear.yaml \
  --out runs/deception_probe.json
```

If the trace has no activation fields, the command creates synthetic activations.
The result records `activations_source`. Check this field before you interpret
the score.

The command can read `activations` or `probe_activations` from trace events.
The probe configuration controls layer indices, pooling, thresholds, GRN, and
the default output path.

### 10.2 Merge deception artifacts

Run:

```bash
gepa deception summary \
  --runs runs \
  --out runs/deception_summary.json
```

The command searches for probe, dual-path, and multimodal artifacts.
You can give explicit paths:

```bash
gepa deception summary \
  --probe runs/deception_probe.json \
  --dual-path runs/deception.json \
  --mm runs/mm_eval.json \
  --out runs/deception_summary.json
```

An absent optional artifact does not stop the summary command.

The hidden `--paired` option is a compatibility alias for `--dual-path`.
Use `--dual-path` in new commands.

### 10.3 Run the dual-path command-line workflow

Use the repository workflow when you have a trusted response hook:

```bash
python -m mindful_trace_gepa.dual_path_evaluator \
  --scenarios datasets/dual_path/data.jsonl \
  --run runs/001 \
  --response package.module:callable

python -m mindful_trace_gepa.dual_path_circuit_tracer runs/001
python tools/merge_run_inspection.py runs/001
```

The response hook can also be a `.py` file. Treat that file as executable code.

## 11. Build CPT pairs

CPT uses public structured reasoning records. It does not require private
chain-of-thought.

### 11.1 Prepare candidate records

Each input row must contain these fields:

- `candidate_id`
- `problem_id`
- `prompt`
- `public_reasoning_summary`
- `structured_reasoning_units`
- `final_answer`
- `reference_answer`
- `model_id`
- `model_scale`
- `checkpoint_id`
- `rollout_id`
- `correctness`
- `confidence`
- `abstained`
- `verifier_status`

The optional `metadata` field contains additional data.

### 11.2 Build the pair file

Run:

```bash
gepa cpt-build \
  --input data/cpt/candidates.jsonl \
  --out data/cpt/pairs.jsonl \
  --seed 17
```

Pair order randomization is on by default. Consensus filtering is on by default.
Use `--randomize-pair-order` or `--no-randomize-pair-order` to set the first
function. Use `--consensus-filtering` or `--no-consensus-filtering` to set the
second function.

The result is a pairwise JSONL file. It is a trainer input. The command does not
fine-tune a model.

## 12. Run bounded SSR

SSR evaluates and repairs public reasoning units. It keeps the original trace.
It does not add a new 17-case category.

### 12.1 Prepare reasoning units

Use a JSON array, a JSON object with `reasoning_units`, or a JSONL file.
Each unit uses these fields:

- `unit_id`
- `parent_unit_id`
- `sequence_index`
- `sub_question`
- `sub_answer`
- `evidence_summary`
- `assumptions`
- `uncertainty_markers`
- `confidence`
- `verifier_status`
- `repair_status`
- `dependencies`
- Optional `metadata`

### 12.2 Run SSR

Run:

```bash
gepa ssr-run \
  --input data/ssr/units.jsonl \
  --out runs/ssr/report.json \
  --mode evaluation \
  --max-iterations 2 \
  --run-id ssr-001
```

Use `--initial-answer-reference` to attach an answer identifier.
The result is one `SSRRunReport` JSON file.

## 13. Work with synthetic datasets

### 13.1 Validate a dataset

Run:

```bash
python scripts/synthetic_dataset_tool.py validate \
  data/synthetic/gold/superalignment_gold_v1.jsonl
```

Success gives exit status 0 and a `validation passed` message.

### 13.2 Print a summary

Run:

```bash
python scripts/synthetic_dataset_tool.py summary \
  data/synthetic/gold/superalignment_gold_v1.jsonl
```

Use `--allow-invalid` only when you must inspect an invalid draft. Do not use an
invalid draft for training.

### 13.3 Create a case scaffold

Run:

```bash
python scripts/synthetic_dataset_tool.py scaffold \
  data/synthetic/templates/new_case.jsonl \
  --case-id syn-new-001
```

Edit the new record. Add all required sections. Score each item on the 0-to-4
scale. Then run validation again.

### 13.4 Rebuild the reward-integrity RL data

The rich source is:

```text
data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl
```

Validate the rich source. Then rebuild the derived pair file and manifest:

```bash
python scripts/synthetic_dataset_tool.py validate \
  data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl

python scripts/build_reward_integrity_rl_dataset.py
```

The builder must print:

```text
built 8 cases and 48 preference pairs
```

The builder writes byte-identical files for identical source input.

### 13.5 Generate constitutional data splits

Run:

```bash
python scripts/generate_constitutional_dataset.py \
  --constitution docs/GEPA_Mindfulness_Constitution.md \
  --input data/constitutional_training/examples.jsonl \
  --schema data/constitutional_training/schema.json \
  --out-dir runs/constitutional \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 17
```

The three ratios must describe the intended split. Keep the constitution as the
canonical source. The derived data does not replace it.

## 14. Run canonical RL on a CPU

Use `gepa rl` when you must update Transformers policy weights.
Do not use the compatibility CPU demo as weight-update evidence.

### 14.1 Install the RL runtime

Run:

```bash
python -m pip install -e '.[rl]'
```

The extra installs bounded versions of PyTorch, Transformers, and PEFT.
It does not install TRL, Datasets, or Accelerate.

### 14.2 Prepare a local model

The model directory must contain:

- `config.json`
- Tokenizer assets that `AutoTokenizer` accepts.
- A `.safetensors` or `.bin` weight file.

The canonical loader uses `local_files_only=True`. It does not download the
model.

Copy a preset:

```bash
cp configs/rl/pytorch_cpu_ppo.yaml run.cpu.ppo.yaml
```

On Windows PowerShell, run:

```powershell
Copy-Item configs\rl\pytorch_cpu_ppo.yaml run.cpu.ppo.yaml
```

Replace this placeholder:

```yaml
policy:
  model_name: /absolute/path/to/local-transformers-model
```

Use an absolute path.

### 14.3 Validate the model directory

Set `MODEL_DIR` to the local path. Then run this check:

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
    raise SystemExit("MODEL_DIR must contain a candidate weight filename")
AutoConfig.from_pretrained(model_dir, local_files_only=True)
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
print(model_dir)
PY
```

Exit status 0 confirms that the configuration and tokenizer are readable.
Model construction checks the full weight contents.

### 14.4 Check runtime capabilities

Run:

```bash
gepa rl doctor --config run.cpu.ppo.yaml
```

The doctor does not load the model. It prints one `AVAILABLE` or `UNAVAILABLE`
line for each required capability.

- Exit status 0 means that all required capabilities are available.
- Exit status 2 means that one or more required capabilities are unavailable.

### 14.5 Run one PPO update

Force offline model access. Then run one optimizer step:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl train --config run.cpu.ppo.yaml --max-steps 1
```

On Windows PowerShell, run:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
gepa rl train --config run.cpu.ppo.yaml --max-steps 1
```

The command prints one JSON result. For one successful update, verify these
conditions:

- `global_step` is 1.
- `policy_parameters_updated` is `true`.
- The policy checksums before and after training are different.
- The result identifies a checkpoint.
- The result identifies a log directory.

The general `parameters_updated` field also includes the value head.
Use `policy_parameters_updated` as policy-weight evidence.

### 14.6 Run GRPO

Copy `configs/rl/pytorch_cpu_grpo.yaml`. Replace its model path. Then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl train --config run.cpu.grpo.yaml --max-steps 1
```

GRPO requires `policy.do_sample: true` and `algorithm.group_size` of at least 2.
The shipped preset uses group size 4.

If all responses get the same reward, GRPO applies `zero_variance_policy`.
The `skip` value omits the response group. A run can then finish without an
optimizer step.

### 14.7 Collect without training

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl collect --config run.cpu.ppo.yaml
```

Collection generates trajectories. It does not score them. It does not update
weights.

You can override the dataset and log directory:

```bash
gepa rl collect \
  --config run.cpu.ppo.yaml \
  --dataset data/prompts.txt \
  --output runs/collection
```

Collection can use `dataset.format: text`. Each non-empty line becomes one
prompt.

### 14.8 Evaluate without training

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl evaluate --config run.cpu.ppo.yaml
```

Evaluation generates and scores trajectories. It does not run an optimizer step.
Evaluation requires the strict chosen/rejected JSONL dataset.

### 14.9 Resume a checkpoint

Select the checkpoint explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl resume \
  --config run.cpu.ppo.yaml \
  --checkpoint runs/rl_cpu_ppo/checkpoint-00000001 \
  --max-steps 1
```

`--max-steps` is relative to the restored global step.
Use `--max-steps 0` to verify restoration without a new rollout.

The checkpoint must match the configuration hash and dataset hash.
The CLI does not select the latest checkpoint for you.

### 14.10 Inspect RL artifacts

A checkpoint directory contains:

```text
checkpoint-00000001/
|-- backend.pt
|-- manifest.json
`-- training_state.pt
```

`backend.pt` contains model, value-head, optimizer, and random state.
`training_state.pt` contains engine and algorithm state.
`manifest.json` contains hashes, lineage, and global step.

Each invocation creates a log directory:

```text
rl-<run-id>/
|-- metrics.jsonl
|-- run_manifest.json
`-- trajectories.jsonl
```

Keep the manifest files with their related artifacts.

## 15. Canonical RL dataset format

Training and evaluation use a closed JSONL schema. Unknown or missing fields
cause an error before model construction.

Each row contains these main fields:

- `record_id`
- `source_case_id`
- `source_case_version`
- `source_path`
- `source_line`
- `source_sha256`
- `pair_rule`
- `prompt`
- `chosen`
- `rejected`
- `chosen_class`
- `rejected_class`
- `chosen_reward_components`
- `rejected_reward_components`
- `diagnostics`
- `schema_version`

Use this schema version:

```text
reward-integrity-rl-pairs-v1
```

The component maps contain eight bounded values:

- `objective_fidelity`
- `feedback_integrity`
- `skill_transfer`
- `reality_contact`
- `exploit_disclosure`
- `long_horizon_agency`
- `benign_creativity`
- `repair_quality`

Use the derived dataset in this path for a repository test run:

```text
data/synthetic/reward_integrity/rl_pairs_v1.jsonl
```

Do not edit the derived file directly. Edit the rich source. Then rebuild the
derived file.

## 16. Canonical RL configuration reference

The configuration has eight sections and one top-level seed. Unknown keys cause
an error.

### 16.1 `runtime`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `backend` | String | `pytorch` | Use `pytorch`, `cuda`, `llama-cpp-vulkan`, or `mojo-vulkan-llamacpp`. |
| `device` | String | `cpu` | Use `cpu`, `cuda`, or `cuda:<index>`. |
| `precision` | String | `fp32` | Use `fp32`, `fp16`, or `bf16`. Mixed precision requires CUDA. |
| `distributed` | Map | Single process | See the next table. |

### 16.2 `runtime.distributed`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `strategy` | String | `none` | Use `none`, `ddp`, or `fsdp`. |
| `world_size` | Integer | 1 | Use 2 or more for a distributed strategy. |
| `rank` | Integer | 0 | Value must be less than `world_size`. |
| `local_rank` | Integer | 0 | Value must be less than `world_size`. |
| `sharded_optimizer` | Boolean | `false` | Use `true` only with FSDP. Restore is not supported. |

The values can use the literal strings `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`.
The loader then reads the related environment variables.

### 16.3 `policy`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `model_name` | String | `demo-model` | Must not be empty. Use a local path for canonical RL. |
| `max_new_tokens` | Integer | 256 | Must be greater than zero. |
| `do_sample` | Boolean | `true` | GRPO requires `true`. |
| `temperature` | Number | 1.0 | Must be finite and greater than zero. |
| `top_p` | Number | 1.0 | Must be greater than zero and not more than 1.0. |

### 16.4 `algorithm`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `name` | String | `ppo` | Use `ppo` or `grpo`. |
| `learning_rate` | Number | `1e-5` | Must be greater than zero. |
| `batch_size` | Integer | 1 | Must be greater than zero. |
| `gradient_accumulation_steps` | Integer | 1 | Must be greater than zero. |
| `max_steps` | Integer | 100 | Must be greater than zero in the file. |
| `group_size` | Integer | 8 | GRPO requires 2 or more. |
| `kl_coef` | Number | 0.05 | Must not be negative. |
| `clip_range` | Number | 0.2 | Must be greater than zero. |
| `value_coef` | Number | 0.1 | Must not be negative. |
| `gamma` | Number | 0.99 | Must be from 0 through 1. |
| `gae_lambda` | Number | 0.95 | Must be from 0 through 1. |
| `group_normalization_epsilon` | Number | `1e-8` | Must be greater than zero. |
| `zero_variance_policy` | String | `zero` | Use `zero`, `center_only`, or `skip`. |
| `max_grad_norm` | Number or null | 1.0 | A number must be greater than zero. Null disables clipping. |

The CLI `--max-steps` option can be zero. That value disables new rollout work
for the current invocation. It does not change the file constraint.

### 16.5 `reward`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `weights.alpha` | Number | 0.3 | Must not be negative. |
| `weights.beta` | Number | 0.3 | Must not be negative. |
| `weights.gamma` | Number | 0.2 | Must not be negative. |
| `weights.delta` | Number | 0.2 | Must not be negative. |
| `overlay_weight` | Number | 0.0 | Must not be negative. |
| `integrity_overlay_enabled` | Boolean | `false` | A true value requires a positive overlay weight. |

The four base weights must have positive total mass.
If the overlay is off, `overlay_weight` must be 0.

### 16.6 `dataset`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `train_path` | String | Empty | Training requires a usable path. |
| `validation_path` | String or null | Null | Use a valid optional path. |
| `format` | String | `jsonl` | Use `jsonl` or `text`. Training and evaluation require strict JSONL pairs. |

### 16.7 `checkpoint`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `output_dir` | String | `runs/default` | Must not be empty. |
| `save_steps` | Integer | 100 | Must be greater than zero. Hybrid training requires 1. |

### 16.8 `logging`

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `log_dir` | String | `runs/logs` | Must not be empty. |
| `level` | String | `INFO` | Use `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

### 16.9 `hybrid`

This section applies to the experimental hybrid path.

| Key | Default | Constraint |
| --- | --- | --- |
| `model_id` | `local-policy` | Use one safe identifier. Do not use path separators. |
| `expected_actor_backend` | `mojo-coordinator` | Use one safe identifier. |
| `adapter_store` | `runs/hybrid/adapters` | Must not be empty. |
| `training_mode` | `lora` | Only `lora` is accepted. |
| `staleness_policy` | `reject` | Use `reject` or `down_weight`. |
| `max_policy_lag` | 0 | Must not be negative. |
| `downweight_decay` | Null | Use null with `reject`. Use a value between 0 and 1 with `down_weight`. |
| `lora` | Empty map | Supports `r`, `lora_alpha`, `lora_dropout`, `bias`, `task_type`, `target_modules`, and `modules_to_save`. |

### 16.10 `seed`

The top-level `seed` is an integer. The default is 42.

## 17. Run RL on one NVIDIA GPU

The repository did not qualify this path on its source verification host.
Run the acceptance tests on your target hardware.

### 17.1 Install a CUDA-enabled PyTorch build

Use the official PyTorch selector. Select a build that matches the operating
system, driver, and CUDA requirement.

Do not assume that the ordinary `rl` extra installs a CUDA build.

### 17.2 Prepare the configuration

Copy this file:

```text
configs/rl/cuda_single_gpu.yaml
```

Replace `LOCAL_MODEL_PATH` with an absolute local model path.
Keep `runtime.device: cuda:0`.
Start with `runtime.precision: fp32`.

Set `checkpoint.save_steps: 1` for a one-step acceptance run.

### 17.3 Check and train

Run:

```bash
gepa rl doctor --config run.cuda.ppo.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl train --config run.cuda.ppo.yaml --max-steps 1
```

Verify the same JSON fields that the CPU procedure specifies.

### 17.4 Diagnose out-of-memory errors

Record the complete error. It contains allocator data when PyTorch supplies it.

Run this command to inspect other GPU processes:

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
```

Stop a process only when you own it.

To reduce memory use, do one or more of these actions:

1. Select a smaller local model.
2. Reduce `policy.max_new_tokens`.
3. Reduce `algorithm.batch_size` when it is greater than 1.
4. Increase gradient accumulation after you reduce the step batch.

The runtime does not change the configuration automatically. It does not retry
with weaker settings.

### 17.5 Run the CUDA acceptance test

Run:

```bash
python -m pytest --strict-markers -m cuda tests/test_rl_cuda.py -q -rs
```

A skipped precision is not a pass. Treat it as unsupported on that device.

## 18. Run two-GPU DDP

Copy `configs/rl/cuda_ddp.yaml`. Replace `LOCAL_MODEL_PATH`.

Then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m mindful_trace_gepa rl train \
  --config run.cuda.ddp.yaml \
  --max-steps 1
```

`torchrun` supplies the topology environment variables.
The configured device must match `cuda:LOCAL_RANK`.

FSDP configuration is validated by the runtime. The repository does not support
sharded-optimizer checkpoint restore.

## 19. Experimental llama.cpp and Vulkan collection

This path is for inference and collection only.

It does not support these functions:

- Training.
- Resume.
- Evaluation.
- Backward passes.
- Optimizer steps.
- Value heads.
- Full-weight updates.

### 19.1 Start a loopback server

Build llama.cpp with Vulkan. Start `llama-server` on `127.0.0.1`.
Make the server alias equal to `policy.model_name` in the YAML file.

Example:

```bash
export GGUF_MODEL=/absolute/path/to/model.gguf
./build/bin/llama-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model "$GGUF_MODEL" \
  --alias LOCAL_GGUF_MODEL_ID
```

### 19.2 Run the doctor

Run:

```bash
gepa rl doctor \
  --backend llama-cpp-vulkan \
  --endpoint http://127.0.0.1:8080
```

Exit status 2 is expected because training capabilities are unsupported.
`supports_vulkan: UNKNOWN` means that Vulkan proof is absent.

### 19.3 Collect trajectories

Run:

```bash
gepa rl collect \
  --config configs/rl/llama_cpp_vulkan_collect.yaml \
  --backend llama-cpp-vulkan \
  --endpoint http://127.0.0.1:8080 \
  --dataset data/synthetic/reward_integrity/rl_pairs_v1.jsonl \
  --output runs/llama_cpp_vulkan/logs
```

Fields such as token IDs, log probabilities, values, rewards, advantages, and
returns stay null unless the server supplies validated evidence.

## 20. Experimental hybrid actor and PyTorch learner

This path uses an external actor coordinator and a local PyTorch LoRA learner.
It is not pure Mojo training.

Before you start, complete these actions:

1. Use a source checkout.
2. Copy `configs/rl/hybrid_vulkan_grpo.yaml` to `run.hybrid.yaml`.
3. Set the local Transformers learner path.
4. Set a safe `hybrid.model_id`.
5. Set the exact `hybrid.expected_actor_backend`.
6. Supply an external configured coordinator.
7. Bootstrap and publish the initial LoRA adapter.
8. Verify the current adapter manifest and checksum.

Do not use `mojo/rl_coordinator/main.mojo` as the training coordinator. That
file is a non-generating protocol reference. It returns `actor_unconfigured`.

Do not put a coordinator command, an actor endpoint, or secrets in the YAML file.

Run training only after the bootstrap checks pass:

```bash
gepa rl train \
  --config run.hybrid.yaml \
  --backend mojo-vulkan-llamacpp \
  --learner pytorch \
  --coordinator-command /absolute/path/to/configured-coordinator \
  --actor-endpoint http://127.0.0.1:8080 \
  --max-steps 1
```

The command runs the coordinator argument vector directly. It does not use a
shell.

The learner publishes a PyTorch LoRA state file. It does not convert the file to
GGUF. It does not deploy the adapter to llama.cpp. It does not prove that the
actor loaded a new adapter.

Use the complete bootstrap and recovery procedure in
`docs/rl/README.md` before you operate this path.

## 21. Use the compatibility demos

### 21.1 CPU demo

Run:

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

Select GRPO or a different output directory when necessary:

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo \
  --trainer grpo \
  --output runs/grpo_cpu_demo
```

The demo accepts these options:

- `--trainer ppo|grpo`
- `--output PATH`
- `--root PATH`

> CAUTION: This demo uses compatibility trainers. The trainers update
> simulator-owned Python scalar tables. They do not update Transformers policy
> weights. Use `gepa rl train` for a real policy update.

### 21.2 vLLM demo

Configure the endpoint in `gepa_mindfulness/configs/vllm.yaml`.
The default endpoint is `http://localhost:8000`.

Start the vLLM server. Then run:

```bash
python -m gepa_mindfulness.examples.vllm_demo.run_vllm_demo
```

The demo generates two sample responses. It prints reward and trace summaries.
It does not use the canonical RL engine.

### 21.3 Full pipeline shell script

On a POSIX shell, run:

```bash
./scripts/run_full_pipeline.sh
```

On Windows, use Windows Subsystem for Linux or Git Bash. The file is a shell
script and is not a PowerShell script.

## 22. Use the ontology workbench

The workbench is a local web application. It keeps the canonical ontology
read-only.

### 22.1 Install and start the workbench

Use Node.js 22.13.0 or later.

On Linux or macOS, run:

```bash
cd apps/gepa-ontology-workbench
npm install
npm run dev
```

On Windows PowerShell, the package script uses POSIX environment syntax.
Use this equivalent command after `npm install`:

```powershell
$env:WRANGLER_LOG_PATH = ".wrangler/wrangler.log"
npx vinext dev
```

Open the local URL that the command prints.

### 22.2 Check the canonical digest

Find the digest status in the application header.

- `Verified` permits a governed export after proposal checks pass.
- A failed or unavailable digest disables export.

Do not bypass a failed digest check.

### 22.3 Use Explore mode

1. Select `Explore`.
2. Use the global search to find concepts, aliases, or relations.
3. Select `Normative`, `Operational`, or `All`.
4. Select a relation predicate when necessary.
5. Select a concept.
6. Review its definition, relations, assessments, maturity, and guardrails.
7. Select `Copy ID` when you need the canonical identifier.

A relation selection does not replace the selected concept.

### 22.4 Use Assess mode

1. Select `Assess`.
2. Review support and opposition separately.
3. Read each uncertainty statement.
4. Read each provenance warning.
5. Read each correlated-evidence warning.
6. Read each maturity gap.

Do not add correlated evidence as if it were independent confirmation.
An unavailable quantity stays unavailable.

### 22.5 Use Invariants mode

1. Select `Invariants`.
2. Search by number, key, statement, explanation, or forbidden inference.
3. Select `Clear search` to restore the full ordered list.

The workbench shows 22 ordered invariants.

### 22.6 Check a proposal in Improve mode

1. Select a target concept.
2. Select `Improve`.
3. Enter a canonical ID.
4. Enter a label.
5. Select the layer.
6. Enter a registered type.
7. Select the epistemic status.
8. Select the governance classification.
9. Enter qualitative uncertainty.
10. Enter a definition.
11. Enter resolvable provenance.
12. Enter optional relation data as one complete set.
13. Enter an operational mapping.
14. Enter maturity evidence.
15. Select `Run semantic checks`.

The lifecycle is fixed to `proposed`.

If you enter relation data, supply all three values:

- Relation predicate.
- Relation target.
- Relation family.

The validator checks required fields, registered types, layer rules, duplicate
IDs, alias overlap, semantic overlap, relation domain and range, protected
kernel changes, provenance, uncertainty, and maturity evidence.

### 22.7 Resolve findings

The results separate blockers from warnings.

- A blocker disables export.
- A warning permits export after review.
- An invariant reference opens the related invariant.

Use `Explicit normative revision` only for a governed normative revision.
It does not automatically approve a protected-kernel change.

### 22.8 Generate a governed bundle

1. Resolve all blockers.
2. Verify the canonical digest.
3. Select `Context` or `Training`.
4. Select JSON, YAML, or Markdown.
5. Select `Generate governed bundle`.
6. Review the read-only preview.
7. Select `Copy bundle` or `Download bundle`.

A training bundle requires exactly one selected target. The target must have
curated behavior examples.

The downloaded name is `gepa-governed-bundle` with the selected extension.
Copy and download errors do not remove the preview.

### 22.9 Validate the workbench source

From the workbench directory, run:

```bash
npm test
npm run lint
npm run build
npm audit --omit=dev
```

The production dependency audit must return exit status 0.
The full `npm audit` is informational for the applicable source version.
It has known development dependency findings.

## 23. Use notebooks

The repository supplies fine-tuning and evaluation notebooks.

Start Jupyter from the repository root:

```bash
jupyter notebook notebooks/ft_phi3_mini_unsloth_gepa.ipynb
jupyter notebook notebooks/ft_llama3_8b_unsloth_gepa.ipynb
jupyter notebook notebooks/eval_deception_acl2025.ipynb
```

Run from the repository root so that relative dataset paths resolve.
Review notebook package and hardware requirements before you run a cell.

## 24. Complete CLI reference

### 24.1 General command form

```text
gepa <command> <subcommand> [options]
```

Run `gepa --help` for the current command tree.
Run `gepa <command> --help` for command-specific help.

### 24.2 DSPy commands

| Command | Required options | Optional options and defaults |
| --- | --- | --- |
| `gepa dspy run` | `--input`, `--trace` | `--context`; `--model`; `--enable-optim`; `--dual-path`; `--with-logprobs` on; `--log-topk 3`; `--log-every 16`; `--long-context`; `--shard-threshold 10000`; `--enable-value-decomp`; `--enable-dvgr`; `--use-grn-value-decomp`. |
| `gepa dspy compile` | `--out` | `--config configs/policies/dspy.yml`; `--dataset`; `--enable-optim`. |
| `gepa dspy contrastive-run` | `--data`, `--out` | `--context general`; `--probes`. |

### 24.3 Data and refinement commands

| Command | Required options | Optional options and defaults |
| --- | --- | --- |
| `gepa cpt-build` | `--input`, `--out` | `--seed 0`; pair randomization on; consensus filtering on. |
| `gepa ssr-run` | `--input`, `--out` | `--mode evaluation`; `--max-iterations 2`; `--run-id ssr-run`; `--initial-answer-reference ""`. |

### 24.4 Viewer and score commands

| Command | Required options | Optional options and defaults |
| --- | --- | --- |
| `gepa view` | `--trace`, `--tokens`, `--out` | `--deception`; `--dual-path`; `--page-size 200`; `--max-points 5000`; `--manifest`. |
| `gepa score` | `--trace`, `--out` | `--policy`; streaming on; `--sharded`; `--manifest`; `--zstd`. |
| `gepa score-auto` | `--trace`, `--out` | `--policy`; `--config`; `--judge`; `--classifier`; `--classifier-config`; `--classifier-artifacts`; `--no-print`. |
| `gepa judge` | `--trace`, `--out` | `--model`; `--mock`. |
| `gepa clf train` | `--labels`, `--config`, `--out` | None. |
| `gepa clf triage` | `--scores`, `--out` | `--threshold 0.6`. |

### 24.5 Deception commands

| Command | Required options | Optional options |
| --- | --- | --- |
| `gepa deception probes` | `--trace`, `--model`, `--probe`, `--config` | `--out`. |
| `gepa deception summary` | `--out` | `--probe`; `--dual-path`; `--mm`; `--runs`. |

The viewer and deception summary parsers also accept the hidden compatibility
option `--paired`. It has the same effect as `--dual-path`.

### 24.6 RL commands

| Command | Required options | Optional options |
| --- | --- | --- |
| `gepa rl train` | `--config` | `--backend`; `--endpoint`; `--learner`; `--coordinator-command`; `--actor-endpoint`; `--max-steps`. |
| `gepa rl resume` | `--config`, `--checkpoint` | The common backend options and `--max-steps`. |
| `gepa rl collect` | `--config` | The common backend options; `--dataset`; `--output`. |
| `gepa rl evaluate` | `--config` | The common backend options. |
| `gepa rl doctor` | None | `--config`; `--backend system|llama-cpp-vulkan`; `--endpoint`. |

The common engine backend values are:

- `pytorch`
- `cuda`
- `llama-cpp-vulkan`
- `mojo-vulkan-llamacpp`

The learner values are `pytorch` and `mojo`. Only `pytorch` is supported.

## 25. Artifact reference

| Artifact | Producer | Use |
| --- | --- | --- |
| `trace.jsonl` | `gepa dspy run` | Event review and scoring. |
| `tokens.jsonl` | `gepa dspy run` | Token display in the viewer. |
| `summary.json` | DSPy or contrastive commands | Run summary and counts. |
| `deception.json` | DSPy run | Per-input deception result. |
| `deception_probe.json` | `gepa deception probes` | Linear probe result. |
| `deception_summary.json` | `gepa deception summary` | Merged deception result. |
| `scores.json` | `gepa score-auto` | Tiered wisdom scores and confidence. |
| `score.html` | `gepa score` | Simple mean-score report. |
| `viewer.html` | `gepa view` | Self-contained offline review file. |
| `manifest.json` | Trace sharding or RL checkpoint | Hash, shard, or lineage data. Interpret it in its directory context. |
| `metrics.jsonl` | Canonical RL | Step metrics. |
| `trajectories.jsonl` | Canonical RL | Generated trajectory records. |
| `run_manifest.json` | Canonical RL | Configuration and capability evidence. |
| `backend.pt` | Canonical RL checkpoint | Backend-owned saved state. |
| `training_state.pt` | Canonical RL checkpoint | Engine and algorithm saved state. |

## 26. Troubleshooting

### 26.1 The shell cannot find `gepa`

Cause: The virtual environment is not active, or the script directory is not on
`PATH`.

Action:

```bash
python -m mindful_trace_gepa --help
```

If this command works, activate the environment again.

### 26.2 The DSPy command reports an unavailable component

Cause: The optional DSPy dependency is absent.

Action:

```bash
python -m pip install -e '.[dspy]'
```

Then run the command again.

### 26.3 An input row has no inquiry

Message:

```text
Each input row must contain an 'inquiry' or 'prompt' field.
```

Action: Add one of the two fields to every JSONL row.

### 26.4 The viewer is empty

Possible causes:

- The trace path is wrong.
- The trace file is empty.
- `--page-size` is too small for the expected data.
- The token file is absent, so only the token panel is empty.

Action: Confirm each path. Validate the JSONL. Then rebuild the viewer.

### 26.5 A sharded score cannot find the manifest

Message:

```text
Manifest not found at <path>
```

Action: Supply the correct `--manifest` path. Confirm that each shard path in
the manifest is relative to the manifest directory.

### 26.6 The classifier option fails with a null path

Cause: The parser does not provide a usable implicit path in this source version.

Action: Supply both options:

```text
--classifier-config configs/classifier/default.yml
--classifier-artifacts artifacts/classifier
```

### 26.7 The RL model tries to access the network

Cause: Offline environment variables are absent, or another library call does
not use the canonical loader.

Action: Set both variables:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Use an absolute local model path.

### 26.8 The RL doctor reports `UNAVAILABLE`

Cause: A required package, device, or precision is not available.

Action: Read every doctor line. Install or configure the missing capability.
Run the doctor again before model construction.

### 26.9 A checkpoint does not resume

Possible causes:

- The configuration changed.
- The dataset changed.
- The checkpoint is incomplete.
- The selected checkpoint belongs to a different run.

Action: Compare the configuration hash and dataset hash in the manifests.
Select the exact matching checkpoint.

### 26.10 GRPO finishes without an update

Cause: All responses have the same reward, and the zero-variance policy skips
the group.

Action: Use a model and dataset that produce reward variation. Inspect the
trajectory and metric logs before you change the policy.

### 26.11 CUDA runs out of memory

Action: Use the procedure in section 17.4. Do not delete another user's process.

### 26.12 The Vulkan doctor returns exit status 2

Cause: The backend intentionally does not support training capabilities.

Action: Inspect the inference and Vulkan evidence lines. Use the backend only
for collection.

### 26.13 Hybrid training reports `actor_unconfigured`

Cause: The checked-in Mojo protocol reference was used as a coordinator.

Action: Supply an operator-configured coordinator. Do not use
`mojo/rl_coordinator/main.mojo` for training.

### 26.14 The workbench export button is disabled

Possible causes:

- Proposal checks did not run.
- A blocker is present.
- The canonical digest is not verified.
- A training target has no curated examples.
- The selected target does not resolve.

Action: Read the message below the export controls. Correct the stated cause.
Run semantic checks again.

### 26.15 `npm run dev` fails on Windows

Cause: The package script uses POSIX environment-variable syntax.

Action: Set `WRANGLER_LOG_PATH` in PowerShell. Then run `npx vinext dev`.

## 27. Data protection and reproducibility

### 27.1 Protect sensitive data

Do these actions:

1. Review prompts before you store them.
2. Remove credentials and personal data from example files.
3. Restrict access to trace, token, viewer, and checkpoint files.
4. Do not publish a viewer until you inspect its embedded data.
5. Do not put secrets in YAML files or command histories.

### 27.2 Preserve provenance

Keep these items together:

- Source commit identifier.
- Configuration file.
- Dataset hash.
- Run manifest.
- Checkpoint manifest.
- Metric and trajectory logs.
- Model identifier and local model hash, when available.

### 27.3 Make a reproducible run

1. Use an explicit seed.
2. Use an explicit model path.
3. Set offline environment variables.
4. Keep the input dataset unchanged.
5. Keep the configuration unchanged.
6. Record the command.
7. Keep all manifests.
8. Verify checksums after copy or transfer.

## 28. Repository map for users

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and quick start. |
| `pyproject.toml` | Package metadata, dependencies, and CLI entry point. |
| `src/mindful_trace_gepa/` | Main trace, score, deception, and viewer package. |
| `gepa_mindfulness/training/` | Canonical and compatibility training code. |
| `configs/rl/` | Canonical RL presets. |
| `docs/rl/README.md` | Detailed RL procedures and limits. |
| `data/synthetic/` | Synthetic source data, schemas, and derived data. |
| `datasets/` | Evaluation and example datasets. |
| `modules/` | Optional robustness, CPT, and SSR modules. |
| `scripts/` | Dataset, analysis, and pipeline scripts. |
| `notebooks/` | Fine-tuning and evaluation notebooks. |
| `apps/gepa-ontology-workbench/` | Local governed ontology workbench. |
| `runs/` | Conventional run output location. |

## 29. Operational checklist

Use this checklist before a research result or training claim.

1. Identify the source commit.
2. Identify the command and all options.
3. Identify the model and dataset.
4. Identify the maturity level of the selected path.
5. Run the applicable doctor or validation command.
6. Keep the result JSON and all manifests.
7. Check for synthetic or simulated data labels.
8. Check `policy_parameters_updated` for a weight-update claim.
9. Compare the policy checksums.
10. Inspect the checkpoint and log directories.
11. Record skipped hardware tests as limitations.
12. Protect all sensitive artifacts.

## 30. Verification record for this manual

The manual author completed these checks against the applicable source commit:

- Read the main README, package metadata, entry points, command handlers,
  configuration validators, user guides, module guides, workbench source, and
  selected tests.
- Executed `gepa --help` and the help command for all 19 leaf CLI commands.
- Installed the basic package in an isolated virtual environment.
- Executed `gepa dspy contrastive-run` with the supplied dual-path dataset.
- Executed `gepa score` with the supplied sample trace.
- Executed `gepa view` with the supplied sample trace.
- Executed the synthetic gold-dataset validator.
- Confirmed that the source checkout had no changed files after the checks.

The verification environment did not contain a local Transformers model, CUDA
hardware, Vulkan hardware, Mojo, or a vLLM server. The related procedures come
from source validation, repository tests, and checked-in operator guides. They
are not hardware execution evidence.
