# GEPA Mindfulness Superalignment

GEPA Mindfulness Superalignment pairs GEPA-inspired reflective reasoning with
traceable honesty instrumentation. The repository contains a Python package with
training utilities, DSPy-style declarative pipelines, deception diagnostics, and
an offline trace viewer that can be embedded into reports.

The project targets Python 3.10+.

## Highlights

* **GEPA scoring primitives** – `gepa_mindfulness/core` models the contemplative
  principles (Mindfulness, Empathy, Perspective, Agency) and paraconsistent
  alignment imperatives used throughout the training code.
* **Training orchestration** – `gepa_mindfulness/training` wires PPO utilities,
  abstention checks, optional deception heuristics, and CLI tooling into a
  reproducible training loop.
* **DSPy-style pipelines** – `src/mindful_trace_gepa/dspy_modules` provides
  guardrailed declarative chains that emit GEPA checkpoints and can be compiled
  into manifests when the optional `dspy-ai` dependency is installed.
* **Dual-path deception probes** – `src/mindful_trace_gepa/deception` ships
  heuristics, circuit-fingerprint shims, and dual-path detectors.
* **Offline trace viewer** – `src/mindful_trace_gepa/viewer` bundles a static
  HTML viewer that stitches traces, token confidence curves, and deception
  overlays into a single portable file.
* **Self-contained graph analytics** – the in-tree `networkx` stub mirrors the
  features we depend on, including an iterative strongly connected component
  traversal so attribution metrics stay consistent without external
  dependencies.
* **Curated datasets** – the `datasets/` and `gepa_datasets/` directories include
  JSONL corpora for ethical QA, abstention calibration, OOD stress prompts,
  anti-scheming probes, and dual-path reasoning scenarios. The full bundle is
  also provided as `gepa_datasets.zip` for distribution.

## Repository layout

```
README.md                       Project overview and quick-start
pyproject.toml                  Build metadata and extras
gepa_mindfulness/               Reference training pipeline
  adapters/                     Inference bridges (HF + vLLM)
  configs/                      Minimal YAML presets
  core/                         GEPA scoring, abstention, rewards
  examples/                     CPU + vLLM demo entry points
  training/                     CLI, PPO utilities, orchestrator
src/mindful_trace_gepa/         Trace tooling, DSPy modules, viewer
examples/                       Thin wrappers + sample traces
scripts/                        End-to-end helpers and analyses
datasets/, gepa_datasets/       Bundled evaluation and training corpora
notebooks/                      Unsloth/PEFT fine-tuning notebooks
```

Every subdirectory that contains runnable components includes its own README for
more detail.

See `docs/grn_integration.md` for configurable Global Response Normalization
usage across scoring, PPO training, and probe evaluation.

## Requirements

* Python 3.10+
* `pyyaml` (installed automatically via `pip install -e .`)
* Optional extras:
  * `torch`, `transformers`, and `trl` – required for PPO training and the CPU
    demo.
  * `requests` – required for the vLLM example.
  * `dspy-ai` – unlocks DSPy pipelines and compilation (`pip install -e .[dspy]`).
  * `weasyprint` – enables PDF export (`pip install -e .[pdf]`).
  * `circuit-tracer` – optional third-party package for full circuit logging;
    the repository falls back to lightweight shims when it is absent.

## Installation

From the repository root (`pwd` should end with `GEPA-Mindfulness-superalignment`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dspy]  # install without the brackets if you do not need DSPy extras
```

If you received the project as a ZIP or plain folder, run `git init` before
installing so you can track local changes. The `.gitattributes` file normalises
line endings to LF during commits; no additional configuration is required on
most systems.

## Quick start

1. **Run the CPU demo** – executes a minimal PPO loop using the bundled prompts
   and configuration:

   ```bash
   python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
   ```

   Results are written to `training_logs/rollouts.jsonl` (or to the directory
   you pass via `--log-dir`).

2. **Generate a trace with the DSPy pipeline** – create a trace and token log
   that can be inspected offline. Requires the `[dspy]` extra but uses only the
   lightweight fallback implementation by default:

   ```bash
   gepa dspy run --input examples/self_tracing_sample.jsonl \
                 --trace runs/dspy_trace.jsonl
   ```

   This command writes `runs/dspy_trace.jsonl`, `runs/tokens.jsonl`, and
   `runs/summary.json` (plus a deception summary when detectors are enabled).

3. **Render the offline viewer** – bundle the trace into a single HTML report:

   ```bash
   gepa view --trace runs/dspy_trace.jsonl \
            --tokens runs/tokens.jsonl \
            --out runs/dspy_trace_view.html
   ```

4. **End-to-end smoke test** – run the shell helper to execute the CPU demo and
   associated analyses in sequence:

   ```bash
   ./scripts/run_full_pipeline.sh
   ```

5. **vLLM example (optional)** – point the training orchestrator at an existing
   vLLM server configured in `gepa_mindfulness/configs/vllm.yaml`:

   ```bash
   python -m gepa_mindfulness.examples.vllm_demo.run_vllm_demo
   ```

   Ensure the endpoint (`model.vllm_engine`) is reachable and that `requests` is
   installed in your environment.

## CLI overview

The editable install registers the `gepa` console script. Key commands include:

* `gepa dspy run --input INPUT.jsonl --trace runs/trace.jsonl` – execute the
  GEPA DSPy-style pipeline (use `--dual-path` to enable dual-path reasoning;
  requires `dspy-ai`).
* `gepa dspy compile --out dspy_artifacts/ --enable-optim` – produce guarded
  prompt manifests from the DSPy modules.
* `gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl --out report.html`
  – build the static offline trace viewer.
* `gepa score --trace runs/trace.jsonl --out summary.html` – aggregate principle
  and imperative scores from a trace.
* `gepa deception probes --trace runs/trace.jsonl --config configs/deception/probes_linear.yaml`
  – run deception heuristics against stored traces.

Run `gepa --help` for the complete command tree.

## Deception and dual-path analysis

Honesty is rewarded directly via the GEPA reward weights while deception signals
are surfaced for manual review. Recommended workflow:

1. Train or run rollouts with honesty rewards enabled (`gepa_mindfulness` CPU
   demo or your custom configuration).
2. Explore single-prompt dual-path reasoning datasets:

   ```bash
   gepa dspy run --dual-path --input datasets/dual_path/data.jsonl \
                 --trace runs/dual_path_trace.jsonl
   ```

   or use the lightweight contrastive baseline that operates without model
   inference:

   ```bash
   gepa dspy contrastive-run --data datasets/dual_path/data.jsonl \
                             --out runs/contrastive/ --context general
   ```

## Dual-Prompt + Adversarial Integration

Run GUI:

```bash
python -m app.main
```

Or command line:

```bash
python src/adversarial_evaluator.py --scenario safety_lab_001
python src/adversarial_circuit_tracer.py runs/001 --tokenizer mistral-instruct
python tools/merge_run_inspection.py runs/001
```

All artifacts saved under `runs/<id>/`.

## Notebooks and fine-tuning

The `notebooks/` directory provides Unsloth/PEFT workflows for Phi-3 Mini and
Llama-3 8B. Launch from the project root so relative dataset paths resolve:

```bash
jupyter notebook notebooks/ft_phi3_mini_unsloth_gepa.ipynb
jupyter notebook notebooks/ft_llama3_8b_unsloth_gepa.ipynb
```

Each notebook relies solely on the bundled datasets and renders both an HTML
score report and an offline viewer via the `gepa` CLI.

## License

MIT License.
