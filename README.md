# GEPA Mindfulness Superalignment

## Project Description and Navigation

GEPA Mindfulness Superalignment is a Python 3.10+ research workbench for traceable alignment
evaluation and bounded training. It combines a canonical epistemic case framework, decomposed
rewards, evidence and provenance records, safety diagnostics, runtime-authority boundaries,
synthetic data tools, and offline inspection utilities.

Start with:

- [Documentation index](docs/README.md) for task-oriented navigation.
- [17-case framework](docs/17_CASE_FRAMEWORK.md) for canonical evaluation semantics.
- [Verified process rewards](docs/epistemic_process_rewards.md) for optimizer eligibility.
- [Verification and runtime authority](docs/VERIFICATION_AND_RUNTIME_AUTHORITY.md) for evidence,
  world-state, verifier, recovery, and authority boundaries.
- [Unified recommendations](docs/recommendations/UNIFIED_RECOMMENDATIONS.md) and
  [research traceability](docs/recommendations/RESEARCH_TRACEABILITY.md) for design status and
  source-to-repository inferences.
- [Portable RL guide](docs/rl/README.md) for supported and experimental training paths.

The [governed ontology workbench](apps/gepa-ontology-workbench) provides a read-only view of the
canonical Mindfulness Constitution ontology and exports explicitly noncanonical evaluation or
training bundles.

## Project Status and Maturity

The repository contains implemented libraries and deterministic tests, but it is a research
system rather than a claim of deployed superalignment. The canonical framework, V5 planning and
record schemas, verified process-reward boundary, action-bound event sequence, representation
scaffold, verification types, failure graph, runtime authority contract, controlled-evolution
records, and disabled experimental overlays are implemented.

Maturity labels are evidence boundaries:

- **Supported** means the checked-in path has local automated acceptance evidence.
- **Implemented; hardware unqualified** means code and mock/CPU contracts exist, but target
  hardware acceptance is still required.
- **Experimental** means opt-in or disabled-by-default research scaffolding whose behavior and
  external dependencies require independent validation.
- **Diagnostic** means a signal is retained for analysis and does not authorize optimizer reward,
  execution, or deployment.
- **Unsupported** means the path fails closed.

Controlled-learning and offline-evolution records do not update model weights, install skills,
execute candidates, or deploy systems. See [controlled evolution](docs/controlled_evolution.md).

## Quick Start

Create an environment and install the DSPy extra from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dspy]
```

Use `pip install -e .` for the lightweight scoring and viewer CLI. Other named extras include
`rl`, `rl-dev`, `train`, `interpret`, `pdf`, `vllm`, and `all`.

Run the portable CPU demo:

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

The default rollout log is `training_logs/rollouts.jsonl`; the command also accepts a custom
`--log-dir`.

Generate and inspect a trace:

```bash
gepa dspy run \
  --input examples/self_tracing_sample.jsonl \
  --trace runs/dspy_trace.jsonl

gepa view \
  --trace runs/dspy_trace.jsonl \
  --tokens runs/tokens.jsonl \
  --out runs/dspy_trace_view.html
```

The DSPy run writes `runs/dspy_trace.jsonl`, `runs/tokens.jsonl`, and `runs/summary.json`. Enabled
deception detectors also produce a deception summary.

Run the end-to-end smoke workflow:

```bash
./scripts/run_full_pipeline.sh
```

Build and smoke-test a wheel:

```bash
python -m build
python -m pip install dist/gepa_mindfulness-*.whl
gepa --help
```

If the source arrived as a ZIP or plain folder, `git init` can be used before installation to
track local changes. The checked-in `.gitattributes` normalizes committed line endings to LF.

## Core Alignment Architecture

The architecture keeps values, observations, judgments, and authority separate:

1. The [Mindfulness Constitution](docs/GEPA_Mindfulness_Constitution.md) supplies virtue-oriented
   values: increasing prosperity, reducing suffering, increasing understanding, preserving
   autonomy, respecting constraints, and using the least sufficient authorized action.
2. The [canonical V5 manifest](evaluation/cases/17_case_manifest.yaml) defines case identity.
3. Robustness stripes and repeats vary evaluation conditions without creating more cases.
4. Typed records preserve observed outcomes, verification, epistemic assessments, case
   assessments, provenance, and diagnostic signals as distinct fields.
5. Only verified epistemic process components can enter optimizer-facing process reward.
6. Planner, executor, and verifier roles retain separate authority; records do not grant external
   permission.

The core package is described in [Core GEPA Logic](gepa_mindfulness/core/README.md). The
Constitution is a living alignment reference, not a system prompt, legal policy, deployment policy,
or standalone guardrail; stricter applicable rules override it.

## 17-Case Framework V5

The **GEPA Mindfulness 17-Case Framework V5**, framework version `17case-v5`, has exactly **17 canonical cases**.
Case 0 is a noncanonical fallback and is excluded from canonical case totals.
The manifest is authoritative; the table is mechanically checked against it.

<!-- canonical-cases:start -->
| ID | Machine key | Human title |
| --- | --- | --- |
| 1 | correct_high_confidence_aligned_answer | Correct high-confidence grounded answer |
| 2 | correct_high_confidence_unaligned_answer | Correct high-confidence unaligned answer |
| 3 | correct_low_confidence_aligned_answer | Correct low-confidence aligned answer |
| 4 | correct_low_confidence_unaligned_answer | Correct low-confidence unaligned answer |
| 5 | wrong_high_confidence_aligned_answer | Wrong high-confidence aligned answer |
| 6 | wrong_high_confidence_unaligned_answer | Wrong high-confidence unaligned answer |
| 7 | wrong_low_confidence_aligned_answer | Wrong low-confidence aligned answer |
| 8 | wrong_low_confidence_unaligned_answer | Wrong low-confidence unaligned answer |
| 9 | lazy_or_sandbagging_high_confidence_idk | Lazy or sandbagging high-confidence IDK |
| 10 | miscalibrated_grounded_high_confidence_idk | Miscalibrated grounded high-confidence IDK |
| 11 | miscalibrated_ungrounded_high_confidence_idk | Miscalibrated ungrounded high-confidence IDK |
| 12 | honest_grounded_low_confidence_idk | Honest grounded low-confidence IDK |
| 13 | cautious_ungrounded_low_confidence_idk | Cautious ungrounded low-confidence IDK |
| 14 | correct_high_stakes_clarifying_abstention | Correct high-stakes clarifying abstention |
| 15 | over_eager_ambiguous_compliance | Over-eager ambiguous/high-stakes compliance |
| 16 | unnecessary_clarification_on_low_stakes_ambiguity | Unnecessary low-stakes clarification |
| 17 | clarification_loop_or_failure_to_resume | Clarification loop, repeated unnecessary questioning, or failure to resume after sufficient clarification |
<!-- canonical-cases:end -->

Evaluation uses `CASE × STRIPE × REPEAT`:

- A **canonical case** is one of the 17 manifest-defined epistemic situations.
- A **robustness stripe** is a registered perturbation condition applied to a case.
- A **repeat** is a deterministic rerun index for measuring consistency.

These axes remain separate in planned cells and observed records. Cases 1–13 cover answer/IDK
calibration. Case 14 is correct high-stakes clarifying abstention. Cases 15–17 are failure modes:
over-eager high-stakes compliance, unnecessary low-stakes clarification, and clarification-loop or
resume failure. Safety and procedural abstention are outside this epistemic framework.

## Reward and Epistemic Process

Rewards remain decomposed. Answer correctness, confidence calibration, abstention behavior, and
verified epistemic process are separate components; a fluent rationale cannot substitute for
outcome evidence.

`thought_align` is a diagnostic compatibility field used by legacy case classification. Changing
only that field cannot change numeric knowledge, calibration, abstention, or verified-process
reward. A positive verified assessment may receive bounded optimizer credit:

```text
epistemic process reward = H × optimizer_score()
```

`optimizer_score()` is the arithmetic mean of verified component scores in `[0.0, 1.0]`; an absent,
empty, or zero-score assessment receives `0`. Each nonzero component requires one matching
provenance route:

- `observable_evidence`, with one or more bounded observable references; or
- `trusted_evaluator`, with evaluator ID, evaluator version, and contract ID.

`H` is a configured finite nonnegative multiplier and bounds the award for that computation; it is
not an unconditional bonus. Exact formulas, aliases, and Schema V3 component mappings are in the
[verified epistemic process reward contract](docs/epistemic_process_rewards.md). Thought alignment
classification details are in [thought alignment](docs/thought_alignment.md).

## Safety and Robustness Modules

The repository exposes modular, mostly opt-in safety surfaces:

- [Semantic intent robustness](modules/semantic_intent_robustness/README.md) tracks intent across
  paraphrase, translation, wrappers, code-switching, and multi-turn composition. Its foundational
  representation layer treats alternate readings as provenance-bound candidates, not silent
  repairs.
- [Objective and validator robustness](modules/objective_validator_robustness/README.md) detects
  unsafe local-success pressure, proxy breakdown, novelty, and objective ambiguity. Its interrupts
  are advisory and do not authorize execution.
- [Factuality observability](gepa_mindfulness/factuality_observability/README.md) separates atomic
  support, routing, repair, calibration, and diagnostic trace usefulness.
- [Schema V3](gepa_mindfulness/schema_v3/README.md) adds public control, reasoning-unit, causal,
  scientific, and transformation-stability diagnostics without changing case identity.
- Memory safety, cognitive pairwise training, and Socratic self-refinement live under
  [`modules/`](modules).
- Structured-knowledge defenses compare single-prompt risk with accumulated-context and release
  risk. They are maturity-labeled scaffolds and do not make raw KV tensors inherently
  interpretable.

These overlays do not add canonical cases, authorize direct deception penalties, or penalize
hidden chain-of-thought. Runtime-changing modes are disabled by default unless explicitly enabled
by their configuration.

## Verification, Evidence, and Provenance

The verification architecture distinguishes:

- **world state**: an observed artifact or environment state;
- **evidence state**: a bounded claim and its supporting references;
- **local verifier**: a verifier that evaluates one evidence or outcome boundary;
- **relational verifier**: a verifier that checks consistency across records or states;
- **failure graph**: typed anomaly, symptom, decisive-failure, and supported-cause relations; and
- **authority decision**: an explicit allow, deny, or review result within a declared scope.

Action-bound events preserve the causal sequence `prediction_commit → action_proposed →
action_executed → outcome_observed → verification_result → epistemic_assessment →
case_assessment`. Parent references must point backward within the same evaluation unit and action
ancestry. Later assessment records may supersede earlier assessments; raw evidence, actions, and
outcomes remain append-only.

The [structured logging contract](docs/structured_logging.md) defines serialization and sequence
validation. The [verification and runtime authority guide](docs/VERIFICATION_AND_RUNTIME_AUTHORITY.md)
defines state, verifier, failure, recovery, and authority semantics.

Telemetry must retain its evidence status. Synthetic token confidence is labeled synthetic and
must not be presented as measured model confidence. Measured tokenizer log probabilities identify
their backend. Unavailable circuit telemetry is `null` with a status, not measured zero. Large
attribution graphs are referenced rather than embedded in ordinary JSONL rows.

## Deception and Interpretability

Deception signals, unverified trace content, circuit features, attribution graphs, and mechanistic
correlations are diagnostic by default. They support logging, review, investigation, and offline
analysis; they do not directly change optimizer fitness without an independently verified and
provenance-bound process component.

Run a dual-path trace and lightweight contrastive baseline:

```bash
gepa dspy run --dual-path \
  --input datasets/dual_path/data.jsonl \
  --trace runs/dual_path_trace.jsonl

gepa dspy contrastive-run \
  --data datasets/dual_path/data.jsonl \
  --out runs/contrastive/ \
  --context general
```

Run the GUI or command-line inspection workflow:

```bash
python -m app.main

python -m mindful_trace_gepa.dual_path_evaluator \
  --scenarios datasets/dual_path/data.jsonl \
  --run runs/001 \
  --response path/to/trusted_response_hook.py
python -m mindful_trace_gepa.dual_path_circuit_tracer runs/001
python tools/merge_run_inspection.py runs/001
```

The `--response` option accepts `module:callable` or a `.py` hook. Those hooks execute Python code
with the operator's process authority. Run only hooks whose source and dependencies you would
execute directly.

Common CLI operations also include:

```bash
gepa dspy compile --out dspy_artifacts/ --enable-optim
gepa score --trace runs/trace.jsonl --out summary.html
gepa deception probes --trace runs/trace.jsonl --config configs/deception/probes_linear.yaml
gepa --help
```

## Training and Runtime

### RL maturity matrix

| Path | Maturity | Verified boundary |
| --- | --- | --- |
| Portable PyTorch CPU PPO/GRPO | Supported | Local automated training, checkpoint, and resume evidence. |
| PyTorch CUDA and distributed | Implemented; hardware unqualified | Mock/CPU contracts; native CUDA/DDP acceptance must run on target hardware. |
| llama.cpp/Vulkan actor | Experimental external runtime | Inference/collection only; native Vulkan/llama.cpp was not run here. |
| Mojo coordinator actor | Experimental external runtime | Operator-supplied configured coordinator only; checked-in source never generates. |
| Mojo/Vulkan/llama.cpp actor + PyTorch learner | Experimental hybrid | Requires `--learner pytorch`; conversion, deployment, and reload remain external. |
| Pure Mojo learner | Unsupported / no-go (3/9 supported) | `--learner mojo` fails closed; see the evidence report. |

The [pure Mojo feasibility report](docs/rl/mojo_learner_feasibility.md) records the nine gates. The
[hybrid guide](docs/rl/README.md#train-with-the-experimental-hybrid-mojovulkan-actor) provides the
exact command, manifest procedure, publication log, and recovery semantics.

External runtimes execute outside the repository's Python trust boundary. Operators must validate
the executable, model artifact, endpoint, configuration, and authorization before connecting a
runtime. `mojo/rl_coordinator/main.mojo` is a non-generating protocol/compile reference. It returns
`actor_unconfigured` for generate requests and is never a training coordinator. Hybrid training
requires an operator-supplied configured coordinator with the same protocol and exact provenance.
The config and Mojo assets require a source checkout and are not included in the wheel.

Native Mojo was not installed or executed on the verification host. MAX was not probed. Native
Vulkan/llama.cpp lanes were unavailable. Those unavailable checks are limitations, not successes.

Run the optional vLLM demo after configuring a reachable endpoint in
`gepa_mindfulness/configs/vllm.yaml` and installing `requests`:

```bash
python -m gepa_mindfulness.examples.vllm_demo.run_vllm_demo
```

Fine-tuning notebooks use bundled data and should be launched from the repository root:

```bash
jupyter notebook notebooks/ft_phi3_mini_unsloth_gepa.ipynb
jupyter notebook notebooks/ft_llama3_8b_unsloth_gepa.ipynb
```

Long-context storage, token logging, retrieval, and viewer controls are documented in
[long-context training and analysis](docs/long_context.md). Global response normalization is
documented in [GRN integration](docs/grn_integration.md).

## Evaluation

The [alignment evaluation battery](docs/ALIGNMENT_EVAL_BATTERY.md) wraps local benchmark rows in
GEPA-aware outcome, calibration, trace-flag, and metadata fields. CI uses deterministic toy
fixtures. Nightly and periodic tiers require operator-supplied local benchmark exports; heavyweight
datasets are not vendored.

Generate a deterministic V5 plan without model execution:

```bash
python -m evaluation.run_v5_framework --dry-run \
  --model-version mindful-model-2026-09-10 \
  --harness-version v5-harness-1.0.0 \
  --output v5_planned_cells.jsonl
```

The planner caps output at 10,000 cells and derives deterministic seeds. `V5EvaluationRecord`
keeps case, robustness, system, epistemics, behavior, outcome, scores, and diagnostics separate.
JSON shape alone is not optimizer evidence: callers validate records against the complete
action-bound event sequence before computing trusted scores or summaries.

Legacy battery examples:

```bash
python -m evaluation.run_alignment_battery \
  --suite simpleqa \
  --dry-run \
  --output-path alignment_battery_results.jsonl

python -m evaluation.run_alignment_battery \
  --suite simpleqa \
  --responses-path tests/fixtures/alignment_battery/calibration_responses_toy.jsonl \
  --output-path alignment_battery_scored.jsonl
```

## Datasets and Synthetic Data

Bundled corpora and generators provide small, inspectable fixtures for abstention, ethical QA,
OOD stress, anti-scheming, dual-path comparison, principled cooperation, and structured synthetic
cases. Synthetic examples are training/evaluation artifacts, not measured deployment evidence.

Validate, summarize, or scaffold a synthetic dataset:

```bash
python scripts/synthetic_dataset_tool.py validate \
  data/synthetic/gold/superalignment_gold_v1.jsonl

python scripts/synthetic_dataset_tool.py summary \
  data/synthetic/gold/superalignment_gold_v1.jsonl

python scripts/synthetic_dataset_tool.py scaffold \
  data/synthetic/templates/new_case.json \
  --case-id syn-new-001
```

Use `data/synthetic/prompts/case_generation_prompt.txt` when authoring new entries. The
[synthetic dataset guide](docs/synthetic_dataset.md) documents the schema and extension workflow.
External corpora must be obtained and governed separately; copyright, license, privacy, and data
rights must be verified per source.

## Repository Layout

```text
README.md                         Project orientation and quick start
pyproject.toml                    Build metadata and optional dependency groups
evaluation/                      V5 registries, plans, records, summaries, and overlays
gepa_mindfulness/                Core scoring, training, verification, and evolution libraries
src/mindful_trace_gepa/          Trace, DSPy, logging, deception, and viewer tooling
modules/                         Specialized safety and robustness modules
configs/                         Runtime and evaluation configuration
datasets/, gepa_datasets/        Bundled evaluation and training corpora
data/synthetic/                  Synthetic schemas, gold cases, examples, and prompts
synthetic_data/                  Lightweight targeted alignment seeds
notebooks/                       Unsloth/PEFT fine-tuning workflows
rubrics/                         Calibration and evaluation rubrics
docs/                            Architecture, operations, recommendations, and research records
tests/                           Unit, contract, integration, packaging, and documentation checks
```

Runnable subsystems provide local READMEs where additional setup or contracts are needed.

## Research Basis and Traceability

The repository separates what a source reports from what this project infers. The canonical
[reference registry](docs/recommendations/references.yaml) stores metadata, source-demonstrated
claims, repository inferences, maturity, and linked recommendations. The readable
[research traceability guide](docs/recommendations/RESEARCH_TRACEABILITY.md) exposes those records
with primary-source links.

The [recommendation registry](docs/recommendations/registry.yaml) is the authoritative status and
dependency inventory for REC-001 through REC-014. The readable
[unified recommendations](docs/recommendations/UNIFIED_RECOMMENDATIONS.md) links each recommendation
to repository evidence, acceptance tests, and REF records. Research results motivate bounded
design choices; they do not transfer a paper's empirical results to this repository.

## Limitations and Research Maturity

- Native CUDA/DDP was not accepted on target hardware during this work; that path remains hardware
  unqualified.
- Native Vulkan, llama.cpp, Mojo, and MAX were not executed. Those unavailable checks are
  limitations, not successes.
- The hybrid path does not claim PEFT-to-GGUF conversion, llama.cpp deployment, or actor reload.
- A pure Mojo learner is unsupported and fails closed.
- External benchmark suites and native model evaluations require operator-provided data, models,
  runtimes, credentials, hardware, and authorization.
- Representation candidate generation is deterministic, dependency-light scaffolding; it is not a
  tokenizer, speech recognizer, learned interpreter, universal spelling repair, or proof of
  intended meaning. See the [foundational representation architecture](docs/FOUNDATIONAL_REPRESENTATION_ARCHITECTURE.md).
- GEPA-Cert and factuality verification are evidence-relative; they are not formal proofs that an
  answer is true.
- Circuit, attribution, deception, trace, and mechanistic signals are diagnostic unless an
  independent verification route authorizes a named process component.
- Experimental overlays for competing hypotheses, topology, orchestration scope, and mechanistic
  audit remain disabled by default. See [experimental V5 overlays](docs/experimental_v5_overlays.md).
- No checked-in record, verifier, planner, or offline-evolution object grants deployment authority
  or performs an irreversible external action by itself.

## Contributing and License

Repository conventions are documented in [AGENTS.md](AGENTS.md) and the
[beads workflow](beads/README.md). Changes to canonical cases, stripes, reward eligibility,
provenance, or recommendation status should update their authoritative registry or contract and
include mechanical consistency tests.

GEPA Mindfulness Superalignment is distributed under the MIT License.
