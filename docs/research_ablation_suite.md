# Unified research studies and reproducibility

PR 10 adds **Experimental** study preparation and **Diagnostic** host-private capture/audit
integration. It prepares the master's six study rosters and ordinary existing improvement
protocols. It runs no model, trainer, external service, tool, shell command, scheduler or
deployment. A configured runtime is an operator declaration; qualification remains
`not_assessed`. No empirical improvement or full paper reproduction is established by this code.

All entry points require exact `enabled=True` before accessing inputs or callbacks. The
Constitution, canonical 17-case V5 manifest, case 0 fallback, CASE × STRIPE × REPEAT, protected
verifiers, reward eligibility, human review and runtime authorization retain their existing
owners. Study condition identifiers are overlays, not additional canonical cases.

## Prepare a declared study

`evaluation.research_study_records.STUDIES` holds immutable condition rosters:

| Study | Declared comparison conditions |
| --- | --- |
| A: correction and retention | Baseline fine-tuning; ActionOPD; LoCD; LoCD + ActionOPD; LoCD + SST; LoCD + ActionOPD + SST |
| B: knowledge, progress and diversity | Outcome-only GRPO/DAPO; KnowGPO; PRM4RL; VPO; KnowGPO + PRM4RL; VPO + KnowGPO; VPO + KnowGPO + PRM4RL |
| C: anti-loop behavior | PEO baseline; action ledger; DeLoop; DeLoop + PEO memory; DeLoop + LoCD; DeLoop + PEO memory + LoCD |
| D: adaptive inference | Ordinary bounded routing; generative routing; Jev decisions; compatible native LEV depth |
| E: data and improvement | Fixed data/harness; AutoSFT; RSIFlow; harness evolution; AutoSFT + RSIFlow; AutoSFT + RSIFlow + harness evolution |
| F: evidence access | Structured retrieval; Librarian workspace; optional latent memory; hybrid memory |

Each `StudyRuntime` identifies its condition, availability and reason. `unsupported` and
`not_requested` entries remain in the roster without fabricated system/version records.
`configured` requires a `SystemConfig`, external runtime version/config digest, common source
checkpoint and public source references. This records provenance; it does not establish that an
implementation is faithful, independently qualified, authorized or effective. A configured
comparison requires the declared first-condition baseline. No unavailable method silently
becomes another baseline. All-unconfigured and baseline-only plans remain `not_run`.

`StudyConfiguration` pins the existing `DatasetManifest`, `MetricSpec` records,
`TrustedEvaluatorContract`, `ResamplingPolicy`, evidence references, all seeds/repeats, training
and inference budget digests, initial checkpoint, source training data, protected manifest and
source-code revision. Preserve the actual referenced artifacts; a hash does not authenticate
their contents. The external runtime configuration must retain optimizer, precision, dependency
versions, hyperparameters, data transformations, model/harness/checkpoint/evaluator versions,
curriculum and method-specific controls. Credentials stay outside these records.

`prepare_research_study` validates the existing manifest lineage/partition rules and compiles
ordinary `CandidateSpec`, `EvaluationSlot` and `ImprovementProtocol` records. Cases, metrics,
conditions, seeds and repeats retain explicit identities. Seed belongs to the paired condition
coordinate, while repeat has its own index. Baseline and candidate share the seed and inference
budget digest. Baseline slots are declared separately for each contrast: do not assume their
actual evaluation cost is free or fabricate duplicate billing. The host meters actual spending
and enforces training/inference caps; preparation checks declarations, not a worker process.

Limits are 128 manifest cases, 16 component metrics, 2–16 seeds, 1–8 repeats, at most 4,096
prepared slots and the existing complete-envelope bound. Check the slot count before expanding
the Cartesian product. Split larger studies into preregistered contrast/metric batches with
other arms explicitly `not_requested`; preserve the complete roster and report outstanding
conditions. Keep data, checkpoint, task distributions, relevant budgets and split policy matched
across the declared contrasts. Planned metadata is not a claim that any comparison ran.

```python
from evaluation.research_studies import prepare_research_study

# Typed configuration comes from operator-owned immutable experiment declarations.
plan = prepare_research_study(configuration, enabled=True)
print(plan["availability"], plan["status"])
# Export the existing protocol to an independently authorized host runtime if available.
# Preparation itself neither imports nor calls that runtime.
```

The disabled [operator template](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/main/configs/research_ablation_suite.example.json)
is not an automatic loader or a spending authorization. Model training, paid endpoints,
protected evaluation, deployment and irreversible operations require separately configured
operator authority. The package provides none of those services.

## Capture decomposed observations

`StudyMetricCapture` contains one planned slot, value, unit, status, cohort, consequence stratum,
public evidence, optional binary outcome/reason and measurement applicability (`eligible`).
Allowed statuses are `observed`, `unresolved`, `failed` and `not_run`. Unresolved, failed,
unrun or inapplicable measurements have value `None`, never an invented zero. A verified task
failure may be captured as an observed false outcome with its evidence. An execution failure
without an outcome stays missing, with its failed attempt and actual costs retained in the journal.
Applicability does not grant training or reward eligibility.

Declare each metric batch's eligible case cohort before evaluation. An inapplicable observation
does not satisfy a complete paired roster; do not remove its slot after inspecting outcomes.
The independently protected 17-case suite remains fixed even when a component metric uses a
predeclared subset of opportunities.

`prepare_research_capture` recompiles and compares the exact plan before creating a private
`StudyCaptureEnvelope`. The embedded explicit source schema binds plan/protocol/slot/case/budget,
model/harness/repeat/seed, evaluator/rubric, metric/unit and evidence. The existing audit's closed
source reader validates this schema before authentication; legacy source schemas retain their
existing behavior. Boolean repeat/seed aliases, malformed capture statuses, substituted
references and mismatched provenance fail closed.

Metric captures, envelopes and private reports are `HIDDEN_EVAL`, `optimizer_input=False` and
confer no authority. The shared training gate rejects their retained dictionaries, including
an outer `TRAIN` label around a restricted source. Preserve the envelope: its internal legacy
`DiagnosticEvidence` stores source JSON as text and stripping the outer label loses that guard.
Only the host adapter unwraps it for existing audit operations. Hashing or labeling a capture
does not verify an outcome. Do not admit hidden test data or diagnostics for training.

| Metric family | Distinct observable measures to declare and retain |
| --- | --- |
| Capability | Independently verified task success, OOD transfer, generalization, best-of-N; retain N and opportunity denominators. |
| Epistemics | Correctness, calibrated probabilities with independent outcomes, appropriate/false abstention, source fidelity, verified knowledge transitions. |
| Alignment | Protected test regressions, authorization/autonomy outcomes, observable non-deceptive/non-coercive behavior, semantic-intent invariance and correction acceptance. |
| Efficiency | Actual tokens, latency, tool repetitions, measured/estimated training FLOPs and verified progress per intervention; unknown cost stays unknown. |
| Continual learning | Prior/new capability, KL drift, applied spectral-update diagnostics and cross-domain interference. |
| Governance | Unauthorized proposal disposition, protected evidence integrity, rollback-reference checks and human-review routing. |

Use separate `MetricSpec` identities, directions, units and rubric digests; no opaque composite
score or optimizer reward is computed. Existing audit transform/calibration semantics apply.
For probability calibration retain the independently observed outcome and calibration artifact;
agreement with a teacher or deeper prediction is a different diagnostic. Self-reported honesty,
fluent explanations, hidden reasoning, model agreement and suspicious text spans alone cannot
establish factual, causal or moral claims. Uncertain moral judgments remain unresolved.

Measurements can refer to the same evidence. That does not create additional optimizer credit;
any future reward admission must still use the existing [credit accounting](research_credit_accounting.md)
and verified eligibility boundary. No teacher/knowledge/progress/process credit is paid here.

## Independent audit, uncertainty and private tests

`audit_research_study` unwraps validated envelopes only on the host and delegates to
`audit_improvement`. It requires the existing attempt journal, exposure history and event-specific
final-test authorizations. The existing host authentication contract must independently verify
the exact protocol, sources, slot bindings, provenance, complete attempts/exposure history,
evaluator independence and final permission. A callable returning true is only a software fixture
unless the host actually performs those checks. Callbacks are not a malicious-Python sandbox.

The adapter reserves complete output capacity before authentication. Oversize inputs are rejected
before callbacks; batching must be explicit. Missing, failed, cancelled or unauthorized outcomes
remain visible in planned denominators and cannot make a complete improvement comparison.
Full paired-audit output, raw component values, verified evidence, failure distributions,
severe events, costs and independence reasons remain separate. Private numbers never feed back
into the diagnosis selector or data generator. Existing private promotion remains the admission
interface; this report issues no validation receipt or acceptance/deployment decision.

A plan within the slot cap may need smaller audit batches once source and exposure-history sizes
are known. Preregister that batching and retain all failures; capacity checks are not permission
to discard unfavorable observations.

Uncertainty reuses `estimate_paired`: average repeats and seed conditions within cases, then
cases within declared dependency clusters, and bootstrap clusters. Repeats and shared-case seeds
do not increase the number of independent task families. Insufficient clusters, incomplete pairs
or unknown dependencies suppress intervals. Selection and independently held-out estimates use
disjoint clusters; final results additionally require their pinned authorization. Intervals are
conditional on fixed configurations and are not simultaneous or post-selection guarantees.

```python
from evaluation.research_study_capture import prepare_research_capture, audit_research_study

private_capture = prepare_research_capture(plan, observed_component, enabled=True)
report = audit_research_study(
    plan, complete_journal, private_captures,
    exposures=complete_exposure_history, final_authorizations=host_final_authorizations,
    authenticate=independent_host_authenticate, enabled=True,
)
```

## Controls and reproducibility protocol

Study A additionally separates conventional SFT, off-policy and on-policy context distillation,
localized correction and semantic-action losses. Declare both alignment/reasoning/coding orders,
SST/no-SST skill acquisition and LoCD+SST at matched data/compute. Record every stage checkpoint
and existing epoch lineage. Compare endpoints from the same initialization; do not describe
different intermediate parent states as identical. Evaluate all protected epistemic/constitutional
behavior, calibration/abstention, task completion, general skills, KL drift and necessary corrections.

Study B keeps constitutional admissibility independent of objective weights. Validate eligible
knowledge/progress contributions separately before combinations; use existing overlap accounting.
Study C includes useful retries, changed observations, answered clarifications, duplicate blocking,
DPO/verified GRPO controls and premature termination. Avoid optimizing repetition rate alone.
Study D requires a suitable native loop backbone for LEV; a conventional repeated model remains
Unsupported as a faithful LEV backend. Report error/escalation recall separately from latency
and deeper-answer agreement. Study E retains rejected/no-change interventions and independent
dataset/procedure admission. Study F preserves raw recoverable sources and leaves unimplemented
memory backends explicitly unsupported. No AGG geometry, SSOH architecture or quantum code is imported.

For every major intervention also use the separate [model–harness 2×2 report](improvement_workflows.md#paired-modelharness-attribution):
original/original, original/candidate, updated/original and updated/candidate. Hold tasks and
relevant caps fixed and report model, harness and interaction effects. No family average can
cancel a protected failure. Existing human governance decides promotion after protected checks.

Archive configuration/source/environment hashes, raw immutable datasets, split/lineage manifests,
checkpoints/model/harness/evaluator versions, all seeds, method controls, dependency-cluster policy,
actual costs, failed runs, exposure history, review/authorization records and raw/verified reports.
Never edit held-out cases during optimization. Run new suites with `python -m pytest -q
tests/test_research_studies.py tests/test_research_study_capture.py`, then related improvement
audit/statistics and documentation/packaging checks. CPU fixtures test software contracts only;
`scripts/validate_ablation.py` contains placeholder zero measurements and is not an empirical backend.

Method definitions and adaptation limits remain in the existing [source matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md)
and [traceability](recommendations/RESEARCH_TRACEABILITY.md). This suite is original research
coordination software over those adapters, not a reproduction of all methods. Unrun training,
native/optional backends and empirical effectiveness remain **Unsupported**.

The [MEMORILLA source/runtime qualification protocol](latent_memory_research.md) supplies the
optional latent-memory intake boundary for Study F. Source qualification adds no runtime;
unimplemented latent and hybrid conditions remain explicitly `unsupported` in the existing roster.

## MDAC curriculum and RL comparison

Run `python -m evaluation.mdac_comparison_fixture --output /tmp/mdac-study.json --report-dir /tmp/mdac-study-report` for a CPU comparison using a deterministic table learner. Six arms, two paired seeds and five private evaluator cases produce 12 cells. Each receives 12 practice examples and executes 12 table updates. The grader includes an unlabeled case, a known case with no planted mechanism, and a partly incorrect mechanism classifier. The recorded effects describe this fixture.

| Arm | Sampling rule |
| --- | --- |
| A | Cycle through the frozen fixed catalog. |
| B | Draw uniformly with the same sample and compute budget. |
| C | Weight each practice lesson by its model-attributed development failure rate. |
| D | Pool failures and opportunities within the declared behavioral category. |
| E | Weight by failure rate times positive paired diagnostic recovery score from a retained calibration intervention. |
| F | Apply the PR4 allocator with target shortfall, directional decline, mastery retention, severe-case reserves, exploration, generalization, baseline maintenance, and correlation caps. |

Each lesson belongs to a training family. Its diagnostic and intervention sources belong to their dedicated partitions. F requests separately authored practice prompts for diagnostic exploration and novel generalization. The host must review these variants as members of the indicated training family. Zero total weight in C, D or E falls back to the fixed catalog. Nested labels, curricular reasons, private references, and evaluation prompts stay outside the trainer request. Only the independent grading callback receives the private case record. Host code is trusted and must isolate real training and evaluation processes. Keep final and OOD judgments out of subsequent model selection.

Feed `Study(**settings)` the JSON configuration after converting the `seeds` and `methods` lists to tuples. Replace the illustrative model configuration digest with the digest of the actual serialized configuration and replace the checkpoint path with an immutable local snapshot and supply the frozen PR16 family manifest. The proposed single-GPU configuration is unexecuted. It requires reviewed eligible data, calibrated independent judging, a resource meter, installed backend dependencies, and a fresh starting checkpoint for every cell. One compute unit has the same physical meaning throughout a comparison. The host rejects conditional comparisons when an arm cannot match sample, generation, training and evaluation constraints.

Implement the training callback using `rl_training_config` with the existing PPO or GRPO `RLTrainingEngine`, or `dapo_training_config` with the existing optional DAPO hybrid trainer. Materialize only admitted examples corresponding to the selected prompts, seed every backend from the request, and keep outputs in separate directories. Return `TrainingRun` with actual metered costs, the exact request digest, checkpoint identities and an inference callback. The runner rejects a mismatched receipt, a different model configuration digest, or actual training tokens different from the frozen budget. Each inference callback receives the same per-case token limit and must return a metered completion. Real access-controlled evaluations need an independent host to authenticate these receipts, grade predictions and authorize final-test use. The CPU fixture does not exercise an RL update.

Results retain opportunity counts for diagnosis precision, recall, exact multilabel agreement, calibration of exact-set confidence, false M15 attribution, unknowns, held-out alignment, OOD alignment, correction, honest disclosure, and reward–verified behavior divergence. Repair latency needs a verified recovery and an observed step count. Absent measurements remain unavailable. Stability compares diagnosis sets for the same case across seeds. Each seed and method has paired alignment and capability deltas relative to A, plus gains per sample and compute unit. Per-mechanism regression counts reveal alignment loss in other annotated categories. Report DRS from the intervention-calibration stratum independently of held-out curriculum effects.

The comparisons test whether mechanism-directed sampling improves alignment and control of forgetting under fixed resources. Additional studies should freeze an independent contrast for verified versus unverified repair, experience reuse, clarification coverage, ethical transfer, or mechanistic memorization indicators before revealing final judgments. The toy table learner supplies no evidence on those model-backed questions or on the relative merits of PPO, GRPO, and DAPO.

For teacherless PPO/GRPO, use the default canonical `runtime_options="{}"`. To freeze an existing teacher-assisted GRPO setup, provide canonical JSON containing the host's `actionopd` and `research_credit` sections, including teacher and verifier versions and contract IDs. To freeze the existing Mojo/Vulkan GRPO hybrid, supply its `backend` and `hybrid` sections. The canonical engine validates the combination and the comparison binds it to every request digest. DAPO uses its distinct GEPA feedback adapter. No teacher service or hybrid accelerator was exercised in the table experiment.

### Matched CPU comparison of PPO and GRPO

Run the optional PyTorch control from the checkout.

```bash
python -m evaluation.mdac_comparison_fixture --cpu-rl \
  --output rl-controls.json --report-dir rl-controls
```

The control runs arms A–F on two seeds for each teacherless method. Each cell starts from
the same zero-initialized categorical policy and critic. It draws four one-action rollouts
for each of twelve scheduled prompts, takes one SGD step per prompt and uses the same
five independently graded probes. Training uses the repository's PPO and GRPO losses.
For PPO the episodes terminate after one action, so the return is the observed reward.
GRPO normalizes the sampled rewards within each prompt group. One update per rollout
batch exercises the on-policy objective. It gives no evidence about the effect of clipping
over multiple optimization epochs. Optimizer steps and action tokens measure the budget.
Wall-clock time and floating-point operation counts can differ between methods.

The toy task tests selecting the requested operation or abstaining. The model can observe
the operation word. The training signal is one for the registered action and zero otherwise.
The evaluator scores exact task success and avoidance of a wrong assertion separately.
The diagnostic development inputs are authored fixtures. Private cases carry no latent
mechanism annotation. The report marks these synthetic controls illustrative and leaves
unsupported measurements unavailable. They test comparative infrastructure and weight
updates. The configuration for a language-model study with DAPO and GEPA feedback still
requires its separate backend, data and compute budget.

## Alignment-first lifecycle

`evaluation.mdac_alignment_first.run_lifecycle` starts from one declared policy, evaluates an independent alignment, ethical generalization, and capability profile, then trains three stages. Each order gets the same first two stages' total exposures. Alignment-first starts with four constitutional and reasoning examples per unit, followed by three capability examples and one alignment example. Capability-first reverses those stages. Mixed training distributes the same five alignment and three capability examples across both stages. Every stage ends with an independent curriculum-validation profile.

Development diagnosis runs a matched control, repair, positive control, and sham against the current checkpoint through the PR3 replay runner. The fixture uses a separately created development-calibration ledger family. Its evaluator grades actual public actions. A failed control or inconclusive replay sends the attribution to exploration. PR4 allocates the corrective round. One quarter of this round always replays reviewed alignment anchors. Before a mechanism-specific slot reaches training, the PR5 verifier replays an alternative decision and a retained control through the continuation horizon. Only the verified actor input and target pass the training eligibility check. The host provides the private admission store and independent evidence to PR11. The final and OOD probes run after the advisory decision and remain excluded from training. The tutorial lacks the twelve-dimension private study required for promotion, so the incumbent stays active.

Run the three CPU fixtures with `python -m pytest -q -s tests/test_mdac_alignment_first.py -k cpu_cycle`. To execute the tiny model path without a download, create a fresh local checkpoint with `python -m gepa_mindfulness.examples.mdac_small_model --initialize /tmp/mdac-toy-model`, then run `MDAC_LOCAL_TEST_MODEL=/tmp/mdac-toy-model OMP_NUM_THREADS=1 python -m pytest -q -s tests/test_mdac_alignment_first.py -k local_checkpoint`. The test uses PyTorch, Transformers, a local SQLite admission catalog and the repository GRPO objective. It trains real model weights, hashes and saves each checkpoint, and reloads it for the next stage. Its two permitted action tokens and four sampled responses per update make costs directly measurable. It performs 12 optimizer steps, using 48 sampled action tokens. Positive rewards come from the reviewed training target. The synthetic checkpoint has no established language competence. These tests verify the training, measurement, repair, retention and governance interfaces.

The single-GPU study template is `configs/mdac/alignment-first.json`. Replace the illustrative snapshot path and configuration digest with the actual local coherent base model before use. Supply reviewed full-text ethical/reasoning and capability training data, an appropriate actor, and independent evaluators. The single-GPU template and teacher-assisted studies remain unexecuted. The one-token CPU smoke uses teacherless GRPO. Other study runtimes carry their pinned settings through the PR15 training request. The measured toy controls do not estimate long-run ordering effects or frontier-system behavior.

To save the three full native reports, split manifests, PR16 JSONL, summaries and figures from the repository root, run `PYTHONPATH=. python tests/mdac_alignment_fixture.py --report-dir /tmp/mdac-orders`. Add `--model /tmp/mdac-toy-model` to run all three orders with trained local GPT-2 weights, or `--no-plots` for a lightweight export. Use a fresh output directory. This repository fixture uses the same disposable private catalogs as the admission integration tests. Independent run and trial provenance remain in each native report.

The ledger intervention measures sensitivity on its own calibration family. The separate lossy-compression development case runs its own preregistered replay before diagnosis. The case records the missing ledger fee directly from the public prompt. Retrieval and internal use remain unmeasured. The table policy recovers when the fee is restored, while the tiny policy remains inconclusive. A supported development claim requires a separately registered case-specific treatment, matched initial public input and response, independent evaluator, and positive observed paired effect. Repair references bind the current weight digest, family, actor input and supervision target to the complete independently authenticated trial roster. The tiny model performs the training and replay phases. Its repair-continuation checks use a separately labeled deterministic simulator.
