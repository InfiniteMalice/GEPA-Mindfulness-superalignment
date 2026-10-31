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
