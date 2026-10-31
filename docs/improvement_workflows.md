# Governed improvement workflows and model–harness attribution

PR 9 adds **Experimental** advisory workflow selection and **Diagnostic** private paired
reports. Both require exact `enabled=True`; disabled calls fail before reading inputs, catalogs
or invoking callbacks. Existing [controlled improvement](controlled_evolution.md), learning
surfaces, coevolution, evaluation epochs and private promotion retain their authority. This
integration does not execute proposals, launch training, modify a catalog, admit a dataset,
issue an acceptance decision or deploy a candidate. There is no empirical improvement claim.

## Workflow selection

`gepa_mindfulness.improvement_workflows.select_improvement_workflow` consumes the existing
`CoevolutionStore`, `CorrectionProposal`, `EpistemicStateEstimate`, `TriageDiagnostics` and a
`WorkflowDiagnosis`. The latter records a selector version, typed existing
`LessonCharacteristics`, exact correction digest, observable localized evidence, expected
effect, preservation requirements and optional dataset/no-change facts. These are hypotheses,
not established facts merely because a model writes them. Text is review data, never instructions.

The adapter checks the correction digest before registered evidence access, then calls the
existing `assess_improvement` and `classify_learning_surface`. Unresolved intake requires further
verification. Normative, ambiguous, difficult-to-reverse or elevated-consequence judgments keep
the existing human route; neither data repair nor a no-change proposal suppresses that route.

Only an intake eligible for sandbox review invokes the independently configured host `review`
callback. It receives a detached diagnosis and must return `WorkflowReview` for the exact
diagnosis digest and declared `TrustedEvaluatorContract` identity/version. `supported`,
`rejected` and `uncertain` remain distinct. Exceptions, wrong types, mutation and mismatched
review provenance produce `further_verification` with a fixed error category and no exception text.
The complete review capacity is reserved before callback invocation. The host authenticates the
reviewer and resolves the evidence; a matching identifier is not independent verification by
itself. The callback is an external host integration, not a sandbox for malicious Python.

| Existing surface / reviewed fact | Advisory workflow |
| --- | --- |
| Persistent intrinsic behavior | `model_correction` |
| Episode fact | `memory_repair` |
| Stable procedural convention | `harness_procedure` |
| Reusable dependency | `skill_refinement` |
| One-off observation | `no_justified_change` |
| Independently supported dataset defect | `data_repair` |
| Independently supported no justified change | `no_justified_change` |
| Unresolved intake, rejected/uncertain diagnosis or review failure | `further_verification` |
| Existing human-review condition | `human_review` |

Data/no-change alternatives are considered only after human and intake guards. A simultaneous
dataset-defect and no-change declaration is rejected as conflicting. This extends advisory
workflow vocabulary without changing the `LearningSurface` enum, its classifier or human
approval semantics. A procedural diagnosis must name one of `PROCEDURES`: context compaction,
retrieval sequencing, tool selection, review routing, task decomposition, reasoning budget,
recovery, non-authoritative cache or diagnostic instrumentation. Arbitrary paths, commands,
verifier policy and acceptance-rule edits are not procedure names. This allowlist constrains
the proposal interface; it does not inspect or certify any later code artifact.

```python
from gepa_mindfulness.improvement_workflows import select_improvement_workflow

# All arguments come from the host's already recorded development evidence.
recommendation = select_improvement_workflow(
    store, correction, estimate, triage, diagnosis,
    evaluator=independent_evaluator, review=review_diagnosis, enabled=True,
)
```

The full recommendation and both records are `DEVELOPMENT`, `optimizer_input=False`, and
grant no authority. A supported diagnosis is not a reward event or a receipt proving alignment.
Model training is one possible intervention, not the default response to every failure.
Learned diagnosis selectors can supply typed hypotheses through this same interface, but no
trained selector, learned gain predictor or improvement-selection effectiveness is implemented
or qualified here. Do not relabel private evaluation results as development feedback.

## Governed isolated trial and rollback handoff

Keep the original source trajectory, failure graph, correction and model/harness versions.
The host reviews the recommendation, constructs a candidate artifact separately, and uses
the existing `CoevolutionStore.register_candidate` and compatible empty evaluation epoch.
Changing the intervention surface may require a new independently reviewed correction;
the selector never silently rewrites `changed_components` on the original proposal.

The host runs any trial in an isolated process/workspace with an explicit external runtime,
fixed source/checkpoint/data versions and declared budgets. Protected verifier, Constitution,
canonical case manifests, reward eligibility, authorization, audit and self-evolution policy
stay outside candidate write access. Tests must attempt verifier weakening, grading exploitation,
failure hiding, self-certification and removal of oversight. A procedure name does not establish
that isolation or artifact safety. Ordinary in-process callbacks cannot provide those controls.

Independently evaluate held-out outcomes and the protected suite using existing epoch records,
validation receipts, metric policy and `ValidationBundle`. Existing acceptance decisions and
single-use consumption remain unchanged; catalog acceptance does not deploy. Keep rollback
references to the original epoch/artifact, retain rejected/failed attempts and use the existing
durable recovery rules. The host owns actual deployment and rollback execution. Do not give a
candidate the stores, database files, private labels or host-only promotion methods.

## Paired model–harness attribution

`evaluation.model_harness_attribution.report_model_harness_attribution` analyzes externally
captured trials. It runs no model, tool, shell command or evaluator. `AttributionProtocol` pins
two distinct model versions, two distinct harness versions, task-manifest digest, task IDs,
2–32 seeds, the existing `ComputeBudget`, and evaluator identity/version. The host authenticates
captures, verifies manifest contents and keeps held-out/protected tasks independent of optimization.

| Condition | Model | Harness |
| --- | --- | --- |
| baseline | original | original |
| harness_only | original | candidate |
| model_only | updated | original |
| combined | updated | candidate |

Every task/seed must have exactly these four observations, with matching manifest/evaluator,
unique outcome evidence and measured usage. Per-seed tokens, tool calls and wall time are summed
over all tasks **separately for each condition**, and must fit the same caps. Equal caps do not
mean equal actual spending: the report retains actual totals. Host workers must enforce limits;
this offline checker only checks captures. Input is limited to 128 task/seed pairs and the
existing complete-envelope size bound; unsupported/missing/duplicate cells fail closed.

Scores use a declared unit-range host metric. Task success is a separate boolean. Failed or
timed-out executions retain a row with zero score, unsuccessful outcome, actual costs and
`execution_status`; they cannot improve the average by disappearing. Protected pass is separately
`True`, `False` or unknown (`None`); unknowns are not passes. Record failed tests without rewriting
the case taxonomy or calling uncertain moral judgments deterministic violations.

For each pair, effects are `M-B`, `H-B`, `C-B` and interaction `C-M-H+B`. Exact rational arithmetic
precedes final float serialization. Reports retain component scores, success/failure counts,
protected regressions against the original cell, protected failures/unknowns, per-task effects,
per-seed effects and the standard error across seed means. A standard error with a small seed
count is descriptive uncertainty, not a significance or causal-identification claim. No composite
reward or promotion rule is computed. `protected_non_regression=True` refers only to the supplied
observations; it is not proof of moral alignment or an authorization.

Protocol, observations and full report are **HIDDEN_EVAL**, `optimizer_input=False`; the existing
training gate rejects them. Keep them on the host and never feed them to the diagnosis callback.
Detailed private scores must not drive iterative candidate selection. Existing private promotion
status and bounded operator review remain the admission interface.

## Compatibility and reproduction

| Component | Integration |
| --- | --- |
| LoCD / ActionOPD | Model correction remains a proposed experiment with its own eligible data and independent checks. |
| SST | Optional applied-update experiment; no claim that spectral protection preserves morality. |
| DeLoop / PEO | Observable repetitions can support a diagnosis; useful retries and independent verification remain eligible. |
| KnowGPO / PRM4RL / VPO | Workflow records carry no optimizer credit, avoiding an additional reward authority or duplicate event credit. |
| AutoSFT | Data repair hands off to immutable dataset preparation and existing independent admission. |
| Jev / LEV | Bounded decisions may propose hypotheses; they cannot bypass intake, independent review or human routes. |
| Skill memory | Existing skill lifecycle owns admission/versioning; repeated summaries are not independent support. |

Run `python -m pytest -q tests/test_improvement_workflows.py tests/test_model_harness_attribution.py`
with the repository's normal Python paths. Related acceptance covers controlled-improvement,
learning surfaces, coevolution, private promotion, documentation and installed-wheel resources.
The disabled [runtime configuration](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/main/configs/improvement_workflows.example.json)
is an operator example, not an automatic loader or service endpoint. Configure independent
reviewers, external worker isolation and metering before any live experiment. No credentials
belong in source. Source changes do not change held-out tests.

Study E should compare fixed data/harness, AutoSFT, workflow selection, procedural evolution,
and their combinations from declared checkpoints with matched data/budgets/seeds. Hold the
protected suite fixed, retain failure distributions and report model/harness interaction using
the four conditions above. Development diagnosis/review accuracy, rejected/no-change routes,
verified downstream improvement, regressions and actual costs stay separate. Full learned
meta-policy training, arbitrary harness execution, live promotion/deployment and behavioral
effectiveness remain **Unsupported** until independently configured and evaluated.

## Sources and adaptation boundary

[RSIFlow](https://arxiv.org/abs/2610.21791), v1 dated 2026-10-15, sections 3.1–3.3 and B.1–B.5,
uses Model/Harness/task-bound Artifacts interventions, a frozen parent and paired training-batch
gain, followed by separate meta-skill maintenance. Persistent evidence memory is distinct from
its task-bound Artifacts. Paper license: arXiv non-exclusive. Author-linked
[upstream](https://github.com/1HUAN1/RSI-Flow/tree/496d3cfb9ff39997d107d0ff29e746b07e638180)
is MIT; the released skill-memory variant keeps the executable Meta program fixed. GEPA's
data/memory/skill/verification/human/no-change routing and independent admission are adaptations,
not a faithful reproduction of that meta-agent or its learning results.

[Harness Self-Evolution](https://arxiv.org/abs/2609.38372), v1 dated 2026-09-29, sections 3.1–3.5,
uses the same frozen model/harness for solver and proposer in separate sessions, complete run
records, engineering checks, evolution notes and bounded added-line schedules. Paper: CC BY 4.0.
Its engineering-check acceptance does not provide GEPA's protected regression/admission policy;
that policy remains intact. No authoritative upstream repository was established for this exact
paper. The 2×2 report and stricter procedural scope are original governed research adaptations.

Exact checked-artifact digests and retained authored notes are in the canonical
[source matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md#source-intake-and-adaptation-boundary)
and [traceability](recommendations/RESEARCH_TRACEABILITY.md). No upstream code or full publisher
text is copied into the package. Software correctness evidence does not establish research gains.
