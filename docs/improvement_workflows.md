# Governed improvement workflows and model–harness attribution

PR 9 adds **Experimental** advisory workflow selection and **Diagnostic** private paired
reports. Both require exact `enabled=True`; disabled calls fail before reading inputs, catalogs
or invoking callbacks. Existing [controlled improvement](controlled_evolution.md), learning
surfaces, coevolution, evaluation epochs and private promotion retain their authority. This
advisory adapter retains those limits. A separate opt-in runtime now dispatches eligible
host workflows and runs JSON-only procedural trials through existing catalog admission.
It does not deploy candidates or automatically launch training. There is no empirical improvement claim.

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
meta-policy training, arbitrary executable harness evolution, deployment and behavioral
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


## Active dispatch and governed procedural execution

[`governed_improvement_runtime.py`](../gepa_mindfulness/governed_improvement_runtime.py)
adds two explicitly enabled host paths. `dispatch_improvement_workflow` reruns the existing
selection/intake and independent diagnosis review, then chooses an installed model, data,
memory, skill or harness handler. A fresh complete-packet host check must return exact True
before that handler runs. Human-review, further-verification and no-justified-change routes
never invoke a mutating handler. Missing handlers withhold execution. The handler receives a
detached selection; its return is an observation, not an accepted model, dataset, reward or
deployment grant. Use the corresponding existing feature's independent admission at each
handoff. The dispatcher does not fine-tune the model for every failure or train a meta-policy.

`run_harness_improvement` implements an actual isolated-process procedural trial. Use an
enabled `HarnessTrialConfig`, original and candidate `HarnessProcedureConfig`, a registered
HARNESS-only correction, supported procedural diagnosis, explicit candidate ID/artifact root,
independent diagnosis reviewer, fresh host authentication and independent candidate evaluator.
The source epoch must be closed, the separate candidate epoch empty/open, the model version
unchanged and the two harness versions distinct/exactly registered. One eligible procedure
changes; protected policy fields cannot be encoded as configuration options.

| Procedure | Closed settings |
| --- | --- |
| Context compaction | `max_tokens`; `preserve_sources=True`; `preserve_action_history=True` |
| Retrieval sequencing | `order`: permutation of the original eligible source-ID pool |
| Tool selection | `preferred_tools`: permutation of the original eligible tool-ID pool |
| Review routing | `preferred_reviewers`: permutation of the original independent reviewer-ID pool |
| Task decomposition | Positive `max_subtasks` up to 128 |
| Reasoning budget | Positive `max_tokens` up to 65536 |
| Recovery | `max_retries` from 0 to 3 |
| Non-authoritative cache | `max_entries` from 0 to 1024; `provenance_required=True` |
| Diagnostic instrumentation | Bounded unique `extra_events`, additional to protected audit |

Preference lists select no new source, tool or verifier. They are advisory configuration for
the installed worker, and never waive existing eligibility/authorization, mandatory review or
verification. Names are IDs, not paths or commands. Unknown parameters, verifier/acceptance
policy edits, disabled provenance or history preservation, and replacement reviewer pools fail
closed. This intentionally supports bounded configuration evolution; arbitrary candidate
Python, harness source rewriting and policy changes remain outside the executable interface.

The operator installs a trusted standalone worker script with an absolute path and SHA-256.
It receives one UTF-8 JSON object on stdin, containing `harness` (the versioned procedure
configuration), pinned `model_version`, public `task={task_id,prompt}` and `seed`. It returns one
bounded JSON object on stdout. The worker loads the operator's configured backend/procedure
and executes it; no external client, model checkpoint or credentials are installed here.
Authentication must independently establish that the script/config actually implement the
pinned backend and procedure. A model-version string or worker hash alone is not competence,
permission or scientific provenance. Installed workers must apply existing runtime grants
before any tool action and record actual PEO effects; stdout is not a verifier finding.

Every call disables the user site (`-s`), removes the current-directory import entry,
uses a new working directory and starts with a minimal controlled environment. PYTHONPATH,
PYTHONHOME and credential variables are not inherited. The configured PYTHONHASHSEED is
honored (unlike `-I`, which ignores it), making string hashes/set ordering repeatable under
the pinned interpreter. The existing interpreter's LD_LIBRARY_PATH is retained when present
for its dynamic libraries. Environment keys and a fingerprint are bound before host admission;
raw values are not logged. The child rechecks the exact bounded script bytes
before executing their immutable snapshot. POSIX CPU/file-size limits and a wall timeout bound
the process; stdout is file-backed and size-limited, stderr is discarded, and the process group
is killed/reaped. Duplicate/malformed/oversized JSON, nonzero exit and timeout are retained as
failed observations. Output must fit both its raw/reencoded byte budget and the enclosing
audit depth; invalid nested output is recorded as failed with no retained payload. Maximum
first-output/task/decision capacity is checked before candidate registration, and each later
call reserves remaining complete-audit capacity before launch. Complete audit/output/decision capacity is reserved before launch, and
pending attempts count before execution. The public task is copied so later callback mutation
cannot rewrite evidence. The evaluator's worker callable expires immediately when evaluation
returns or raises. No catalog/store handle or private labels are passed to the process.

This is process and data separation for **trusted installed worker code**, not a hostile-code
security sandbox: the OS account still has ambient filesystem/network permissions. Configure
an external container/OS sandbox before running arbitrary untrusted executables. The candidate
interface here accepts only closed data and cannot choose an executable. The host enforces
real model/tool spending, model identity, source relevance and any stronger isolation policy.
Non-POSIX execution is Unsupported; no CUDA/distributed qualification is claimed.

The admitted artifact is `harness.json` in a new candidate-ID-hash directory. Existing
`CoevolutionStore.register_candidate` binds its exact bytes and rollback source epoch.
`evaluate(candidate, run_public_task)` supplies the existing independently issued held-out and
protected `ValidationBundle`. The callback retains its store/epoch access on the host; the
worker receives only public tasks. A failed protected outcome cannot receive a passing
validation receipt. Missing/failed worker observations withhold acceptance; changed artifact
or worker bytes, wrong bundles or invalid receipts fail closed. Existing
`decide_candidate_acceptance` alone decides catalog admission. The return contains a live
configuration only when accepted, plus the existing decision. `execute_candidate=False` stays
unchanged. No acceptance decision is consumed for deployment, and no original settings are
mutated. Retain rejected/failed artifacts and start a new candidate epoch/version for a retry.

Full trial audits are HIDDEN_EVAL, contain actual worker evidence and remain host-private.
They cannot be optimizer input or iterative selector feedback. Only existing bounded governed
admission status can cross that boundary. Incomplete filesystem writes/failed evaluations may
leave a retained pending candidate; there is no silent rollback, audit deletion or auto-retry.
The host uses the existing durable recovery and rollback references.

Software checks use `tests/test_governed_improvement_runtime.py` with the existing workflow,
coevolution and attribution suites. Authored CPU worker fixtures demonstrate actual subprocess
execution and authority boundaries; their declared scores are not research measurements.
Research runs are omitted at operator request. The existing private 2×2 protocol above explains
why/how later fixed-model, harness-only, model-only and combined trials separate attribution:
use the same public task/seed roster, fixed independently installed model/harness versions,
matched actual budgets and private verifier captures, then call the existing attribution report.
Do not feed held-out measurements to the workflow selector or call a process exit an improvement.
Learned meta-policy training, arbitrary source-code evolution, deployment and empirical gains
remain unqualified. This is an original governed GEPA adaptation of the already cited sources.
