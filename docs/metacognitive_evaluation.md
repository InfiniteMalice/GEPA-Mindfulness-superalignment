# Metacognitive structural evaluation

`evaluation.metacognitive.evaluate_metacognitive` is disabled by default. An enabled call accepts
an exact tuple of `ChallengeFixture` records and an identical proposal step cap for each arm.
Each fixture carries a canonical `V5EvaluationCell`, a causally captured `RecordedEpisode`, and
separate host-owned expected finding/proposal labels. Labels are read only after policy output.
No label is included in `MonitoringInput` or passed to monitoring or selection.

The canonical 17 cases and 11 stripes are reused as evaluation coordinates; no canonical task
or case definition is changed. Rebinding synthetic records across these coordinates validates
provenance and deterministic repeats. It does **not** apply the actual robustness transformations
to model prompts or establish generalization across the canonical tasks.

## Four conditions

| Condition | Computation | Execution and final behavior |
| --- | --- | --- |
| A | Retain the supplied existing baseline record; no exported new assessment | Baseline supplied record only; new model rollout NOT_RUN |
| B | Derive monitoring from the same recorded prefix | No recommendation exported or action taken |
| C | Monitoring and one bounded repertoire proposal | Proposal only; no action taken |
| D | Replay C's proposal as simulation data | SIMULATION_ONLY; no authorized action dispatch |

These are retrospective comparisons of diagnostic software, not four independent model runs.
The same source bytes, versions, seed, root objective and proposal cap are used within each unit.
Diagnostic computation differs by arm; tokens, runtime cost and matched model compute are
unmeasured. No existing execution or reward interface is extended.

## Challenges and controls

Ten families cover shared unsupported assumptions, contradictory live records, semantic
laundering, unexplained evidence discontinuity, excessive confidence, unnecessary abstention,
high-stakes ambiguity, repeated ineffective methods, false verification and reward pressure.
A healthy negative control has supported premises, independent alternatives and no intervention.
The continuity family uses the existing PEO audit and explicitly makes no claim about motive.
The repetition family uses the existing DeLoop audit; structural repetition is not proof of
failure or grounds for stopping execution.

Semantic laundering tests that changed descriptive language cannot promote an unsupported
record. It is not an automated semantic equivalence evaluator: hosts must retain original
claims and give meaning-changing claims distinct identities. The unnecessary-abstention family
checks that sufficient supported answer evidence does not encourage further abstention. It
does not measure whether a model actually resumes answering. Reward-pressure text does not
change source status, optimizer eligibility or authority.

Eight ablations retain per-unit results: baseline/monitoring, monitoring/recommendations,
shared analysis on/off, continuity on/off, a declared fixed evidence-verification strategy versus
state-guided selection, trusted label admission enabled/disabled, full diagnostics versus a
simple unsupported-claim counter, and intact versus byte-corrupted input. The fixed and simple
baselines are diagnostic comparators with no action execution or measured benefit. Corruption
must fail the protected-capture checks. The label-admission ablation retains the original
verification events and bindings while removing host permission to treat outcome labels as
independent accuracy/progress measurements; it does not score calibration against self-reports.
Additional critiques and answer-label exposure do not
count as improved task behavior.

## Metrics and limitations

Monitoring quality is exact agreement with synthetic finding-kind labels. Strategy quality is
agreement with separately declared acceptable proposal kinds. Unsupported-premise detection,
false-positive premise warnings, dependency detection and contradiction retention are reported
with explicit sample counts. Existing calibration summaries expose Brier score, ECE and selective
accuracy at 0.5 and 0.8 for independently labeled synthetic observations only. Partial coverage
and omitted diagnostics remain visible in assessments.

Verified task success, hallucinations, appropriate/incorrect abstention and clarification,
evidence recovery, repeated-failure reduction and model generalization remain null with reasons:
no independently measured post-intervention behavior is available. Held-out tags protect fixture
admission and label isolation; rebinding the same synthetic construction is not empirical
held-out generalization. No aggregate alignment score is produced.

All exports use the existing `REGRESSION` training restriction. Held-out hypothesis and captured
trajectory records use `HIDDEN_EVAL`; no TRAIN fixture is generated. Expected labels and result
records cannot be admitted to optimization by this API. There is no model download, private
reasoning input, evaluator fallback, network call, reward update or accepted-evidence mutation.

The bundled public PEO templates in `gepa_mindfulness/data/metacognitive_templates.json` are
synthetic adaptations of the existing PEO continuity, epistemic reconciliation and DeLoop test
fixtures. Factories rebuild their typed contracts and pre-action digests for each cell. They are
not empirical observations of a model, and their independent verifier identity is a fixture
declaration rather than authentication of an external service.
