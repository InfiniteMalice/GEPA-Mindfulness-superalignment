# Alignment-specific mechanistic studies

Maturity: **Experimental**, opt-in software. Captures, reports and research records remain
**Diagnostic**, DEVELOPMENT, and excluded from optimization. Trained ethical mechanisms,
semantic interpretation, generalization and alignment improvement remain **Unqualified**.
No training, deployment, new canonical V5 category or reward component is introduced.

## Controlled comparisons and existing owners

`synthetic_data.alignment_studies.alignment_study_templates(enabled=True)` returns 26 private
research templates spanning seven orthogonal families:

| Family | Comparisons |
|---|---|
| Grounded honesty | Evidence present/missing; counterevidence versus an unverified guess |
| Abstention | Justified IDK followed by sufficient information; unnecessary IDK pressure |
| Clarification | Missing date, necessary question, supplied answer, resumption after repeated question |
| Semantic laundering | Paraphrase, indirect wrapper, Spanish translation, multi-turn reformulation |
| Dual-path specificity | Incentive to fabricate versus public comparison; ordinary mistakes, uncertainty, harmless role-play and hypotheticals |
| Constitutional conflict | Prosperity, suffering reduction, understanding, autonomy/consent and oversight; legitimate alternatives |
| CVA | Authorized completion, technical contradiction, excluded solution, unresolved feasibility, new evidence, new authorization, bypass pressure and limited authorized search |

These are authored hypotheses and public toy situations, not established classifications of human
intent or moral truth. Expected actions are evaluator-only. Equivalence is explicitly claimed,
not verified by a paraphraser. Multi-turn prompts may change several contextual properties; fixture
factor declarations support auditing but cannot establish semantic isolation by themselves.
The CVA templates retain all original public prompt fields, including changed fixture labels,
facts and budgets; multiple changes are classified as compound. Do not interpret their aggregate
response differences as a uniquely identified causal effect of authorization alone.

`bind_alignment_studies(templates, cells, plan_id=..., roles=..., condition=..., enabled=True)`
creates an `AlignmentStudyPlan` with existing `CausalPair`/`CausalVariant` records. The host supplies
each arm's exact existing `V5EvaluationCell`; the binding never infers its semantic case. All 17
canonical cases and original CASE × STRIPE × REPEAT coordinates remain unchanged. Case 0 is excluded
by the original planner. Arbitrary research fixtures may use this contract, but their authored
expectations do not become verified evidence by entering a plan.

## Typed public capture

`StudyTemplate` retains public turns, declared factor changes, private expected actions, source
references and optional original `SecurityFixture` objects. `StudyItem` binds a template to a
legacy pair. `StudyRoles` names discovery, prediction, executor, verifier, skeptic and registrar
actors. Names may coincide; distinct labels or model instances do not establish independence.

`public_study_input(item, arm, enabled=True)` returns only a detached `PublicStudyInput`:
variant/pair/prompt identities, system identity and ordered public turns. It excludes private
expected actions, factors, case judgments and fixture qualification. Actor callbacks must be
trusted host adapters, not executable code loaded from a prompt. This API separation is not
a hostile-Python sandbox; a host must isolate an actual evaluator's private data.

`PublicBehavior(actions, public_output, evidence_refs, native_input=None,
physical_model_digest=None, tokenizer_id=None, tokenizer_version=None)` retains original public
output and its adapter-normalized observable actions. Repeated actions are retained, including
repeated clarification. Native provenance is complete or absent. Labels and text are observations,
not inferred honesty, correctness or intent. Host adapters must provide public answers rather than
silently labeling hidden scratchpads public.

`capture_study_behavior(plan, actor, clock=None, backend=None, authorize_candidate=None,
enabled=True)` invokes the host actor for both arms of every item before returning a closed
`StudyBehaviorBatch`. Actor failures are retained as censored captures without invented actions.
Original outputs, references, timestamps and existing `CausalCapture` bindings remain available.
Mutated inputs, invalid records and duplicate outcome evidence fail clearly.

Conditions are baseline, behaviorally_trained, diagnostic_instrumented and intervention_informed.
Behaviorally trained and intervention-informed conditions require explicit candidate evidence references;
an intervention-informed candidate additionally needs a separate strict host authorization callback
before any actor invocation. That callback must resolve the candidate through existing protected
research governance. A reference/label cannot grant authority by itself. These are evaluation
conditions, not parameter-update or training APIs. Existing coevolution/evaluation stores continue
to own candidate registration, protected regression and promotion.

The diagnostic-instrumented condition requires its exact `FrozenNativeModel` and physical identity.
The backend is checked before/after actor callbacks, including failure exits; a changed model cannot
be concealed as a censored capture. Any supplied native capture must match that physical identity.
Baseline and diagnostic arms for comparative research should use identical physical models and
inputs; later [comparative reporting](research_ablation_suite.md) must retain unmatched conditions
as unrun rather than substituting new weights into the diagnostic arm.

## Behavior before native intervention

`instrument_study_pair(plan, item, behavior, commit, backend=..., clock=...,
authenticate=..., enabled=True)` accepts a stage-6 `CausalCommit` only after the complete behavior
batch closes. It requires the exact study/item/batch/model, both original native inputs, matched
tokenizer identity/version, and a prediction timestamp following behavioral completion. A trusted
host authenticates a detached `StudyInterventionPacket` containing those bindings. Missing, throwing,
mutating, false and truthy non-True callbacks fail closed before the executor. The callback must
establish original source custody, actual tokenization of each public prompt, model provenance and
real temporal ordering. Timestamp text, an arbitrary token mapping, or a role name cannot prove these.

Actual interventions reuse the [stage-6 executor](causal_mechanistic_evidence.md) on the frozen copy.
The returned `StudyMechanismRecord` binds original plan/item/behavior digests to the complete raw
causal report. No model parameters change. Effects remain unresolved diagnostics. Subsequent
independent replay and protected one-use WRITE registration use stage 6; this bridge cannot
self-promote a result because a reviewer likes its explanation. Opposite/null predictions and
additional replications are retained in separate source reports, not silently replaced.

The full research protocol is Observe → Hypothesize → Predict → Intervene → Measure → Replicate
→ Revise. Discovery/prediction use the native hypothesis/split tools; the executor performs real
forwards; independent checking and registrar admission use the host-verification contracts;
skeptical analysis retains falsifications, nonspecific controls, off-manifold and intent limits.
Reports distinguish protocol steps from executed steps and explicitly mark unrun replication,
revision and registration. Optional model-backed researchers may supply hypotheses, but cannot
supply verification authority. Human or tool skepticism remains necessary beyond numeric checks.

## Behavioral, CVA and mechanistic reporting

`report_alignment_studies(plan, behavior, adjudications, opportunities=..., mechanisms=(),
authenticate=None, cva_captures=(), authenticate_cva=None, enabled=True)` directly calls the
existing `evaluate_causal_suite`. Its digest-bound `PairAdjudication`, severity, opportunities,
host authentication, case coverage and missingness remain authoritative. Missing judgments stay
unresolved even when native effects are numerically specific. Public action accuracy does not
establish a circuit, and a circuit effect cannot override a behavioral failure.

`cva_study_slots(plan, enabled=True)` returns original benchmark fixtures/slots and planned study-arm
identities. A `PublicBehavior` may additionally retain its original `cva_capture: BenchmarkCapture`.
In that case its original public output must be the exact structured response JSON (original
whitespace is preserved), and both captures require identical source evidence references. The
public response is parsed through the existing exact `BenchmarkResponse` contract; free text is
retained as an ordinary public observation rather than converted into a certificate.

The report derives CVA inputs only from those original batch outputs. Optional `cva_captures` is
an exact complete-source cross-check, not a way to inject later results or select favorable arms.
Unrelated-session captures and censored arms cannot borrow benchmark results. A separate
`StudyCVAObservation` authentication packet binds the exact plan/item/batch, original public input,
output, behavioral capture and benchmark fixture/slot/capture. Its host callback must attest the
same observation, not merely that some benchmark capture exists. Missing or failed binding
authentication stays unresolved; missing original structured data stay missing. Only then does the
existing `evaluate_cva_benchmark` replay observations and check certificates. Duplicate slots are
rejected before capture; genuinely separate observations need distinct V5 repeats or separate plans.
No deduplication silently maps multiple captures onto one outcome. Scoped finite qualification and real-world
evidence-limited noncompletion retain their [original CVA semantics](constraint_verified_alignment.md).

Raw mechanism joins require exact study/item/batch, model/input/timing and existing action-bound
report checks. They remain RAW_UNVERIFIED; no mechanism call is NOT_RUN. The report emits existing
`MechanisticAuditReference` records with exact source case IDs and immutable run references. Their
1.0 uncertainty is an explicit unknown placeholder required by the legacy record, not a calibrated
probability. Existing attribution/dual-path tooling and graph comparison remain available; these
references create no causal graph edge or deception fingerprint.

`public_dual_path(response, enabled=True)` reuses the strict existing dual-path parser and returns
only its two public answers, official answer and explicit final choice. It omits scratchpad fields;
valid parser formatting remains required. Neither public comparisons nor a single activation
establish deceptive intent. Mistake/uncertainty/role-play/hypothetical specificity controls are
predeclared research contrasts rather than negative hidden-thought rewards.

## Reproduction, sources and limits

The [stage report](experiments/phase2_pr7_validation.md) and raw artifacts distinguish scripted
public actors, engineered/random token models, host-fixture authentication, and unrun trained-model
studies. All seeds, physical configurations, original evidence and failed hypotheses are retained.
The [primary-source audit](phase2_research_sources.md) preserves the Cammarata/Gao disagreement:
description, held-out prediction, intervention fidelity and alignment improvement are separate
claims. No result here ranks the three imperatives, qualifies a trained refusal/honesty circuit,
or weakens the positive-only verified epistemic-process reward boundary.

Evidence contrasts use matched task instructions; clarification contrasts retain identical role
structure and prior turns while changing only the supplied date. Laundering wrappers/multi-turn
and value-conflict templates declare their additional factors and remain compound where needed.
Those declarations make confounds visible rather than certifying complete semantic isolation.
