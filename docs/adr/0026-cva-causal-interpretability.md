# ADR 0026: Constraint verification and falsifiable mechanistic research

Date: 2026-11-04. Maturity: **Experimental** design; evidence products are **Diagnostic**.
Status: Phase 2 architecture, subject to stage-specific implementation and review.

## Intent and boundaries

Distinguish legitimate task completion from mere objective achievement, and test mechanistic
explanations by held-out prediction and controlled interventions. Software acceptance, predictive
validity, causal validity and alignment improvement are separate claims. None follows from a
passing unit test alone. The [baseline audit](../experiments/phase2_pr0_audit.md) identifies the
existing owners that this phase extends.

The Constitution remains normative. Increase prosperity, reduce suffering and increase
understanding stay co-equal in paraconsistent, mutually constraining tension. Mindfulness,
Emptiness, Non-Duality and Boundless Care remain intact, together with honesty, autonomy, consent,
corrigibility, intelligent disobedience, humility, proportionality, oversight and Least Action.
No objective weight, learned reviewer, circuit description or harness candidate overrides them.

Canonical V5 remains exactly `CASE × STRIPE × REPEAT`, cases 1–17; case 0 is noncanonical.
New task conditions and mechanistic statuses are orthogonal research overlays. Existing stripes,
case semantics, reward components and serialized record versions retain their owners.

## Approaches considered

1. Replace inference/reward/governance with one constraint solver and circuit-based evaluator.
   Rejected: it duplicates existing authority, presumes moral decidability and confounds
   instrumentation with training.
2. Add opaque model-reviewed success/failure labels and semantic neuron descriptions.
   Rejected: a plausible label cannot establish a witness, infeasibility or causal influence.
3. Extend existing verification and interpretation packages with optional, typed research paths.
   Selected by the implementation author under the supplied staged specification: deterministic
   bounded checkers first, independently recorded interventions second, no default inference or
   optimizer changes. The tradeoff is deliberately narrow supported guarantees and explicit
   unsupported backends rather than universal claims.

## CVA formalization and guarantee scope

For a candidate action/plan `a`, define
`S_valid = {a | G(a) AND T(a) AND P(a) AND E(a)}`. `G` is objective achievement, `T` modeled technical
validity, `P` applicable authorization/scope, and `E` independently established enforceable
ethical/safety conditions. Values that cannot be formalized remain human-review obligations;
this formula is not a complete moral decision procedure.

Candidate verification and task feasibility are different operations and different typed
subjects. Rejecting one candidate never proves the task infeasible. A search timeout, unavailable
checker, unsupported proof format or absence of a witness remains unknown. Scoped exhaustive
checking can establish infeasibility only over its exact finite domain and modeled assumptions.
Technical and authorization-constrained infeasibility retain separate scopes.

Versioned records retain task/constraint identity and version, authority ordering, candidate
digest, objective, checker identity/version/method, certificate kind, original evidence references,
verdict, completeness assumptions, uncertainty, permitted-next-action descriptions, provenance and
aware timestamps. Next-action descriptions grant no permission. Guarantees distinguish exact,
certified approximation, statistical estimate, heuristic and unknown. Approximation bounds and
confidence statements require separately checked prerequisites, estimator/sampling assumptions
and explicit limitations; no optional solver's `UNSAT` string is a certificate.

Initial certificates are finite assignments and exhaustive coverage of bounded discrete domains.
Compact symbolic predicates use a closed typed vocabulary, never `eval`, generated code or
network endpoints. Construction/search and checking are distinct paths. Rechecking a claimed
solution validates the declared guarantee, including authorization and applicable enforceable
conditions. Contradictory task constraints do not establish anything about unmodeled tasks.

For open-world failure use `NO_AUTHORIZED_SOLUTION_ESTABLISHED`, an epistemic noncompletion
assessment rather than `VERIFIED_INFEASIBLE`. Retain attempted operations, observations, encountered
constraints, any actual contradiction, unknowns, needed information/authorization and legitimate
alternatives. An authorized search that fails must not be relabeled impossible. Useful remaining
authorized options prevent a failure report from implying that further work is futile.

## Trust and evidence flow

Untrusted actor proposal → bounded immutable input snapshot → host-owned checker → exact
subject-bound result → independently authenticated evidence admission → existing governed store.

The host owns formalization, permissions, evidence resolution, checker configuration and clocks.
Input labels/digests, a deserialized record, different reviewer strings or several agents using
the same model do not authenticate any of those facts. No model-supplied callback or code may be
loaded as a verifier. Missing authorization fails closed; unknown ethical conditions prevent
verified success. Typed results are not bearer credentials.

Reuse `EvidenceReference`, `TrustedEvaluatorContract`, `DiagnosticRecord`, action-bound prediction/
action/outcome identities, `LocalVerificationResult`, `RelationalVerificationResult` and existing
runtime grants where applicable. Proposals remain distinct from accepted evidence. Any evidence
commit still uses `commit_verified_claim` and its exact-scope one-use WRITE authority. Model/harness
promotion still uses existing epochs, protected manifests, receipts and Coevolution acceptance;
acceptance does not deploy. A new certificate cannot issue or consume execution permission.

All diagnostic exports retain `training_eligibility="DEVELOPMENT"` (or the stronger original
holdout restriction), no optimizer input and no authority. Preserve the complete envelope at
admission: the legacy training gate cannot recognize stripped provenance. No new negative
hidden-thought penalty, inferred deception punishment, circuit-target reward or apparent-honesty
bonus is introduced. Future training use requires separate authorization and the existing
positive-only verified epistemic-process reward contract.

## Mechanistic research architecture

Preserve both hypotheses: native neurons/circuits may admit coherent explanations, while their
weak/signed activity and computation can remain incompletely understood. The aim is measurable
prediction and causal validation, not a cleanliness score. See the
[source traceability](../phase2_research_sources.md).

Extend `gepa_mindfulness.interpret`, existing attribution graphs, comparison and metric APIs.
The legacy gradient-times-activation graph remains a discovery product; its topology and
approximate gradients do not establish intervention effects. Edge-level evidence supplements
legacy fields with conservative defaults. Never upgrade a graph on language-model review alone.
Use the existing `MechanisticAuditReference` and `mechanistic_circuit_audit` flag for V5 joins.

Architecture adapters declare exact model family/version, module path, layer, tensor, hook side,
shape, token positions, activation transformation, gating and normalization semantics. Begin
with locally instantiated tiny transformers and explicit supported families. GPT-style ungated
MLPs and gated MLPs require different hook definitions; unsupported families/hooks raise clearly.
Attention-head and residual measurements identify the actual component and projection, not a
name-matched tensor guessed to have neuron semantics. Optional torch/model dependencies remain
lazy outside the explicit research path.

Observe → Hypothesize → Predict → Intervene → Measure → Replicate → Revise.

Discovery samples high positive/negative, medium/weak, near-zero and inactive regimes, local
context/offsets and upstream/downstream evidence. Competing relation, transition, conjunction,
hierarchical, multiple-mechanism and unknown hypotheses remain possible. Split identities and
content commitments separate discovery, development and locked evaluation. Freeze the explanation,
permitted-feature predictor and predictions before locked outcomes are revealed. Prediction
functions do not receive target activations. Report false positives/negatives, strength error,
near-miss failures, OOD degradation and signed-regime coverage.

Interventions run on a copied frozen model with hooks removed on every exit. Validate shape,
dtype, device, positions and exact supported hook before substitution; preserve the source weights,
mode and RNG state. Compute `KL(P_original || P_substituted)` directly from matched logits, together
with held-out NLL change, activation error, behavior/calibration and per-example measurements.
Compare identity/null, mean, ablation, random/shuffled predictors, random neurons/directions and
matched-complexity baselines on identical model versions/inputs. Grouped neuron/path experiments
address tiny individual effects, redundancy and interactions. Low KL from a constant is not an
explanation certificate. Large interventions may be off-manifold; retain direction and magnitude.

Predeclared contrasting pairs and expected effects precede patch/replacement/ablation,
restoration, path patching, swaps or mediation measurements. Distinguish necessity and sufficiency,
retain controls, replicate across independent examples and preserve null/contradictory results.
Evidence levels are `DISCOVERED_CORRELATION`, `PREDICTIVE_HYPOTHESIS`, `HELD_OUT_SUPPORTED`,
`INTERVENTION_SUPPORTED`, `REPLICATED_CAUSAL_EVIDENCE`, `CONTRADICTED`, `UNRESOLVED`.
An evidence registrar accepts verified measurements only through trusted host admission; role
names alone provide no independence. Original records are append-only research artifacts.

## Integration and empirical qualification

Offline synthetic white-hat fixtures model bounded endpoints and rules of engagement as data,
with no network, subprocess, executable payload or real exploit interface. Include authorized
solvability, contradiction, excluded solution, missing information, changed authorization and
unsuccessful authorized search. Semantic wrappers, new evidence, valid new authorization and
bypass pressure are distinct manipulations. Expected answers remain evaluator-only.

Studies A–H cover feasibility, graceful failure, discovery, simulation, causal graphs, behavioral
versus mechanistic change, generalization and resource tradeoffs. Alignment comparisons reuse
public dual-path records, causal diagnostics, V5 cells and the existing 2×2 model–harness report.
The diagnostic condition cannot update weights. Any intervention-informed training condition
requires a separate protected candidate and regression suite, including honesty, reasoning,
factuality, abstention, generalization and constitutional conflicts.

Native, contextual-native and optional decomposition conditions share model/data/intervention
budgets and actual cost accounting. Decomposition reconstruction error is measured separately.
Unsupported SAE/backend conditions stay present as unrun, never replaced with fabricated results.
No superiority conclusion follows from one synthetic model or benchmark.

Resource trials retain immutable evidence and audit sources while comparing retained compact
state, recomputation time and verification fidelity. They establish no transformer time–space
theorem. Reports preserve versions, seeds, missingness, failed attempts, null results and the
existing dependency-cluster uncertainty policy. No protected failure is averaged away.

## Delivery and migration

PRs 0–9 follow the supplied dependency order. Each stage has its own diff, review, tests, maturity
and limitations. Ordinary inference has no dependency on these optional research paths.
Legacy serialized objects keep their wire formats; new versioned records use explicit adapters.
New attribution fields use defaults and retain unknown evidence. Original content digests survive
any explicit migration. No current runtime or loss is reconstructed or silently replaced.

The supplied request authorizes staged implementation, check monitoring and merging passing PRs.
The maintainer additionally approved the documented quality-gate fallback because the named skill
is unavailable: task specification, design, implementation, tests, formatting/lint/type checks,
fresh review and a recorded final gate. This approval does not authorize relaxation of runtime
constraints, empirical evidence standards or protected evaluation.
