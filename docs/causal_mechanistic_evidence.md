# Causal mechanistic evidence

Maturity: **Experimental**, opt-in software; outputs and accepted research registries remain
**Diagnostic**, `DEVELOPMENT`, and excluded from optimization. Semantic understanding, intent,
alignment improvement and trained-model generalization remain **Unqualified**. This extends
[native discovery](mechanistic_research.md), [simulation](activation_simulation.md), and the existing
[action-bound sequence](structured_logging.md), not the canonical 17-case taxonomy.

## Typed API and execution

`native_causal_records.py` provides exact, versioned `CVARecord` contracts with bounded canonical
serialization/digests. `InterventionSite(hook, component_indices, token_positions)` uses the actual
architecture-specific hook. Attention-head selections expand to contiguous head coordinates;
groups and complete residual vectors are supported up to 64 expanded columns. A scalar edge cannot
inherit support from an entire group. `LogitContrast(position, target_token, reference_token)`
declares the output metric. `EffectPrediction(direction, minimum_effect, maximum_effect,
minimum_control_gap)` freezes a signed interval and random-component specificity requirement.

`CausalPlan` binds hypothesis, physical model digest, matched donor/recipient inputs, sites, mode,
metric, prediction, evidence references, seed, optional literal replacement, and evidence budget.
`freeze_causal_plan(..., prediction, action, clock, enabled=True)` binds the existing immutable
`PredictionCommit` and `ActionRecord` to that exact plan digest. The host must retain prediction
custody before execution; self-authored timestamps and different role names cannot establish it.

`execute_causal_plan(backend, commit, clock, enabled=True)` uses an existing `FrozenNativeModel`.
Only its verified GPT-2/Llama hooks are accepted; unsupported models and hook locations fail.
It performs real copied-model forward passes, validates shape/dtype/device/positions, records hook
order, retains original/applied activation values, and removes temporary hooks on failure.
No weights, source model, optimizer or runtime permissions are changed.

| Mode | Primary measured effect |
|---|---|
| `ACTIVATION_PATCH` | Donor source values inserted in recipient; changed minus recipient contrast |
| `ACTIVATION_REPLACE` | Declared finite values inserted in recipient |
| `ABLATION` | Donor contrast minus donor contrast with source zeroed; bounded necessity test |
| `COMPONENT_SWAP` | Donor-to-recipient effect; reverse recipient-to-donor also retained separately |
| `RESTORATION` | Downstream donor mediator restored after source ablation; recovery over ablated state |
| `PATH_PATCH` | Source patch changes mediator; only changed mediator inserted in recipient |
| `MEDIATION` | Total source-patch effect minus effect with mediator restored to recipient baseline |

Path patching can aggregate multiple routes to a mediator. Controlled mediation is not full natural
causal identification; interactions, alternative paths and off-manifold states remain possible.
Necessity and sufficiency concern the declared manipulation and output contrast. Null single-unit
ablation does not rule out redundant mechanisms. Whole residual swaps are bounded component swaps,
not parameter surgery or an architecture-independent layer replacement.

Every run retains donor/recipient baselines, exact identity, zero ablation, random components at a
matched component budget, and magnitude-matched random directions. A full-width selection may have
no random alternative; the missing control prevents a specific supported result. Singleton random
directions are degenerate sensitivity controls. Reports include raw logits, actual contrasts,
effects, activation L2 magnitudes, assumptions, model/version, seed, references and prediction result
(`MET`, `FALSIFIED`, `NOT_SPECIFIC`). Raw evidence remains `UNRESOLVED` with authentication
`NOT_PERFORMED`. Its prediction/proposed/executed/outcome events bind the measurement digest through
the existing action-bound contract. No description or gradient score grants causal status.

## Independent checking and protected registration

`mechanistic_evidence.py` distinguishes `DISCOVERED_CORRELATION`, `PREDICTIVE_HYPOTHESIS`,
`HELD_OUT_SUPPORTED`, `INTERVENTION_SUPPORTED`, `REPLICATED_CAUSAL_EVIDENCE`, `CONTRADICTED`,
and `UNRESOLVED`. These are evidence levels, never V5 cases or optimizer rewards.

`MechanisticVerifierContext(backend, evaluator, actor_id, clock, authenticate)` is trusted host
configuration, never a model-loaded callback. The host authentication callback must establish source
custody, prediction timing, held-out isolation where applicable, and actual researcher/verifier
independence. It receives a detached `VerificationPacket`; missing, throwing, mutating, false or
truthy non-True callbacks fail closed. Distinct actor labels alone are insufficient independence.

`verify_causal_runs(reports, context=..., enabled=True)` checks exact report/event bindings and
independently recomputes metric arithmetic from raw logits, then re-executes every intervention on
the exact frozen physical model. A coherent forged logit trace fails replay. A positive specific
run can support `INTERVENTION_SUPPORTED`; multiple distinct observed causal prefixes can support
`REPLICATED_CAUSAL_EVIDENCE`. Renamed inputs, future-only and masked-token changes do not establish
replication. Replication here is a bounded software criterion, not statistical independence or
cross-model replication. Failed predictions remain `CONTRADICTED`; nonspecific/null results remain
unresolved. All original reports are retained in `MechanisticFinding`.

`verify_predictive_run(report, context=..., enabled=True)` re-captures actual locked native
activations and recomputes the frozen explanation test. At most it yields `HELD_OUT_SUPPORTED`
under the preregistered numeric criterion, or `CONTRADICTED`; it never grants causal support.
Retrospective discovery predictors may use future context. Causal-decoder simulation separately
requires the permitted-prefix policy. Numeric fit alone establishes neither semantics nor superiority
over matched predictors, and this verifier does not certify generation-process isolation itself.

`MechanisticRegistry(registry_id, revision, entries)` retains append-only supported and contradicted
findings. Serialized records are inert, not bearer credentials. `mechanistic_update_scope(before,
finding)` binds exact before/after snapshots. `admit_mechanistic_finding(..., context, action,
authorization, grant_registry, clock, enabled=True)` re-verifies original sources and consumes the
existing one-use host `WRITE` grant scoped to that update. No new authority mechanism is invented.
The host owns the current snapshot, serialization of updates and durable storage. It must not accept
an actor-selected stale registry as current. Registration retains fixed diagnostic training
ineligibility and never overrides constitutional constraints or governed promotion.

## Graph overlay

`mechanistic_graphs.py` provides `GraphEvidenceBinding(edge_index, finding_id, uncertainty=None)`
and `attach_graph_evidence(graph, registry, bindings, trusted_registry_digest=..., enabled=True)`.
The host pins the accepted registry digest. The returned graph is copied; original weights and
legacy metrics are preserved. Exact model, primary context, hook/site and output-contrast mappings
are required. Patch/replace/swap/path/mediation support belongs to the recipient context;
ablation/restoration support belongs to the donor. Reverse swaps require separate primary evidence
before annotation. Explicit group nodes may represent group findings; individual members do not
inherit those effects. Prediction-only findings cannot promote graph edges.

Known gradient/native observations remain `DISCOVERED_CORRELATION`; unknown methods, unmapped
edges and missing evidence remain `UNRESOLVED`. Related contradictory findings remain visible even
if an older positive finding is selected. Edge uncertainty may remain null. These annotations report
the registered experimental scope, not causal strength, semantic understanding or deceptive intent.

## Reproduction and limits

The stage's [recorded results](experiments/phase2_pr6_validation.md) retain all predeclared modes,
positive constructed paths, nulls, falsified predictions and raw logits. No network targets, model
downloads, learned parameter updates or alignment effects are involved. CPU exact replay is tested;
CUDA, other architectures, trained behavioral mechanisms and semantic generalization remain unrun.
See the [primary-source audit](phase2_research_sources.md) for the competing Cammarata/Gao standards.


The full-evidence executor reserves `20 × sequence_length × vocabulary_size` distribution values
before execution, bounded by the plan's budget (default 250,000; hard maximum 1,000,000), plus the
bounded record encoding. Thus vocabulary sizes above 50,000 are unsupported even at one token;
32,000 permits only one token at the maximum budget. `FrozenNativeModel` defaults to a 50M
parameter budget on CPU and CUDA; callers may explicitly configure up to 1B parameters.
Only tiny CPU models were exercised by these probes. Standard large pretrained-model tracing
is unsupported. The prototype fails
explicitly rather than truncating logits or lifting audit limits. A scalable immutable external
measurement store needs a separate reviewed design and regression evaluation.

A nonspecific independently replayed result remains `UNRESOLVED`; successful replay does not
establish specificity against controls. Contradicted predictions receive an explicit falsification
reason; replay alone cannot make them supported. Historical `base_commit` fields identify the
predecessor, not the tested revision. The [snapshot index](experiments/phase2_manifest_snapshots.json)
identifies the tested full revisions: stage 6 is `5fe5453122bfbe11ee1a303af207f3be001076e3`.
Current reviewed hashes are separate from these retained historical snapshots. See the final-stage
publication provenance guide and original-history bundle. Artifact checksums identify retained bytes. Numerical reproduction need not reproduce a
wall-clock timestamp, runtime version or measured timing cost byte for byte; preserve those actual
values instead of assigning a historical clock to new measurements.
