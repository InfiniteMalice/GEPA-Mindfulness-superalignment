# Phase 2 stage 6 implementation refinement

This refines the existing approved phase plan, not an issue tracker. Base is stage 5 commit
`428678a`; stages 0–5 remain separate local review boundaries. Stage 6 is not implemented yet.
Initial `tests/test_native_causal_interventions.py` fails on the missing module, as expected.

## Execution contract

Reuse `FrozenNativeModel`, exact hook factory, native public inputs, scalar tensor replacement,
existing action-bound `PredictionCommit`/`ActionRecord`, evidence references and authority grants.
No model parameters are updated; all interventions operate on a frozen copy. Default-off APIs
must fail before inspecting inputs. Unsupported hooks/modes/layouts must fail clearly.

The initial test names the proposed records: `CausalPlan`, `InterventionSite`, `LogitContrast`,
`EffectPrediction`, `InterventionMode`, and `freeze_causal_plan` / `execute_causal_plan`.
Each site binds an exact HookSpec, component indices and token positions. Scalar selections
use actual tensor columns. Head selections may be expanded explicitly to contiguous head
coordinates when within the declared tensor budget; larger/vector-unsupported paths fail.
The pair consists of donor/recipient NativeInput records, not semantically certified clean/corrupt
labels. The metric is a predeclared target-minus-reference output logit at a declared position.
Prediction specifies signed direction, nonzero minimum effect, finite maximum and control gap.
Original source/model/input references, seeds, assumptions and actual raw logits remain available.

The frozen commitment binds the unbound plan digest through an existing PredictionCommit:
`{"causal_plan_digest": plan.digest}`. Its ActionRecord scope is `mechanistic-plan:` plus that
digest, and its prediction ID must match. Reuse the existing immutable action/prediction
snapshot pattern; do not confuse self-authored timestamp text with trusted temporal custody.
Execution refuses an unbound plan, mismatched model, forged hook or incompatible geometry.
Precommit/execute/outcome events should reuse existing action-bound envelopes where feasible.

## Actual supported interventions

Activation patch: donor values inserted at source site in recipient context.
Activation replace: declared finite values inserted at the source site.
Ablation: zero source components in donor context; necessity score is donor minus ablated score.
Component swap: patch the donor site into recipient and the recipient site into donor; full
residual selections implement bounded layer swaps. Report both directional effects.
Restoration: ablate source in donor, then restore a distinct downstream mediator to donor values;
compare the recovered output to the ablated state. Trivial source self-restoration is an identity
control rather than an explanatory result.
Path patch: first source-patch recipient, capture changed mediator values, then insert only those
mediator values into an otherwise unpatched recipient. This tests a controlled functional path,
not every edge or all possible routes from source to mediator.
Mediation: source-patch recipient, then restore mediator to recipient baseline. Report total
effect, controlled direct effect, and their difference as the controlled mediated effect, with
interaction/off-manifold assumptions explicit. Do not imply full natural-causal identification.

A multi-hook trusted executor should attach hooks together, verify actual order/source before
mediator, capture original tensors and remove every hook on success/failure. Reuse existing
replacement/identity checks rather than cloning a second attribution implementation. Keep
directions, applied values and L2 magnitudes explicit; compare against identity, ablation and
seeded matched/random controls. Evaluate necessity/sufficiency only for the declared experiment.
Null single-unit effects do not establish irrelevance under redundancy.

Metric arithmetic is recomputable from retained logits. Measured prediction result is MET,
FALSIFIED, or NOT_SPECIFIC when controls fail specificity. The raw run's mechanistic evidence
status remains UNRESOLVED and measurement authentication NOT_PERFORMED until host registration.
No language-model endorsement upgrades it. Repeated examples must be genuinely distinct input
contexts and maintain matched model/sites/prediction meaning, not renamed duplicate data.

## Evidence levels and protected registration

New `mechanistic_evidence.py` should represent DISCOVERED_CORRELATION, PREDICTIVE_HYPOTHESIS,
HELD_OUT_SUPPORTED, INTERVENTION_SUPPORTED, REPLICATED_CAUSAL_EVIDENCE, CONTRADICTED and
UNRESOLVED. These are orthogonal evidence levels, never V5 cases. Attribution-only data cannot
be promoted by accepting a description or assigning a level in serialized model output.

Retain edge-specific uncertainty, assumptions, source references and provenance. Prefer an
opt-in overlay keyed to exact existing graph/node/edge identities so old graph constructors,
serialization and metrics remain compatible. Missing/ambiguous aggregate-to-native mappings
stay unresolved. Native observational graphs have no invented causal edges.

Only trusted host verification may admit records. Context authenticates source custody,
prediction-before-measurement and actual researcher/verifier independence; different role/model
labels alone are insufficient. Verify numeric effects/control arithmetic and source binding
independently, retaining falsified hypotheses/nulls as well as supported findings. Missing,
throwing, mutating or truthy non-True callbacks fail closed. Host-owned one-use WRITE grants
must bind exact before/after registry snapshots and publication action through the existing
runtime authority machinery; never load callbacks/grants from model/scenario content.

Keep the accepted mechanistic registry itself a fixed DEVELOPMENT/Diagnostic record so admission
cannot launder internal features into optimizer inputs. General `EvidenceState` claims alone do
not carry training eligibility, so do not replace a typed diagnostic registry with an untyped
claim string that could lose this boundary. Reuse existing grant consumption/evidence refs,
not a new parallel authority implementation. The host persists/serializes protected snapshots;
deserializing an apparently accepted record is not a bearer credential.

Known infrastructure: `verification/state.py` provides `evidence_update_scope` and
`commit_verified_claim` (host-owned Local/Relational verification, accepted references and
consumed WRITE authorization). `learning_surfaces.py` owns `EvaluationEpochStore`; `coevolution.py`
owns trajectory/candidate/protected-suite admission. `action_bound_events.py` provides immutable
prediction/action/outcome records. Do not rebuild these stores or weaken their contracts.

## Validation intent

Tests cover every implemented mode on real tiny GPT-2/Llama forwards, independent contrast and
mediation arithmetic, prediction binding, exact hook order/geometry, no hook/source/RNG leak,
distinct-example replication, rejected hypotheses, no reviewer-only upgrade and diagnostic-only
training exclusion. Use an engineered tiny model with a known selected MLP-to-logit path for
positive software controls: zero other weights, preserve norm gains, set one MLP writer/reader
and distinct embedding/readout axes. Declare synthetic construction; do not infer trained semantic
or ethical understanding. Random-model null experiments and all controls remain retained.

The accepted registry needs adversarial tests against forged receipts, self-review, changed
measurement/plan/control/graph digests, missing/replayed grants and stale before snapshots.
Fresh independent review, relevant/full regression, packaged protocol and raw reproducibility
artifacts complete the stage. Scientific generalization, human intent and alignment improvement
remain separate unqualified claims.
