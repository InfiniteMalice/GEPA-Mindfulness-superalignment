# DeLoop + Predict–Execute–Observe research protocol

Maturity: **Experimental**. This stage implements a default-off PEO repetition audit,
independently reviewed recovery-pair preparation, and a separate optional DPO numerical kernel.
It does not run a model trainer, deploy a duplicate blocker, award GRPO penalties, admit a dataset
or change runtime authority. Behavioral improvement is unestablished.

## Source and adaptation boundary

[Chen et al., arXiv:2610.19376v1](https://arxiv.org/html/2610.19376v1), submitted 2026-10-14,
defines exact repeated action/observation signatures in section 3.1, first-recovery mining under
the original prefix in section 4.1, and DPO in section 4.2. Appendix G.1 clarifies that the chosen
action must also be a different call, even if a repeated call later returns a new observation.
These sections and primary metadata were checked on 2026-10-27. Paper license: CC BY 4.0.
The text says code/data will be publicly available; no authoritative implementation was identified
or copied. See [REF-DELOOP](recommendations/RESEARCH_TRACEABILITY.md#ref-deloop) and the retained
[equation excerpt](recommendations/source_artifacts/deloop-2610.19376v1-equations.txt). The source
digest and representation are recorded in the [integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md#source-intake-and-adaptation-boundary).

The paper's raw miner does not need reward, success labels or an independent judge. Mindfulness
adds conservative state-version boundaries, source-bound recheck exemptions, independent review,
original-state applicability and existing local/relational findings before emitting a reviewed
proposal. These are original safety/epistemic adaptations, not a faithful reproduction of the
paper's complete training experiment. No source hyperparameter is silently adopted; DPO beta is
an explicit operator input.

## Capture existing PEO records

`LoopStep` reuses `RewardRequest` and the ActionOPD `SemanticAction`. The request's trajectory ID
is the action ID. Its public response must encode that semantic action as JSON. Its actual prompt,
optional prompt token IDs, model identity/version and evidence remain intact. The environment's
`state_version` is a host-observed external-state identity, distinct from a growing prompt digest.
Action arguments use canonical JSON; whitespace inside argument strings remains meaningful.
String observations are compared exactly; structured observations use canonical JSON, an explicit
adaptation from the paper's exact observation matching.

Before execution, the existing `PredictionCommit.predicted_outcome` contains
`{"deloop": step.commitment()}`. This binds the public-state digest, semantic action, environment
version, repeat context and its evidence. The action-bound `ActionRecord.action_class` must match
the semantic decision kind. Existing proposed/executed/outcome/verification events retain their
original schemas; no payload fields or canonical taxonomies are redefined.

```python
from gepa_mindfulness.training.deloop import LoopStep, audit_loops

step = LoopStep(request, semantic_action, state_version="host-observed-state-v1")
# The host logs step.commitment() in its pre-action prediction, then records execution/outcome.
report = audit_loops(events, captured_steps, task_id="declared-task-id", enabled=True)
```

The complete supplied trace must pass the existing action-bound validator. Every executed action
must have exactly one supplied step in execution order and one unambiguous observation. Source
events must be chronological and share run/repeat/model/harness/conversation/trace/rollout identity.
Observation references must belong to the source request. Conflicting source-kind declarations
for one evidence identity fail before review or mining. Source inputs are bounded to 512 steps,
4,096 events and 4 MiB aggregate JSON before any reviewer runs. Source envelopes are snapshotted.
The host retains the immutable raw event store; the audit retains its digest and action-bound IDs.

Legacy verification results and existing typed local/relational events are understood. Failed or
unresolved source checks keep a repeat out of pair mining. Typed local checks require successful
execution, schema/arguments, authorization and intended-operation observation. Typed relational
checks require intact provenance, valid authorization scope, supported outcome and no contradiction.
These structural checks do not authenticate the issuer, dereference evidence or prove relevance.

## Repetition, useful retries and case 17

The raw signature combines canonical action, observed result and environment version. A match
means a **structural repeat**, initially `unresolved_repeat`. New observations or changed external
state are reported separately. `human_recheck`, `independent_verification` and `productive_repeat`
contexts require observable references retained in the source request and pre-action prediction;
they remain `exempt_repeat` and never become a negative preference automatically. The host owns
those classifications and state versions. A model-generated exemption is not self-authenticating.

Clarification repeats can carry `repeated_clarification_unchanged_state` or the source-bound
`repeated_clarification_after_answer` diagnostic subtype. Repeated verification can carry
`verification_restart_without_new_observation`. These annotations can accompany an independently
classified case-17 record; they never assign a canonical case. Detecting every failure to resume
without an action repeat is outside this adapter's implemented scope.

The existing `ExperimentalOverlayConfig(repeated_action_audit=True)` declares the allowed audit
and recovery-proposal outputs. Like other overlay declarations it does not execute the adapter.
Callers explicitly enable the adapter separately; both defaults are off. No canonical case,
robustness stripe, confidence or abstention semantics change.

## Independent recovery review

`mine_loop_pairs(events, steps, task_id=..., reviewer=contract, review=host_callback, enabled=True)`
revalidates the source itself. It does not accept a caller-authored audit as evidence. For each
nonexempt repeat with verified original and repeated observations, it selects the first later
structurally non-looping action whose canonical call differs. If there is no such action, no pair
is emitted. A same call returning a new observation is not a different positive action.

The immutable `LoopCandidate` retains both source requests for host audit, but its scorer-facing
`prompt` and prompt token IDs come from the **rejected step's original state**. The chosen step's
later prompt is never substituted. Independent reviewers may inspect later evidence; they must
confirm that the proposed recovery is supported and applicable in the original state, without
requiring information that became available only later.

Before calling the reviewer, the miner reserves space in the 4 MiB output envelope for the
complete candidate and a maximum-size valid serialized review, including all source references
in every allowed finding binding. Prior accepted and rejected records consume that budget.
Both output routes are checked; insufficient capacity raises before the callback. This is
conservative: a smaller actual review might fit when the reserved maximum does not. The sizing
shape is never emitted as evidence, and the completed output is still validated.

`LoopPairReview` binds the exact candidate digest and existing `TrustedEvaluatorContract`.
The identity must match the host-configured reviewer and differ from the generating model IDs.
The review reuses existing local/relational result records for the rejected loop and recovery.
Unproductive-repeat support must include the original and repeated observation references.
Recovery outcome/operation findings must include the chosen observation references. Matching
IDs alone does not establish truth; the trusted host authenticates the callback, evidence and
evaluator independence. This library is not a sandbox against arbitrary Python callers.

Acceptance requires a confirmed repeated failed route, intact provenance/authorization, no
contradictory loop finding, original-state applicability, successful local checks, satisfied
recovery dependencies, supported recovery outcome and no contradictory/repeated failed recovery.
Irreversible recovery records additionally require the existing local permission finding; this
does not grant permission for a new action. Unsupported, premature, verification-skipping,
future-dependent or unresolved recoveries yield no pair. A rejected first recovery is recorded;
the miner does not search farther ahead for a more favorable positive. Malformed, foreign-subject
or foreign-evidence reviews fail closed.

All audit, candidate, step and review outputs remain versioned DEVELOPMENT records. An outer
TRAIN tag cannot make these proposals optimizer-eligible. Independent review does not admit data,
authorize an action or change reward eligibility. The host must separately derive and admit any
training dataset under existing controls, preserving the original proposals and source provenance.

## DPO kernel and ablations

`deloop_dpo_loss` consumes four aligned one-dimensional batches of **summed action-sequence log
probabilities**: student chosen/rejected and reference chosen/rejected, plus explicit positive
`beta`. It uses stable softplus of the negative reference-relative margin, averages instances and
detaches reference inputs. It does not score text, inspect hidden thoughts, run backward, update
an optimizer or admit a pair. PyTorch is lazy and optional; unsupported shapes or nonfinite inputs
fail. Its existence is not evidence of a completed DeLoop training pipeline.

Run `python -m pytest tests/test_deloop.py tests/test_peo_continuity.py tests/test_experimental_overlay_registry.py tests/test_experimental_overlay_flags.py`.
The CPU contracts cover state/observation changes, rechecks, original prefixes, review bindings,
contradictions, source chronology/coverage and resource bounds. Real-PyTorch loss tests skip
explicitly if unavailable. Contract fixtures are authored tests, not behavioral efficacy evidence.

Unexecuted protocols compare no intervention, action ledger, inference-time duplicate blocker,
loop/recovery DPO, verified-penalty GRPO, DeLoop + PEO memory, and DeLoop + LoCD. Study C also
includes the three-way combination. Match models, tasks, compute and source versions, use multiple
paired seeds and independently held-out tests, and retain failure distributions and uncertainty.
Report loop rate alongside informative-repeat rate, clarification recovery, premature termination,
verified completion, tokens and time. Include necessary rechecking, partial answers, changing tool
results and forbidden shortcuts. Do not promote a lower loop rate that sacrifices verification or
task success. No blocker, GRPO reward path or behavioral experiment is implemented by this stage.
