# Decision comparisons and local context distillation

## Status

The adapters provide **Experimental** opt-in ActionOPD comparisons, numerical reward/advantage
helpers, localized token preparation and a separate optional PyTorch forward-KL objective.
The [native LoCD trial](#native-cpu-local-correction) now performs isolated CPU context
self-distillation and existing governed catalog validation. The [native ActionOPD adapter](#native-verified-actionopd)
now binds a configured black-box teacher callback to independently verified decision credit and
canonical CPU GRPO training. It installs no external API client. Catalog acceptance never deploys either model.
Default training, rewards, runtime authority and canonical cases are unchanged.

## Sources and attribution

Primary texts checked on 2026-10-27:

- [ActionOPD v1](https://arxiv.org/html/2610.19820v1), sections 2 and A.2–A.4;
  [official source](https://github.com/thu-nics/ActionOPD/tree/d447bdf2ad3158ae7c315cb96ce0a180f94459b1),
  Apache-2.0, particularly teacher_reward.py and action_matching.py.
- [LoCD v1](https://arxiv.org/html/2610.21427v1), section 4;
  [official source](https://github.com/zz-haooo/LoCD/tree/33d950954f14c0f4e087274ae9365c7f2d2cbd80),
  MIT, particularly core/rewriter.py and core/dtypes.py.

Both papers use CC BY 4.0. No upstream source code is copied or vendored. Bibliography metadata
and repository inferences extend the existing [reference registry](recommendations/references.yaml)
and [traceability reader](recommendations/RESEARCH_TRACEABILITY.md). The source links to REC-001/REC-010 do not change recommendation status or governance.

The ActionOPD numerical reward follows its Eq. 6: invalid -1; valid nonmatch 0; first match 1,
with an additional frequency bonus proportional to subsequent teacher matches. K=0 is no
supervision; K=1 has unit match credit. Eq. 7 centers each retained group and uses one population
standard deviation across the batch. These are diagnostic objective values, not authorized reward.

LoCD preserves the student prefix and regenerates only a local region. The supplied forward-KL
kernel detaches teacher distributions, averages tokens within each span, then averages instances.
This is one numerical path, without the paper's complete training, clipping or other divergences.
The paper studies shared-model context self-distillation; this interface does not enforce a
same-model teacher or claim to reproduce the full training setup.
Segment SFT would be a distinct adaptation and is not mislabeled as the published divergence loss.

## ActionOPD adapter

[actionopd.py](../gepa_mindfulness/training/actionopd.py) accepts existing `RewardRequest` student
rollouts and `TeacherActionSample` objects. The caller collects the actual student states and
queries a separately configured black-box teacher; no teacher token probabilities are needed.
Teacher sample IDs, generation identity/version/contract, sampling configuration and public-state
fingerprints are retained. Repeated actions with different sample IDs keep their multiplicity.
Use `nominal_teacher_count` for a partially retained teacher multiset; invalid samples remain
audit-only, and an empty valid multiset supplies no supervision.

`SemanticAction` distinguishes grounded/qualified answers, IDK, necessary clarification,
refusal/redirection, verification, authorized tool use, evidence-update proposals, escalation and
resumption. Labels are host-parsed declarations; a TOOL_USE label grants no tool permission.
The bounded matcher compares exact typed canonical JSON content. It normalizes object-key order,
but does not infer arbitrary paraphrase equivalence or use an LLM judge. This is an explicit
adaptation of the upstream semantic matcher. Hosts retain observable parsing/canonicalization
evidence and do not treat lexical matching as semantic truth.

The public-state fingerprint includes the retained prompt/token IDs, student model/policy and
sampling configuration, including visible tool configuration when supplied. Hosts must capture
all visible state; omitted context cannot be reconstructed or certified by a digest. All students
must have the same fingerprint, and teachers must name that exact state. A candidate's trajectory
ID is also its action identity for optional `RelationalVerificationResult` attachment. Independent
verification evidence must occur within that request's observable boundary.

```python
from gepa_mindfulness.training.actionopd import evaluate_action_group

assert evaluate_action_group(object(), object(), object(), object()) is None
report = evaluate_action_group(
    student_requests, parsed_actions, teacher_samples, independent_outcomes,
    frequency_bonus=0.5, enabled=True,
)
```

The 0.5 diagnostic default is sourced from the paper, not a validated Mindfulness hyperparameter.
The report preserves agreement and independent outcome records separately. Agreement never
populates a correctness field or validates provenance. A matched prohibited or unsupported action
can retain a failed/unknown outcome. Results are DEVELOPMENT data with no authority or training
admission. Invalid-action penalties describe observable protocol failures, not hidden thoughts.

`actionopd_advantages(groups, epsilon=...)` skips unsupervised/equal-reward groups. It is distinct
from ordinary GRPO's per-group normalization. The native opt-in adapter below uses this helper
before the existing clipped GRPO update. Full upstream rollout selection and benchmark runs remain
unimplemented. Hosts independently admit actions, rewards and datasets before optimization.

## Local segment and correction-source adapter

[locd.py](../gepa_mindfulness/training/locd.py) provides `build_local_segment()` as a default-off
token-layout helper. It requires recorded prompt/response token IDs, a strict original region
`[start,end)`, a teacher continuation exactly that region's token length, and teacher-only context.
Whole-trajectory regions are rejected. The student receives original public input and prefix plus
the local continuation; original suffix tokens are not teacher-regenerated. The mask is false on
the original prefix and true only on the continuation. The source trajectory is retained unchanged.

This exact-length token convention is a bounded repository adaptation, not a character-to-token
localization algorithm. Hosts supply a consistent tokenizer/version and independently verify the
region and continuation. The helper cannot prove that a teacher quote identifies mechanistic cause.
Multiple regions should be separate examples. Hints remain in the teacher context field and are
never appended to generated student inputs. Caller-supplied prompts must themselves be appropriate.

`prepare_registered_local_segment()` additionally reuses
`CoevolutionStore.correction_source_events()` and `CorrectionProposal`, preserving source events,
failure localization, verifier references and teacher metadata. Source rollout/trajectory IDs must
agree, and the source policy must match the registered closed epoch. This validates the registered
source binding; it does not establish the semantic correctness of token offsets or correction text.
The host must also establish that the supplied rollout content corresponds to the registered
experience; shared IDs and model versions alone cannot prove that correspondence.
There is no parallel failure registry or new admission policy.

`localized_forward_kl(student_logits, teacher_logits, loss_mask)` lazily imports PyTorch and accepts
aligned `[batch,time,vocabulary]` logits and a boolean `[batch,time]` mask. Each instance needs
at least one local token. The teacher is detached; masked prefix positions receive no gradient.
Inputs must be finite floating-point tensors on the same device. This kernel does not run model
forward passes, inject hints, apply optimizer updates or enforce host policy by itself. ActionOPD
and LoCD remain separate functions and objectives.

## Reproduction and limits

```bash
PYTHONPATH=src:modules:reasoning-generalization-tracer/src \
  python -m unittest discover -s tests -p test_distillation_adapters.py -v
```

The local adapter tests use real repository contracts. PyTorch is unavailable in the current
source-snapshot environment, so gradient/equal-instance-weight checks are explicitly skipped
locally and require full CPU RL CI. Existing coevolution integration coverage exercises registered
source handoff. Tests establish software behavior, not research improvement or moral alignment.
The current focused local suite passes 365 tests with four explicit PyTorch skips, including
existing reference, coevolution, eligibility, verifier and canonical-case regressions. Pytest 8.4.2
and Pluggy 1.6.0 were loaded from pinned upstream source tags into an external tools directory,
with ordinary generated version metadata; no test dependency is vendored or substituted in the
repository. Local full-wheel, lint/type and real-PyTorch validation still require CI.

Before any training experiment, configure immutable splits/checkpoints, teacher/evaluator versions,
multiple seeds, budgets and the protected 17-case/stripe/repeat suites. Compare baseline, ActionOPD,
LoCD and sequential combinations separately; reserve eligible event credit through existing host
accounting. Do not optimize exported diagnostic reports or infer source correctness from agreement.
No empirical transfer, SST retention result or protected-case non-regression is claimed here.

## Native CPU local correction

`gepa_mindfulness.training.locd_runtime.run_local_correction()` supplies an opt-in, single-region
training trial over `TorchPolicyBackend`. Its versioned `LoCDTrainingConfig` is disabled by default.
This is the shared-model context self-distillation subset of REF-LOCD: the unchanged source
policy generates a local continuation with trusted evidence, then its detached contextual
distribution teaches an isolated copy of the student through the existing masked forward KL.
There is no ActionOPD agreement loss, scalar reward, external teacher service or automatic action.

The operator supplies these existing host objects and roles:

- An exact `RewardRequest` with explicit TRAIN provenance, original token IDs, registered
  trajectory/model identity, and sampling metadata `source_record_sha256` and
  `policy_parameter_checksum`. The latter must match the actual current backend. Nested
  held-out/development/regression or failed V5 provenance stays ineligible. The declared source
  digest does not authenticate capture; the host source reviewer must check the immutable record,
  source dataset, tokenizer and actual recorded experience against the coevolution evidence.
- A `CoevolutionStore`, registered `CorrectionProposal` changing only MODEL, and a separately
  opened candidate epoch. Teacher evidence references must be observable records. The trusted
  host authenticates their contents, applicable constraints and supported localization; reference
  labels or suspicious text spans do not establish truth or mechanistic cause.
- Fresh `authorize_source` and `authorize_correction` callbacks installed by the operator.
  They see detached complete review envelopes including source/correction, teacher evidence and
  actual generated local target. Only literal `True` on an unchanged envelope admits that stage.
  Exceptions, mutation, missing roles and truthy substitutes fail closed. Configuration names the
  review policy, reviewer, independent evaluator, source dataset digest and seed. Identity strings
  record scope; the trusted host must enforce real separation of roles.
- An independent `evaluate(candidate, generate)` host function. The candidate identifies the
  actual saved artifact; `generate(RolloutRequest(...))` supplies bounded inference only, without
  teacher context, corrected training labels or optimizer methods. The host evaluates held-out
  tasks and the pinned protected case/stripe/repeat suite, creates the existing epoch records,
  closes the candidate epoch, and returns a store-issued `ValidationBundle`. It must evaluate
  actual outputs under an independently defined policy. The existing `decide_candidate_acceptance`
  remains the only catalog acceptance owner; generated summaries and booleans cannot replace it.

Call `run_local_correction(backend, request, store, correction, config=config, start=start,
end=end, teacher_context=trusted_context, authorize_source=host_source_review,
authorize_correction=host_correction_review, evaluate=host_protected_evaluation,
candidate_id=new_candidate_id, artifact_root=operator_artifact_root)`. Host functions in this
example must be configured independently; there is no default reviewer that approves everything.

Generation requires a Transformers-compatible `generate` interface. Single-beam/single-sequence
overrides disable expanded generation outputs and clear supported inherited contrastive,
constrained, assisted and token-healing mode selectors. Original configuration and effective
overrides are retained.

The function copies model, reference, value head, optimizer and SST state and checks storage and
optimizer ownership. It retains the original student prefix, generates exactly the declared local
span with greedy source-policy decoding, masks the prefix, and applies the backend's real backward
and optimizer step. Teacher context is supplied only to teacher generation and target forward
passes. Mandatory post-step generation uses the original public prompt without that context.
Loss computation does not establish that the corrected continuation is safe or correct; the
independent correction and protected-evaluation roles remain necessary.

The original backend stays unchanged whether the candidate is accepted, rejected or raises.
The returned `LocalCorrectionResult.backend` is the isolated backend only after catalog acceptance;
its decision still has `execute_candidate=False`. Rejected candidates retain their checkpoint and
the existing source-epoch rollback reference. An unchanged policy (including an SST freeze) returns
`no_model_change`, no admitted candidate, and no improvement claim. No automatic model promotion,
deployment, protected manifest rewrite or reward eligibility change occurs.

Each changed trial writes the actual native checkpoint and a DEVELOPMENT `trial.json` under a
new directory named by the candidate ID's SHA-256. Existing directories are never overwritten.
The registered artifact digest binds checkpoint bytes. Native checkpoint loading retains optimizer
and optional SST reference state; this is not a resumable multi-example curriculum manager.
Audit JSON cannot recreate reviewer authority or replace store receipts. Teacher evidence/context
is retained in operator-controlled trial artifacts and must receive appropriate access controls.
Inference call/token/logit bounds and finite FP32 parameter checks fail closed; model and artifact
digests are rechecked before acceptance. The host must keep its evaluator and filesystem trusted.

The configured parameter, token, logit and evaluation-call limits are engineering budgets, not
published training hyperparameters. This implementation supports portable CPU full-weight FP32
models with a known output-vocabulary head. CUDA, mixed precision, distribution, LoRA, external
teacher APIs and complete paper reproduction remain Unsupported. The native update, checkpoint,
SST zero/full-rank and protected rejection tests qualify software behavior; behavioral retention,
constitutional stability and research effectiveness remain unqualified. Maturity: **Experimental**.

Run `python -m pytest -q tests/test_locd_runtime.py tests/test_distillation_adapters.py
tests/test_model_harness_coevolution.py` in the CPU RL development environment. The weight-driven
seven-token fixture verifies an actual update, prefix masking, teacher-only hints, source isolation,
independent receipt admission and checkpoint round-trip; it is not a research benchmark.
The operator excluded research runs. Future qualification would use the documented Study A
conditions and protected 17-case/stripe/repeat suites with matched data/compute, independent
held-out evaluations and multiple seeds, measuring correction, forgetting and calibration separately.

## Native verified ActionOPD

`gepa_mindfulness.training.actionopd_runtime` connects typed semantic decisions to the canonical
CPU GRPO learner through the existing `ResearchRewardProvider` and `ResearchCreditLedger`.
`ActionOPDConfig` is disabled by default and becomes part of checkpoint/resume identity when
enabled. It pins teacher/verifier versions and contracts, source-review policy, explicit teacher
sample count, sampling configuration, frequency bonus, whitening epsilon and cache/candidate
budgets. Engineering budgets are not paper hyperparameters. Ordinary GRPO and default hashes
remain unchanged.

Configure these host roles through `build_actionopd_provider()`:

- A pinned `ResearchRewardPolicy` and durable `ResearchCreditLedger`. Use the single coefficient
  `("verified_decision", method.credit_weight)`, where `credit_weight=1/(1+frequency_bonus)` bounds
  Eq. 6 scalar credit. A zero baseline keeps this method separate from knowledge, progress and
  process objectives. Their later combinations require separately validated accounting.
- An admitted `DatasetProvider` through the engine's existing `dataset_factory`. Source rows must
  retain explicit TRAIN eligibility, at the root or canonical pair `metadata` location, and all
  original provenance. The host authenticates that materialized rows belong to the immutable file
  captured by the engine and its pinned dataset digest. No protected test is rewritten.
- A `capture_source(Trajectory) -> RewardRequest` host observer. The engine binds actual native
  outputs to selected admitted rows before capture. Capture may add observable evidence; output,
  token IDs, policy identity, sampling and retained source metadata stay exact. Its complete
  observable boundary is explicit. Nested held-out or failed V5 provenance cannot be removed or
  relabeled. Capture does not grant training permission or certify truth.
- Fresh `authorize_source(envelope)` review before teacher access, including cache hits. It
  authenticates capture, tokenizer/public-state completeness, source eligibility, applicable
  constraints and configured permission/budget for proposed teacher calls. Missing, denied,
  mutating, throwing and truthy non-boolean reviews fail closed. Paid APIs and large training need
  separately configured operator authority; this adapter includes no network client or credentials.
- A `sample_teacher(public_packet, index) -> TeacherActionSample` callback. The packet contains
  public prompt/token IDs, student model/policy, public sampling fields, configured teacher
  identity/sampling and an opaque state fingerprint. It excludes sampled student responses, source
  records, private answer keys and evaluation labels. The host retains complete public task
  context in the original prompt. A fingerprint cannot recover omitted context or authenticate
  evidence. Credentials belong outside logged sampling configuration.
- Host `parse_action(request)` and independent `verify_action(request, action)` roles returning
  existing `SemanticAction` and evidence/action-bound `RelationalVerificationResult` records.
  Task fit, dependencies, provenance, authorization and supported claimed outcomes must be
  affirmative, contradiction resolved as none, and a repeated failed route absent. Uncertainty
  or adverse results reject the proposal. Fresh host review authenticates the records; a flag or
  declared identity does not prove moral alignment or truth.
- `account_event(request) -> (semantic_event_id, layers)` and fresh existing `authenticate_claims`,
  `authorize_optimizer_use` and `authorize_accounting` callbacks. Normalize the actual event
  across methods and declare every relevant prior reward layer. Unknown, missing or counted
  required layers fail closed. The existing ledger prevents event replay and concurrent duplicate
  reservations. Callbacks are installed by the host, never certified by a teacher or candidate.

The resulting object is an exact `ResearchRewardProvider`. Pass it through the existing
`reward_factory`, with `config.actionopd=method` and `config.research_credit=credit_policy`.
The engine checks matching method/provider/batch/loss settings and actual CPU FP32 execution.
Use the existing native full-weight backend, admitted dataset factory, checkpoint store and
optional SST. Undeclared or mismatched services fail before teacher sampling. Changed
teacher/config/data identity prevents incompatible resume. One bounded rollout batch per
optimizer step supplies global whitening; cross-microbatch whitening is unsupported.

For example, with independently installed host functions, construct
`provider = build_actionopd_provider(method, policy=credit_policy, ledger=host_ledger,
capture_source=host_capture, authorize_source=host_source_review,
sample_teacher=host_teacher_sampler, parse_action=host_parser, verify_action=host_verifier,
account_event=host_event_accounting, authenticate_claims=host_claim_review,
authorize_optimizer_use=host_training_review, authorize_accounting=host_accounting_review)`.
Run `RLTrainingEngine(config, dataset_factory=host_dataset_factory,
backend_factory=host_native_cpu_backend, reward_factory=lambda _: provider).train()`.
There are no default approving reviewers.

Samples are cached only for the same retained student state and pinned generation/policy
configuration, with sample/state/byte bounds. Invalid teacher samples retain provenance and
supply no match. Eq. 6 agreement is recomputed separately from independent outcomes. A prohibited
shortcut cannot obtain credit merely because a teacher agrees. An invalid or uncertain candidate
fails closed; appropriate observable behavioral penalties remain separate from hidden thoughts.

When the student model/policy scope changes, obsolete ephemeral samples are discarded and cache
capacity resets. Memory bounds continue to apply within the current scope; they do not cap the
number of optimizer steps. Existing durable evidence and admission/accounting records remain
intact. Source authorization still precedes cache rotation and subsequent sampling.

New `research-credit-claim-v2` verified-decision proposals retain the exact source, typed action,
teacher samples and independent verification, and reproduce the score. Existing claims/policies
keep their v1 schemas; decision-enabled policies/reports use v2. Exports remain DEVELOPMENT data.
Raw `teacher_agreement` and `diagnostic` kinds remain ineligible even with provenance. Only fresh
live authorized decisions supply optimizer access; JSON cannot recreate that capability or
authorize execution.

The batch preparer verifies live target/policy/method bindings, groups by public student state,
centers each retained group and whitens across the complete retained batch using Eq. 7. Scalar
credit and epsilon use the same positive scale, preserving raw Eq. 7 normalization up to floating
rounding. Unsupervised/equal-reward groups are skipped without forcing an update. The existing
clipped GRPO loss uses actual student probabilities; teachers never need token probabilities.
Native checkpoints retain optimizer/SST state and method identity. Fresh caches after resume
require identical generation provenance; bit-identical external sampling also needs a
deterministic host sampler. The ledger is not refunded on later failure.

Raw/weighted verified-decision metrics use closed schema v4. Metric schemas v1/v2/v3 keep their
prior meanings. Agreement, independent outcomes and host eligibility remain inspectable.
Training creates checkpoints; protected evaluation and governed promotion remain separate
mandatory operator processes. Protected evaluation uses student-only inference without teacher
supervision. Native ActionOPD capture/scoring operates only on admitted TRAIN sources.

Run `python -m pytest -q tests/test_actionopd_runtime.py tests/test_actionopd_verified_credit.py`.
The offline fixture samples from actual current weights, trains with canonical GRPO and resumes
native checkpoints using host evidence and a probability-free teacher callback. These are
software checks. Maturity: **Experimental**. CUDA/AMP/distribution, LoRA, complete reproduction,
external-service qualification and behavioral effectiveness remain Unsupported or unqualified.
Research runs were excluded by the operator. Future Study A qualification would compare baseline,
agreement, independently verified decisions and stripe augmentation with matched budgets,
held-out 17-case suites and multiple seeds, reporting calibration, completion and regressions.
