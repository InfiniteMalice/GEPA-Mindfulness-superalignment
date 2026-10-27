# Decision comparisons and local context distillation

## Status

PR 2 provides **Experimental** opt-in ActionOPD comparisons, numerical reward/advantage helpers,
localized token preparation and a separate optional PyTorch forward-KL objective. It does not
provide a teacher service, active model training, environment execution, optimizer admission or
model promotion. Default training, rewards, runtime authority and canonical cases are unchanged.

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
from the current GRPO helper, which normalizes each group separately. No change to canonical GRPO
is made. The full upstream rollout selection/clipped update and benchmark run are unimplemented.
Hosts must independently admit actions, rewards and datasets before using any proposed objective.

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
