# Verified credit runtime implementation plan

Goal: let the existing canonical CPU trainer consume explicitly host-authorized knowledge,
progress and verified process contributions through the existing reward pipeline and durable
research-credit ledger. This closes a prerequisite for the remaining requested runtime features;
it is software integration, not a research experiment or an effectiveness claim.

Architecture: a default-off ResearchRewardProvider wraps RewardPipeline. It snapshots a proposal,
checks the existing eligibility/audit contracts and exact host scope, scores the baseline once,
then requires separate fresh semantic and optimizer-use callbacks. ResearchCreditLedger performs
its existing independent accounting review and durable event reservation. Only a privately
constructed live ResearchRewardDecision exposes optimizer_reward and target-bound trajectory
binding. Exported records remain DEVELOPMENT. No new authority registry or acceptance policy.

Tech stack: Python 3.10+, existing dataclasses/JSON/SQLite contracts; optional existing PyTorch
backend only in the native CPU acceptance test. No new package dependencies.

Spec: original master prompt sections 3, 8–10, 17 and 22, as narrowed by the operator's request
to skip research studies and enable working features. Existing research_credit_accounting.md,
VERIFICATION_AND_RUNTIME_AUTHORITY.md and epistemic_process_rewards.md retain authority.

Approaches: reusing the existing pipeline, audit and ledger preserves the established owners.
Directly training from exported diagnostics would violate eligibility. Creating a separate reward
or governance registry would duplicate authority. The selected approach adds explicit live host
hooks using the same defensive callback pattern as CausalRewardAdapter.

Execution: Superpowers planning, native inline implementation, TDD, one fresh whole-branch
review and final evidence. repo-quality-gate is unavailable; the maintainer already authorized
spec/design → implementation → tests → quality review → evidence. Design/execution/merge approval
is covered by the supplied specification and standing instruction to continue without waiting.
No bd or .beads operations. Repository guidance keeps planning here and forbids Markdown task
checkboxes, so this plan uses task paragraphs instead of the skill template's checklist.

Global constraints: default-off compatibility; no diagnostic or raw teacher agreement credit;
explicit TRAIN and recursive held-out checks; positive-only process bonuses; immutable source
requests and evaluator/version/split provenance; independent admissibility and optimizer review;
unknown judgment withholds optimizer access rather than inventing a moral violation. Coefficients
only weight eligible research objectives, never normative commitments. No downloads, paid APIs,
large-model training or deployment. Canonical 17/case-0/CASE × STRIPE × REPEAT remain unchanged.

Review focus:
- Caller/callback mutation or serialization must not mint or change live authority.
- Reusing an event after restart, changing use IDs or selecting overlapping contributions must fail.
- A matched teacher, diagnostic, private source or nested holdout must not reach optimization.
- Signed progress must retain observable negative evidence; baseline validation must remain intact.
- A valid-looking claim with stale action/model/dataset/run/task/harness/policy binding must fail.

Task 1: live provider and policy (research_credit_runtime.py; test_research_credit_runtime.py).
Consumes RewardRequest, ResearchCreditClaim/Use/Ledger, RewardPipeline, require_training_eligible,
audit_research_credit and the existing mutation-safe host callback helper. Produces
ResearchRewardPolicy, ResearchRewardSubmission, ResearchRewardProvider.score(request) and a
non-restorable ResearchRewardDecision. The policy pins policy/run/task/harness/dataset identities,
explicit coefficients for knowledge/progress/process, enabled=False and a finite adjustment budget
at most 1. Score validates all selected claims and bindings before callbacks, keeps raw and weighted
components separate, and reserves only after both semantic and optimizer-use review succeed.
Tests first: default-off exact pipeline behavior, accepted/rejected live scalar access, immutable
source/callback snapshots, all strict policy fields/versions, unknown/rejected/private/held-out
and teacher-only signals, foreign system/evidence bindings, positive-only process, budget overflow,
durable replay, overlapping events, missing/exceptional/truthy/mutating callbacks. Observe RED,
implement, then run all credit/ledger/reward/eligibility/causal regression tests. Expected: all pass.

Task 2: canonical trainer consumption and operational guide (engine.py, test_rl_engine_cpu.py,
research_credit_accounting.md and existing integration matrix/guide).
Consumes Task 1's exact live decision type. _reward_value calls optimizer_reward only for that
exact type; _bind_reward requests its original-target-bound immutable scored trajectory. Existing
numeric/RewardAssessment behavior and staleness guards remain unchanged. The provider prepares
its trajectory using existing baseline validation and observable signed-component provenance;
no exported diagnostic total gets a generic optimizer path. Native acceptance performs a bounded
CPU PPO step with host-supplied reviewed claims and asserts changed policy parameters and retained
component provenance; a rejected review leaves policy parameters unchanged. Add tests for request
substitution, negative progress, negative baseline without evidence and generic forged totals.
Observe RED, implement and run related engine/config/checkpoint/credit suites. Document the Python
host configuration (callbacks are not loaded from candidate-controlled YAML), why/how, defaults,
rollback/replay limits and source adaptation. Existing docs are already wheel resources.

Pre-flight: Task 2 consumes Task 1's exact live type/optimizer_reward/trajectory binding. Both use
one immutable JSON review payload and the same target request digest. Baseline binding delegates
to the current engine checks, so no weaker negative-reward path is added. Live decisions cannot
be restored or nested inside an ordinary reward pipeline; exported diagnostics stay DEVELOPMENT.

Acceptance: focused deterministic CPU suites, canonical native train step, Python 3.10 syntax,
100 columns, Ruff/Black and wheel resources; fresh whole-branch review; final-head Python
3.10/3.12 CI and Codex/CodeRabbit threads. Publish ready, fix only verified review findings, merge
with expected head, verify main's exact accepted tree. Keep evidence outside the source tree when
possible to avoid documentation-only commits retriggering long CI. Do not claim research gains.

Task 1: implemented after the missing-runtime RED check. Baseline 133 checks passed; initial
provider/engine coverage reached 162 passes, then 397 credit/causal/eligibility/logging/native
checks and 640 extended backend/config/checkpoint checks passed. Callback rejection, mutation,
source restrictions, no-restoration, durable replay and signed progress are covered. Complete
report sizing happens before reviewer calls. Unselected held-out intake and private proposer
errors each failed before their guards passed (80 targeted credit/ledger checks).

Task 2: Ruling: checkpoint policy identity and decomposed logging require optional canonical
configuration and metric-record v3. Without them, resume could change coefficients silently and
logs would collapse raw versus weighted contributions. Absent/inactive configuration keeps the
existing default hash and legacy checkpoints. The existing v1/v2 metric contracts stay closed.
Configuration/native/metric integration tests observed 8 missing-feature failures before passing.

Task 2: Ruling: native torch.__version__ is a TorchVersion string subclass; the bounded JSON
adapter rejects it. The generated backend_version now uses str(torch.__version__), preserving
its serialized bytes while producing the declared public JSON primitive. The native acceptance
failure supplied RED evidence before this conversion passed. No JSON validator was weakened.

Task 2: Ruling: ledger structural validation is shared through _prepare_allocation, not copied
into another admission policy. Unknown/already-counted/missing prior layers and missing host
binding failed before the shared preflight and early model guard passed. Collection alone does
not need reward callbacks; it still cannot optimize or reserve an event. Float-sum overflow is a
bounded input error, not a usable score. Both edge tests were RED before these changes passed.

Verification: software effectiveness only; no research study, paid endpoint, GPU/model download,
large-model training or live deployment. Local full pytest stopped after 1195 passes/1 skip at
`test_controlled_evolution_package.py::test_wheel_bundles_readable_controlled_evolution_resources`:
isolated build dependency installation is sandbox-network limited. No test was weakened.
Non-isolated wheel build succeeds. Full final-head GitHub CI remains mandatory before merge.

Final fresh whole-branch review: 641 checks independently passed in 12.01 seconds. No Critical
or Minor findings. Two Important findings were independently reproduced: (1) an ordinary provider
wrapper could introduce live research decisions without checkpoint-bound configuration; (2) a
structured baseline replaced original evidence references required by contributions. Fixes enforce
matching enabled configuration/exact provider at actual result consumption, preserve baseline and
source evidence, and reject conflicting reference identities. Four native PPO/GRPO wrapper cases
and three positive/signed/conflicting structured-baseline cases were RED before their fixes passed.
Final: Ruling: empirical research gains and alignment effectiveness remain outside the operator's
revised scope; software fixtures establish execution only. Claiming more would create false trust.
Final: Ruling: real evaluator independence, evidence truth and constitutional admissibility remain
trusted host responsibilities. Fresh semantic/use/accounting reviews are mandatory, but a
classifier or identifier cannot prove them. A dishonest host could otherwise pay false credit.
Final: Ruling: arbitrary trusted Python/private-state/object-allocation forgery is not sandboxed;
callbacks and stores must remain outside candidate control. Treating this as a sandbox would let
candidate code mint authority in a threat model this library does not implement.
Final: Ruling: cross-journal replay, rewinding and semantic aliases require host canonicalization,
durable storage and the existing rollback/audit procedure. Same-journal replay is enforced;
ignoring those host duties could duplicate credit across distinct journals or false event IDs.
Final: Ruling: GPU/distributed/hybrid paths are explicitly Unsupported here; automatic teacher
services and method-specific objectives are subsequent runtime PRs. Assuming current support
would overstate the implemented feature and could omit required checks.
Final: Ruling: full remote-tree preservation, package contents and final remote CI are mandatory
publication/merge gates because the local snapshot is partial and isolated builds are network
limited. Skipping them could drop upstream files or miss integration failures.

Final fix-pass evidence: all 648 focused checks pass in 14.44 seconds. The two reproduced
Important findings have 7 RED→GREEN regression cases; no minor was deferred and no second
review was dispatched. The broader RL/credit/causal run passed 1298 checks with 5 explicit skips;
three environment failures remain: test_rl_adapters.py::test_adapter_import_succeeds_from_an_installed_wheel
(isolated dependency networking), test_rl_cuda.py::test_ddp_multi_trajectory_evaluation_converges_across_ranks
and test_rl_cuda.py::test_ddp_grpo_skips_all_ranks_when_only_one_rank_has_reward_variance.
An explicit GLOO_SOCKET_IFNAME=lo diagnostic confirms socket creation fails with Operation not
permitted for both DDP cases; no test or runtime networking policy was changed. Post-fix full
pytest again reached 1195 passes/1 skip before the named isolated package test. The isolated
build independently reports proxy Errno 1 Operation not permitted before project build.
Final nonisolated wheel builds and its 6 changed runtime files plus packaged guide/matrix match
source. Outside-checkout wheel import/default identity/enabled policy/v3 metric/resource smoke
passes without importing Torch/NumPy. Wheel SHA256:
895b3052b81329173c696d01bb714f862ff49525a030f0f25638f59e66b2f98b.
Full final-head remote CI, full remote-tree preservation and public review threads remain
mandatory publication/merge gates. Research effectiveness is unqualified.
