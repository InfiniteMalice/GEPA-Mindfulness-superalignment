# Enable SST in canonical training

Goal: make the existing SST optimizer transformation usable through `gepa rl` configuration,
real CPU PPO/GRPO updates and checkpoint/resume. The operator changed scope: implement requested
features; keep research to brief why/how notes. No research ablations or effectiveness claims.

Base/main: 18c056f4ce3145bc7258f7694b648534f5033749 (merged PR819), tree
bee69c35bbc2708999a559234c43b88bcca3729b. Work in isolated verified snapshot
/workspace/gepa-feature-sst-runtime; no usable Git metadata. Preserve remote-only files via
the established GitHub tree overlay. No bd/.beads operations.

Use Superpowers planning, inline execution, meaningful functional tests and one fresh final
review. repo-quality-gate is unavailable; its maintainer-approved fallback remains in force.

## Design

Reuse SSTConfig/SSTStepAdapter; no new objective, reward eligibility, case, verifier or
authority. Add an optional top-level `sst` section to RLRunConfig, disabled by default.
Only portable CPU FP32 full-weight, non-distributed runs are accepted for this integration;
the existing adapter also validates actual tensor dtype, ownership, shape and element limits.
Explicitly reject incompatible configurations before runtime/model loading where possible.

Protect every trainable two-dimensional policy parameter. Biases/nonmatrix parameters and
the separate value head use native optimizer updates and are identified as outside projection.
The adapter projects the actual native AdamW/Adam/SGD delta; gradients are not the projection
target. Existing clipping occurs before the native step. Existing numerical diagnostics stay
non-authoritative; expose applied-update magnitude/leakage through step results and logs.

Retain default backend checkpoint v2 and legacy v1 compatibility. SST-enabled backend
checkpoints use v3 with the existing SST state schema. Validate configuration, reference,
targets and counters before any restore mutation. Include SST state in existing transactional
preflight/rollback, and never recreate the protected basis from resumed weights. Default-off
configuration hashes retain old values; enabled SST settings participate in checkpoint identity.
Injected backends cannot silently ignore an enabled SST configuration.

## Task 1: configuration and compatibility

Extend real configuration/hash tests first, observe missing SST configuration failures, then
implement strict optional parsing, supported-mode checks and stable disabled hashes. Factories
pass the same immutable SST configuration; no heavy import or model load for default parsing.
Files: runtime_config.py, engine.py, backends/torch_portable.py and relevant runtime tests.

## Task 2: applied updates, checkpoint/resume and practical use

First add native backend tests comparing projected applied AdamW updates against an ordinary
optimizer control, including weight decay. Test rank extremes, refreshed/fixed reference resume,
enabled/disabled checkpoint mismatch, corrupt references/counters, rejected modes and restore
rollback. Then attach the existing adapter to TorchPolicyBackend, versioned SST checkpoint
payloads and diagnostic step records/logs. Add a real small offline engine train/checkpoint/
resume acceptance case, using existing fixtures rather than downloading a model.

Update the existing SST/portable-RL guides with configuration and one source-checkout example;
briefly explain why/how SST operates and that it does not prove behavioral retention/alignment.
No new paper, research-only PR, ablation runner or dependency is added. CPU PyTorch and existing
Transformers dependencies are installed in /workspace/gepa-sst-runtime solely for functional
verification, with no model/data/GPU download or paid runtime call.

## Acceptance and review focus

Functional optimizer projection, restart equivalence, failure atomicity, strict configuration,
legacy resume/hash preservation and no silently ignored enable flag are required. Preserve
existing reward/verifier/human authority and all 17 canonical cases. Run relevant native CPU
suites locally, existing CLI/engine acceptance and mandatory CI. One fresh whole-branch review,
ready PR, final checks/review monitoring and expected-head merge are already authorized.

Ruling: CPU integration is the next independent feature supported by existing code. CUDA,
AMP, distributed and LoRA modes must fail explicitly until implemented; treating them as
enabled would silently remove the requested protection. Training effectiveness is not an
acceptance claim. Research ablations are excluded by the operator's latest instruction.

Pre-flight: Task2 consumes Task1's exact SSTConfig field/hash semantics; checkpoint and
transaction state must change together. Default-off format/hash behavior cannot change.

Baseline evidence: 119 SST/config checks passed; native backend/engine 115 passed after
restoring original omitted 13,226 byte backend.pt and 17,661 byte training_state.pt legacy
fixtures from merged main via GitHub plugin, verifying both Git blob SHA and provenance
SHA256. No fixture/test was modified. CPU torch 2.9.1+cpu and existing Transformers
dependencies installed locally; no model or GPU artifact was downloaded. Configuration RED
5 failures missing SST setting → GREEN 158 checks, including original default config golden
hash and genuine legacy resume. Native backend RED 15 missing SST constructor cases → GREEN
103 applied-update/backend checks. Canonical engine/factory RED 4 cases showed ignored
config/missing SST report; now wiring canonical factories and metrics.

Task 2 integration found the existing closed metric schema rejected new SST names. Ruling:
add explicit SST metric record version 2, retaining v1 closed names and all other record
version 1 behavior, instead of silently changing old logs. The new version requires SST
diagnostics. Regression test failed for unsupported v2 → passed after versioned extension.
New native backend and engine/factory functional checks now pass: 471 checks 5.82 seconds,
including actual PPO/GRPO train/resume, original basis retention, rejected silent backend
downgrade, legacy checkpoints and closed metric records. No ablation/research-effectiveness
run. Documentation now explains operational why/how/usage.

2026-11-03 resumed after interrupted verification. Related suite 530 passed 15.77 seconds;
includes canonical PPO/GRPO SST train/resume, real native SGD/Adam/AdamW projections, rank
extremes, corrupt state/rollback, legacy checkpoint/hash and versioned metrics. Legacy
translation test caught silently dropped top-level SST request: RED enabled false → GREEN
after preserving the SST section. Default legacy translation and non-SST hash remain
unchanged. Python 3.10 AST/100 columns/Black passed. Whole local pytest 1195 passed/1 skipped
then stopped at test_controlled_evolution_package.py::
test_wheel_bundles_readable_controlled_evolution_resources. Independent isolated build
confirmed sandbox proxy connection Errno 1 Operation not permitted whilst installing
wheel/setuptools, before project build. No failing test weakened; full remote CI remains
mandatory. Nonisolated wheel build and 46 required resource/6 runtime byte checks passed;
outside checkout config/hash/v2 metric/doc smoke passed without Torch/NumPy imports. No
research/ablation run; this is functional verification.

Final fresh whole-branch review: all 16 paths/hashes scrutinized; 530 checks independently
passed 15.34 seconds. No Critical/Important runtime finding. One valid minor operator
documentation finding: old recovery paragraph contradicted canonical v3 automatic SST
persistence. Corrected paragraph to match tested runtime; standalone callers still own
external checkpoint coordination. No runtime change or new prose-only test; existing native
checkpoint tests provide the behavioral evidence.
Final: Ruling: research/behavioral retention/alignment/hyperparameter effectiveness/ablations
remain outside revised user scope; claiming these from software tests would overstate
evidence.
Final: Ruling: CUDA/AMP/distributed/LoRA remain unsupported and explicitly rejected; assuming
execution support would silently remove protection on untested modes.
Final: Ruling: large-model throughput/memory scaling remain unqualified; bounds are explicit
and no representative benchmark was run. Removing limits without engineering could exhaust
resources.
Final: Ruling: cross-version/degenerate SVD bitwise reproduction is not guaranteed; saved
anchors and runtime versions remain required. Assuming cross-platform bit parity would
misstate continuation.
Final: Ruling: arbitrary injected Python backends/hooks/concurrent mutation remain trusted
host code; configuration equality cannot certify arbitrary implementation behavior. Treating
it as certification would misplace trust; runtime authorization and host admission remain
independent.
Final: Ruling: full isolated packaging/full-suite/remote integration are mandatory CI gates
because local dependency installation is network blocked; merging before green would leave
integration unverified.
Final: Ruling: unchanged governance/reward/evidence/canonical contracts receive existing
regression coverage, not new empirical alignment qualification. Treating tests as alignment
proof would be unsound.

Documentation follow-through: updated existing REF-SST repository-inference/maturity
sentences in references.yaml and RESEARCH_TRACEABILITY.md because they also incorrectly
declared canonical integration absent. No source identity/citation/registry edges/policy
changed. These two matching status edits follow fresh runtime review; final remote
Codex/CodeRabbit review covers the 18 path PR. All 6 runtime file hashes remain those reviewed
and covered by 530 focused tests.
