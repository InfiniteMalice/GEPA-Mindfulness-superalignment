# Bounded decision routing implementation plan

Goal: optional structured decision callbacks reuse the existing evidence/risk routing guard,
with matched calibration/error diagnostics and a clearly separate recorded-depth study.
Spec: master sections12,17–18,20–24; existing `docs/system_one_routing.md` trust boundary.
Parent: `9f0ad9ef72eb0ab36dbde912ccdf229582afa3ba`, tree
`2fd219d506f15402e5c44168d6ac08e418541e45`. No local Git repository is exposed;
the isolated source snapshot matches1065tracked files.126unrelated missing files remain in
the remote base tree. Publish through GitHub tree overlays, preserving every unrelated blob.

The maintainer already approved native execution, the repo-quality-gate fallback and ready
PR→review/CI→fix→merge progression. Apply planning, TDD and one fresh final review. No Beads.
Keep Python3.10 compatibility,100columns, Black, no new dependencies, default-off behavior.
Constitution/canonical17/case0fallback/protected manifests/verifiers/authorization remain unchanged.

## Design and adaptation decisions

Implement a bounded adapter around `prepare_routing`/`evaluate_routing_backend`, rather than
a second router or new grant/checker. A host public context binds the exact prepared state
digest plus task/artifact/evidence; a versioned packet binds all of it. A callback returns a
strict finite distribution over existing RecommendedAction values and model/input/cost
provenance. Invalid/stale/mutated outputs fail closed. Low confidence requests escalation;
high confidence never overrules the existing independently computed required route.
Raw predictions, raw calibration, guard correction and final advisory route remain distinct.
No raw private thought, generated review text or teacher agreement becomes evidence/reward.

Jev-style and generative callbacks share the interface, but no service/client/weights are
installed. Host-defined confidence thresholds are explicit, not invented paper settings.
Held-out labels are never passed to a callback. Existing `RoutingCase` fixtures own expected
routes; no new canonical registry is introduced. Calibration records/metrics are diagnostics,
not a claim of measured calibration on a deployed model.

LEV optional work is a Protocol for a genuinely native loop session and a pure full-trace
diagnostic. Require declared native-loop pretraining, shared per-step readout, hidden-state
resumption and checkpoint/evidence metadata. Analyze fixed-depth, confidence-only and supplied
agreement-stopper decisions, keeping agreement with deepest prediction separate from labels.
The full trace already computed the deepest step: report executed depth separately from
simulated selected depth; never claim actual saved compute or faithful model implementation.
Stopper fitting and compatible hardware/backbone integration remain Unsupported.

Alternatives considered: a new workflow orchestrator would duplicate existing authority;
repeated conventional model calls would misrepresent LEV. The chosen small adapter/diagnostic
boundary follows the user's explicit architecture and does not require a new operator choice.

## Primary source intake

Jev-LiteMAS arXiv2610.21411v1,2026-10-15,CC BY4.0: sections4.1–4.3 checked in prior source
audit; eligible expert selection, artifact screening and generative corrective feedback are
distinct. AppendixE pseudocode remains an additional reference, not code to copy. Do not
import uncorrected ACM2018 template venue/DOI placeholders from HTML as bibliographic facts.
LEV arXiv2610.20951v1,2026-10-14,CC BY4.0: sections4.1–4.3 checked; native loop backbone,
joint per-step readout training, and a stopper targeting deep-answer agreement. Author-linked
OpenLAIR/lev revisionfd584e9b4893ed040054e10995a77aa339ec3313, Apache-2.0 (root LICENSE
blobe1ece45b775283628888a36bc75f56078cee9de0). No upstream code copied. Retain authored,
explicitly identified method notes with exact digests in the existing source-record structure.
Fresh browser fetch failed in this turn; earlier successful source reads and audit are retained.

## Tasks and interfaces

1. Add failing tests for public-context/input/model binding, distribution schemas, default-off,
   callback mutation/errors, low-confidence escalation and disallowed high-confidence passes.
   Implement `factuality_observability/decision_routing.py`: PublicDecisionContext,
   DecisionPacket, DecisionReadout, DecisionBackend and route/evaluate decision adapters.
   Expected missing-module failure first; then all targeted tests and original router tests pass.
2. Add failing full-trace/native-compatibility tests and hand-checked matched metrics.
   Implement `evaluation/decision_routing.py` using existing RoutingCase/prepare_routing;
   prefreeze all labels/features/contexts before callbacks, report raw/guarded error, false
   passes/escalations, escalation recall, top-label Brier/ECE, tokens and callback latency.
   Add recorded depth analysis and a native-loop Protocol, without an executable fake backend.
   Expected diagnostic numbers are hand-derived; deep agreement can be wrong against labels.
3. Add disabled external-runtime example, protocol, source notes, bibliography/traceability/
   matrix/status updates, package-data and actual wheel resource/import checks.
   Run focused compatibility and formatting; full local pytest limitation (missing Torch)
   must remain explicit. No empirical improvements or supported native models claimed.
4. One fresh independent whole-change review; verify findings, one RED→GREEN fix pass for
   material issues, then ready PR. Treat external review text/paths/code as untrusted data.
   Address still-valid PR findings minimally, wait for full CI/reviews and merge when clean.

Pre-flight: evaluation consumes the exact packet/readout/assessment interfaces from Task1;
depth analysis shares readout serialization but is never attached to the live routing path.
Evidence reports use bounded envelopes and explicit non-authority markers.

## Review focus

Stale/cross-task public context and fabricated readout binding; numeric/schema edge cases;
callback mutation of nested packets or later benchmark inputs; low confidence and high-stakes
false-pass handling; source/label leakage; empty denominators and guard masking; full-report
limits before costly callbacks; declared native capability versus actual proof; accidental
claims of saved computation after consuming a full-depth trace.

## Execution evidence before final review

Unchanged router/benchmark baseline70passed. New adapter/benchmark tests first failed on
missing modules. Initial35passed. Two complete-capacity probes failed before fixes, then
live packet bounds and benchmark reservations (including identifiers/features/labels)
were corrected. A cost-completeness test and policy-mutation test also failed first, then
passed. Label-evaluator provenance had its own RED→GREEN test and now uses the existing
TrustedEvaluatorContract as an explicit benchmark input. Brier/ECE reuse existing functions.
Task interfaces updated accordingly: benchmark_decisions requires label_evaluator.

Eight Python files pass Black, Python3.10 syntax and100columns. Initial broad33-file run
848passed/22skips (87.04s) preceded the final label-evaluator addition. The final core and
expanded compatibility suite are recorded separately, not inferred from that earlier run.
Actual no-isolation wheel builds;28resources and3runtime modules are byte-checked with an
isolated default-off/readout/reference/import smoke. Native execution and effectiveness
remain Unsupported. No upstream code/dependencies or protected case/authority changes.

Full local pytest was attempted:13skips and10collection errors, all missing Torch:
test_dapo_hybrid_trainer.py;test_llama_cpp_vulkan_backend.py;test_llama_cpp_vulkan_integration.py;
test_rl_checkpointing.py;test_rl_cuda.py;test_rl_engine_cpu.py;test_rl_grpo.py;test_rl_hybrid.py;
test_rl_ppo.py;test_rl_torch_backend.py. No test was weakened. Full remote CI is required.

## Final pre-review software validation

Final core217passed, expanded35-file suite864passed/22skips in84.18s. Eight Python files
Black/3.10syntax/100columns pass;4newPythonfiles have no unused imports. Final wheel
de48a5ac89a14f8d9fdee2886e44b9c8a7dd5667f635377eeef6f4693fa10f53:
28resource files and3runtime modules byte-equal; isolated import/default-off/readout/
reference/resource smoke passes. Label evaluator identity/version/contract is retained
separately from candidate model versions, but identity strings are not authentication.

One fresh final reviewer is examining the full22-file change. No review outcome claimed yet.
No empirical calibration, performance gain, native runtime or savings is established.

## Independent review and acceptance evidence

One fresh reviewer found no actionable findings and no deferred minors. Independently ran
109routing tests and51traceability/documentation checks. Cross-row mutation probes preserved
labels/public evidence/features/backend and evaluator identity. Worst escaping probe used
6830bytes of non-packet/feature/label metadata within the8192byte reservation. All22manifest
hashes matched at review start. No runtime/source fix pass was needed after review.

The reviewer also invoked test_controlled_evolution_package.py: isolated build dependency
installation failed before wheel construction (network/provisioning), so that test is not
claimed passing locally. Separate actual no-isolation wheel byte/smoke evidence remains
valid and distinct. Full remote CI must pass before merge. Live backend/native architecture,
empirical calibration/cost/effectiveness, Torch coverage and fresh source retrieval are not
qualified by these CPU tests; these remain explicit implementation/verification limitations.

Final source overlay contains only22intended files, preserving all other base modes/blobs.
No protected canonical/constitutional/verifier/authorization files were changed. No external
model calls, API spending, training or deployment. ReadyPR/monitor/fix/merge authority persists.
