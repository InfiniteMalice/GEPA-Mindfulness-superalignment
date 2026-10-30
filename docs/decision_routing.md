# Structured decisions and native-depth research

**Maturity: Experimental structured-routing adapter; Diagnostic recorded-depth analysis.**
Live Jev/LEV services, native loop models, trained stoppers and empirically qualified
calibration remain **Unsupported**. This integration installs no models, clients or new
dependencies. It executes only explicitly configured host callbacks and grants no runtime,
training, verifier or promotion authority.

## Source and adaptation boundaries

[Wang et al., Jev-LiteMAS, arXiv:2610.21411v1](https://arxiv.org/abs/2610.21411v1),
2026-10-15, CC BY 4.0, separates bounded coordination from generative work and corrective
feedback. Sections 4.1–4.3 were checked 2026-10-30. Its HTML contains uncorrected ACM
template venue/DOI text; that placeholder is not bibliographic evidence. No authoritative
implementation repository has been established and no upstream code is copied.

[Song et al., LEV, arXiv:2610.20951v1](https://arxiv.org/abs/2610.20951v1),
2026-10-14, CC BY 4.0, uses a natively looped backbone and per-depth decision readouts.
Sections 4.1–4.3 were checked 2026-10-30. The paper links
[OpenLAIR/lev](https://github.com/OpenLAIR/lev), checked revision
`fd584e9b4893ed040054e10995a77aa339ec3313`; its root LICENSE is Apache-2.0,
Git blob `e1ece45b775283628888a36bc75f56078cee9de0`. No upstream code is copied.

The [Jev method note](recommendations/source_artifacts/jev-litemas-2610.21411v1-method.txt)
and [LEV method note](recommendations/source_artifacts/lev-2610.20951v1-method.txt) are authored
summaries, not publisher bytes. Their exact identities/digests are recorded in the
[integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md#source-intake-and-adaptation-boundary).
Earlier successful primary-source reads are retained in the source audit; a later fetch
failure does not imply the sources were never checked.

This implementation is a GEPA adaptation of the decision interface, not a full MAS or LEV
reproduction. It scores the eight existing `RecommendedAction` values, retains raw errors,
then applies the existing trusted guard. Unlike a permissive screening cascade, a model
pass cannot waive verification or authorization. Expert dispatch, memory-specific selection,
generative repair execution and independent evidence collection remain host responsibilities.
The Constitution, canonical 17 cases, case 0 fallback and `CASE × STRIPE × REPEAT` stay intact.

## Live adapter and public state

[decision_routing.py](../gepa_mindfulness/factuality_observability/decision_routing.py)
reuses `prepare_routing()` and `evaluate_routing_backend()` from the
[existing router](system_one_routing.md). The guard remains the sole owner of its required
route and allowed actions. `ACCEPT` proposes continuation; it is not an execution grant.

`PublicDecisionContext` binds the prepared state's SHA-256 input digest to public task text,
an artifact and observable evidence references. The host must establish that this context
actually belongs to the task; a hash does not authenticate a source. Do not include private
thought, hidden labels, privileged reviewer context, credentials or protected test feedback.
`DecisionPacket` adds the exact canonical prepared routing features. Changes to the task,
artifact, features or source references change the packet digest.

`DecisionBackend(name, version, model_version, kind, predict)` is a configured host callback.
Kinds `jev` and `generative` identify experiment arms; a kind string proves no model architecture.
The callback receives a detached packet and returns `DecisionReadout`. Probabilities follow
`ACTIONS = tuple(RecommendedAction)` in the existing enum order, with one finite probability
per action, a sum within binary64 absolute tolerance `1e-12`, and stable enum-order tie breaks.
Readouts bind the exact packet digest, model version and depth 1. They also retain captured
input/output token counts and an optional calibration-artifact digest. These captures are
not independently verified cost or calibration evidence.

Call `route_decision(request, public=..., backend=..., confidence_threshold=...,
policy=RoutingPolicy(enabled=True))`. With the default policy, it returns `None` before
reading any input or invoking a callback. The confidence threshold must be explicitly
selected by the host using development/calibration data; it is not a paper hyperparameter.

Low-confidence predictions propose human escalation. Malformed, misbound, wrong-model,
wrong-depth, mutated or exceptional outputs also escalate. High-confidence inadmissible
proposals fall back through the established guard; raw predictions remain separately visible.
Exceptions retain a fixed category rather than untrusted exception text. No proposed route
creates reward credit, a verifier finding or an authorization token. The host enforces callback
timeouts, spending/resource limits and downstream grants; this Python interface is not a sandbox.

All new record schemas round-trip with exact fields and fixed DEVELOPMENT/non-authority markers.
Context limits are 8192 task characters, 65536 artifact characters, 32 evidence references and
65536 routing-feature JSON characters. The existing 4 MiB/depth-32 envelope limit also applies.
The complete retained live result and benchmark packet/feature/label metadata capacity are
reserved before callbacks. Oversized work must be explicitly sharded, not silently truncated.

## Matched error and calibration diagnostics

[benchmark_decisions](../evaluation/decision_routing.py) uses the existing
`evaluation.system_one_benchmark.RoutingCase` and prefreezes every state, label and public
context before any callback. Supply one public context per case and distinct versioned
backends. Supply `label_evaluator` using the existing `TrustedEvaluatorContract` to retain the label
owner's identity, evaluator version and contract separately from candidate model versions.
This declared identity is not evidence authentication. The comparison never
downloads models or starts services. Cases/arms are limited to 128/8 and the bounded report
size. Use consistent case cells, seeds, public inputs and budgets across shards/arms.

The report retains each raw prediction, confidence escalation, final guarded result, backend
failure and callback latency. It reports raw/guarded correctness separately so successful guard
correction cannot conceal a poor backend. A pass means `ACCEPT`. A review route means external
checking, decomposition/verification or human escalation. Review-required cases have only review
routes among their independently supplied expected actions; recall is `None` when there are none.
False passes are ACCEPT proposals outside expected routes. False escalations require an expected
set containing no review route. Expected sets that permit either review or ordinary continuation
do not count as review-required or unnecessary-review cases.

Top-label Brier score and equal-width ECE reuse the existing calibration suite, on valid raw
readouts only. They evaluate confidence in the selected action against independently supplied
acceptable routes, rather than conflating route confidence with task-answer truth. Empty samples
remain `None`. Expected labels never reach the backend. Token totals sum valid captured readouts;
`token_counts_complete=false` marks missing counts after failures. Counts are not authenticated
billing/FLOP measurements. Latency covers the callback plus bounded transport/readout validation,
and excludes preparation. No automatic winner, calibration certificate or promotion is produced.

Compare ordinary bounded/rule routing, generative decisions and Jev-style callbacks on the same
existing cells. Match the old rule baseline through `benchmark_routing` and report both raw and
guarded metrics. Add compatible native-depth arms only when actually available. Include high-stakes
ambiguity, necessary refusal/escalation, stale evidence, clarification/resumption and option-order
robustness. Prefer verified error and escalation recall over speed; audit erroneous deep agreement.
Use multiple seeds and paired uncertainty estimates with the existing improvement statistics.
No comparative effectiveness experiment is established by CPU fixture tests.

## Optional LEV interface and recorded traces

`NativeLoopBackend` is a Protocol, not an implemented model. An operator-owned runtime must
encode a request once, resume hidden state at each `step`, and share its checkpoint/readout head.
`LoopedBackendSpec` captures architecture, checkpoint digest, model version, evidence and depth
cap (1–16), plus declarations of native loop pretraining, state resumption and shared readout.
An operator must independently verify these facts; booleans and architecture names are not proof.
Arbitrarily repeating a conventional model is not a faithful LEV implementation.

`analyze_depth_trace(..., enabled=True)` requires a full contiguous, packet/model-bound trace
through the declared deepest step. It rejects incompatible declarations and partial/mixed traces.
It compares fixed-depth readouts, an explicit confidence threshold and supplied per-depth
agreement-stopper estimates. The stopper target is agreement with the deepest prediction,
which can be wrong. Optional expected actions remain an independent diagnostic label input.
No stopper fitting, automatic label acquisition or routing execution occurs in this function.

The analyzer reports actual executed depth separately from the simulated selected depths.
Because the full trace already ran, `actual_compute_savings_established` is always false.
It does not equate fewer selected steps with saved wall time, tokens or FLOPs. Real adaptive
execution needs a compatible native runtime, actual metering and protected behavioral evaluation.
Calibration identity, memory/context integrity and backend architecture must be independently
reviewed before attaching that runtime. No accepted baseline or protected verifier is changed.

CPU checks: `python -m pytest -q tests/test_decision_routing.py tests/test_decision_routing_benchmark.py`.
The [disabled example](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/blob/1d6b29a8092573baed9e125c988737af06d273e7/configs/decision_routing.example.json) leaves operator thresholds,
calibration bins and external backend configuration unspecified. No external API spend or model
training has been run for this integration.
