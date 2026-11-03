# Live bounded routing and authorized host dispatch

Implement the operator's requested Jev-style separation of bounded decision,
generative corrective review, and actual execution. User has opted out of research
runs and permission/planning round trips. Use the approved quality-gate fallback,
inline TDD, one independent read-only review, and source/wheel/CI evidence.

Reuse PublicDecisionContext, DecisionBackend/Readout/Assessment, RoutingPolicy and
the existing causal-evidence guard. Do not introduce action taxonomy or an alternate
runtime authority. Default off. An enabled call performs one cheap bounded decision,
optionally one separately configured generative review on uncertainty/error, with
public corrective feedback. Every proposal goes through the same existing guard.
No unbounded clarification/review loops and no model self-certification.

Prepare a bounded detached DEVELOPMENT execution packet retaining both readouts,
backend/calibration versions, public source/features, and chosen route. Execute
only the corresponding operator-installed handler, only after fresh exact-context
host approval and existing one-use RuntimeAuthorization consumption. Owner-issued
action scope binds the entire packet and final route. No route, reviewer agreement,
export or proposal confers permission; absent/misbound/expired/replayed grants do
not invoke a handler. Uncertain high-stakes routes remain external/human review.
The host handler must log real PEO observations and independently verify outcomes.
No reward, task-success assertion, canonical-case mutation or deployment follows.

The native LEV backend Protocol and recorded-depth diagnostics stay explicitly
optional; without a compatible pretrained looped model, do not relabel conventional
repeated calls as LEV or claim computation savings. Record why/how future error,
escalation, latency/token and protected-case comparisons would qualify routing.

Initial verification: 159 existing routing/authority baseline checks passed; new
module import failed RED before implementation. 181 focused checks now pass,
including bounded review, actual matched dispatch, conservative verification
guard, source-authentication denial, missing handlers, stale bindings, consumed
grants, mutation, reviewer failure and oversized feedback. The latter was
reproduced RED before retaining escalation without invoking the reviewer.

One fresh independent whole-feature reviewer found no required issues and
independently passed all 181 checks. Declined empirical calibration/value claims,
unsupported native LEV backbones and trusted callback sandbox guarantees, matching
the documented scope. Black, Python 3.10 syntax, 100 columns and pure import/closed
config without Torch/NumPy pass. The corrected dataset dependency is incorporated;
its public Unicode/path findings are fixed in the parent. Complete remote CI and
public review remain required before merge. No research run or speed/accuracy
improvement is claimed.

Public review found the stale-scope test depended on unequal real callback
latencies. Equal bound timer captures reproduced a PermissionError rather than
the assumed binding mismatch RED. The test now controls all six timestamps
across its three calls; distinct scope latencies are deterministic. The separate
consumed-fresh-grant test still independently checks one-use rejection. No
production behavior or assertion strength is changed.
