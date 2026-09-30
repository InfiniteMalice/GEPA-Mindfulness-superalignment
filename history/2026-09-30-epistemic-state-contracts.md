# PR-1: temporal epistemic record contracts

## Reconciliation at main d33dc7c

Existing: `action_bound_events.py` freezes predictions, actions, observations and verifier
records. `event_sequence.py` validates causal order, run/repeat identity and fixed model/harness
versions. `ConfidenceSource` is shared by runtime and evaluation. `EvidenceReference`,
`EvidenceState`, verification interfaces and runtime governance already separate information,
assessment and permission. Verified process reward, skill lifecycle, offline epochs and
coevolution have independent admission contracts.

Partial: `fuse_confidence()` combines current signals heuristically; it has no temporal state.
SoT snapshots, public epistemic commitments, memory safety and motivated-forgetting audits retain
continuity diagnostics, but do not represent prior/measurement/posterior uncertainty. Synthetic
generators and participatory-agency phases supply cases and curricula, not longitudinal worlds.
Experimental overlays retain hypotheses and topology diagnostics. No JEV backend was found.

Missing: typed temporal estimates, measurements, innovations and uncertainty-update records.
PR-1 is the earliest missing prerequisite. Later causal reconciliation, estimation, fusion,
training and promotion integration remain separate work.

Redundant: a new event stream, evidence vocabulary, canonical case, authority subsystem, reward
interface, curriculum framework or research registry would duplicate existing ownership.

Experimental: new records are diagnostic, explicitly constructed offline and unused by runtime.
SoT and existing overlays remain gated. No automatic producer or runtime flag is needed for an
inert record contract.

## Design and trade-offs

Add `gepa_mindfulness/verification/epistemic_state.py` beside the existing evidence state.
Reuse `EvidenceReference`, `ConfidenceSource`, `EvaluatedSystemVersion` and `EventEnvelope`.
An `EpistemicContext` groups the existing run/repeat/version identity and checks an explicitly
supplied event's identity. This check neither resolves references nor validates causal order.

Four frozen records preserve evidence and nonempty producer provenance:
`EpistemicStateEstimate`, `EpistemicMeasurement`, `InnovationRecord`, `UncertaintyUpdateRecord`.
World/model/monitor uncertainty remain separate optional normalized [0,1] diagnostics.
Availability is numeric availability, not evidence support, source trust or authority.
Unavailable records contain null numeric fields; missing variance never means zero variance.

Prefer a `DiagonalState` with a named representation, ordered dimension names, finite values and
optional nonnegative diagonal variances. This is symmetric positive semidefinite by construction.
Reject full matrices rather than silently discard cross-covariances. A general matrix validator
and a new numerical dependency are unnecessary for this contract stage. Semantic normalization
is producer-defined by estimator version; numerical variances are in squared dimension units.

Innovations retain scalar predicted/actual measurements, computed signed residuals, prediction
and observation IDs, mismatch status and optional positive innovation variance. Normalization
requires an explicit basis reference; the record does not validate that statistical assumption.
Update records retain complete prior/posterior snapshots and measurements, require matching
evaluation identities, preserve measurement evidence, and label correlation treatment explicitly.
Unknown correlation defaults to unresolved; no mode executes fusion.

## Non-goals and compatibility

No changes to event types, sequence validation, existing JSON, `fuse_confidence()`, routing,
rewards, training, memory persistence, execution or authority. No mathematical estimator,
source-reliability learner, verified-measurement promotion or causal reconciliation adapter.
Exactly 17 canonical cases remain. Existing callers need no migration.

## Research

The canonical WMLLM/DWM summaries support prospective predictions and distinct observations.
Kalman (1960), DOI 10.1115/1.3662552, provides recursive linear estimation and error covariance
under explicit model assumptions. The primary paper was read at the CMU-hosted ASME copy.
Transfer here is record structure only. Semantic state estimation remains a design hypothesis.
Add Kalman to the existing reference registry with a DOI URL and null arXiv ID; do not fabricate
an arXiv identifier. Preserve the current registry's reciprocal recommendation checks.

## Implementation and validation plan

1. Add failing tests in `tests/test_epistemic_state.py` for round trips, immutable snapshots,
   unavailable versus zero, malformed numbers/diagonal covariance, shape and identity mismatches,
   provenance, unresolved correlation and compatibility with existing events/rewards.
2. Implement the inert records and narrow validation helpers. Run the contract suite.
3. Add DOI-only registry tests; extend `evaluation/recommendations.py` and the existing canonical
   YAML/Markdown registries for Kalman under REC-002. Preserve previous reference metadata.
4. Document the API and its trust limits in `docs/epistemic_state.md`, link the docs index and
   existing event/verification guidance. Run focused existing event, reward, authority and
   continuity regressions, registry consistency, full pytest where available, Ruff, Black,
   type checks and wheel build/smoke.
5. Review the complete diff and documentation precision, fix actionable findings, and create
   a draft PR with scope, limitations, quality evidence and PR-2 dependencies.

Review focus: booleans masquerading as numbers; unknown correlation becoming independence;
mutable caller containers changing saved predictions; run/repeat/version drift; unsupported
normalization or covariance being represented as measured certainty. Tests cover these inputs.

The user explicitly authorizes proceeding without intermediate approval unless inspection finds
a genuine architectural ambiguity. Superpowers planning/TDD and repo-quality-gate are applied;
planning lives under `history/` per AGENTS.md, and no Beads commands or tracking files are changed.

## Quality-gate evidence

Baseline: 121 existing action-event, sequence and verified-process tests passed before the new
module was present. The new contract suite initially failed because the requested module did not
exist. The new Kalman registry expectation also failed before registry implementation.

After implementation, the full repository suite passed: 3,345 passed, 18 skipped, 16 warnings
on Python 3.12, with HF/Transformers offline. After the independent review, 10 additional
regression tests were added; the latest focused event/evidence/authority/reward/continuity,
confidence and registry suite passed 455 tests. These extra tests cover invalid repeat IDs,
unsupported schema versions, detached JSON payloads and caller-owned measurement lists.

Repository-wide Ruff and Black checks passed. The changed record and registry modules passed
mypy. A wheel built and installed into a separate environment; outside the checkout, the new
module imported from that environment, numerical/context records round-tripped, the packaged
guide existed, the Kalman registry entry loaded and `gepa --help` succeeded.

A fresh read-only reviewer found no actionable correctness or documentation issues. The review
did not independently run tests. Deferred concerns are the explicitly later-stage causal,
estimator and correlation algorithms. Full covariance is deliberately unsupported. Different
estimator versions may be retained across snapshots; model/harness evaluation identity cannot
change within an update. This preserves diagnostic provenance without treating method identity
as runtime authority.

Documentation precision: PASS after correcting REC-002's exact registry-derived evidence list.
The contract guide identifies actors, fields, failure behavior, tests and host-review limits.
No unresolved BLOCK findings. No runtime behavior, reward weight, canonical case or permission
changed. Statistical effectiveness, hardware qualification and deployment authentication remain
unestablished by these local tests.
