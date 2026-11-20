# Recorded episode metacognitive monitoring

`verification.metacognitive_peo.monitor_recorded_episode` is disabled by default. Enabled calls
require a `RecordedEpisode` with a protected causal cutoff, the original source snapshot and
explicit existing diagnostic inputs. It returns an immutable `RecordedEpisodeReport` containing
the derived assessment and detached diagnostic records. It performs no execution or evidence write.

## Capture and temporal boundaries

The trusted host calls `bind_source_snapshot` with the known source snapshot, earlier events and
the exact diagnostic inputs at the decision cutoff, then protects the returned event and original
bytes in its append-only trace. The helper proposes a record; it does not write or authenticate a
trace. The optional `metacognitive_source_snapshot` event uses the existing immutable research
envelope requirements and binds run, case, stripe, repeat, seed and system versions.
Hosts must record earlier events under one nonblank `trace_id`, with consistent conversation
and rollout IDs, before capture. A valid PEO sequence alone does not establish that trace scope.

Three digests bind the original source snapshot, earlier event bytes, and diagnostic inputs
(outcome bindings, continuity requests, loop steps, evidence-use records and failure graph).
The adapter rejects edited sources, rewritten events, omitted bindings, forward parents,
nonchronological prefixes, mismatched evaluation coordinates and future evidence-quality records.
It reads only through the cutoff; even an unreadable later suffix is ignored. Source event IDs
must name the exact captured prefix. At most 1,024 events are inspected.
Archive growth after the cutoff does not affect replay. Original evidence-use assessments remain
in the report alongside separate cutoff-time freshness assessments. Calibration coverage counts
all bound outcomes and keeps the domain partial when any label or confidence is unavailable.

Digests establish byte integrity against a host-protected capture. They do not authenticate an
actor-controlled replacement of both data and digests. The host must protect capture ownership,
event custody, source classification and the association of an evaluation cell with its episode.

## Existing contracts reused

The adapter runs existing action-bound sequence and epistemic reconciliation validation. A
nonzero innovation does not determine whether the model, assumption or environment is wrong.
Original estimates, conflicting evidence, hypotheses and supersession history remain retained.

Existing `PEOContinuityRequest` / `audit_peo_continuity` determine retention, retrieval, influence,
unresolved omission and legitimate revision/scope-change classifications. Their events must be
unchanged members of the captured prefix, and their evidence-use state must match original
claims. Findings preserve the existing `causal_or_motive_claim=False` boundary.

Existing `LoopStep` / `audit_loops` distinguish novel, unchanged-state, exempt and informative
repeats. Unresolved repeats recommend comparison; they do not automatically establish
unproductivity, penalties or branch stopping. Existing `FailureGraph` snapshots retain their
localization and declared causal status. Nodes must bind captured event times and observable
source references; graph status is not upgraded into semantic proof by the adapter.

`EvidenceUseAssessment` retains reliability, trust, authority, integrity, freshness, original
claims and measurements. Numeric limitations remain diagnostic evidence requests, not reward
changes or authorization. The adapter neither reweights variances nor deletes inconvenient data.

## Outcome and strategy diagnostics

`OutcomeBinding` connects a recorded prediction, action, optional observation and optional
verification event. Strategy selection must have been declared in the prediction. Reports keep
selected, executed, observed, observation-verified, label-verified and verified-progress states
separate. A verified observation need not show progress. Unverified success claims remain unknown.

An observed JSON `correct` or `progress` label is not independent verification by itself. Using
those labels requires a separate host-owned `TrustedEvaluatorContract` with contract ID
`metacognitive-observed-labels-v1` and matching verifier identity/version. This policy input must
not be enrolled from model output or an untrusted episode file. It is a declaration the host
authenticates, not a Python sandbox. Local/capture verification cannot silently become accuracy
verification. Conflicting verifier results are preserved and cannot be cherry-picked.

Calibration reuses existing accuracy, Brier and ECE conventions. High-confidence warnings require
an explicitly supplied existing host threshold; the adapter changes no calibration threshold.
Missing labels or verifier support remain unavailable. No self-report or private reasoning is
used as independent accuracy, progress, reward or authority evidence.

## Verification and limitations

Run `python -m pytest -q tests/test_metacognitive_peo.py tests/test_peo_continuity.py
tests/test_epistemic_reconciliation.py tests/test_deloop.py tests/test_failure_graph.py`.
The tests cover cutoff isolation, missing/future references, source and diagnostic-input edits,
unknown labels, false verification, verifier conflicts, continuity, repetition, original graph
status and detached report round trips. Existing specialized records retain their own schemas;
importing a report validates its wrapper and assessment, not authenticity of its diagnostic tables.

Source-based and specialized findings share the bounded partial view and exact omission counts.
The adapter accepts up to 64 continuity/evidence-use records, 512 loop/outcome records, and a
failure graph with at most 512 nodes and 4,096 edges. Original host history is not pruned.
Host resource and storage controls still apply. No model effectiveness or alignment improvement
has been measured by these CPU contract tests.
