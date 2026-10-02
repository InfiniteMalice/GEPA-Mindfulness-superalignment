# PR-16: Multi-hypothesis / Pareto epistemic state

## Intent and reconciliation

Preserve competing explanations externally when evidence does not justify a single estimate.
Expose bounded diagnostic views to the actor without granting deletion or persistence authority.

| Status | Existing evidence and decision |
| --- | --- |
| Existing | `evaluation.experimental_records.HypothesisSet` and the disabled `competing_hypotheses` flag. Reuse for actor diagnostics and gating. |
| Partial | `UncertaintyUpdateRecord.unresolved_hypotheses` retains names, not competing assessments or provenance history. Leave compatible. |
| Existing | EvidenceReference, EpistemicContext, non-TRAIN eligibility and canonical case IDs. Reuse with exact snapshots. |
| Missing | Immutable external hypotheses, assessment history, conservative Pareto diagnostics and bounded projections. Add. |
| Redundant | New scalar estimator, hypothesis generator, actor deletion tool or runtime authority gate. Do not add. |
| Experimental | Host-declared scores and triggers; no empirical proof that preserving alternatives improves decisions. |

## Design

Inert frozen/slotted records represent Hypothesis, HypothesisScores, HypothesisAssessment,
HypothesisTrigger and HypothesisState. Keep all public statements and observable evidence refs
externally, with context, canonical case, protocol ID, complexity/compute units and non-TRAIN label.
Require at least two distinct hypothesis IDs and at least one recorded trigger. Statements are public
claims, not private reasoning. Bound identifiers to 128 UTF-8 bytes and statements to 512 bytes.

Seven separate scores: evidence_fit, uncertainty, complexity, risk, reversibility, compute_cost,
transfer. Fit/reversibility/transfer maximize; the other four minimize. All except complexity and
compute cost are unit values [0,1]; those two use shared declared units and nonnegative finite values.
None means unavailable. No summed fitness, Gaussian collapse, mixture weight or reward is inferred.
The host's protocol defines how it measures and normalizes these diagnostic dimensions.

Each assessment carries an assessor ID, status (supported, challenged, context_limited, unresolved),
scores and observable evidence. An explicit supersedes link can replace only an earlier active
assessment for the same hypothesis and assessor; all old records remain. Multiple unsuperseded
assessments remain live. Disagreeing score/status records make a candidate conflicted and
incomparable. Missing assessments/dimensions also make it incomparable. Complete agreed vectors
use strict Pareto dominance: no worse in all dimensions and better in at least one. Equal vectors
remain tied. Even dominated or challenged hypotheses are retained and pageable.

Triggers record host-reported persistent innovation, multimodal evidence, verifier conflict, regime
shift or structural alternatives with evidence. They do not automatically diagnose causes or change
authority. Existing temporal-estimator decisions remain unchanged.

`append_hypotheses` extends all three immutable histories and increments revision, with no deletion
or record-edit interface. `validate_extension` checks a proposed successor against the authoritative
prior: fixed identity/protocol/units/eligibility/context, one revision increment, exact history prefixes,
at least one addition. Hosts use this check before atomic storage and authenticate callers; Python
objects and hashes are not access control. No persistence callbacks, I/O or actor tools are installed.

`pareto_hypotheses` returns separate frontier, dominated and incomparable ID lists plus conflicts.
`project_hypotheses` exposes 2..32 candidates in insertion-order pages, including dominated entries;
scores/status/conflict are diagnostics, not authority. It wraps the existing HypothesisSet and accepts
an explicit host-supplied diagnostic_uncertainty for that legacy field; it never computes a scalar
posterior from candidates. Include totals, omissions and next_offset. The last page may overlap one
entry to satisfy the existing two-alternative contract. All alternatives must be reachable by paging.
No external evidence IDs, assessor IDs, triggers or context provenance enter actor payloads.
Public measurement protocol ID and unit declarations accompany numeric scores. Hosts supply the
protocol definitions to the actor and choose these bounded labels for public exposure. JSON-quote
both IDs and statements in legacy labels to preserve valid whitespace, and normalize validated
numeric uncertainty to a float at the legacy boundary.

Use `ExperimentalOverlayConfig.competing_hypotheses`, disabled by default, for mutation proposals,
analysis and projections. Record construction/serialization is inert. Strict JSON import/export
preserves full history and validates graph links; no caller-owned serialization method is invoked.

## Scope, research and verification

Exactly 17 canonical cases; existing schema/default estimator/training/reward/authority unchanged.
Python >=3.10, 100-column Python, no dependency. Specs/plans in history. One inline task and one
fresh whole-branch review; draft PR only. Existing next-PR authorization covers execution/publication.

Sources: SRHarness (2609.35501), DoAtlas-2 (2609.35107), Dual-Frontier (2609.26293), Physical
Representation Languages (2609.23381), AbGaze (2609.35296). First/second/fifth need registry entries.
SRHarness supports persistent candidates and compact views; DoAtlas-2 motivates evidence statuses;
the others motivate explicit unresolved structure and separation from decision authority. The local
score axes, conservative comparison and retention rules are design inferences, not paper results.

Tests: Pareto trade-offs/ties/dominance, unknowns/conflicts, supersession provenance, deletion/edit/
stale revision rejection, actor projection bounds/paging/privacy, exact primitives/enum/evidence
validation, serialization and hostile instance methods, strict flags and non-TRAIN. Include existing
diagnostic deserialization/admission integration and full suite/build/wheel example. Require >=80%
new-code coverage, Ruff/Black, scoped types, Python3.10 syntax and documentation consistency.

Risks: hosts authenticate evidence and score semantics, control storage concurrency and choose
public statements. Exact agreement is deliberately conservative and may enlarge the frontier.
Full history is unbounded; this opt-in API does not implement pruning or durable storage.
