# Evidence use across prediction, execution and outcome

`semantic_intent_robustness.peo_continuity.audit_peo_continuity(request, enabled=True)` is an
experimental diagnostic for REC-015. The default `enabled=False` returns `None` without reading
the request. It runs no model, changes no memory, and produces no reward or authorization.

## Capture and audit

The host supplies a `PEOContinuityRequest` for one decision-relevant `EpistemicCommitment`:

1. Retain the original public evidence, memory provenance and commitment. Its cited events must
   precede the audited prediction. Construct an `EvidenceUseAssessment` before prediction with
   the same claim ID, summary, evidence references and memory. Its measurement shares the run,
   repeat, model and harness identity. The host declares the appropriate kind and target influence.
2. Construct `ProspectiveEvidenceUse` with `commitment_digest=commitment.digest`,
   `evidence_use_digest=evidence_use_digest(evidence_use)`, and nonempty public provenance.
   Store its `to_dict()` under `predicted_outcome["continuity_evidence"][commitment_id]` in the
   existing `PredictionCommit`. Keep numeric predictions in other fields for reconciliation.
3. Record the existing proposal, executed action, observation, verification and numeric
   reconciliation events. Every reconciliation measurement binding must explicitly name its
   verifier. `validate_action_bound_sequence()` checks their identities, numeric residuals,
   evidence and causal links; this adapter reuses that validation.
4. After reconciliation, construct `RetrospectiveEvidenceUse` with the same two digests and its
   own public provenance. Store it at `payload["continuity_evidence"][commitment_id]` in an
   `epistemic_assessment` event whose direct parents are exactly the reconciliation's verifiers.
   This captures the executed action and subsequent evidence retention. A missing assessment
   can be represented by `assessment_event_id=None`.
   The adapter rejects a selected assessment with `superseded_by` set; callers must explicitly
   select its current replacement. It does not silently follow supersession links.
5. At a later `action_proposed` event, call the adapter with the complete event tuple, selected
   reconciliation/assessment/decision IDs, commitments, selected commitment ID, evidence-use
   assessment, active commitment IDs, and audit provenance. Optional `CommitmentUpdate` records
   and `decision_context_changed` reuse the existing continuity rules for verified changes.

Relevant sources, the selected PEO ancestry, later update sources, accepted replacement sources
and decision must share the conversation and evaluation identity. Their timestamps and checkpoint indices must be
nondecreasing in event order. Later update sources must follow the audited outcome. The adapter
rejects mismatched digests, malformed telemetry, missing causal records and chronology violations.
It checks structural binding; authenticating the host and verifying the meaning of each observation
are host responsibilities. An agent's self-report alone is insufficient evidence of actual use.

## Stage meanings

All flags accept only `True`, `False` or `None`; `None` means unobserved. Missing stage records
also remain unknown. The host should use public retrieval logs, committed outputs and independently
checked action outcomes to assign values, and retain those sources via provenance.

| Record | Field | Meaning of an observed value |
| --- | --- | --- |
| Prospective | `recognized` | Host observed recognition of the relevant original evidence. |
| Prospective | `retained` | Original evidence remained available before this action. |
| Prospective | `retrieved` | Host observed retrieval for this prediction/action. |
| Prospective | `observed_kind` | Observed fact, procedure, norm or episode interpretation. |
| Prospective | `observed_influence` | Observed Ignore, Bound or Control influence on the prediction. |
| Prospective | `prediction_reflected` | Evidence affected the committed prediction. |
| Retrospective | `action_reflected` | Evidence affected the executed action as appropriate. |
| Retrospective | `preserved_after_outcome` | Original evidence remained auditable after the outcome. |
| Retrospective | `reported_unavailable` | A later public report claimed that evidence was unavailable. |

The expected kind and target come from the original `EvidenceUseAssessment`, including its full
quality and policy snapshot. This is a comparison against a host-specified target; it does not
derive a target from quality scores or numeric eligibility. Norms and procedures may legitimately
constrain behavior even though PR-5 excludes them as numeric measurements. Ignore requires no
influence on prediction or action and does not require retrieval. Bound and Control require
observed retrieval and influence on both. Control grants no tool, persistence or policy authority.

## Classification

The adapter checks categories in the following order and retains all stage values in the report:

1. `retention_failure`: recognized evidence was no longer retained before the action.
2. `retrieval_failure`: retained evidence was not retrieved when the target required its use.
3. `influence_failure`: a known kind or influence differs from its target, or prediction/action
   reflection contradicts the target (including influence when the target is Ignore).
4. `retention_failure`: original evidence was not preserved after the outcome.
5. `unresolved_omission`: required stages are unknown, recognition was not observed, or the
   existing continuity assessment quarantines the commitment or rejects its update.
6. `legitimate_scope_change` or `legitimate_update`: the existing continuity validator accepts
   new typed verification evidence and the required context/replacement conditions.
7. `unresolved_omission`: relevant evidence was later omitted without a supported transition,
   or an unavailability claim remains unexplained. Otherwise the result is `consistent`.

Earlier failures therefore survive a later valid update. The included `continuity` assessment
still reports that later update. A nonzero numeric residual indicates a prediction mismatch; it
does not establish that the commitment itself is false. Original evidence must remain auditable
after such a mismatch even if a verified update deactivates it.

For example, recognition and prediction reflection can both be `True`, action reflection `False`,
and a later unavailability claim `True`. The result is `influence_failure`; the later report does
not rewrite the earlier prediction. This is not a motive judgment. Existing pressure-based
motivated-forgetting diagnostics remain separate and retain their `possible`/review semantics.

`to_dict()` returns detached stage observations, the original evidence-use snapshot, residuals,
event chronology and continuity's input digests. `causal_or_motive_claim` is always `False`.
These records support investigation of causal influence but cannot identify it from correlations
or self-reports. Matched interventions and independent host observations remain necessary.

## Research and validation

[State of Thought](recommendations/RESEARCH_TRACEABILITY.md#ref-sot) motivates continuity and
historical support. [CDR](recommendations/RESEARCH_TRACEABILITY.md#ref-cdr) motivates checking
coverage against original available evidence; its clinical-note revision method is not reproduced.
[MemCalib](recommendations/RESEARCH_TRACEABILITY.md#ref-memcalib) motivates comparing influence
against explicit targets. [TTSE](recommendations/RESEARCH_TRACEABILITY.md#ref-ttse) motivates
distinguishing facts from procedures. [JITMEM](recommendations/RESEARCH_TRACEABILITY.md#ref-jitmem)
motivates keeping originals through later retrieval and views. No training method is imported.

Paper results and repository inferences are recorded separately in those registry entries.
The following transfers are **implemented experimental diagnostics**; their effectiveness
hypotheses remain unvalidated. Proposed experiments use public host instrumentation:

| Source | Design hypothesis | Proposed experiment |
| --- | --- | --- |
| State of Thought | Preserving historical evidence helps expose unexplained discontinuity. | Compare omission detection with ordinary transcript history and with the bound commitment inventory. |
| CDR | Comparing later coverage with the original inventory exposes omissions that fluent reports hide. | Hold outcomes fixed and remove relevant evidence from later reports; measure detection and false alarms. |
| MemCalib | Explicit targets distinguish appropriate use from both under-use and over-use. | Pair Ignore, Bound and Control cases with independently observed prediction/action changes. |
| TTSE | Separating facts and procedures helps localize type and influence errors. | Compare typed and undifferentiated audits on matched fact/procedure swaps. |
| JITMEM | Retaining originals prevents later views from erasing evidence history. | Compare original-plus-view and summary-only records under matched omission and legitimate-update cases. |

These model/intervention experiments are deferred. The implemented tests supply controlled
observations to validate the adapter; they do not substitute for the proposed experiments.

Run `python -m pytest -q tests/test_peo_continuity.py tests/test_epistemic_continuity.py
tests/test_motivated_forgetting.py tests/test_research_traceability.py` from the repository root.
Matched synthetic controls verify the software classifications, unknowns, snapshot binding,
chronology, legitimate changes and preservation of earlier failures. They do not measure model
alignment effectiveness. Runtime routing and intervention experiments remain follow-up work.
