# Observable revision and honesty diagnostics

`evaluation.epistemic_revision` records public commitments and joins them to the existing
ClaimGraph, check results and one canonical V5EvaluationRecord. It preserves the canonical
record's serialization. `validate_episode_events` delegates chronology and record provenance to
the existing V5 validator and returns DEVELOPMENT metadata. It never returns optimizer scores.

The V5 record binds the initial prediction's confidence. A revised final confidence binds a
separate later `prediction_commit`; it cannot rewrite the first prediction. Check evidence must
join the executed action's observation and a successful matching verifier event before that final
prediction. This validates joins, not the semantic truth of the public answer. Answer text, mode
and confidence-source declarations remain sidecar diagnostics because historical prediction
payloads did not encode them. Leveled verifier-only streams without the legacy observation/verifier
join currently fail closed in this adapter and need an explicit compatible integration.
Any changed commitment content requires a later prediction event, even if confidence is unchanged.
An unchanged commitment may reuse the initial prediction with a different commitment ID.

`revision_diagnostics` reports answer/mode/confidence changes, unsupported replacement premises,
counterevidence retention, unresolved-claim preservation, clarification proportionality and resume
behavior separately. Host-supplied decisive-premise findings distinguish underreaction from
appropriate stability. Unsupported replacement premises after a decisive falsification flag
rationale migration. An unchanged answer alone never implies deception. Unknown decisiveness
or unknown alternative support leaves the evaluator uncertain.
Rationale migration also applies when the answer changes but new premises remain unsupported.
Initially unresolved claims must remain explicit unless a check supports or contradicts them;
an unresolved check result always keeps its claim in the preservation requirement.

APPROPRIATE_REVISION means a response reacted to designated decisive evidence; it does not
certify the revised answer. Correctness, calibration, abstention and action validity remain in
the canonical assessment. Case-specific focus adapters cover 1–8 (answer/calibration), 9–13
(abstention) and 14–17 (clarification/resume), without assigning a new case identity.

Candidate A/B roles are generation conditions, never truth labels. External checking should
assess both disputed and shared premises before attaching any separate verified process credit.
No hidden reasoning or trace appearance enters these diagnostics. Tests:
`tests/test_epistemic_revision.py`, `tests/test_v5_provenance.py`, and existing reward gates.

Sources: *VeriHarness: Scaling Agentic Verification for Long-Horizon Tasks* (arXiv:2610.00972)
motivates evidence-backed revision; *How to Have a Sensitive Debate: An Instance-Optimal Protocol
for AI Debate* (arXiv:2610.02557) motivates decisive-premise scrutiny. The revision labels and V5
adapters are repository inferences. Their diagnostic value requires evaluator validation.

`compose_revision_example` joins the episode to typed stakeholder/perspective state, candidate
claim partitions, selected challenges and check requests. It checks claim, stakeholder, action
and request identities and retains validated event IDs. Candidate A/B labels do not identify truth.
The exported example is DEVELOPMENT-only and contains no private trace or raw event payload.
