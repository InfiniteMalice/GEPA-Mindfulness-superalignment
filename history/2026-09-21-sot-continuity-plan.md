# State of Thought continuity implementation plan

## Task specification

Implement the supplied SoT integration brief inside semantic_intent_robustness. Extend the
existing trajectory interface; preserve the semantic record, representation candidate, memory
trust, action-bound sequence and verified-process reward contracts. No AGG, controller, optimizer,
private reasoning capture, raw tensor persistence, or automatic enforcement is in scope.

The baseline has layer trajectory and transcript adapters, semantic clusters with negative controls,
provenance-bound representation candidates, memory retrieval assessment, and action-bound events.
It has no SoT measurement contract or explicit evidence-omission diagnostic.

## Design and execution

Use frozen bounded records and pure offline assessments. Implement inline with superpowers
test-driven-development and repo-quality-gate. The detailed user brief supplies the intended
behavior; this plan resolves integration choices without changing its scope.

1. Extend internal_state_trajectory.py with SoTStateSnapshot, measurement/source labels and a
   protocol. Backend-calibrated normalized features are comparable only within the same model,
   backend, adapter, layer selection, feature schema and measurement status. Missing features
   remain null. Test missing, nonfinite, out-of-range, synthetic and transcript values first.
2. Add semantic_state_continuity.py using existing SemanticSafetyRecord and decomposition fields.
   Compare aligned trajectories (not arbitrary hidden representations); report endpoint and
   transition distances separately from public decomposition and policy agreement. Topic-matched
   controls must expose collapse. Test paraphrase, wrappers, translation, code switching,
   multi-turn decomposition and intent-shift controls.
3. Add epistemic_records.py and epistemic_continuity.py. Commitments contain bounded public
   summaries and observable EvidenceReference values, with source EventEnvelope IDs. Validate
   the existing action-bound sequence and bind references to its evaluation unit before use.
   Append explicit update records; do not overwrite evidence. Recall uses RetrievedMemory and
   assess_retrieved_memory, retaining source identity, trust and representation provenance.
   State similarity ranks recall but cannot remove independently relevant commitments.
4. Add motivated_forgetting.py. Only an unexplained omission plus later provenance-bound
   directional pressure and evidence that it favors the proposed action can produce POSSIBLE.
   Missing, cross-unit, future or incomplete evidence yields review/insufficient evidence.
   Retention and supported updates are matched no-signal controls; no causal motive claim.
5. Add continuity_audit.py integration through SemanticIntentPipeline with four false-by-default
   flags and immutable research/shadow maturity labels. Offline helper calls are explicit analysis,
   with no runtime side effects. Add separate metrics and a synthetic fixture evaluation command.
6. Register REF-SOT and experimental REC-015 (dependencies REC-002/005/006/010/011/014), extend
   strict registry inventories, and update reader docs plus privacy/reward/logging documentation.

## Validation and review focus

Tests must catch cross-conversation evidence, future supersession/pressure, duplicate IDs,
missing provenance, mismatched state scales, empty metric denominators, and retained uncertain
candidate hypotheses. Matched fixtures cover legitimate update, scope change, unexplained omission,
pressure-correlated omission, retention under pressure and three-turn euphemistic laundering.

Run targeted tests, Ruff, Black, relevant mypy and broad pytest. Record unavailable optional
dependencies separately from failures. A final independent review checks the entire diff.
Expected improvement is inspectable continuity diagnostics, not demonstrated model robustness.
The three empirical experiments in the supplied brief remain future experiments.
