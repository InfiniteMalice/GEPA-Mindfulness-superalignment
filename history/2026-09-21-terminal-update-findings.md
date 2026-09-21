# Terminal-update verifier binding correction

CodeRabbit identified that generic verified evidence can retire a commitment even when its
declared transition conflicts with the verifier finding. Preserve the public-evidence architecture
and add a required named finding to terminal-update source resolution. Contradiction requires
`contradiction_status=contradicted`; scope exclusion requires evidence-bound `task_fit=False` plus
the existing explicit context change; replacement requires `claimed_outcome_supported=True` plus
the existing grounded replacement record; withdrawal requires `claimed_outcome_supported=False`.

All update references must occur in bindings for the required finding. Legacy verification remains
usable for ordinary prior commitments but cannot authorize a terminal transition. Negative findings
must also carry observable bound evidence. The host remains responsible for relating the verified
action/claim to the declared commitment and authenticating the verifier; this change does not infer
semantic truth from field names.

Use the existing review/TDD and repo-quality-gate workflow. Add regressions for wrong findings,
untyped fallback, unrelated bindings, every supported terminal status, and public-source constraints.
Update deterministic controls to include typed relational verification, document the transition
contract, and run continuity, semantic, memory, reward and registry checks before pushing the fix.
