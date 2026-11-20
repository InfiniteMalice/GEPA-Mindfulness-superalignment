# Shadow metacognitive control proposals

`verification.metacognitive_control.propose_control` is disabled by default. Enabled calls
require an exact immutable assessment and an explicit `ControlBudget` under an unchanged root
objective. The controller returns one `ControlProposal`; it accepts no executor, evaluator,
network callback, mutable evidence store or optimizer.

The proposal retains the full source-linked assessment, selected finding and repertoire method,
rationale, unknown or estimated information gain, resource estimate, stopping condition and
required separate host authorization. Current rules leave information gain unknown and estimate
only diagnostic steps. They do not estimate model tokens or validate scientific utility.

Selection prioritizes material ambiguity, contradiction and evidence continuity before less
urgent diagnostics, with deterministic finding-ID tie breaking. Clarification, missing-evidence
requests, abstention, assumption verification, comparison, sensitivity, reconciliation and
continuity review remain distinct outputs. Methods must declare both the diagnostic kind and
the selected proposal output. The evidence-verification method also accepts factual evidence-gap
findings, so the source engine's requests are not lost during composition.

An exhausted diagnostic budget or round limit, unavailable/superseded assessment, or absence of
applicable findings produces a zero-cost no-intervention proposal. This continues the existing
investigation only under its separate host limits; it is not a claim that all domains are healthy.

Repetition alone cannot justify stopping. A stop proposal additionally requires completed
repetition coverage, multiple source events and retained no-progress claim declarations marked
contradicted with observable evidence. The host must authenticate these declarations and
independently confirm branch unproductivity before any stop. Construction cannot authenticate
the declarations or prove their semantic meaning. All stopping remains conditional on separate
host confirmation and the existing runtime authorization path; it cannot alter the root objective.

Every imported or constructed proposal must remain `shadow_only=True`,
`confers_authority=False`, and diagnostic. Changing a label, importing a proposal or selecting a
strategy cannot grant permission, change reward weights or promote evidence. Strict versioned
serialization rejects unknown fields, invented strategies and resource overspend.

Run `python -m pytest -q tests/test_metacognitive_control.py` for default-off behavior, method
selection, unknown gain, distinct outputs, deterministic priority, no-intervention paths,
budget/round limits, conditional branch stopping and authority-laundering rejection.
