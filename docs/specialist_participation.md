# Advisory specialist participation

PR 11e adds an **Experimental** CPU selector and **Diagnostic** reports for optional
MPT-inspired participation research. It is disabled by default. Native MPT action/bidding
training, specialized model backends, counterfactual environments and empirical effectiveness
remain **Unsupported**. No provider API, trained specialist, executor or new dependency is added.

The implementation is an original adaptation of [Xu et al., Learning Self-Orchestration in
Decentralized Agent Swarms](https://arxiv.org/abs/2610.22322v1), not a training reproduction.
The [source record](recommendations/RESEARCH_TRACEABILITY.md#ref-mpt) and
[matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) retain the checked version, date,
licenses, pinned code artifact and authored-note digest.

## Contract and trust boundary

`gepa_mindfulness.factuality_observability.participation_records` reuses strict
`ImprovementRecord`, existing `SystemConfig`, `TrustedEvaluatorContract`, public
`EvidenceReference` and `PublicDecisionContext`. The exact v1 wire format rejects unknown
fields, authority extensions, private evidence, booleans as numbers, nonfinite/out-of-range
bids and malformed arrays. Every record is DEVELOPMENT with `optimizer_input=False` and
`confers_authority=False`.

| Record | Meaning |
| --- | --- |
| `SpecialistCandidate` | Declared specialist/principal, model/harness/config, checkpoint, specialty and public provenance. A description is not verified competence. |
| `ParticipationPacket` | Request ID, public context, declared scope and fixed roster of 1..16 candidates. No private evaluation labels or reward fields are required. |
| `ParticipationReview` | Independent host assessment bound to exact packet/candidate, reviewer principal/evaluator, unique review ID, eligible/rejected/uncertain status, reason and public evidence. |
| `SpecialistBid` | Exact packet/candidate, specialist/system, finite willingness in [0,1] and public capture provenance. This is neither a probability nor permission. |
| `ParticipationConfig` | Explicit enable flag, reproducible seed and output byte budget. |

`select_specialist` accepts a host's current state digest, configured reviewer identity,
independent review callback and bid callback. It checks state binding and rejects duplicate
or case-alias specialist/principal IDs before callbacks. Reviewer principal must differ from
every candidate principal, including ASCII case aliases. Identity strings and digest equality
are structural controls; the host must authenticate principals, checkpoints, state, evidence
and the actual independent reviewer. The host owns the meaning of eligibility under applicable
constraints. A classifier or reviewer agreement does not prove moral alignment.

All candidates are independently reviewed before any bid is requested. Rejected and uncertain
candidates receive no bid call and cannot win. Uncertain judgments retain their status; they
are not counted as deterministic constitutional violations. A positive eligible candidate can
be recommended while another candidate is quarantined. If no positive eligible bid remains,
an uncertain review yields `further_verification`; otherwise `no_participation`.

The selector samples uniformly among sorted exact positive maxima using `random.Random(seed)`.
It does not add jitter, merge near ties, discard tiny positive values, or reopen zero bids.
The seed is an experimental reproducibility setting, not adversarially secure randomness.
Use independently predeclared seeds and retain the roster when measuring participation.

Each callback receives detached packet/candidate records. Mutation, callback exceptions,
malformed or substituted responses, changed system bindings and duplicate review IDs abort
the complete selection. Earlier valid bids remain observable but cannot win a partial failed
request. Error reports contain fixed categories, not exception text. Returned records are
detached before subsequent calls. Raw callback integrations are trusted host code: this is
neither a Python sandbox nor a timeout/spending mechanism. Hosts must enforce deadlines,
isolation, external-runtime configuration and budgets before enabling model-backed callbacks.

Before the first callback, the selector reserves the complete report including both a 64 KiB
serialized review and 64 KiB bid per candidate, selection identities and failure metadata.
Every returned record is independently capped at 64 KiB. Configurable output limits range
from 4096 bytes to the existing 4 MiB maximum; conservative reservation may reject a budget
even when a particular short response would fit. Text/provenance limits also bound inputs.
The final report is checked against the same budget.

Reports include all roster rows, supplied reviews/bids, selected ID or null, exact top ties,
route, failure, configuration and provenance. They explicitly set `authority_granted=False`,
`execute_specialist=False`, `optimizer_input=False` and `model_effectiveness_established=False`.
They do not allocate reward or certify a task outcome. No bid, source handle or review becomes
verified optimizer feedback. No single event is newly credited through this pathway.

## Host integration

Restore the disabled [example configuration](../configs/specialist_participation.example.json)
with `ParticipationConfig.from_dict`. With the default config the selector returns `None`
without inspecting the packet, current-state argument or callbacks.

```python
from gepa_mindfulness.factuality_observability.participation_records import ParticipationConfig
from gepa_mindfulness.factuality_observability.specialist_participation import select_specialist

# packet, independent_review and capture_bid are host-configured public integrations.
report = select_specialist(
    packet,
    current_state_digest=current_state_digest,
    evaluator=host_evaluator,
    reviewer_principal_id=host_reviewer_principal,
    review=independent_review,
    propose_bid=capture_bid,
    config=ParticipationConfig(enabled=True, seed=7),
)
```

`capture_bid` can return saved public readouts or an explicitly configured external backend.
It must bind the declared model/harness/config and checkpoint through the exact candidate
digest. Evidence labels alone do not establish authentic capture. Existing
[Jev/LEV routing](decision_routing.md), verifier independence and
[runtime authority](VERIFICATION_AND_RUNTIME_AUTHORITY.md) remain controlling. A suggestion
cannot waive escalation or required verification. Hosts must obtain and consume current
action-scoped authorization using the existing registry before any actual execution; this
selector neither creates nor consumes those grants. It cannot appoint itself as verifier.

## Source differences and research protocol

The primary v1 paper (2026-10-15), sections 2.1–2.3 and Appendix A, describes highest-bid
turn allocation, uniform ties, skipped all-zero turns, backward value propagation for action
learning and separate counterfactual bidding preferences. It alternates action and bidding
updates. These learning mechanisms are absent here; bids are never a new reward authority.

Author-linked `Embodied-Minds-Lab/MPT` at `3b56087a2e436e04f077672cf8cffd91b8820b08`
was checked through README, `src/economy_engine/auction.py`, LICENSE and NOTICE. The repository
has Apache-2.0 code licensing; the paper has the arXiv perpetual non-exclusive distribution
license. No upstream code or data was copied, imported or executed. The pinned auction code
reopens all-pass rounds at a positive floor and uses jitter. Our abstention/exact-max rules,
continuous [0,1] bids, independent host review and advisory-only report are explicit adaptations.

Compare predeclared fixed selection, round-robin and saved/model-derived bids using the same
independently eligible specialist roster, tasks, public context, action policies, evaluator and
resource caps. Use the existing [reproducibility/protected evaluation](research_ablation_suite.md)
contracts for external experiments; this PR does not alter Studies A–F or create canonical cases.
Keep independent held-out outcomes and review labels out of bid inputs. Retain all planned
requests, quarantines, missing/failed calls, zero bids, escalations and unsuccessful outcomes.
Do not select a policy using its final held-out evaluation.

Measure verified downstream task completion, protected behavior, false eligibility passes,
uncertain/rejected reviews, escalation/recovery, participation distribution, bid/action tokens,
latency and costs separately. Include bid generation for every eligible participant in compute
accounting; unknown costs stay unknown. Use paired task comparisons, multiple predeclared seeds
and cluster-aware uncertainty via the existing evaluator. A lower coordination cost or higher
bid is not success. No comparative model experiment has been run by this implementation.

Adversarial CPU controls cover a rejected would-be winner, false self-certification, an
uncertain eligibility judgment, all-pass abstention, stale context, switched evaluator/model,
callback mutation, partial failures and complete report reservation. The
[acceptance tests](../tests/test_specialist_participation.py) establish software contracts,
not learned participation quality, correct moral classification or checkpoint competence.
