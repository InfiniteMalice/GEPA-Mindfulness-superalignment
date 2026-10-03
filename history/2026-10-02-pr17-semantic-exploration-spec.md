# PR-17: Semantic exploration control

## Intent and reconciliation

Choose what evidence-gathering strategy to propose when uncertainty remains. Keep semantic distance
separate from sampling temperature; high uncertainty alone must not select the broadest strategy.

| Status | Existing contract and decision |
| --- | --- |
| Existing | InformationGainQuestion and expected_information_gain_inquiry flag; reuse. |
| Existing | EvidenceReference, EpistemicContext, canonical case IDs, non-TRAIN labels; reuse. |
| Partial | Epistemic routing selects coarse continuation/retrieval actions; leave compatible. |
| Partial | HypothesisState preserves competing explanations; hosts may derive evidence gaps and diversity from this external history, without automatic scalar collapse. |
| Missing | Bounded comparison of semantic SEARCH/DEBATE/SPARK candidates using gain, budget, stakes and reversibility. |
| Redundant | New estimator, tool executor, retrieval model, reward or runtime authority gate. |
| Experimental | All gain estimates, diversity measures and selection thresholds are host declarations and local hypotheses, not validated behavioral results. |

## Contracts

Add one pure module `gepa_mindfulness.verification.semantic_exploration` with frozen/slotted records:
- ExplorationRequest: request_id, context, source_case_id, world_uncertainty, model_uncertainty,
  monitor_uncertainty, stakes, evidence_gap, hypothesis_diversity, remaining_compute, compute_unit,
  gain_protocol_id, evidence_changed, evidence_refs, training_eligibility=DEVELOPMENT.
- ExplorationCandidate: id, mode, target, distance, question, expected_information_gain,
  compute_cost, reversibility, evidence_refs.
- ExplorationPolicy: max_distance=3, uncertainty_threshold=0.5, breadth_threshold=0.5,
  high_stakes_threshold=0.8, minimum_gain=0.05, minimum_reversibility=0.5.

Modes SEARCH/DEBATE/SPARK; targets world/model/monitor. Distances are exact integers 1..5.
SEARCH runs direct, adjacent, related domain, cross-domain, remote analogy. DEBATE runs same-topic
specialist, adjacent specialty, related field, cross-disciplinary panel, distant discipline.
SPARK challenges a local constraint, method, approach, domain convention, or paradigm assumption.
These descriptors constrain the host's proposed action; the library neither generates nor executes it.

IDs/protocol/units/context strings <=128 UTF-8 bytes, questions <=512. Exact numeric primitives,
finite unit values [0,1], no booleans. None allowed for uncertainty/gain/reversibility; missing values
never become zero. Budget/cost are JSON-safe integer compute units; cost >=1. Candidate count 1..32,
unique IDs; each record retains 1..32 observable evidence refs. Exact typed nested records are copied
by declared class fields, never instance serializers. Revalidate at every operation.

## Selection

`propose_exploration(request, candidates, *, policy=ExplorationPolicy(),
config=ExperimentalOverlayConfig()) -> dict[str, Any]` requires the existing inquiry flag.
No proposal for unchanged evidence, zero budget, or unknown monitor uncertainty.
Distance cap: 1 initially; 3 when evidence_gap >= breadth_threshold; 5 when both gap and diversity
meet that threshold. Clamp by policy.max_distance. High stakes (>= high_stakes_threshold) cap at 2.
Thus uncertainty alone cannot widen distance. Defaults cap exploration at 3.

Reject candidates above cap, above remaining budget, below max(minimum_reversibility, stakes),
missing/low gain (gain must be positive and >= minimum_gain), missing/low target uncertainty
(must be positive and >= uncertainty_threshold), or missing reversibility. When monitor uncertainty
is positive and meets uncertainty_threshold, only monitor-targeting candidates may be proposed.
Rank remaining candidates by gain descending, cost ascending, distance ascending, ID ascending.
Select at most one. Gain is a normalized host-declared expected information gain under one public
gain_protocol_id; it is not measured realized gain or reward. Costs share one compute_unit.

Return bounded detached JSON with schema, request ID, eligibility, measurement labels, distance cap,
reason, per-candidate rejection reasons, selected candidate or null, and legacy InformationGainQuestion
or null. Include authority_granted=False. No external evidence/context IDs enter this actor payload;
public IDs/questions/protocol/units must be chosen by the host for disclosure. Legacy question strings
are JSON-quoted to preserve valid whitespace. Each result is an advisory snapshot: hosts recheck
state relevance and budget at execution, authenticate measurements, retain provenance, and enforce
their existing authority controls. No budget reservation, clock, I/O, callbacks or persistence.

## Research and validation

Night Science (2609.35706) separates semantic action/level guidance from temperature; transfer only
interpretable action levels, not its training/reward algorithm. LADDER (2609.24346) motivates avoiding
repeated retrieval without changed evidence; a host boolean is not its graph/entity detector.
Use the repository's canonical Reasoning Topology Matters source (2609.24710), which studies
task-dependent topology; do not replace it with the similarly named 2603.20730. DoAtlas-2
(2609.35107) motivates evidence-directed inquiry. Only Night Science needs a new source: 69 total.
Keep REC-011 experimental, 19 recommendations and exactly 17 canonical cases.

Tests cover every selection factor, all 15 descriptors, missing inputs, budget/tie/permutation cases,
inquiry serialization, non-TRAIN eligibility, bounded payload/privacy, nested mutation and hostile
serializers, invalid flags/primitives/Unicode. Compare matched fixtures where only one diagnostic
changes. Require >=80% new-module coverage, full suite, static checks and installed-wheel example.

## Decisions

Use explicit host candidates rather than a learned generator or fixed action chosen from uncertainty
alone. Cost: candidate quality/calibration stays host-owned. Use separate target uncertainties rather
than a summed confidence. Cost: unknown monitor state defers proposals. Use deterministic local
thresholds/ranking rather than an unvalidated learned reward. Cost: defaults need empirical tuning.
