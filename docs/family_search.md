# Evaluator-guided family-search preparation

PR 11f adds **Experimental** CPU plan preparation and **Diagnostic** attempt reports for
structurally different improvement hypotheses. It is disabled by default. Live generation,
autonomous search orchestration and comparative model effectiveness remain **Unsupported**.
There is no new dependency, evaluator, model/provider call, candidate execution or admission policy.

This is an original methodological adaptation of
[Gu et al., Discovering state-of-the-art quantum circuits with large language models](https://arxiv.org/abs/2610.22367v1).
The [source record](recommendations/RESEARCH_TRACEABILITY.md#ref-multi-family-discovery) and
[matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) retain checked versions, artifacts,
licenses and the authored-note digest. No quantum code or circuit geometry is imported.

## Existing authority and contracts

The adapter reuses `CandidateSpec`, `ImprovementProtocol`, `DatasetManifest`, `AttemptJournal`,
`summarize_journal` and strict `ImprovementRecord` serialization. It does not extend canonical
cases, rewrite manifests, change the fixed Studies A–F roster or interpret family labels as
eligibility. `DatasetCase.family_id` remains a task-dependency clustering coordinate; a new
`SearchFamily.family_id` names a candidate hypothesis. They have different roles.

`audit_improvement` remains the outcome-audit path with its independent host authentication,
source-to-slot/evaluator bindings, held-out exposure checks, paired statistics, component metrics
and explicit uncertainty limits. The new report does not consume an alleged verification flag
or re-score an outcome. Generating, proposing, evaluating, retaining and accepting a candidate
remain distinct operations. A journal status of `selected` is a supplied search declaration,
not a governed admission or deployment receipt.

Actual deficiencies still pass through existing [RSIFlow intake](improvement_workflows.md)
and [controlled evolution](controlled_evolution.md). Human-review routes, protected checks,
model/harness version separation, action grants, evidence integrity and reward eligibility
retain their current authorities. A model-generated proposal never changes those rules.

## Records and preparation

`evaluation.family_search_records` defines exact v1 records. Every record is DEVELOPMENT with
`optimizer_input=False` and `confers_authority=False`; unknown fields, versions, private
references, malformed arrays and primitive coercion are rejected.

| Record | Meaning |
| --- | --- |
| `FamilySearchConfig` | Explicit operator enable flag, false by default. It configures no external runtime. |
| `SearchFamily` | Stable ID, model/data/memory/harness surface label, structural hypothesis and observable source references. A label or description proves neither novelty nor admissibility. |
| `FamilyMembership` | Existing candidate ID, candidate-family ID, synchronous proposal round index and round ID. |
| `FamilySearchPlan` | Existing protocol; fixed task/protected/checkpoint/inference-budget/training-budget digests; families/members; public source references; separation/sharing schedule and common per-family caps. |

`prepare_family_search(plan, manifest, *, config=FamilySearchConfig())` returns `None` while
disabled, before inspecting the plan or manifest. When enabled it snapshots the existing
records and validates:

- Exact manifest/protocol binding, existing task lineage and protocol indexes.
- One known family membership per candidate, with unique normalized identities.
- An independent round-zero root for every family; parent rounds and freeze ordinals strictly
  precede descendants. Round IDs map consistently and uniquely to their indices.
- Cross-family edges only at `round_index >= independent_rounds` and
  `(round_index - independent_rounds) % sharing_interval == 0`.
- A common baseline and inference-budget digest; complete baseline/candidate pairs for the
  same case, metric, repeat and shared seed, including paired condition identity.
- The same declared case/metric/repeat coverage for every candidate and each of 2..16 seeds.
  Every evaluated non-training manifest case and declared metric appears. Training cases
  cannot become evaluation slots.
- Common per-family candidate and planned evaluation-slot caps. Caps and planned rosters
  do not prove equal actual resource use; missing trials and costs remain observable.

The graph uses the existing `CandidateSpec.parent_candidate_id` exclusively. A scheduled donor
transfer creates a new descendant in the recipient family with the donor as its sole parent.
The existing transitive exposure audit therefore follows it across family labels. This is a
copy/improve transfer, not multi-parent fusion. Combining multiple donors, hidden influences,
or a donor plus an unrecorded recipient parent is unsupported. Hosts must retain complete
source lineage and exposure history; graph validation cannot detect concealed inputs.

Bounds: 2..8 families, 1..64 candidates/members, rounds 0..15, at most 128 manifest cases,
16 metrics, 2048 planned slots and 4096 journal events. Hypotheses and source identities are
bounded. A plan is at most 1 MiB; inputs and reports use the existing 4 MiB JSON envelope.
Pure processing has no callback or side effect; oversized reports fail closed.

## Retained search history

`summarize_family_search(plan, manifest, journal, *, config=...)` validates the existing journal
before reporting per-family progress. Proposal event round IDs must match memberships;
chronological proposal rounds cannot go backward, and a child proposal follows a recorded,
frozen parent. Evaluation round IDs remain independently recorded. These are declared
synchronous research rounds, not claims that a live runtime was supervised.

The report retains original events and all failed, rejected, cancelled, pending, unattempted
and supplied-selected candidates. It reports proposal/attempt/completion counts, incomplete
evaluations, transfers and existing cost groups. `known_total` is the existing known subtotal;
`partial` and `unknown_count` remain visible, and unmeasured attempts are counted. These are
provided journal entries, not authenticated billing or full compute accounting. Do not add
global totals to per-family totals: the latter decompose the same events.

Reports explicitly set `optimizer_input=False`, `confers_authority=False`,
`promotion_eligible=False`, `execute_runtime=False` and `model_effectiveness_established=False`.
No winner, constitutional score, composite reward or verified-improvement certificate is
emitted. Independently audited negative outcomes must remain negative and protected regressions
cannot be offset by unrelated gains.

## Host workflow and reproducibility

Restore the disabled [example configuration](../configs/family_search.example.json) using
`FamilySearchConfig.from_dict`. The host supplies existing immutable protocol/task records;
the adapter does not generate cases, candidates, slots, observations or journal events.

```python
from evaluation.family_search import prepare_family_search, summarize_family_search
from evaluation.family_search_records import FamilySearchConfig
from evaluation.improvement_audit import audit_improvement

config = FamilySearchConfig(enabled=True)
preparation = prepare_family_search(plan, manifest, config=config)
progress = summarize_family_search(plan, manifest, journal, config=config)
outcomes = audit_improvement(
    plan.protocol, manifest, journal, evidence,
    exposures=exposure_history,
    final_authorizations=event_specific_final_permissions,
    authenticate=independent_host_authenticator,
    enabled=True,
)
```

Plan task/protected/budget/checkpoint digests are commitments, not authentication. Hosts verify
the actual protocol, source/evaluator versions, permitted changes and complete provenance.
They isolate any external model/harness trials, meter all generation/evaluation costs, enforce
spending and action permissions, and preserve raw evidence and rollback references. No live
runtime or spending is enabled by these Python functions.

For an external comparison, preregister a single-family baseline and separate data/model/
memory/harness hypotheses, common tasks/metrics/seeds, operator-approved budgets and protected
checks. Measure verified task outcomes, protected regressions, attempted/failed proposals,
source diversity and actual resources separately. Candidate selection uses only its permitted
selection partition. Keep independent audit/final/OOD tests sealed from generation and
selection; exposures to a donor also affect its descendants. Existing cluster-aware intervals
do not confer simultaneous, training-seed or post-selection guarantees.

The paper's separate structural families, evaluator feedback, retained lineage/failures and
periodic sharing inform this pattern. Its circuit-class correctness reduction, bounded-error
allowance, TQ resource objective and MAP-Elites archive geometry do not transfer to moral
eligibility. The primary v1 paper (2026-10-15), framework/Figure 1 and methods were checked
2026-11-02; paper CC BY-NC-SA 4.0. Author-linked `andigu/uncompute` at
`64b5a13aa249f1360846657392e6c66817b26d0f` has Apache-2.0 code licensing; README and LICENSE
were read. That release supplies circuits/tasks/evaluator; no search-runtime code was
established by README/root inspection. Nothing upstream was copied, imported or executed.

Synchronous rounds, fixed caps, matched rosters and single-parent transfers are original
adaptations. The [CPU acceptance tests](../tests/test_family_search.py) establish software
contracts, including real existing-audit composition. No native search or model-effectiveness
experiment has been run, and no component becomes Supported merely because tests pass.
