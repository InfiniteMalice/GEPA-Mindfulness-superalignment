# PR-6 design: reliable self-improvement evaluation

Status: Written design and PR-5 deferral approved on 2026-10-10; implementation plan and native execution also approved on 2026-10-10.
Date: 2026-10-10.
Reviewed code: `9002bbbb0afa877d18485362942c32fd37ac6149`.
Workflow: superpowers brainstorming (architectural), with repo-quality-gate.

## Approved decision

The maintainer explicitly approved deferring PR-5 implementation and this PR-6 written design
on 2026-10-10. Proceed with a separate implementation plan. Implementation begins only
after that plan and its execution method receive approval.

The original assignment specifies PR order, although PR-6 depends on the earlier diagnostic
PRs. Those diagnostic PRs are complete. This exception changes the order only: PR-5's
independent-evidence and nonduplicated-reward prerequisites remain in force. Repository reward
defaults govern the completed pilot, and the additional causal penalty remains withheld.
PR-7 remains dependent on the earlier work; this proposal does not release PR-7.

## Problem and outcome

Selection-set gains can overstate independently evaluated improvement. PR-6 will provide an
opt-in offline report that exposes data reuse, attempted candidates, paired changes, uncertainty,
cost, and failures. It will consume host-captured records without executing an optimizer,
calling a model, changing reward weights, admitting training data, or promoting a candidate.

The requested distinctions are separate report sections: optimization progress, evaluated
behavior, verified improvement evidence, and deployment eligibility. One section never grants
the status or authority of another. In particular, a positive interval is not an acceptance rule.

## Alternatives

| Approach | Benefit | Cost or limitation |
| --- | --- | --- |
| **Recommended: compose an offline report with existing diagnostic contracts** | Adds the missing accounting and partition checks while preserving existing authority boundaries | Hosts must supply authenticated manifests and capture records |
| Extend the durable coevolution lifecycle and catalog | Could enforce attempt registration inside that lifecycle | Broad schema and authority changes; excludes external optimizer attempts unless also integrated |
| Documentation and manual spreadsheet only | Smallest initial change | Does not enforce lineage separation or reproducible candidate accounting |

The recommendation adds no second candidate lifecycle and requires no new package dependency.

## Existing boundaries and proposed modules

`gepa_mindfulness/coevolution.py` already owns candidate registration, comparisons, and
acceptance receipts. `gepa_mindfulness/learning_surfaces.py` owns evaluation epochs and the
`ValidationSplit` values `held_out` and `protected`. PR-6 will not expand those values or
reinterpret a lifecycle receipt as independent audit evidence.

Existing `evaluation/causal_diagnostics.py`, `evaluation/debate_analysis.py`,
`evaluation/pluralistic_comparison.py`, `evaluation/evidence_topology_comparison.py`, and
`evaluation/ladder.py` remain the sources of diagnostic semantics. Reuse `SystemIdentity`,
`TrustedEvaluatorContract`, public evidence references, canonical digests, and training
eligibility restrictions. Preserve existing canonical case identities; never manufacture a
V5 identity to accommodate an incomplete capture.

Proposed additions are `evaluation/improvement_records.py` for immutable inputs and serialization,
`evaluation/improvement_statistics.py` for paired aggregation and resampling, and
`evaluation/improvement_audit.py` for validation and report assembly. Public names are proposed
interfaces, not existing APIs. Tests will mirror these responsibilities. Add a user guide and
a PR-6 decision to ADR 0021 after approval, preserving PR-1 through PR-4 status details.

## Input contracts and trust

The host supplies a versioned protocol, dataset manifest, append-only attempt journal snapshot,
source diagnostic records, and independent authentication callbacks. The protocol binds the
baseline, candidate configurations, metric definitions and directions, expected evaluation slots,
data digests, pairing policy, statistical parameters, and permitted split use.

Hashes establish integrity, not independence or completeness. Host authentication must bind the
full protocol and journal snapshot to an external evidence reference. Serialized `verified`
flags, model-authored explanations, and filenames cannot authenticate themselves. Missing or
failed authentication leaves the corresponding evidence unverified and prevents a verified
improvement claim. The library cannot prove that a host disclosed every real-world attempt.

Structural ambiguity, such as duplicate IDs, invalid finite numbers, cycles, or conflicting
bindings, rejects the input with a specific error. Missing, censored, failed, disputed, and
unauthenticated expected observations remain visible with reasons; they never become zero
failures or successful outcomes. Reports retain a DEVELOPMENT restriction and grant no
training or execution authority. File paths and source text are data, never commands to run.

## Five partitions and transformation lineage

The dataset manifest uses five distinct purposes: `synthetic_training`, `optimizer_selection`,
`independent_audit`, `final_test`, and `ood_combinations` (out-of-distribution combinations).
These purposes do not replace `TrainingEligibility` or the lifecycle `ValidationSplit` enum.
Training manifest entries describe provenance only; including an entry does not admit it to training.

Each case binds its content digest, scenario family, parent identifiers, transformation chain,
and partition. The manifest contains ancestry metadata through every root, including ancestors
that are not evaluated. The validator rejects unknown parents and cycles. Cases connected by
ancestry or belonging to the same scenario family cannot span partitions. Repeated content
digests across partitions also fail validation. An OOD label requires a predeclared combination
definition and host provenance review; the label alone is not evidence of novelty.

The host attests provenance completeness. The validator detects declared overlap; it cannot
discover an unrecorded paraphrase relationship from a hash. Missing provenance certification
therefore prevents a claim of verified split independence.

Audit and final-test exposure are recorded against candidate freeze and selection-round records.
When results inform a later proposal or selection, that dataset cannot support an independent
claim for the later candidate. Its original purpose and exposure history remain visible.
Final-test evidence requires separate host authorization bound to the frozen candidate,
protocol, dataset digest, and evaluation event. Without that authorization, final-test results
are excluded from statistics and labeled unauthorized; an absent result is labeled not run.
PR-6 does not load or execute final-test cases itself.

## Every attempted candidate and evaluation cost

The journal records candidate ID, proposal/configuration digest, baseline model/configuration,
candidate model/configuration, parent candidate, selection round, attempt event, terminal status,
and evidence references. Terminal states include selected, rejected, failed, and cancelled;
unfinished entries remain pending. Re-evaluations have separate event IDs and retain the same
candidate identity. Snapshot validation checks event ordering and rejects conflicting histories.

The report derives unique candidate count, evaluation-attempt count, and selection-round count
from the full journal rather than accepting caller-supplied totals. An attempt without a result
still appears. Unattempted planned candidates are reported separately and excluded from the
attempted-candidate count. The host certifies the journal against its external optimizer log.

Each candidate exposes exact data identity, selection delta, audit delta, authorized final-test
delta, OOD results, alignment regressions, severe events, uncertainty, and cost. Cost includes
failed and cancelled attempts when measured; unknown costs remain unknown. Costs with different
units or currencies are not summed together. Estimated and billed costs stay separate.

## Paired statistics and scope of uncertainty

Pair baseline and candidate observations only when case content, rubric, evaluation budget,
intervention condition, and declared repeat/seed policy match. The changed system/configuration
is the comparison target. Different or unknown seeds must be declared explicitly; a repeated
index does not imply a shared random seed. Unmatched observations remain descriptive only.

For a metric, compute candidate-minus-baseline deltas on matched observations. Preserve the
metric's direction so that a positive failure-rate delta is displayed as a regression. First
average repeats within the same case and intervention condition, then cases within each
dependency cluster, then give each cluster equal weight. Report raw opportunity counts and raw
failure rates separately; the cluster-weighted estimate has a different denominator.

Construct dependency clusters by connected components of shared scenario family, ancestor,
or intervention chain. Keep each exact baseline/candidate configuration comparison in its own
stratum. Do not pool model versions or candidates as independent replications; intervals are
conditional on the recorded system configurations and do not imply generalization to new models.
Host-declared additional shared dependencies merge clusters. Unknown dependence prevents an
interval rather than silently treating rows as independent.

Use a paired cluster percentile bootstrap within each stratum: resample complete cluster means
with replacement and calculate the mean delta. The protocol fixes the seed, resample count,
confidence level, and minimum cluster count before results are inspected. Proposed defaults are
10,000 resamples, 95% confidence, and a minimum of 20 clusters. The minimum is a conservative
reporting policy, not a statistical guarantee. Below the minimum, report the estimate and
`insufficient_clusters` instead of an interval. Retain all repeats within their clusters.

Report cluster count, matched coverage, omitted observations, protocol parameters, and interval
limitations. Incomplete paired coverage permits descriptive matched-subset estimates but no
full-roster interval. Intervals are per-comparison summaries, without simultaneous coverage or
post-selection guarantees; PR-6 makes no population-wide verified-improvement decision.

Optimizer overstatement is selection improvement minus independent-audit improvement, with the
same metric, direction, and weighting. Report final-test overstatement separately when authorized.
If both independent partitions support intervals, resample their clusters independently and
subtract the replicate deltas. Otherwise report the point difference with an unavailable interval,
or no difference when either estimate is absent. Never substitute a reused audit for a final test.

## Failure reporting and authority

Always include causal invariance failures, required-update failures, laundering susceptibility,
debate fault-localization errors, unjustified abstentions, incorrect clarification resumption,
calibration degradation, and severe safety/authorization events. Use the existing diagnostic
definitions and units. Where a source metric measures success, derive failure only from its
known eligible outcomes and expose that denominator. Missing metrics remain missing.

Keep every severe-event evidence reference and adjudication status, including events on missing
or otherwise excluded pairs. Aggregate improvement cannot suppress them. Calibration comparisons
require matching source definitions and bins where applicable; do not invent confidence values.

The report separates observed selection progress from evaluated behavior and authenticated
independent improvement evidence. Deployment eligibility is `not_assessed` by this module.
Existing deployment/acceptance receipts may be referenced for audit, but PR-6 issues none and
never changes a catalog or training pipeline. The completed PR-5 pilot remains evidence for its
recorded scope; it is not retroactively certified as an independent five-partition experiment.

## Verification contract for implementation

| Requirement | Required verification |
| --- | --- |
| Partition separation | Adversarial fixtures for descendants, paraphrases with shared ancestry, family overlap, cross-partition duplicate content, unknown parents, cycles, and distinct OOD families |
| Complete accounting | Frozen external-journal fixtures with selected/rejected/failed/cancelled/pending candidates, missing results, repeated evaluation, duplicate events, and mismatched counts |
| Independent evidence | Unauthenticated inputs, forged status fields, changed digests, audit reuse, and final results without candidate-bound authorization cannot support verified claims |
| Paired estimates | Hand-calculated fixtures for metric directions, mismatched rubrics/budgets, explicit seed policies, missing pairs, and repeat aggregation |
| Cluster uncertainty | Fixed-seed reproducibility, invariant results when identical repeats are duplicated, transitive dependency merges, minimum-cluster suppression, and separate model strata |
| Overstatement | Known selection/audit/final deltas, independent partition resampling, and unavailable inputs or intervals |
| Failure visibility | Each requested failure family plus severe failures on excluded pairs remains visible despite improved aggregate scores |
| Authority preservation | No optimizer/model calls, catalog changes, reward changes, training admission, or deployment receipts; nested restrictions survive serialization |
| Compatibility and docs | Existing diagnostic/lifecycle regression suites; formatting and lint checks on changed Python; static review of guide and ADR against actual APIs |

Before/after evidence will use offline fixtures: existing reports show local diagnostic rates;
the extension additionally identifies selection overstatement and split contamination. Those
fixtures demonstrate software behavior, not improvement in a deployed model. The implementation
plan will specify exact tests and commands after this design is approved.

## Research basis and limits

The [Winner's Curse paper](https://arxiv.org/abs/2610.09239) reports selection-set inflation,
uncertain individual independent-audit estimates, and tested acceptance alternatives that did
not consistently improve whole-run outcomes. Its abstract was rechecked on 2026-10-10.
This design does not reproduce its experiments. The five partitions, contracts, thresholds,
and cluster aggregation above are repository design choices, not claims established by that paper.

[SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
describes paired resampling and percentile intervals. PR-6 will resample cluster summaries,
not individual dependent observations. No SciPy dependency is required; deterministic standard
library resampling is sufficient for the specified method. Statistical validity still depends
on the declared sampling unit and independence between clusters.

## Preparation validation

This proposal was checked against the original PR-6 requirements and the current source modules
named above. No product code, reward configuration, captured pilot evidence, or branch was
changed during preparation. Implementation tests have not been run for this proposed feature.
