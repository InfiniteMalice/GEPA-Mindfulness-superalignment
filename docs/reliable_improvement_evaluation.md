# Reliable improvement evaluation

`evaluation.improvement_audit.audit_improvement()` audits host-captured improvement attempts.
It reports selection effects separately from independent audit, authorized final-test and OOD
(out-of-distribution) effects. It does not run a model, optimizer, final test or training pipeline.

The API is experimental and opt-in (`enabled=True`). Its DEVELOPMENT report grants no authority.
`verified_improvement_evidence` means authenticated independent comparison evidence, including
negative effects; it is not a certificate of improvement. Deployment eligibility remains
`not_assessed`. An interval excluding zero never promotes a candidate.

## Inputs and trust

Construct frozen records from `evaluation.improvement_records`. Each has strict `from_dict()`
and detached `to_dict()` methods. Unknown fields, invalid finite numbers, duplicate identities,
ambiguous event histories and conflicting bindings are errors. `record_digest()` hashes exact
canonical contents; hashes alone do not establish authenticity, completeness or independence.

| Input | Host responsibility |
| --- | --- |
| `ImprovementProtocol` | Freeze exact candidate/baseline model, harness and configuration identities, metrics, expected slots, seed policy, budgets and resampling policy before inspecting results |
| `DatasetManifest` | Supply complete ancestry through every root, scenario families, concrete transformation chains, extra dependencies and partition purposes; certify that undeclared paraphrase relationships are absent |
| `AttemptJournal` | Include proposals, starts, finishes and terminal decisions for every attempt, including losers and failures; certify completeness against the external optimizer log |
| `DiagnosticEvidence` | Bind a complete source report and exact row selector to the slot, original case content, configuration, rubric and budget; retain original restrictions |
| `ExposureRecord` | Disclose who used audit/final results for evaluation, proposal or selection, including affected rounds and descendant candidates |
| `FinalTestAuthorization` | Independently authorize the exact frozen candidate, protocol, final partition and evaluation-start event |

Call `audit_improvement(protocol, manifest, journal, evidence, exposures=(),
final_authorizations=(), authenticate=host_callback, enabled=True)` with tuples of records.
The callback receives an `AuthenticationRequest` containing purpose, subject digest, evaluator
identity and public evidence references. It must resolve those references against independently
controlled host records and return exactly `True` only when the complete binding is authenticated.
Missing references, no callback, non-boolean approval, exceptions or changed callback inputs
leave evidence unverified. Exception text is not exported. Callback execution itself remains
the host's responsibility; the reporting library performs no filesystem or network operations.

Authentication purposes and hashed subjects are:

- `protocol`: the complete protocol dictionary.
- `provenance`: the complete manifest dictionary, including dependency declarations.
- `journal`: the complete journal dictionary and external log completeness references.
- `exposure_history`: `{protocol_digest, exposures}`; an empty history also needs certification.
- `evidence`: `{protocol_digest, evidence, slot}`. The host verifies configuration/budget/content
  bindings that a legacy source report does not itself carry.
- `final_authorization`: the complete authorization record with evaluation-start event ID.

`rubric_digest` is the canonical digest of the source `TrustedEvaluatorContract` dictionary.
Changing data, model or source protocol cannot be hidden behind that stable rubric identity:
the separate evidence authentication binds the full source report to the complete PR-6 protocol.
A host that simply trusts a serialized `verified` flag does not satisfy this contract.

## Partitions, attempts and missing data

The five purposes are `synthetic_training`, `optimizer_selection`, `independent_audit`,
`final_test`, and `ood_combinations`. A partition may be empty. Family/ancestor/chain overlap,
shared content, and declared dependency links cannot cross partitions. Unknown parents and
cycles fail validation. Ancestor-only entries participate even with `evaluated=False`.
An OOD case binds a canonical JSON combination definition by digest. Hash checks cannot discover
an unrecorded paraphrase or establish that a declared OOD combination is novel.

Training entries are provenance metadata only and cannot occupy evaluation slots. PR-6 does not
change `TrainingEligibility` or lifecycle `ValidationSplit`. Audit or final results used for
proposal/selection cannot support independent claims for the affected candidate or descendants.
Evaluation-only exposure requires host certification of that limited use.

Each journal event has a unique ID and increasing ordinal. Proposals precede evaluations;
evaluation starts occur after candidate freeze. Each slot can start and finish once; use a new
repeat/slot for re-evaluation. Decisions are terminal. A planned candidate without a start is
listed separately and is not counted as attempted. Round counts derive from evaluation starts.
Costs are incremental event charges, not cumulative invoices: do not repeat the same charge on
start and finish. Totals are grouped by unit, currency and estimated/billed basis. Unknown charges
and attempts with no cost measurement remain explicit.

Absent, censored, unverified, incomplete and ineligible observations never become successes or
zero failures. Absent final evidence is `not_run`; provided final evidence without authorization
is `unauthorized`, with no numeric values or source payload exported. Other unverified source
payloads are retained for audit with authentication status; they do not enter numerical effects.
Severe events on otherwise excluded comparisons remain visible with their source status.

## Comparisons, uncertainty and failures

The supported source schemas are `evaluation-ladder-v1` (`rows`, `probe_id`),
`causal-diagnostics-v1` (`rows`, `opportunity_id`) and `sensitive-debate-report-v1`
(`metric_rows`, `opportunity_id`). Selectors are identifiers, never expressions or file paths.
Source digests, evaluator contracts, systems and row metrics are checked. Sources are produced
by existing ladder, causal and debate APIs; PR-6 does not redefine their judgments.

Baseline/candidate pairs need matching case, condition, repeat, metric, rubric, units, budget
and seed policy. A shared policy requires equal known seeds. `per_arm` and `unknown` permit
case pairing but make no claim of coupled randomness. Different or missing seeds are not replaced
with invented values. Unmatched pairs remain in coverage and do not contribute a delta.

The raw delta is candidate minus baseline. Improvement reverses that sign for lower-is-better
metrics. Means give equal weight to repeats within a condition, conditions within a case, cases
within a dependency cluster and clusters within an exact configuration stratum. Raw opportunity
counts/means are also reported; they have different denominators. Repeated cases never increase
the independent cluster count. There is no pooling of candidate/model strata.

`ResamplingPolicy(seed=...)` defaults to 10,000 paired cluster percentile resamples, 95% intervals
and at least 20 clusters. The interval uses linear interpolation between sorted resample values.
The host declares parameters before results. Missing pairs, unknown dependencies or too few
clusters suppress the interval and expose the reason. Twenty clusters is a reporting policy,
not a validity guarantee. Intervals are conditional, per-comparison summaries without simultaneous
or post-selection coverage. Their validity depends on actual independence between clusters.

Overstatement is selection improvement minus independent-audit improvement. Authorized final-test
overstatement is separate. Independent partition clusters are resampled separately; a reused or
unauthenticated audit is not substituted for independent evidence. Missing effects stay null.

Eight failure families are always present: causal invariance, required update, laundering,
debate fault localization, unjustified abstention, clarification resumption, calibration and
severe safety/authorization. Known eligible source outcomes supply denominators. Registered
success metrics can use `one_minus`; arbitrary metrics cannot. Brier contributions require
observed probability/outcome and a host-bound calibration definition digest. ECE remains a
separate source aggregate with its bin-definition binding, never an average of per-case ECEs.
Every severe source event retains its evidence and adjudication status even when a mean improves.

## Runnable offline example

Run this block from an installed checkout. It uses synthetic fixtures and an always-accepting
test callback solely to demonstrate software behavior. That callback is not suitable for real
evidence. The example shows selection improvement of 1, audit improvement of 0, no interval
with one cluster, a severe failed audit observation, and a rejected contaminated manifest.

```python
from dataclasses import asdict, replace

from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import audit_improvement, validate_manifest
from evaluation.improvement_records import (
    AttemptEvent, AttemptJournal, CandidateSpec, DatasetCase, DatasetManifest,
    DiagnosticEvidence, EvaluationSlot, ImprovementProtocol, MetricSpec,
    ResamplingPolicy, SystemConfig, record_digest,
)
from evaluation.ladder import Metric, Observation, Probe, Severity, evaluate_ladder
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility

ref = EvidenceReference("fixture-only", EvidenceSourceKind.EXTERNAL_RECORD)
evaluator = TrustedEvaluatorContract("fixture", "1", "binary-v1")
config_digest = content_digest({"decoding": "fixture"})
cases = tuple(DatasetCase(name, content_digest(name), name, purpose) for name, purpose in (
    ("selection", "optimizer_selection"), ("audit", "independent_audit"),
))
manifest = DatasetManifest("fixture-data", cases, (ref,), dependencies_known=True)
candidate = CandidateSpec("candidate", SystemConfig("baseline", "h1", config_digest),
                          SystemConfig("candidate", "h1", config_digest), config_digest, 0)
metric = MetricSpec("accuracy", "evaluation-ladder-v1", "representation_accuracy",
                    content_digest(asdict(evaluator)), "fraction")
slots, captures = [], []
events = [AttemptEvent("proposal", 0, "candidate", "r1", "proposal")]
for case in cases:
    for arm in ("baseline", "candidate"):
        slot_id = case.case_id + "-" + arm
        slots.append(EvaluationSlot(slot_id, "candidate", case.case_id, "accuracy", arm,
                                     0, config_digest, "condition", 7, "shared"))
        source = evaluate_ladder(
            (Probe("p", Metric.REPRESENTATION_ACCURACY, "all", Severity.CATASTROPHIC,
                   "fraction"),),
            (Observation("p", case.case_id == "selection" and arm == "candidate", (ref,)),),
            protocol_id="fixture", system=SystemIdentity(0, 7, arm, "h1"),
            evaluator=evaluator, training_eligibility=TrainingEligibility.HIDDEN_EVAL,
            enabled=True,
        )
        captures.append(DiagnosticEvidence(slot_id, canonical_json(source), "rows",
                                            "probe_id", "p", (ref,)))
        for kind in ("evaluation_started", "evaluation_finished"):
            events.append(AttemptEvent(str(len(events)), len(events), "candidate", "r1",
                                       kind, slot_id))
protocol = ImprovementProtocol("fixture", record_digest(manifest), (candidate,), (metric,),
                                tuple(slots), evaluator, (ref,), ResamplingPolicy(seed=7))
journal = AttemptJournal("fixture", record_digest(protocol), tuple(events), (ref,))
report = audit_improvement(protocol, manifest, journal, tuple(captures),
                           authenticate=lambda request: True, enabled=True)
contamination_rejected = False
try:
    validate_manifest(replace(manifest, cases=(cases[0], replace(cases[1], family_id="selection"))))
except ValueError:
    contamination_rejected = True
assert report["overstatement"][0]["audit"]["overstatement"] == 1.0
assert contamination_rejected
```

Tests execute this exact block. The other integration fixtures cover authorized and unauthorized
final results, all eight failure families and exception/mutation boundaries. Software fixtures
do not demonstrate an improved model or retroactively certify the PR-5 live pilot.
