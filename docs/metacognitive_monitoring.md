# Metacognitive monitoring and control

## Architecture audit and scope

This experiment derives diagnostic assessments from existing evidence, uncertainty,
hypothesis, and action-bound event records, then select bounded reasoning-method proposals.
Monitoring, strategy selection, and externally verified behavioral improvement are separate
capabilities. All new computation is disabled by default.

The design is inspired by [Sage-Taiyi v2, sections 2.3–2.5](https://arxiv.org/html/2611.12762v2):
source-linked assessments inform selection among reusable methods. The paper's scientific
discovery results do not establish an alignment improvement in this repository.

## Existing capabilities

| Concern | Existing interface and boundary |
| --- | --- |
| Epistemic state | `verification.epistemic_state.EpistemicStateEstimate` separates world, model and monitor uncertainty. `None` means unknown. `EpistemicContext` binds run, repeat and model/harness versions. |
| Evidence and provenance | `core.evidence.EvidenceReference` and `verification.state.EvidenceClaim`/`EvidenceState` retain source identity, existing claim statuses and acyclic supersession history. Structure and source labels do not authenticate truth. |
| Hypotheses | `verification.hypothesis_records.HypothesisState` retains alternatives and assessor histories. `hypothesis_state` detects conflicting live assessments and preserves incomplete alternatives. |
| PEO | `mindful_trace_gepa.action_bound_events`, action-bound sequence validation, and `verification.epistemic_reconciliation` bind predictions, observations, verification and uncertainty updates. |
| Evidence use | `verification.evidence_use.EvidenceUseAssessment` retains original claims, measurements, memory trust, authority, integrity, reliability, distortion and freshness. Diagnostic summaries cannot replace originals. |
| Evidence continuity | Existing PEO continuity and motivated-forgetting diagnostics distinguish missing retrieval, omission, legitimate updates and unsupported unavailability without establishing motive. |
| Strategy selection | `verification.semantic_exploration` already selects bounded inquiry proposals. Reuse its applicable contracts rather than adding a competing uncertainty router. |
| Repeated failure | Existing repeated-action diagnostics and `verification.failure_graph` preserve action/failure provenance; repetition alone is not failure. |
| Runtime authority | `verification.runtime_governance` and governed evidence commit control separately authorized operations. A diagnostic record is not an execution grant. |
| Calibration and abstention | `evaluation.suites.calibration` supplies accuracy, Brier, ECE and selective-answering measures; existing ambiguity/abstention contracts distinguish missing evidence from unclear instructions. |
| Experimental overlays | The exact ordered IDs, allowed outputs and reciprocal research references in `evaluation.experimental_overlays` are a compatibility boundary. |
| Evaluation identity | `evaluation.v5_runner.V5EvaluationCell` retains CASE × STRIPE × REPEAT, deterministic seeds and system versions. The canonical registries contain 17 cases and 11 stripes. |

Narrow monitoring and strategy selection already exist. The missing capability is a typed,
source-linked view that connects those diagnostics to explicit shared dependencies, an inspectable
method catalogue, and matched evaluation of monitoring versus recommendation selection.

## Implemented integration

Focused metacognitive modules in `gepa_mindfulness.verification` provide derived assessments,
monitoring, eight declarative strategies, control proposals and a PEO adapter. Keep original
evidence and hypotheses authoritative; reference their stable IDs and detach snapshots for
validation. They reuse existing evidence statuses. Monitoring lifecycle and unavailability are
separate from the status of a claim.

A separate explicitly enabled evaluation runner, as permitted by the implementation brief,
preserves the fixed overlay registry. All new computation defaults to disabled. Initial control
is shadow-only; the fourth evaluation condition is replay/simulation only. No execution or
parameter-learning path is added.

Data flow is host-owned records → identity/chronology validation at a causal cutoff → derived
assessment → applicable strategy → diagnostic proposal → separate host review. Evaluation-only
labels and later outcomes cannot enter prospective method selection. Unknown metrics remain
unavailable rather than becoming zeros or successes.

The controller cannot alter the constitution, canonical cases, labels, rewards, accepted evidence,
permissions or governance. Assessments and proposals cannot satisfy verifier or reward-admission
contracts. Original contradictory and superseded records remain recoverable.

## Staged delivery and validation

Stages 0–7 deliver audit/baseline, assessment contracts, monitoring, repertoire, control proposals,
PEO integration, evaluation/ablations, and the runner/demonstrations/documentation respectively.
Each stage includes positive, negative and adversarial checks, actual verification evidence and
an independently reviewable PR. The maintainer authorized merge after CodeRabbitAI or Codex has
no outstanding problems and required checks pass.

Planned ablations cover baseline versus monitoring, monitoring versus recommendations, shared
dependencies on/off, continuity on/off, fixed versus guided strategies, verified versus
self-reported evidence, full versus simple uncertainty monitoring, and intact versus corrupted
inputs. Match versions, seeds and budgets; include computation and hidden-label leakage controls.
Report findings by case and stripe without inventing an aggregate alignment score.

The baseline and exact commands are recorded in
[`experiments/metacognitive_baseline.md`](experiments/metacognitive_baseline.md).
Detailed design and implementation decisions are retained in
[`history/2026-11-20-metacognitive-design.md`](../history/2026-11-20-metacognitive-design.md) and
[`history/2026-11-20-metacognitive-plan.md`](../history/2026-11-20-metacognitive-plan.md).

## Known limitations

Typed references and validation do not authenticate callers, establish semantic truth, prove
independence, or measure a model's actual use of evidence. Host-owned provenance and independent
evaluation remain necessary. Synthetic fixture outcomes establish software behavior only.
Trained-model alignment gains, transfer effectiveness, and live authorized intervention effects
are unvalidated until measured under an appropriate protocol.

For example, shared-dependency detection can establish that two declared hypotheses cite the
same unverified premise. It cannot establish that the premise is false or that either hypothesis
is wrong. A diagnostic judgment cannot promote that premise to verified truth.

## CPU demonstrations and reproducible battery

The separate runner preserves every existing experimental overlay ID and its order. It loads
only bundled synthetic data; it requires no model weights, external evaluator, network access,
NumPy or Torch. Its input JSON chooses a factory challenge and canonical V5 coordinates; it
cannot enroll a trusted evaluator, grant authority or provide an execution callback.

```bash
# Disabled before reading a file or constructing an episode:
python -m evaluation.run_metacognitive

# Shared unsupported premise: diagnostic verify_assumption proposal, no action:
python -m evaluation.run_metacognitive --enable --demo unsupported

# Supported independent evidence: no findings, continue with no intervention:
python -m evaluation.run_metacognitive --enable --demo healthy

# All 17 cases × 11 stripes × 2 repeats, ten challenges plus healthy control:
# --output must name a new file. Summary JSON goes to stdout.
python -m evaluation.run_metacognitive --enable --battery \
  --output /tmp/metacognitive-results.jsonl > /tmp/metacognitive-summary.json

# Small selected-coordinate smoke:
python -m evaluation.run_metacognitive --enable --battery \
  --case-id 1 --stripe-id NONE --repeats 1
```

`--input path.json` accepts exactly the `metacognitive-demo-v1` schema illustrated by the two
packaged `gepa_mindfulness/data/metacognitive_*_demo.json` files. Unknown fields, versions,
noncanonical cells and malformed types fail validation. File specifications are limited to
16 KiB. `--output` streams per-unit JSONL with all four condition statuses, metric availability,
source digests and eight ablations. The summary partitions synthetic agreement by challenge,
case and stripe and binds the raw export with SHA-256. Outputs remain REGRESSION data.

See [contracts](metacognitive_contracts.md), [monitoring](metacognitive_engine.md),
[repertoire](metacognitive_repertoire.md), [control](metacognitive_control.md),
[PEO capture](metacognitive_peo.md) and [evaluation limits](metacognitive_evaluation.md) for
individual interfaces. A host may independently investigate a proposal under its existing
governance; this runner never performs that investigation. Neither more critiques nor more
elaborate method descriptions establish improved decisions or alignment.

Open scientific questions include semantic equivalence verification, evaluator independence,
monitor calibration on real episodes, strategy benefit under matched model compute, and whether
authorized interventions improve externally verified outcomes. The synthetic suite answers none
of those empirical questions.
