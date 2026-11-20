# Metacognitive Monitoring and Control Integration Design

Date: 2026-11-20

Repository: https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment

Baseline revision: `c363bd332435cd84180d6c488e30a7ad938db195`.

## Intent and acceptance

Implement the supplied eight-stage brief as independently reviewable PRs. The experiment tests
whether explicit monitoring and method selection improve epistemic behavior. Implemented software,
synthetic contract results, model measurements, and unvalidated research hypotheses must remain
distinct in all reports.

The first release is disabled by default and operates in shadow mode over supplied episode
records. Monitoring produces inspectable assessments; control produces bounded proposals.
Neither component executes actions, writes accepted evidence, or changes rewards.

## Approaches and proposed choice

The proposed approach derives assessments from existing verification records and exposes an
explicitly enabled, separate experimental runner. This is the brief's compatibility path: the
current overlay registry has an exact ordered identifier tuple, output allowlists, and reciprocal
research links. Extending that registry would change its current contract. A separate runner
preserves that contract while permitting the new experiment.

Adding another epistemic-state store would duplicate existing interfaces and is excluded by the
brief. Implementing the experiment as generic prose self-criticism would lose typed provenance,
strict validation, and independently testable monitoring findings. Neither is a viable alternative.

The maintainer approved this design and the quality-gate fallback in this chat on 2026-11-20.

## Existing capabilities to reuse

| Existing contract | Use in the experiment |
| --- | --- |
| `verification.state.EvidenceState`, `EvidenceClaim`, and `EvidenceStatus` | Existing claim status, stable claim IDs, evidence links, and retained supersession history. |
| `core.evidence.EvidenceReference` | Source identity and observable versus private-source classification. Source labels alone do not certify truth. |
| `verification.epistemic_state.EpistemicContext`, `EpistemicStateEstimate` | Run/repeat/system identity and separate world, model, and monitor uncertainty; unknown values remain null. |
| `verification.hypothesis_records.HypothesisState` | Existing alternatives, assessor-specific histories, conflicting assessments, and measurement protocol. |
| `verification.hypothesis_state` | Existing conflict-aware projections and append-only extension checks. |
| `verification.epistemic_reconciliation` and action-bound event sequence validation | Prediction, action, observation, verification, chronology, and uncertainty-update binding. |
| `verification.evidence_use.EvidenceUseAssessment` | Original evidence/memory snapshots, reliability, integrity, authority, stale evidence, and diagnostic limitations. |
| Existing PEO continuity and motivated-forgetting diagnostics | Distinguish omissions, unavailable evidence, legitimate revision, and history preservation without inferring motive. |
| Existing failure graph and repeated-action diagnostics | Preserve failure provenance; distinguish unchanged obstacles from legitimate repeated validation. |
| `verification.semantic_exploration` | Reuse applicable bounded inquiry-selection contracts; do not reimplement existing uncertainty routing. |
| `evaluation.v5_runner.V5EvaluationCell` and case/stripe registries | Exact CASE × STRIPE × REPEAT identity, versioned systems, and deterministic seeds. |
| Existing calibration and abstention metrics | Preserve established confidence thresholds, abstention conventions, and externally labeled accuracy. |

Existing modules already perform narrow diagnostic monitoring and bounded inquiry selection.
What is missing is their source-linked metacognitive assessment, explicit shared-assumption graph,
small strategy catalogue, and matched evaluation of monitoring versus recommendation selection.

## Proposed component boundaries

Add focused modules under `gepa_mindfulness/verification/` for metacognitive records, monitoring,
repertoire entries, control proposals, and the PEO adapter. Add `evaluation/metacognitive.py` and
`evaluation/run_metacognitive.py` for matched evaluation and explicit invocation. Final filenames
and exact signatures will be specified in the implementation plan after design review.

The record layer contains only new diagnostic concepts. It references existing claims,
hypotheses, assessments, and events by stable IDs and retains detached typed source snapshots
where needed for validation. It does not copy the accepted evidence store or introduce a competing
epistemic-status enum. Diagnostic review/supersession lifecycle is separate from evidence truth.

An assessment carries a schema version, full evaluation identity, source event IDs, monitoring
method/version, evidence references, provenance, diagnostic status, findings, uncertainty
dimensions, dependencies, and suggested diagnostic next steps. Claims keep their existing status;
unreviewed or unavailable monitoring inputs are represented explicitly. Serialization rejects
unknown fields, unsupported versions, malformed identities, nonfinite values, and invalid links.

Shared dependencies link hypotheses to explicit assumption or source IDs. Shared support records
do not establish independence. Monitoring reports the common dependency and its declared status;
it does not conclude that every dependent hypothesis is false.

Every finding identifies its affected hypotheses/claims, supporting records, unresolved uncertainty,
and an explicit verification status. Missing inputs produce unavailable or unassessed results.
Confidence alone, self-report, and private reasoning cannot establish verified findings.

The eight repertoire entries follow the brief: assumption audit, discriminative testing,
evidence-grounded verification, robustness analysis, belief revision, clarification/abstention,
evidence recovery, and transfer calibration. Each declares applicability, required evidence,
output, verification requirements, resource implications, prohibited effects, and source.
Clarification and epistemic abstention remain distinct proposal kinds.

Control consumes an assessment and an explicitly declared budget. It chooses only applicable
strategies and returns a proposal containing source IDs, rationale, uncertainty, estimated benefit,
resource estimate, stopping condition, and required authorization. Estimates may be unknown.
There is no callback or capability in this API for executing a proposal. A no-problem episode
recommends continuing without intervention. Unsupported input cannot justify stopping a branch.

## Temporal and authority boundaries

Data flow: host-supplied source records → validation and temporal cutoff → derived monitoring
assessment → applicable strategy selection → diagnostic proposal → separate host review.

Evaluation labels and later outcomes are held outside the prospective controller input. The
runner computes assessments/proposals at a recorded event cutoff, then evaluates against
independently supplied labels. Cross-run, case, stripe, repeat, system-version, and forward-event
references are rejected. Later outcomes must not influence earlier recommendations.

The PEO adapter calls existing sequence and reconciliation validators and references their
original records. Prediction error can support several unresolved explanations; it does not
automatically imply a faulty model or changed environment. Historical evidence remains
recoverable across legitimate revision. Continuity findings do not assert deceptive intent.

No changes are made to the constitution, the 17 canonical cases, Case 0 fallback semantics,
canonical labels, reward weights, reward admission, model parameters, permissions, or runtime
authority. Diagnostic serialization cannot be used as a grant or accepted-evidence update.
No `bd` command or `.beads/*` edit is part of this work.

## Evaluation and ablations

Use matched V5 cells, versions, seeds, and declared resource limits for four conditions:
baseline, monitoring only, monitoring with recommendations, and replay-only authorized-control
simulation. The initial release will not execute real external interventions. Report that
limitation and do not present replay outcomes as deployed control effectiveness.

Cover all ten requested challenge families and the existing robustness stripes, with positive,
negative, and adversarial controls. Report verified success, hallucination, appropriate and
inappropriate abstention/clarification, calibration, assumption-warning precision/recall, shared
dependency detection, contradiction retention, evidence recovery, repeated failure, strategy
quality, verification overhead, tokens/compute, and held-out coverage separately. Unavailable
metrics remain null with a reason; synthetic labels are identified as fixture labels. No new
aggregate alignment score is introduced.

Plan the eight briefed ablations: baseline/monitoring; monitoring/recommendations; shared
dependencies on/off; continuity on/off; fixed/state-guided selection; verified/self-reported
evidence; full/simple uncertainty monitoring; intact/corrupted inputs. Add matched-computation
and evaluation-label leakage controls. Trained-model outcomes remain NOT_RUN unless an explicit,
supported model backend and independently trusted evaluator are supplied and actually executed.

## PR sequence

| PR | Reviewable deliverable and acceptance evidence |
| --- | --- |
| 0 | Architecture audit, reproducible baseline, reused interfaces, limitations, and ablations; no runtime behavior changes. |
| 1 | Immutable derived assessment/dependency contracts; exact round trips, strict schema validation, unknowns, immutability, and no evidence promotion. |
| 2 | Bounded monitoring with source-linked findings; assumption, dependency, contradiction, continuity, calibration, repetition, and outcome-grounding controls. |
| 3 | Eight declarative strategies with applicability, verification, resource, source, and prohibited-effect contracts. |
| 4 | Shadow-only proposals and selection; budgets, stopping conditions, no-intervention path, temporal isolation, and no execution API. |
| 5 | Existing PEO, evidence-use, hypotheses, and verification integration; original history preserved and causal sequence validated. |
| 6 | Matched evaluation conditions, challenge battery, metrics, ablations, and negative controls over canonical identities. |
| 7 | Explicit disabled-by-default runner, CPU demos for a problem and a no-problem episode, packaging/docs, reproducible commands, and measured limitations. |

Each stage includes focused regression tests and compatibility checks. Open and review each PR
against its own final head. Address actionable findings, rerun affected verification, and merge
only after CodeRabbitAI or Codex finds no outstanding problems and required CI passes. Verify
review comments, unresolved threads, and the expected head SHA immediately before merging.

## Baseline and proposed quality-gate fallback

The focused existing-contract baseline passed: **405 tests in 1.79 seconds** on Python 3.12.
The structural dry run produced 374 cells: 17 canonical cases, 11 existing stripes, and two
repeats, with all overlays disabled. Its canonical JSON SHA-256 was
`1fafd40bc6de7c5fa1e48a7b31107a185c0b4fc9e0b89b3a1efa1183ad1597c4`.
Existing synthetic calibration responses produced two results: one correct and one incorrect,
accuracy 0.5, Brier score 0.32500000000000007, and ECE 0.45. These are fixture-scoring results,
not measurements of a newly executed model. Ruff, Black (866 files), and the nine main CI mypy
targets passed against the unmodified baseline.
The complete existing offline suite is running; its result must be recorded before PR-0 delivery.
The existing V5 CLI is a deterministic planner and explicitly does not execute a model. A dry-run
coverage plan is useful structural evidence, not a behavioral baseline or alignment measurement.

Reproduce the focused baseline from the repository root with the project's existing environment:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ../.venv312/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_epistemic_state.py tests/test_hypothesis_state.py \
  tests/test_epistemic_reconciliation.py tests/test_evidence_use.py \
  tests/test_peo_continuity.py tests/test_failure_graph.py \
  tests/test_experimental_overlay_registry.py tests/test_experimental_overlay_flags.py \
  tests/test_17_case_clarifying_abstention.py tests/test_alignment_calibration_metrics.py
```

The repository requires `$repo-quality-gate`, which is absent from the available skill catalogue
and local skill files. The maintainer explicitly approved this fallback before implementation:
retain task specifications, design and implementation notes under `history/`; use Superpowers
test-driven implementation and verification; run focused tests, complete offline tests, Ruff,
Black, the CI mypy targets plus applicable new-module checks, wheel build/package smokes, and
remote CI; document actual results and limitations for every PR. Never substitute skipped checks
for successful checks. Low-risk documentation can use the repository's explicit documentation
exception; code changes require approval of the fallback.

## Research basis and limitations

Sage-Taiyi v2 sections 2.3–2.5 motivate source-linked cognitive assessments, reusable methods,
and separate monitoring/control: https://arxiv.org/html/2611.12762v2.
Its scientific-discovery results are not evidence of alignment improvement in this repository.
The optional hindsight experiment in the brief remains future work and cannot supply rewards.

The host remains responsible for evidence authentication, measurement validity, trustworthy
evaluation labels, and any future authorized execution. Typed records validate structure and
relationships; they do not prove semantic truth. For example, a monitor can correctly flag two
hypotheses as sharing an unverified premise while having no evidence that the premise is false.
