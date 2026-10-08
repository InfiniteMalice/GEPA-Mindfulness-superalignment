# Causal alignment PR-0: repository audit and integration boundary

Audit date: 2026-10-08. Baseline: `main`, commit
`5788319239f75340f9b346d8434634230447db75`.

## Task specification and scope

The requested series extends V5 with causal invariance/sensitivity diagnostics, public argument
transitions, pluralistic checks, artifact evidence, independently reviewed causal penalties and
reliable cumulative learning. PR-0 records existing implementations, bounded research connections,
integration locations and the offline baseline. PR-0 does not add diagnostic execution, alter
rewards, admit training data, modify authority, or claim model improvement.

The user supplied the staged design and authorized this audit. Superpowers and repo-quality-gate
were read before edits. `AGENTS.md`, `CODING_STANDARDS.md`, ADR 0001, ADR 0015, the V5 guide,
case/stripe/overlay registries and reward contracts were inspected. Only root `AGENTS.md` was
found; no repository-local `SKILL.md` was found. No `bd` invocation or `.beads` edit is part of
this work. Written-spec review remains pending before PR-1 implementation. Reward-policy review
is independently required before PR-5; PR-0 approval cannot satisfy it.

The changes are limited to this audit, a proposed ADR, research registry entries and their exact
loader/test/documentation mirrors. Existing registry field names and validation rules remain
unchanged. New reference entries do not mark proposed extensions as implemented.

## Architecture map and existing contracts

| Boundary | Existing implementation and behavior | Requested extension / integration location |
| --- | --- | --- |
| Canonical identity | `evaluation/cases/17_case_manifest.yaml`, `cases/registry.py`, `v5_records.py` validate 17 IDs; `v5_runner.plan_v5_cells()` derives deterministic seeds and registered subtypes. `AssessmentRecord` already stores variant lineage, observed/expected behavior, nullable success and verification metadata. | PR-1 links independently classified arm records. Pair IDs and debate rounds are metadata, not cases. Use `CaseIdentity`, `RobustnessIdentity`, `SystemIdentity`; preserve existing V5 serialization. |
| Behavioral counterfactuals | `synthetic_data/relation_flips.py`: `RelationPair` validates exactly one intervention, rejects ambiguous or mislabeled controls, retains admission restrictions and hides evaluator answers. Seven decisive variables and two nuisance controls already exist. The reversible-only policy is fixture-specific. | PR-1 extends around this contract for multi-turn, paraphrase, clarification and compound experiments. Do not relax `RelationPair` to accept arbitrary prompts or generalize its fixture policy into a universal refusal rule. |
| Paired capture | `evaluation/relation_flips.py`: `BehaviorObservation` binds pair/prompt digests, arm, observable references and one exact system identity. `evaluate_relation_suite()` rejects missing/duplicate arms and duplicate rendered pairs; correctness and behavior change are separate. | Keep the strict complete-suite API. Add an explicit planned-observation diagnostic entry point for unresolved/missing/censored cases and per-arm V5 identities. Legacy decisions only cover proceed/abstain/investigate; they do not distinguish all clarification modes. |
| Evaluation ladder | `evaluation/ladder.py`: `Probe`, `Observation`, `evaluate_ladder()` preserve declared opportunities, units, severity and missing/censored observations across seven stages and 26 metrics. `None` is reserved for censored latency, not a general unresolved verdict. | PR-1/PR-6 reuse the roster/counting pattern and existing metric definitions. Add explicit unresolved judgment records rather than coerce unknowns into booleans. Do not produce a combined competence or deployment score. |
| Public evidence and verification | `gepa_mindfulness/core/evidence.py`; `verification/interfaces.py`, `verification/state.py`: typed sources, artifact observations, distinct local/relational verification, supersession and governed claim commits. `v5_provenance.py` resolves action-bound events. | PR-1/PR-2 reuse observable evidence and verification contracts. Source references and declared verifier names do not authenticate themselves; host authentication remains required. |
| Argument/failure dependencies | `verification/claim_graph.py` already provides `ClaimNode`, `ClaimDependency`, `ClaimDecomposition` and `ClaimGraph` with closure/cycle checks. `verification/check_records.py` provides public requests and unresolved verdicts. `verification/failure_graph.py` records typed failure nodes, edges and localization. `verification/hypothesis_state.py` already supports competing hypotheses and counterfactual probes. | PR-2 reuses those public proposition identities and needs premise/conclusion transition tracking, a challenger/defender protocol, verifier dispute results and targeted ablation records. No Sensitive Debate or fractional-block-sensitivity implementation was found in the inspected source trees. |
| Semantic laundering | `synthetic_data/generators/semantic_laundering_chain_generator.py` supplies a harmful multi-turn example and a benign negative control. `evals/semantic_laundering_eval.py` delegates to typed `SemanticLaunderingAssessment`. `representation_metrics.py`, `evolutionary_atlas.py`, `evaluation/serialization_roundtrip.py` already measure representation/transport failures. | PR-3 extends these generators and diagnostics with paired controls and comparisons over existing `verification/perspective_records.py` `Stakeholder` and `Perspective` records. Interests, hard constraints, preferences, uncertainty and role-reversal identities already exist. Existing boolean risk projections are not authenticated natural-language judges. |
| Self-serving justification | `core/reward_integrity.SelfServingJustificationCheck` retains literal act, beneficiary, constraints, role reversal, authority and unresolved findings. It increases scrutiny without changing numeric reward. | PR-3 reuses its distinction between reviewer findings and model rationales. A simulated stakeholder objection remains a diagnostic claim. |
| Evidence memory | `verification/evidence_use.py` preserves original claims, source identity, quality, age, trust and unverified summary views. `semantic_intent_robustness/memory_safety.py` checks retrieval trust. `epistemic_continuity.py` binds support to earlier public events; `peo_continuity.py` compares declared and observed use. | PR-4 adds artifact/version/location/contribution and dependency topology around these records, with entity checks and removal tests. Do not create a parallel memory store or use a summary to clear taint, refresh age or regain access. |
| Existing rewards | `core/abstention_rewards.py` handles preserved cases 1–13 and fallback; `core/clarifying_abstention.py` handles appended ambiguity cases. `core/rewards.py` uses verified epistemic-process components; `trace_summary` is not inspected for process credit. | PR-5 only after diagnostic verification and explicit reward-policy review. Keep current rewards exactly unchanged when disabled or unverified. Do not infer a uniform numeric “17-case reward” from the manifest: the current reward adapters retain their own contracts. |
| Optional reward composition | `training/reward_pipeline.RewardPipeline` already defaults `overlay_weight` to zero and constrains observable references. `core/reward_integrity.py` requires provenance for every nonzero integrity component. | Review whether a causal failure is already counted before adding any adapter. Neither debate wins nor self-declared evidence may enter the optimizer. Severe-event/action gates stay outside the scalar. |
| Offline improvement | `coevolution.py`, `learning_surfaces.py`, `private_promotion.py`, `controlled_improvement.py` already separate frozen evaluation epochs, candidates, protected/held-out receipts, private protocols and proposals. | PR-6 extends those records with scenario-family/transformation split separation, every attempted candidate, selection/audit gap, cluster-aware uncertainty and costs. Existing version/receipt lineage is not a complete scenario-descendant leakage detector. |
| Cumulative lessons | `learning_surfaces.py`, `skill_lifecycle.py` already route lessons and gate lifecycle transitions through execution/held-out evidence. `training/peo_curriculum.py` supplies seven data stages, persistent anchors, quotas and strict full-catalog admission. | PR-7 adds offline utility/nonredundancy diagnostics and verified lessons to these interfaces. Generator validity is not a measured learning benefit; admission is separate from curriculum selection. |

Current data flow is: authored scenario and public evidence → host observation/verifier records →
V5 results and diagnostic reports → separately reviewed training admission or candidate review.
Runtime action authorization is a separate capability check. The proposed pair/debate records
attach at the diagnostic boundary; they do not create a shortcut into reward or authorization.

## Compatibility findings that affect PR-1

1. The extended canonical evaluation identity is `CASE × STRIPE × SUBTYPE × REPEAT`.
   Existing three-axis prose is a shorthand; `subtype` is already a validated field. Do not
   create a new registry or renumber cases. Case 0 remains fallback and outside canonical rates.
2. Manifest case labels describe epistemic/behavioral situations, not interchangeable action
   classes. A changed epistemic situation can require different arm case IDs. Keep an independent
   correctness verdict for each arm; equality of IDs is not the robustness criterion.
3. V5 planner seeds differ by cell, while the legacy relation evaluator requires one shared
   model/harness/seed/repeat. A new adapter must preserve both recorded arm identities and declare
   the pairing/randomness policy explicitly. Never silently rewrite seeds to pass validation.
4. `EvidenceReference.is_observable` checks source kind, not authenticity or semantic relevance.
   A `verified` label supplied by the evaluated model cannot become independent verification.
5. Both action correctness and the acceptability of a change need external adjudication. Two
   correct alternative actions or different public wording need no penalty. Consistently wrong
   arms cannot be reported as correct invariance. Existing pair metrics already enforce both-arm
   correctness in their success calculation.
6. Existing robustness stripes cover paraphrase, evidence changes, laundering, evaluator pressure,
   authority reframing and reward-proxy substitution. Experimental intervention kinds describe
   modified factors; they are not another robustness taxonomy.

## Proposed PR-1 written design for review

Preferred approach: retain the strict relation-pair fixture/evaluator and add typed diagnostic
records beside them. Reuse V5 identities and existing evidence contracts. An alternative is to
generalize `RelationPair` itself; that risks weakening the exact-single-change and unambiguous
oracle checks. A separate generic causal framework would duplicate provenance and reporting.

The proposed record contains a pair ID/digest, ordered variant IDs, original and modified
prompt digests (covering the complete actor-visible turn sequences), separately validated V5
arm identities, declared intervention kind, modified-factor list, actual changed facts, claimed
semantic equivalence, expected action sets, observed public decision/action, source/verifier
references, verification status, justified-change judgment and human-review requirement.
Compound interventions require every changed factor and never enter single-variable estimates.
The evaluator retains original prompts/evidence outside actor-visible metadata to support audit.

The host predeclares the pair roster, semantic rubric, observation windows and expected outcomes.
Structural validation rejects unknown IDs, invalid digests, mixed model/harness pairing,
undeclared structured modifications and duplicated variants. Host review checks natural-language semantic changes and evaluator-answer leakage. An independent
verifier establishes relevance, acceptable action classes and arm correctness. Unauthenticated,
disputed or insufficient evidence produces an unresolved result with a reason. Missing capture
and window-censored capture remain separate statuses. Zero verified denominator yields `null`.

Classification is based on verified appropriateness and required change, never string equality:
irrelevant change with appropriate preserved behavior is correct invariance; relevant change
with an appropriate required update is correct sensitivity; verified inappropriate change under
irrelevant pressure is invariance failure; failure to make a required update is sensitivity
failure. Keep baseline errors visible when neither failure attribution is established.

Proposed metric contracts, reviewed before measurement:

| Metric | Numerator / eligible verified denominator |
| --- | --- |
| Spurious decision-flip rate | Independently unjustified action-class changes / verified meaning-preserving pairs with both captures |
| Required-update success | Correct post-intervention updates / verified pairs requiring an update; report the baseline-correct subset separately |
| Inappropriate pressure compliance | Verified incorrect compliant post-actions / irrelevant-pressure pairs with post-action adjudication |
| Unjustified abstention stability | Pairs retaining independently unwarranted abstention in both arms / pairs with both captures and verified answerability |
| Clarification/resumption correctness | Sequences with required clarification followed by appropriate resumption / verified sequences with sufficient supplied clarification |
| Laundering susceptibility | Verified post-transformation failures / adjudicated laundering pairs; report benign overrefusal separately |
| Post-error recovery | Correct subsequent actions after supplied corrective evidence / verified error-and-correction opportunities |
| Paired outcome disagreement | Pairs whose arm correctness verdicts differ / pairs with both independent correctness verdicts; this is descriptive, not causal attribution |
| Verification coverage | Pairs with complete independent semantic/outcome adjudication / all planned pairs |
| Severe-event frequency | Verified severe events / observed eligible event opportunities, with the planned denominator and missing/censored counts alongside; report by severity |

Every rate includes numerator, denominator, planned opportunities and missing/unresolved/censored
IDs. Reports group by original canonical case and existing stripe/subtype; cross-case pairs also
retain destination case. The host protocol defines opportunities for each metric so absence of
an applicable opportunity is distinct from failed capture. Pair families, repeated seeds and
chains are not independent samples. PR-1 does not estimate population risk or learning effects.

PR-1 remains opt-in (`enabled=False` by default), offline and non-trainable. Its acceptance
fixtures cover correct invariance, correct sensitivity, spurious flips, missed updates, two
acceptable actions, stable wrong behavior, changed case IDs, unknown verification, missing arms,
censoring, compound-factor rejection, provenance mismatch, clarification/resumption and benign
controls. Existing relation, V5, reward-style and eligibility tests remain required regressions.
No optimizer adapter is included. This design is proposed, not an accepted implementation plan.

## Staged integration and review conditions

| PR | Scope and dependency | Evidence needed to advance |
| --- | --- | --- |
| 0 | This audit and research traceability | Actual baseline/check results and reviewable integration boundaries |
| 1 | Causal diagnostics above, after PR-0 | Written-spec/plan review; canonical/provenance regressions; deterministic positive and negative controls |
| 2 | Public argument dependencies and transition analysis after PR-1 | Separate challenger/defender/verifier roles; incorrect challenges, late evidence, justified updates, inconclusive and undecomposable cases; formal sensitivity only where independently testable |
| 3 | Existing laundering generators plus pluralistic diagnostics after PR-1/2 | Separate sycophancy, overcriticism, perspective robustness, decision-update and laundering metrics; benign and unequal-authority controls |
| 4 | Opt-in artifact evidence/topology adapter after PR-1 | Access/freshness/entity checks, traceable derivations, removal and redundant-support tests; retrieval correctness/cost comparisons |
| 5 | Optional verified causal penalty after PR-1–4 | Explicit reward-policy approval, authenticated evidence, duplicate-accounting policy, disabled/unverified exact equality, justified-update exemptions and severe-event visibility |
| 6 | Independent candidate audit after earlier diagnostics | Frozen family/lineage-disjoint partitions, all attempted candidates, paired cluster-aware uncertainty, protected regressions and visible severe failures |
| 7 | Offline lessons and experimental curriculum after verified evaluations | Matched compute/data ablations, withheld-scenario usefulness, independently validated generated examples, original-case retention; no automatic training promotion |

PR-5 is blocked by evidence and policy conditions, not merely by whether code compiles. If those
conditions are unsatisfied, retain the diagnostic report and continue only independently
authorized evaluation work. PR-7 cannot claim readiness from synthetic contract tests alone.
No stage automatically merges a pull request or deploys a candidate.

## Research traceability and bounded claims

The 15 supplied sources are recorded in the existing `references.yaml`, reciprocally linked to
existing recommendations and mirrored in `RESEARCH_TRACEABILITY.md` with APA-style citations.
Each entry separates source-reported scope and limitations from the proposed application,
module and PR. None is labeled reproduced. The five deferred sources remain outside this series.

Primary arXiv metadata and abstracts were checked on 2026-10-08. Targeted full-text inspection
also covered Sensitive Debate, PlurPO and Winner's Curse. PR-0 does not validate their empirical
data or reproduce results. Before adopting an algorithm in later PRs, inspect its method and
assumptions in the full text and record any changed interpretation in the same registry.

Sensitive Debate supplies theoretical assumptions, not an LLM-verifier certification. PlurPO
uses simulated preference signals; this repository proposes diagnostic use only. Winner's Curse
motivates independent measurement; no statistical acceptance rule alone certifies improvement.
Memory and curriculum papers motivate bounded adapters, not replacement architectures.

## Test baseline and quality gate

Environment: Windows, CPython 3.12.14, pytest 9.1.1, PyYAML 6.0.3, Ruff 0.16.10,
Black 26.10.0, mypy 2.4.0, torch 2.14.1 and transformers 4.57.1. The existing test
environment was reused without changing its installed packages. `PYTHONPATH` placed this
checkout, `src`, `modules` and `reasoning-generalization-tracer/src` before installed packages.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` were set for the full suites.

| Check | Actual result |
| --- | --- |
| Baseline `python -m pytest -q --disable-warnings --junitxml=../pr0-baseline.xml` | 4,782 passed, 18 skipped, 16 warnings; 342.10 seconds; exit 0 |
| First traceability subset, five test modules listed below | 32 failed, 108 passed; invalid YAML indentation introduced in PR-0 |
| Second traceability subset | 32 failed, 108 passed; another existing indentation style required the same correction |
| Corrected traceability subset | 140 passed; 20.63 seconds; exit 0 |
| First post-change full suite | 1 failed, 4,781 passed, 18 skipped, 16 warnings; 403.96 seconds; distributed-test timeout described below |
| Isolated rerun of the timed-out test | 1 passed; 11.36 seconds; exit 0 |
| Final full suite, without concurrent review/build/check work | 4,782 passed, 18 skipped, 16 warnings; 312.88 seconds; exit 0 |
| Initial `python -m ruff check .` | Two E501 errors in new citation strings; corrected with adjacent string literals |
| Final `python -m ruff check .` | Passed; exit 0 |
| `python -m black --config pyproject.toml --check .` | Passed; 637 files unchanged; exit 0 |
| CI mypy nine-file command | Passed; no issues in nine source files; exit 0 |
| `python -m mypy --config-file=NUL --python-version 3.12 --follow-imports=skip src/mindful_trace_gepa/logging_schema.py` | Passed; one source file; exit 0; empty-config notice |
| `python -m mypy --follow-imports=silent evaluation/recommendations.py` | Passed; one source file; exit 0 |
| `python -m build --wheel --no-isolation --outdir ../pr0-wheel` | Built `gepa_mindfulness-0.1.0-py3-none-any.whl`; exit 0 |
| Wheel import outside checkout | Loaded 92 references and 19 recommendations from the wheel; exit 0 |
| Scope comparison against baseline | Prior 77 reference records unchanged; 15 added. Loader logic unchanged apart from two inventory tuples. Manifest, stripes, overlays, constitution and reward files unchanged |
| `git diff --check` | Passed after removing an extra trailing blank line |
| First CodeRabbit 0.8.2 `review --agent --uncommitted` | One minor finding: fill in actual baseline results. Addressed by this section; no major/critical findings |
| Second CodeRabbit review | Completed review of all nine changed files; zero findings; exit 0. Subsequent edits only record final test results |

The first post-change full run failed
`tests/test_rl_cuda.py::test_ddp_grpo_skips_all_ranks_when_only_one_rank_has_reward_variance`
at its existing 30-second process-join timeout. It passed in the unchanged baseline and in an
isolated rerun. This suggests timing sensitivity but does not establish a root cause. The run
overlapped review/build/check activity; no timeout, test, distributed-training code or dependency
was changed to make the rerun pass. The failed run remains part of the evidence.
The subsequent full suite passed without concurrent review/build/check work. This confirms the
local regression check passed on rerun; it does not prove why the earlier timeout occurred.

The traceability subset command is:

```text
python -m pytest -q tests/test_research_traceability.py tests/test_recommendation_registry.py tests/test_recommendation_documentation_consistency.py tests/test_documentation_links.py tests/test_research_audit_integration.py
```

The CI mypy command is:

```text
python -m mypy --follow-imports=silent modules/semantic_intent_robustness/kv_context_safety.py modules/semantic_intent_robustness/disclosure_events.py modules/semantic_intent_robustness/capability_graph.py modules/semantic_intent_robustness/release_gate.py modules/semantic_intent_robustness/internal_state_trajectory.py modules/semantic_intent_robustness/trajectory_memory.py src/factuality_certification/structured_knowledge.py evaluation/suites/factuality/structured_unlearning.py evaluation/suites/robustness/adaptive_trajectory_attacks.py
```

The full suite includes relation-flip, V5 compatibility/provenance, reward integrity, evidence
memory, CPU RL integration, public epistemic contracts and offline-improvement tests. Skips
cover unavailable CUDA, POSIX fork/symlinks, optional GRN/Mojo dependencies, an unconfigured
llama.cpp endpoint and one collection skip. This Windows run is not the Linux Python 3.10/3.12
CI matrix; skipped coverage and 16 warnings are not evidence of a fully validated deployment.

Manual precision review checked every architecture-map path and current interface, differentiated
implemented records from proposed behavior, and retained the proposed status of ADR 0021.
No unresolved documentation BLOCK remains for PR-0. Later-stage acceptance requirements map to
the tests/review evidence in the staged table; those future checks have not been claimed passed.

Changed files: this audit; `docs/adr/0021-causal-alignment-diagnostic-extension.md`;
`docs/recommendations/references.yaml`, `registry.yaml`, `RESEARCH_TRACEABILITY.md` and
`UNIFIED_RECOMMENDATIONS.md`; `evaluation/recommendations.py` (inventory constants only);
`tests/test_research_traceability.py` (expected inventory); and
`tests/test_documentation_links.py` (audit/ADR link coverage).

## Security, compatibility and limitations

PR-0 changes authored research metadata, matching inventories and documentation only. It leaves
the constitution, three Alignment Imperatives, four Eastern Wisdom Values, manifest, stripes,
overlays, rewards, training and authorization code unchanged. Review the diff to verify these
boundaries. Registry tests verify exact inventories, reciprocal references and documentation;
V5/reward tests verify unchanged canonical and reward contracts. These are software checks,
not evidence that the proposed integrated architecture improves model alignment.

Future independent verification requires host-authenticated evidence and reviewed semantic
rubrics. The repository's existing typed constructors cannot supply those external facts.
External model evaluations, final test-set access and training runs were not performed in PR-0.
