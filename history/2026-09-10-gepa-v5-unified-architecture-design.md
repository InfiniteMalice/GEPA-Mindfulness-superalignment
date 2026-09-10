# GEPA Mindfulness 17-Case Framework V5 Unified Architecture Design

**Status:** Approved approach; written specification pending maintainer review

**Date:** 2026-09-10

**Framework name:** GEPA Mindfulness 17-Case Framework V5

**Framework version:** `17case-v5`

## 1. Problem

The repository contains the required ingredients for epistemic calibration, ambiguity handling,
reward-integrity checks, structured logging, semantic-intent robustness, objective robustness, and
training. Those ingredients do not yet form one enforceable V5 architecture.

The current implementation has four material inconsistencies:

1. Case identity is copied across Python modules, tests, and documentation. The repository has no
   machine-readable source of truth for the 17 canonical cases.
2. Generated reasoning prose and self-described uncertainty can increase optimizer-facing reward.
   Several Schema V3 overlay fields also earn credit because they are present, without independent
   verification.
3. Structured logging does not yet model the complete prediction, action, outcome, verification,
   and epistemic-assessment sequence.
4. Semantic-intent robustness operates over prompt-level semantic records but lacks an immutable
   raw-input and bounded alternate-representation layer.

The root README also presents additions chronologically, duplicates subsystem descriptions, and
contains a Version 4 table whose Case 15 through Case 17 meanings conflict with the canonical case
definitions.

## 2. Goals

This change will:

- establish exactly 17 canonical cases under version `17case-v5`;
- represent evaluation as case by robustness stripe by repeat run;
- reward only independently verified epistemic-process properties;
- keep generated reasoning prose and deception or mechanistic signals diagnostic by default;
- record immutable prediction commitments before action and bind them to observed outcomes;
- separate world or artifact state from evidence or belief state;
- add local-execution and relational-evidence verification interfaces;
- add structured, epistemically qualified failure graphs;
- extend the existing semantic-intent module with bounded alternate representations;
- preserve raw input and representation provenance;
- model runtime authority, bounded recovery, learning destinations, and verified skill lifecycle;
- add a recommendation registry and research traceability index;
- reorganize README as orientation, navigation, and conceptual overview.

## 3. Non-goals

This change will not:

- create Case 18 or any later canonical case;
- implement a new tokenizer, foundational model, database, or agent framework;
- infer truth from generated reasoning style;
- subtract reward directly from a deception detector, circuit signal, attribution signal, or
  representation disagreement;
- expose or require private chain-of-thought;
- mutate the model or harness within the evaluation episode being scored;
- implement unrestricted retry, replan, or autonomous self-modification;
- claim that cited research proves the complete repository architecture;
- make experimental Bayesian, multi-agent, orchestration-scope, or mechanistic overlays stable by
  default.

## 4. Architectural invariants

### 4.1 Canonical case invariant

`evaluation/cases/17_case_manifest.yaml` is the canonical source for case IDs, machine keys, human
titles, expected epistemic behavior, and compatibility metadata. The manifest contains exactly 17
entries with stable IDs 1 through 17.

Cases 14 through 17 have these frozen meanings:

- 14: correct high-stakes clarifying abstention;
- 15: over-eager ambiguous or high-stakes compliance;
- 16: unnecessary low-stakes clarification;
- 17: clarification loop, repeated unnecessary questioning, or failure to resume.

Case 0 remains a fallback state and is not a canonical case.

Python code will load case names from the manifest. Compatibility constants such as `CASE_NAMES`,
`ORIGINAL_CASE_IDS`, `FRAMEWORK_CASE_IDS`, and `APPENDED_AMBIGUITY_CASES` remain available, but
their values are derived from or checked against the manifest.

### 4.2 Epistemic process reward invariant

A nonzero optimizer-facing epistemic-process reward requires one of these provenance routes:

1. observable evidence references plus a recorded verification result; or
2. a declared trusted evaluator contract plus the evaluator result and version.

The compatibility field `thought_align` remains available. It records a diagnostic classification
and does not, by itself, establish optimizer eligibility.

Generated reasoning text, public rationale text, `trace_summary`, uncertainty vocabulary,
self-reported honesty, and claims of careful reflection cannot change training fitness while all
verified facts remain constant.

A verified improvement in evidence fidelity, calibration, contradiction handling, prediction
accuracy, justified abstention, or recovery can increase a bounded process reward even when the
final answer is unchanged.

### 4.3 Diagnostic signal invariant

The following remain diagnostic unless a later, separately reviewed change supplies causal
validation and independently verifiable behavioral grounding:

- deception detectors and dual-path divergence;
- circuit, attention, cache, latent-state, and attribution signals;
- generated reasoning alignment;
- semantic-laundering signals;
- representation disagreement;
- mechanistic correlation.

Diagnostic signals may trigger investigation, review, or evaluation routing. They do not directly
change optimizer fitness.

### 4.4 Evidence invariant

Raw evidence records are append-only. A derived interpretation can be superseded through an
explicit `superseded_by` relationship. Supersession does not rewrite or delete the source evidence.

A model statement that an action succeeded is a claim. A world-state observation or verifier
result establishes whether the action occurred and whether the claimed outcome is supported.

### 4.5 Representation invariant

The literal source text is immutable. Conservative normalization, orthographic candidates,
phonological candidates, alternate segmentation, and contextual candidates are derived views.

Every derived candidate records its source span, transform channel, candidate text, component
scores, confidence, generation reason, and provenance. No candidate silently replaces the literal
source in logs, memory, evaluation, or policy input.

Meaning-preserving variants should preserve semantic policy. Meaning-changing controls must remain
separable even when topic or token patterns overlap.

## 5. Components

### 5.1 V5 manifest and registry loader

Create:

- `evaluation/cases/17_case_manifest.yaml`
- `evaluation/cases/robustness_stripes.yaml`
- `evaluation/cases/__init__.py`
- `evaluation/cases/registry.py`

`registry.py` will expose immutable typed records for canonical cases and robustness stripes. It
will validate version, count, unique IDs, unique machine keys, required fields, and stripe names at
load time. Package data configuration will include both YAML files in built wheels.

The stripe registry will contain:

`NONE`, `PARAPHRASE`, `DISTRACTOR`, `MISSING_EVIDENCE`, `CONFLICTING_EVIDENCE`, `STALE_EVIDENCE`,
`UNAUTHORIZED_EVIDENCE`, `TOOL_ERROR`, `TOOL_OUTPUT_INJECTION`, `REWARD_PRESSURE`, and
`TIME_BUDGET_PRESSURE`.

Fine-grained representation phenomena will be stripe subtype metadata rather than new base cases.

### 5.2 Verified epistemic-process rewards

Create `gepa_mindfulness/core/epistemic_process.py` with these concepts:

- `ProcessComponentName` for supported epistemic competencies;
- `VerificationSource` for observable evidence or trusted evaluator routes;
- `ProcessVerification` for one verified property and its provenance;
- `EpistemicProcessAssessment` for diagnostics and optimizer-eligible components;
- a bounded weighted aggregation function.

Modify `gepa_mindfulness/core/rewards.py` so `trace_summary` remains accepted for compatibility but
is diagnostic only. Replace `_honesty_signal` with a verified-process calculation. When no verified
assessment is supplied, the epistemic-process component is zero.

Modify `gepa_mindfulness/core/abstention_rewards.py` so `thought_align` still selects diagnostic
case semantics, but `H` is awarded only when a qualifying verification is supplied. Case identity,
correctness, calibration, and abstention behavior remain behaviorally compatible.

Modify Schema V3 reward augmentation so `r_grounding`, `r_control`, `r_reasoning_unit`,
`r_observability`, and `r_group_theoretic` are zero unless the matching property is independently
verified. Existing overlay values remain available as diagnostics.

The reward-integrity boundary will require observable evidence or a declared trusted evaluator
contract for every nonzero optimizer-facing component, not only negative values. Neutral zero
values require no provenance record.

### 5.3 Action-bound event and evaluation records

Extend `src/mindful_trace_gepa/logging_schema.py` rather than creating another logging framework.
Add event types:

- `PREDICTION_COMMIT`
- `ACTION_PROPOSED`
- `ACTION_EXECUTED`
- `OUTCOME_OBSERVED`
- `VERIFICATION_RESULT`
- `EPISTEMIC_ASSESSMENT`
- `CASE_ASSESSMENT`

Add optional envelope fields for action linkage, parent events, evidence and verifier references,
model and harness versions, case and stripe identity, repeat and seed, authorization, validity,
and supersession.

Create `evaluation/v5_records.py` for a typed V5 evaluation record with distinct case, robustness,
system, epistemics, behavior, outcome, score, and diagnostic sections. Construction validates the
`17case-v5` case version, canonical IDs, registered stripes, non-negative repeat IDs, bounded
confidence and scores, and required model and harness versions.

Prediction commitments are frozen records. An outcome record can reference but cannot mutate the
prediction payload.

### 5.4 V5 cell planning and metrics

Create `evaluation/v5_runner.py` with:

- deterministic enumeration of case by stripe by repeat cells;
- configurable repeat count with default 5;
- explicit model version, harness version, and seed derivation;
- aggregation of `Pass@k`, `Mean@k`, `Pass^k`, and `ConsistencyGap@k`;
- independent correctness and consistency fields.

`Pass@k` means at least one passing repeat in the cell. `Mean@k` is the arithmetic mean of binary
pass indicators. `Pass^k` means every repeat passed. `ConsistencyGap@k` equals `Mean@k - Pass^k`.
The implementation will not label consistency as correctness.

### 5.5 World state, evidence state, verifiers, and failure graphs

Create focused modules under `gepa_mindfulness/verification/`:

- `state.py`: immutable `ArtifactObservation`, `WorldStateChange`, `EvidenceClaim`, and
  `EvidenceState` records;
- `interfaces.py`: `LocalExecutionVerifier` and `RelationalEvidenceVerifier` protocols plus typed
  results;
- `failure_graph.py`: failure nodes, epistemically qualified edges, and localized failure report;
- `recovery.py`: bounded retry, argument repair, bounded replan, clarification, and escalation
  decisions.

Failure reports distinguish `first_anomaly`, `root_cause`, `decisive_failure`, symptoms, and
`recoverable_until`. Relations must state whether they are causal, contributing, preceding,
correlated, or hypothesized. A causal edge requires verifier support.

Create `gepa_mindfulness/runtime_governance.py` for typed Planner, Executor, Verifier, Auditor, and
Human authority capabilities. Only an authorized Executor may perform writes or external actions.
Irreversible actions require explicit authorization evidence.

### 5.6 Representation robustness

Extend `modules/semantic_intent_robustness/` with:

- `representation.py`: source spans, representation channels, candidates, and bounded lattices;
- `representation_views.py`: literal and conservative-normalization views;
- `representation_candidates.py`: orthographic, phonological, segmentation, and contextual
  candidate generation;
- `representation_routing.py`: semantic-hinge routing and disagreement policy;
- `representation_metrics.py`: recall at k, false-repair rate, disagreement, clean regression,
  and latency accounting.

The candidate generator will be deterministic, dependency-light, and bounded by explicit per-span
and global top-k limits. Unicode normalization and removal of known zero-width artifacts are
derived views. Meaningful whitespace remains intact.

Phonological generation will support one-token and phrase candidates through a small explicit
lexicon interface. The known `bone apple tea` hypothesis can appear in top-k when context permits,
while the literal reading remains available. The module will not claim general speech recognition.

The semantic-intent pipeline will evaluate selected candidates through the existing decomposition
and policy path. Material channel disagreement routes to clarification, bounded caution, or
abstention according to stakes; it does not choose an alternate interpretation as proven truth.

`evals/semantic_laundering_eval.py` will delegate to the richer module or become a compatibility
wrapper. It will no longer define a separate semantic-laundering model.

### 5.7 Learning surfaces and skill lifecycle

Create `gepa_mindfulness/learning_surfaces.py` with the destinations `TRACE_ONLY`, `MEMORY`,
`HARNESS`, `SKILL_GRAPH`, `MODEL`, and `HUMAN`. Every lesson proposal has one primary destination,
source evidence, rationale, reversibility, and review status.

Create `gepa_mindfulness/skill_lifecycle.py` with typed transitions for source experience,
verification, procedural family, task-local instantiation, execution, credit, refinement,
held-out validation, commit, and rollback. Skill credit requires execution evidence and cannot be
created from explanatory prose alone.

Model and harness versions are frozen within an evaluation episode. Candidate evolution happens
between episodes and requires held-out evaluation plus the protected V5 regression suite.

Experimental overlays will be declared in a disabled-by-default registry with maturity and feature
flags. The registry will cover competing hypotheses, information-gain inquiry, adaptive small
multi-agent topology, global/focus/local scope, and mechanistic audit. These declarations do not
create new cases or optimizer rewards.

### 5.8 Recommendations and research traceability

Create:

- `docs/recommendations/registry.yaml`
- `docs/recommendations/UNIFIED_RECOMMENDATIONS.md`
- `docs/recommendations/RESEARCH_TRACEABILITY.md`

The recommendation registry will contain REC-001 through REC-014 with stable fields for priority,
status, rationale, targets, supersession, dependencies, research references, repository references,
acceptance tests, and implementation references.

Research metadata will be retrieved from primary paper sources. If an identifier does not resolve,
the traceability entry will record the unresolved status and will not invent authors, venue, DOI,
or findings. Each connection will distinguish source evidence, repository inference, and
experimental proposal.

### 5.9 Documentation consolidation

Rewrite README around this order:

1. project description and navigation;
2. project status and maturity;
3. quick start;
4. core alignment architecture;
5. 17-Case Framework V5;
6. reward and epistemic process;
7. safety and robustness modules;
8. verification, evidence, and provenance;
9. deception and interpretability;
10. training and runtime;
11. evaluation;
12. datasets and synthetic data;
13. repository layout;
14. research basis and traceability;
15. limitations and maturity;
16. contributing and license.

The README will preserve valid commands, runtime caveats, security warnings, hardware qualification,
and supported links. Detailed bibliographies and subsystem manuals will remain outside README.

Update `docs/17_CASE_FRAMEWORK.md`, `docs/structured_logging.md`, the Schema V3 documentation, the
semantic-intent README, core reward documentation, and evaluation documentation to use V5 terms.

Add `docs/FOUNDATIONAL_REPRESENTATION_ARCHITECTURE.md` as a research note. The note will discuss
byte and character access, graphemes, whitespace, phonology, alternate segmentation, latent
patches, adaptive retokenization, and invariance training without presenting an application-layer
implementation as a tokenizer solution.

## 6. Compatibility strategy

- Keep public identifiers such as `thought_align`, `thought_aligned`, `r_thought`, `CASE_NAMES`,
  and existing Schema V3 classes.
- Document `thought_align` as a compatibility diagnostic alias for `reasoning_grounded`; use
  `epistemic_process_verified` for optimizer eligibility.
- Accept old structured event rows and old evaluation result JSONL.
- Make new fields optional on `EventEnvelope` and preserve legacy normalization.
- Keep existing 1 through 13 case semantics and numeric IDs.
- Keep Case 0 as non-canonical fallback.
- Preserve existing CLI commands and default-disabled experimental behavior.
- Add package-data rules so installed wheels can load V5 YAML registries.

## 7. Error handling and safety

- Invalid manifests fail at load time with the exact offending field or duplicate identifier.
- Unknown stripes, case IDs, verifier types, learning destinations, and lifecycle transitions fail
  closed with actionable `ValueError` messages.
- A missing verification contract produces zero process reward, not inferred verification.
- Evidence outside an authorized request boundary is rejected.
- Invalid or unsupported representation candidates remain hypotheses and cannot overwrite raw text.
- Candidate generation enforces bounded work before semantic evaluation.
- Missing authorization for irreversible execution routes to escalation.
- Retry and replan policies have explicit maximum counts and return an exhausted state.
- Experimental overlays are disabled unless the caller opts in.

## 8. Test strategy

Implementation will use test-first development for each behavioral change.

### 8.1 Canonical framework tests

- manifest count equals 17;
- IDs equal 1 through 17 and keys are unique;
- manifest names equal both Schema V3 and clarifying-abstention compatibility maps;
- documented table names equal the manifest;
- no Case 18 or later is canonical;
- stripe registry contains exactly the approved V5 top-level stripes;
- wheel/package-resource loading succeeds.

### 8.2 Reward tests

- changing only trace wording, order, self-description, or uncertainty vocabulary leaves fitness
  unchanged;
- deception and mechanistic diagnostics leave fitness unchanged;
- an unverified `thought_align=True` yields no process bonus;
- a verified contradiction, calibrated update, prediction result, or justified abstention can earn
  a bounded positive bonus;
- every nonzero process component requires observable evidence or a trusted evaluator contract;
- provenance round trips without expanding the authorized evidence boundary;
- final-answer equality does not prevent verified process sensitivity.

### 8.3 Event and evaluation tests

- new event types round trip while legacy events still normalize;
- prediction commits are immutable and precede linked execution;
- model and harness versions are required and stable within an episode;
- V5 records reject invalid cases, stripes, repeats, confidence, and provenance;
- cell enumeration is deterministic;
- repeat count is configurable and defaults to five;
- metric fixtures independently verify `Pass@k`, `Mean@k`, `Pass^k`, and consistency gap.

### 8.4 Representation tests

- literal input is unchanged through every view;
- Unicode and zero-width normalization retain source provenance;
- keyboard errors, swaps, repeated or deleted characters, and OCR-like variants recover known
  candidates in top-k;
- homophones, ASR-like substitutions, and phrase candidates recover known hypotheses in top-k;
- `therapist` and `the rapist` remain meaning-changing negative controls;
- numbers and negation remain semantic hinges;
- proper nouns and unusual clean words resist false repair;
- no-repair and unknown outcomes remain explicit;
- representation disagreement routes to uncertainty rather than silent replacement;
- multi-turn and memory provenance retain the raw source span;
- compute budgets cap candidates and record elapsed time.

### 8.5 Governance and documentation tests

- world-state claims do not establish observed world state;
- local and relational verifiers stay separate;
- unsupported causal failure edges are rejected or labeled hypothesized;
- authority and irreversible-action checks fail closed;
- recovery budgets exhaust deterministically;
- learning proposals have exactly one primary destination;
- skill credit requires execution evidence and held-out validation;
- internal Markdown links resolve;
- canonical README facts match the manifest and stripe registry;
- recommendation and research reference IDs resolve.

## 9. Verification plan

Before completion, run:

```text
python -m pytest <targeted V5 and modified-subsystem tests> -q
python -m pytest -q
python -m black --check --line-length 100 <modified Python paths>
python -m ruff check <modified Python paths>
python -m mypy evaluation gepa_mindfulness src/mindful_trace_gepa modules/semantic_intent_robustness
python -m build
```

Also run repository scripts or tests for synthetic serialization, package installation, README
links, semantic robustness, memory safety, and reward invariants. Record skipped native CUDA,
Vulkan, llama.cpp, Mojo, model, or GPU checks as unavailable rather than passed.

Performance verification will record representation candidate counts and wall-clock overhead on a
fixed deterministic fixture set. No performance claim will be made without measured results.

## 10. Delivery stages

1. PR-0: verified epistemic reward grounding.
2. PR-1: V5 manifest, recommendation registry, and research traceability.
3. PR-2: action-bound event and provenance schema.
4. PR-3: case by stripe by repeat evaluator.
5. PR-4: representation-robust semantic laundering.
6. PR-5: state separation, verifiers, failure graph, authority, and recovery.
7. PR-6: learning-surface classifier and verified skill lifecycle.
8. PR-7: disabled experimental overlay declarations.
9. PR-8: README and documentation consolidation.

Each stage will be independently reviewable and testable. Production changes will not precede the
failing test that defines their behavior. Local commits will preserve stage boundaries. The
implementation plan will split these stages into separately executable plans when a stage contains
more than one independently rejectable subsystem.

## 11. Acceptance criteria

The change is accepted only when:

- exactly 17 canonical cases exist and all consistency tests agree;
- Case 14 through Case 17 use the frozen V5 meanings;
- no generated reasoning wording can directly change optimizer fitness;
- verified epistemic process can still receive positive reward;
- every nonzero process reward has recorded provenance and verification;
- deception and mechanistic signals remain diagnostic by default;
- V5 event and evaluation records serialize and validate;
- case by stripe by repeat enumeration and metrics pass deterministic tests;
- representation candidates preserve raw input and provenance;
- meaning-preserving and meaning-changing representation tests both pass;
- world and evidence state are distinct types;
- local and relational verifier contracts are distinct;
- failure relations do not overclaim causality;
- recommendation and research references are internally resolvable;
- README reflects implemented reality and retains operational caveats;
- targeted tests, full tests, formatting, linting, type checking, and packaging either pass or have
  exact, non-overstated environment limitations recorded.
