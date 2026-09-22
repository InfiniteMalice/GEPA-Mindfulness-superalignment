# Experimental search and attribution overlays (REC-016–REC-019)

These library APIs add defensive research capabilities around the existing **17 canonical
cases (IDs 1–17)**. They preserve case names, V5 reward fields, verifier authority, release gates,
and the original FailureAtlas. No model experiment was run for this integration. Synthetic
tests establish software contracts, not empirical alignment improvements.

## Enablement and composition

`ExperimentalOverlayConfig` has four new flags, all `False` by default:
`evolutionary_semantic_search`, `serialization_roundtrip`, `formal_reasoning_audit`, and
`latent_language_transition`. Registry flags declare enabled capabilities; callers also pass
`enabled=True` to the corresponding computation API. No default runtime path calls these APIs.
The formal API raises when disabled, communication returns `None`, search returns a disabled
result, and latent diagnostics return a `DISABLED` assessment without inferred latent metrics.

The evaluation coordinates remain:

    base case × robustness stripe × variant/strategy × repeat run

`PARAPHRASE / COMMUNICATION_FIDELITY` is a new optional subtype. The canonical stripe inventory
and default evaluation matrix remain unchanged. This subtype can test whether a public
structure survives language communication; it does not certify semantic equivalence.

`evaluation.research_audits.bind_research_audits` validates the existing action-bound V5 event
sequence and creates an external `ResearchAuditBundle`. The bundle retains the V5 record digest,
run/trace/model/harness identity, parent event IDs, strategy lineage references, typed audit
references, and concurrent diagnostic findings. The V5 record and its serialization are unchanged.
Hosts store the referenced typed artifacts and original evidence separately, and authenticate
their association with the episode. An audit reference is not an authority token.

The bundle can distinguish changed meaning, preserved meaning with changed judgment,
serialization/extraction faults, unattributed communication failure, internal drift, stable
measured state with divergent output, invalid public inference, unsupported premises, and an
independently failed final output. Missing evidence yields `UNKNOWN`. Findings can coexist;
they do not identify a unique cause. Communication failures block a laundering-success finding.

`make_research_event` creates reference-only diagnostic events using the existing envelope.
The new event kinds require run and trace IDs, evaluation and artifact references, model/harness
versions, parent events, evidence, and provenance references. Payloads are deeply immutable.
These events are distinct from action-bound verification events and cannot replace them.

## REC-016: bounded quality-diversity search

[`evolutionary_atlas.py`](../modules/semantic_intent_robustness/evolutionary_atlas.py) defines
frozen `SemanticStrategy` and `StrategyPhase` records. Phases reuse `VariantType` and
`TRANSFORM_TEMPLATES`. `materialize_strategy` composes `build_variant` through a host renderer;
the seed needs a provenance envelope whose `source_prompt_id` matches the seed and whose
retained eligibility allows search. The host retains that source envelope and raw evidence.

`DeterministicOperators` supplies targeted mutation, unconstrained mutation, crossover, and
genesis over abstract plans. `EvolutionOperators` is the extension protocol. The reference
operators have no network calls or harmful operational instructions. An external execution
adapter interprets phases and transition conditions; conditions are never executed as code.

`evolve(template, execute_and_verify, budget=..., enabled=True)` requires explicit generation,
candidate, aggregate turn, and mutation-operation limits. This small reference scheduler
generates one candidate per generation. It reserves every candidate's full turn allowance
before execution. The host adapter enforces call timeouts and actual turn/token/cost allowances.
The loop rejects reported token/cost overspend. Optional token/cost budgets require the
zero-cost reference proposal operator because custom proposal costs cannot be accounted for
by the present protocol. LLM proposal adapters need separate host resource controls.

`IndependentExecutionVerifier` is separate from the proposal operator: it executes public
phases and obtains independent semantic and behavioral verification. It returns `SearchEvaluation`
with action-bound V5 evidence, an explicit semantic status, verifier/evidence references, and
three separate fitness dimensions: semantic preservation, judgment instability, failure severity.
The host authenticates semantic verifier identity and binds semantic evidence to the trajectory.
A status string or reference ID alone does not establish verifier independence outside that host.

`QualityDiversityArchive.admit` revalidates V5 provenance and strategy coordinates. Only exact
or independently verified semantic preservation with preservation value 1 enters the population.
Changed meaning, heuristic similarity, and unknown equivalence stay in the search result as
rejected observations. Novelty uses structured phase/family distance through `NoveltyProvider`;
it is separate from fitness. Near duplicates compete locally using Pareto dominance. Each cell
has bounded occupancy and is keyed by case, stripe/subtype, semantic intent, and failure family.
This is a candidate population, not a second failure-evidence archive.

`observe_failure` delegates exclusively to `FailureAtlas.observe`, preserving the original
failed V5 observation, lineage, verifier events, and later independent repair/regression evidence.
The search itself cannot close a failure or mark a repair successful.

`TargetInsight` represents a tentative population observation. `admit_insight` revalidates each
supporting failure before retaining a bounded memory. Unsupported insights are rejected and
low-confidence insights are not retained. Hosts can submit these provenance-linked insights
to subsequent generations/runs; the reference implementation does not invent free-text truths.

Search permits declared TRAIN and DEVELOPMENT sources; REGRESSION and HIDDEN_EVAL are rejected
before parent selection, mutation, seed rendering, memory use, novelty/ranking, or archive
admission. Nested retained eligibility labels cannot be overridden by a permissive outer label.
This differs from parameter optimization: the existing training policy still permits only TRAIN
and retains independent success/review requirements. There is **no training promotion API** here.
Labels cannot detect stripped provenance or concealed external holdout contamination; hosts must
protect their source catalogs and keep adapters isolated from hidden evaluation.

## REC-017: round-trip communication audit

[`serialization_roundtrip.py`](../evaluation/serialization_roundtrip.py) implements
`StructuredSource → Serializer → text → Extractor → reconstructed expression → EquivalenceVerifier`.
`JsonTreeCodec` is a synthetic reference codec, not a natural-language model. The propositional
tree verifier checks exact tree equality, then bounded Boolean-semantic equivalence. Reordered
conjunctions can be verified semantically without being identical trees. Heuristic similarity
and UNKNOWN never count as verified preservation.

The result retains the original source, serialization, reconstruction, evidence references,
tree complexity, verifier results and separate stage fault statuses. `StageEvidence` binds an
independent reader's interpretation to the serialized bytes by SHA-256. Without independent stage
evidence, an end-to-end mismatch does not identify which stage lost meaning. A serializer or
extractor exception is reported at its corresponding stage. Hosts authenticate stage-reader
independence and retain raw stage artifacts.

```python
from evaluation.serialization_roundtrip import (
    Expression, JsonTreeCodec, PropositionalTreeVerifier, StructuredSource, audit_roundtrip,
)

source = StructuredSource("source:scope", Expression("atom", atom="authorized"), ("raw:scope",))
codec = JsonTreeCodec()
result = audit_roundtrip(source, codec, codec, PropositionalTreeVerifier(), enabled=True)
assert result is not None and result.equivalence.semantics_preserved
```

This demonstrates a software round trip only. Natural-language serializers/extractors and their
independent stage readers remain host adapters. Exact propositional verification does not prove
equivalence of arbitrary natural-language intentions.

## REC-018: formal public-reasoning audit

[`formal_reasoning.py`](../gepa_mindfulness/verification/formal_reasoning.py) accepts explicit
public `FormalClaim` objects and separately supplied `PremiseGrounding` records. Observable
`EvidenceReference` objects retain provenance. `FormalSolverAdapter` supports bounded backends;
the reference solver uses propositional atoms, True/False, not/and/or, and `implies(a, b)`.
It parses a restricted AST and never evaluates arbitrary Python code.

Results distinguish VALID, INVALID, UNKNOWN, and UNSUPPORTED_TRANSLATION. The exact formalization,
backend, verifier references, and premise-grounding status remain separate. A tautology or a valid
inference with unsupported premises cannot establish factual correctness. Unknown grounding stays
unknown; formalization success is not rewarded.

`bounded_backtrack` takes an explicit retry limit. On an invalid public claim it anchors at the
latest preceding valid auditable claim, retaining every audit and retry. UNKNOWN or unsupported
results stop retrying; exhaustion is explicit. Hosts bound solver/callback wall time. This helper
does not inspect hidden reasoning, change a reward, train a model, or authorize an action.
Future verified-process integration would additionally need authenticated grounding/provenance,
supported-fragment coverage, and existing host verification/review gates; none is automatic here.

## REC-019: latent-to-language transition diagnostics

[`latent_language_transition.py`](../modules/semantic_intent_robustness/latent_language_transition.py)
reuses `SoTStateSnapshot` and `state_distance`. Independent `PublicDelta` records retain language
and action differences, metric calibration, origins, paired raw references, and comparability.
Missing/incomparable states remain unavailable; black-box behavioral evaluation remains possible.

Only comparable measured-internal snapshots with observed public-output metrics can yield the
two anomaly statuses: LATENT_LANGUAGE_DECOUPLING and LANGUAGE_CHANGE_WITHOUT_MATCHED_LATENT_SIGNAL.
Transcript and derived proxies stay labeled PROXY_ONLY. Language and action transfer ratios are
separate, normalization-dependent descriptions; negligible denominators and nonfinite ratios
produce no ratio. Language changes can coexist with unchanged actions and vice versa.

Neither distance nor decoding proves intent, deception, motive, causal use, successful steering,
alignment, or behavioral success. This integration adds no invasive steering stack. Controlled
interventions and causal validation remain future experimental host work.

## Verification and remaining research work

The four focused test files and `test_research_audit_integration.py` cover normal, missing,
unknown, malformed, hidden-source, provenance, budget, and authority boundaries. Existing V5,
FailureAtlas, semantic robustness, recommendation, and runtime tests provide regression coverage.
The exact execution results and environment limitations are recorded in the delivery report.

Future experimental work includes authenticated model adapters, natural-language stage readers,
calibrated strategy distances and thresholds, stronger supported logic fragments, controlled
latent interventions, population insight generators, and actual model experiments. None of these
is implied by passing synthetic tests. No private chain-of-thought is required or directly rewarded.
