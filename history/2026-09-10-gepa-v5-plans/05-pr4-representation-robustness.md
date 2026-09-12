# PR-4 Representation Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`. Extend
> `semantic_intent_robustness`; do not create a competing semantic-safety package.

**Goal:** Add provenance-preserving, bounded alternate representations before semantic-intent
analysis and reconcile the legacy semantic-laundering evaluator.

**Architecture:** Immutable source documents produce scored candidate lattices through explicit
channels. Semantic-hinge routing limits expensive generation. Existing semantic records evaluate
the selected hypotheses, and disagreement produces uncertainty rather than silent repair.

**Tech Stack:** Python dataclasses, enums, `unicodedata`, `difflib`, regex, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Define immutable sources, views, and candidate lattices

**Files:**

- Create: `modules/semantic_intent_robustness/representation.py`
- Modify: `modules/semantic_intent_robustness/__init__.py`
- Test: `modules/semantic_intent_robustness/tests/test_representation.py`

**Interfaces:**

```python
class RepresentationChannel(str, Enum):
    LITERAL = "literal"
    CONSERVATIVE_NORMALIZATION = "conservative_normalization"
    ORTHOGRAPHIC = "orthographic"
    PHONOLOGICAL = "phonological"
    CONTEXTUAL = "contextual"


class CandidateOutcome(str, Enum):
    CANDIDATE = "candidate"
    NO_REPAIR = "no_repair"
    UNKNOWN = "unknown"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    start: int
    end: int
    raw_text: str


@dataclass(frozen=True)
class RepresentationCandidate:
    source_span: SourceSpan
    candidate_text: str
    transform_channel: RepresentationChannel
    orthographic_score: float
    phonetic_score: float
    contextual_score: float
    semantic_similarity: float
    confidence: float
    provenance: tuple[str, ...]
    generation_reason: str
    outcome: CandidateOutcome = CandidateOutcome.CANDIDATE


@dataclass(frozen=True)
class RepresentationLattice:
    source_id: str
    raw_text: str
    candidates: tuple[RepresentationCandidate, ...]
    max_candidates: int
```

1. Write failing tests for exact source slicing, immutable raw text, score bounds, non-empty
   provenance and generation reason, invalid spans, and maximum candidate enforcement.
2. Add round-trip tests that prove derived candidate text never replaces `raw_text`.
3. Run tests and verify module-not-found failure.
4. Implement frozen records with strict validation and deterministic candidate ordering by
   confidence, channel, span, and candidate text.
5. Export the types and run the focused tests.

## Task 2: Add literal and conservative normalization views

**Files:**

- Create: `modules/semantic_intent_robustness/representation_views.py`
- Test: `modules/semantic_intent_robustness/tests/test_representation_views.py`

1. Add failing tests for literal identity, NFC Unicode normalization, known zero-width removal,
   newline normalization, and provenance retention.
2. Add negative tests proving ordinary spaces, tabs that delimit fields, token boundaries, numbers,
   punctuation, and negation are not collapsed as nonsemantic noise.
3. Run tests and verify failure because the view builder is absent.
4. Implement `literal_view(source_id, text)` and `conservative_views(source_id, text)` using only
   deterministic normalization. Every changed view records the exact transform in provenance.
5. Run the focused tests.

## Task 3: Add bounded orthographic and phonological candidates

**Files:**

- Create: `modules/semantic_intent_robustness/representation_candidates.py`
- Test: `modules/semantic_intent_robustness/tests/test_representation_candidates.py`

**Interface:**

```python
@dataclass(frozen=True)
class CandidateBudget:
    max_spans: int = 8
    max_candidates_per_span: int = 4
    max_candidates_total: int = 24


def build_candidate_lattice(
    source_id: str,
    raw_text: str,
    *,
    context: tuple[str, ...] = (),
    orthographic_lexicon: tuple[str, ...] = (),
    phonetic_lexicon: Mapping[str, tuple[str, ...]] | None = None,
    budget: CandidateBudget = CandidateBudget(),
) -> RepresentationLattice: ...
```

1. Add literal top-k fixtures for adjacent-key errors, swapped characters, repeated/deleted
   characters, OCR-like substitutions, homophones, ASR-like substitutions, and phonetic spelling.
2. Add the phrase fixture `bone apple tea -> bon appétit` and assert only that the intended
   candidate appears in top-k.
3. Add the literal-context control `Put the bone, apple, and tea on the table.` and assert the
   source reading remains highest confidence and no automatic replacement occurs.
4. Add false-repair controls for legitimate proper nouns, unusual clean words, numbers, negation,
   `therapist`, and `the rapist`.
5. Add budget tests proving the generator never exceeds any cap.
6. Run tests and verify module-not-found failure.
7. Implement dependency-free candidate generation using bounded edit-distance comparison and an
   injected phrase lexicon. Treat scores as candidate evidence, not intended-meaning proof.
8. Return explicit `NO_REPAIR` when no candidate exceeds the configured evidence floor.
9. Run the candidate tests.

## Task 4: Route semantic hinges and representation disagreement

**Files:**

- Create: `modules/semantic_intent_robustness/representation_routing.py`
- Modify: `modules/semantic_intent_robustness/modules.py`
- Modify: `modules/semantic_intent_robustness/schemas.py`
- Test: `modules/semantic_intent_robustness/tests/test_representation_routing.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RepresentationDecision:
    selected_candidate_ids: tuple[str, ...]
    disagreement: bool
    policy_action: PolicyAction
    explanation: str


def locate_semantic_hinges(text: str) -> tuple[SourceSpan, ...]: ...


def route_representation_disagreement(
    assessments: Sequence[SemanticSafetyRecord],
    *,
    high_stakes: bool,
) -> RepresentationDecision: ...
```

1. Add failing tests that locate action verbs, targets, negation, numbers, names, authorization
   terms, capability terms, and constraint terms without labeling the spans harmful.
2. Add failing tests for literal/phonetic policy agreement, material disagreement at low stakes,
   and material disagreement at high stakes.
3. Require disagreement to route to clarification, bounded caution, or abstention; forbid a result
   that marks the alternate candidate as verified truth.
4. Run tests and verify failure against the absent routing layer.
5. Add optional representation provenance and disagreement fields to `SemanticSafetyRecord` with
   backward-compatible defaults.
6. Add a `RepresentationRobustnessModule` before semantic decomposition in the existing pipeline.
7. Run routing, schema, serialization, transform, and pipeline tests.

## Task 5: Add metrics and reconcile the legacy evaluator

**Files:**

- Create: `modules/semantic_intent_robustness/representation_metrics.py`
- Modify: `modules/semantic_intent_robustness/evaluators.py`
- Modify: `evals/semantic_laundering_eval.py`
- Modify: `tests/test_reality_contact_scaffolding.py`
- Test: `modules/semantic_intent_robustness/tests/test_representation_metrics.py`

1. Write failing literal-fixture tests for candidate recall at k, false-repair rate, abstention
   precision, abstention coverage, disagreement rate, clean regression, laundering detection, mean
   candidates, and elapsed milliseconds.
2. Implement a frozen `RepresentationMetricSummary` and pure aggregation helpers.
3. Add `evaluate_representation_cases()` to `SemanticRobustnessEvaluator`.
4. Convert `semantic_laundering_risk()` and `intent_tracking_score()` into compatibility wrappers
   over the richer module's typed assessment. Preserve their public signatures and integer ranges.
5. Add a regression test proving the wrappers and rich evaluator agree on equivalent fixtures.
6. Run semantic module and reality-contact tests.

## Task 6: Document, measure, verify, and commit PR-4

**Files:**

- Modify: `modules/semantic_intent_robustness/README.md`
- Create: `docs/FOUNDATIONAL_REPRESENTATION_ARCHITECTURE.md`

1. Document raw-input immutability, channels, candidate lattice, top-k semantics, semantic hinges,
   disagreement, memory provenance, and limitations.
2. Link representation design to the traceability REF-IDs without overstating source claims.
3. Write the foundational-model note described by the specification; do not add tokenizer code.
4. Run all semantic-intent, memory-mediated laundering, KV-context, release-gate, and representation
   tests.
5. Run a deterministic benchmark fixture at least five times. Record fixture count, candidate count,
   median wall time, and maximum wall time in the stage report.
6. Run Black, Ruff, mypy, and `git diff --check` on changed paths.
7. Inspect the stage diff and commit with message
   `feat: add provenance-bound representation robustness`.
