# PR-6 Learning Surfaces and Skill Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`. This
> stage defines controlled records and validators; it does not modify model weights or live skills.

**Goal:** Route lessons to one learning destination and require execution evidence plus held-out
validation before a skill or harness change is committed.

**Architecture:** Two small modules define immutable learning proposals, evaluation epochs, skill
lifecycle states, and transition validation. They consume the evidence and V5 evaluation records
from earlier stages.

**Tech Stack:** Python dataclasses, enums, state-transition maps, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Add the learning-surface classifier

**Files:**

- Create: `gepa_mindfulness/learning_surfaces.py`
- Test: `tests/test_learning_surfaces.py`

**Interfaces:**

```python
class LearningSurface(str, Enum):
    TRACE_ONLY = "trace_only"
    MEMORY = "memory"
    HARNESS = "harness"
    SKILL_GRAPH = "skill_graph"
    MODEL = "model"
    HUMAN = "human"


@dataclass(frozen=True)
class LessonProposal:
    lesson_id: str
    summary: str
    primary_destination: LearningSurface
    evidence_refs: tuple[EvidenceReference, ...]
    rationale: str
    reversible: bool
    review_status: str
```

1. Write failing tests for one-off observations, episode facts, stable procedural conventions,
   reusable dependencies, persistent intrinsic behavior, and normative or hard-to-reverse changes.
2. Add tests rejecting multiple destinations, missing evidence, blank rationale, and automatic
   model routing for a one-off observation.
3. Run tests and verify module-not-found failure.
4. Implement `classify_learning_surface()` as an explicit decision table over typed
   `LessonCharacteristics`; do not classify from prose keywords.
5. Require `HUMAN` for normative, ambiguous, or difficult-to-reverse changes.
6. Run `python -m pytest tests/test_learning_surfaces.py -q`.

## Task 2: Freeze model and harness versions within evaluation epochs

**Files:**

- Modify: `gepa_mindfulness/learning_surfaces.py`
- Test: `tests/test_offline_evolution_epochs.py`

**Interface:**

```python
@dataclass(frozen=True)
class EvaluationEpoch:
    epoch_id: str
    model_version: str
    harness_version: str
    record_ids: tuple[str, ...]
    closed: bool = False


def validate_epoch_record(epoch: EvaluationEpoch, record: V5EvaluationRecord) -> None: ...
```

1. Write failing tests for model drift, harness drift, duplicate record IDs, and attempted harness
   mutation before the source epoch closes.
2. Add a passing test where online operation appends evidence but leaves model and harness versions
   unchanged.
3. Implement exact version checks. A candidate model or harness must have a new version and a new
   evaluation epoch.
4. Run epoch tests plus V5 record tests.

## Task 3: Implement the verified skill lifecycle

**Files:**

- Create: `gepa_mindfulness/skill_lifecycle.py`
- Test: `tests/test_verified_skill_lifecycle.py`

**Interfaces:**

```python
class SkillLifecycleState(str, Enum):
    SOURCE_EXPERIENCE = "source_experience"
    VERIFIED_SKILL = "verified_skill"
    PROCEDURAL_FAMILY = "procedural_family"
    TASK_LOCAL = "task_local"
    EXECUTED = "executed"
    CREDITED = "credited"
    REFINED = "refined"
    HELD_OUT_VALIDATED = "held_out_validated"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class SkillArtifact:
    skill_id: str
    version: str
    state: SkillLifecycleState
    source_refs: tuple[EvidenceReference, ...]
    execution_event_ids: tuple[str, ...] = ()
    validation_record_ids: tuple[str, ...] = ()
    supersedes: str | None = None
```

1. Write failing tests for the complete valid lifecycle and every forbidden skip into `CREDITED`,
   `HELD_OUT_VALIDATED`, or `COMMITTED`.
2. Add a failing test proving a generated explanation that claims success cannot create skill
   credit without action, outcome, and verification event IDs.
3. Add tests for refinement, consolidation, pruning, commit, and rollback provenance.
4. Run tests and verify module-not-found failure.
5. Implement a single `transition_skill()` function with an explicit allowed-transition map.
6. Require execution evidence before `CREDITED`, held-out V5 records before
   `HELD_OUT_VALIDATED`, and a rollback target before `COMMITTED`.
7. Keep rollback available from every state after `VERIFIED_SKILL`.
8. Run skill lifecycle tests.

## Task 4: Model controlled model-harness coevolution

**Files:**

- Create: `gepa_mindfulness/coevolution.py`
- Test: `tests/test_model_harness_coevolution.py`

1. Write a failing fixture for current trajectory, localized failure, teacher correction, candidate
   version, held-out results, protected V5 regression results, accept decision, and rollback.
2. Add tests rejecting wholesale trajectory imitation when no localized failure state is named,
   candidate acceptance without protected regressions, version reuse, and harness changes inside
   the source episode.
3. Implement `CorrectionProposal`, `CandidateSystem`, `ValidationBundle`, and
   `decide_candidate_acceptance()`.
4. Acceptance requires all protected regressions to pass and a non-worse declared primary metric.
   Preserve component metrics rather than a single opaque score.
5. Run coevolution, epoch, failure graph, and V5 evaluator tests.

## Task 5: Document, verify, and commit PR-6

**Files:**

- Create: `docs/controlled_evolution.md`
- Modify: `docs/recommendations/UNIFIED_RECOMMENDATIONS.md`
- Modify: `docs/recommendations/registry.yaml`

1. Document online evidence collection, offline evolution, destination rules, skill credit,
   protected regressions, commit, and rollback.
2. Update REC-009 and REC-010 implementation and acceptance references.
3. Run learning, epoch, skill lifecycle, coevolution, V5, and verification tests.
4. Run Black, Ruff, mypy, and `git diff --check`.
5. Inspect the stage diff and commit with message
   `feat: add controlled learning and skill lifecycle`.
