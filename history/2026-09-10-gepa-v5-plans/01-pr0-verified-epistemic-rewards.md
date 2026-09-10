# PR-0 Verified Epistemic Rewards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` during
> implementation and `superpowers:verification-before-completion` before the stage commit.

**Goal:** Remove optimizer reward from reasoning style while preserving bounded positive reward for
independently verified epistemic process.

**Architecture:** Add one reusable reward-provenance contract and one epistemic-process assessment.
Legacy `thought_align`, `honesty`, and `r_thought` fields remain, but optimizer credit is derived
only from verified component records.

**Tech Stack:** Python dataclasses, enums, typed evidence references, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Define reward provenance and verified process records

**Files:**

- Create: `gepa_mindfulness/core/reward_provenance.py`
- Create: `gepa_mindfulness/core/epistemic_process.py`
- Modify: `gepa_mindfulness/core/__init__.py`
- Test: `tests/test_epistemic_process_rewards.py`

**Interfaces:**

```python
class VerificationRoute(str, Enum):
    OBSERVABLE_EVIDENCE = "observable_evidence"
    TRUSTED_EVALUATOR = "trusted_evaluator"


@dataclass(frozen=True)
class TrustedEvaluatorContract:
    evaluator_id: str
    evaluator_version: str
    contract_id: str


@dataclass(frozen=True)
class RewardProvenance:
    component_name: str
    verification_method: str
    route: VerificationRoute
    evidence_refs: tuple[EvidenceReference, ...] = ()
    evaluator: TrustedEvaluatorContract | None = None


class EpistemicProcessComponent(str, Enum):
    EVIDENCE_FIDELITY = "evidence_fidelity"
    PUBLIC_RATIONALE_FIDELITY = "public_rationale_fidelity"
    CALIBRATION = "calibration"
    CONTRADICTION_HANDLING = "contradiction_handling"
    CONSEQUENCE_PREDICTION = "consequence_prediction"
    JUSTIFIED_ABSTENTION = "justified_abstention"
    MISSING_EVIDENCE_DETECTION = "missing_evidence_detection"
    BELIEF_UPDATE = "belief_update"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class VerifiedProcessComponent:
    component: EpistemicProcessComponent
    score: float
    provenance: RewardProvenance


@dataclass(frozen=True)
class EpistemicProcessAssessment:
    verified_components: tuple[VerifiedProcessComponent, ...] = ()
    reasoning_grounded: bool | None = None

    def optimizer_score(self) -> float: ...
```

1. Write tests that reject empty IDs, non-observable evidence on the observable route, mixed route
   fields, missing evaluator contracts, duplicate components, non-finite scores, and scores outside
   `[0.0, 1.0]`.
2. Run the test file and verify failure because the modules do not exist.
3. Implement frozen records. Require at least one observable reference for the evidence route and
   a complete evaluator contract for the evaluator route. Require provenance component name to
   equal the verified process component value.
4. Implement `optimizer_score()` as zero for no components and the arithmetic mean for verified
   component scores. This preserves decomposition and avoids implicit weights.
5. Add a test proving `PUBLIC_RATIONALE_FIDELITY` requires evidence that compares the public
   rationale with the committed prediction, selected action, and observed outcome; rationale text
   alone is not valid provenance.
6. Export the new public types from `gepa_mindfulness.core`.
7. Run `python -m pytest tests/test_epistemic_process_rewards.py -q` and verify all tests pass.

## Task 2: Make the main reward calculator style-invariant and process-sensitive

**Files:**

- Modify: `gepa_mindfulness/core/rewards.py`
- Modify: `tests/test_rewards.py`
- Test: `tests/test_reward_style_invariance.py`

**Interface change:**

```python
def compute_reward(
    self,
    *,
    response: str,
    reference_answers: Sequence[str] | str | None,
    gepa_scores: Mapping[str, float] | None,
    imperatives: Mapping[str, Mapping[str, float]] | None,
    confidence: float,
    trace_summary: Mapping[str, str],
    abstention: AbstentionAssessment | None = None,
    epistemic_process: EpistemicProcessAssessment | None = None,
) -> RewardBreakdown:
```

1. Add a parameterized failing test that supplies different `trace_summary` wording and ordering
   while holding response, confidence, answer, abstention assessment, and verified process fixed.
   Assert identical `honesty`, `epistemic_process`, and `total` values.
2. Add a failing test where two identical final answers receive different totals because one has a
   verified contradiction-handling component with score `1.0` and valid provenance.
3. Add a failing test proving self-reported uncertainty without verification earns zero process
   reward.
4. Run the three tests and verify they fail against `_honesty_signal`.
5. Add `epistemic_process: float` to `RewardBreakdown`. Keep `honesty` as a compatibility field with
   the same numeric value and document the alias.
6. Replace `_honesty_signal` with a helper that returns
   `epistemic_process.optimizer_score()` or `0.0`. Do not inspect `trace_summary` in reward math.
7. Keep `trace_summary` in the method signature because logging callers still use it.
8. Update the old trace-summary test to assert diagnostic compatibility and zero style reward.
9. Run `python -m pytest tests/test_rewards.py tests/test_reward_style_invariance.py -q`.

## Task 3: Ground `H` and Schema V3 process components

**Files:**

- Modify: `gepa_mindfulness/core/abstention_rewards.py`
- Modify: `gepa_mindfulness/schema_v3/case_v3.py`
- Modify: `gepa_mindfulness/schema_v3/rewards.py`
- Modify: `reasoning-generalization-tracer/src/rg_tracer/schema_v3/case_v3.py`
- Modify: `reasoning-generalization-tracer/src/rg_tracer/schema_v3/rewards.py`
- Modify: `tests/test_abstention_rewards.py`
- Modify: `tests/test_schema_v3.py`

1. Write a failing abstention test showing `thought_align=True` without an
   `EpistemicProcessAssessment` preserves the case ID but sets the thought component to zero.
2. Write a failing companion test showing a verified evidence-fidelity component can produce `H`.
3. Write failing Schema V3 tests showing populated `observed_controls`, `observed_units`,
   observability flags, or group-theoretic fields earn no reward without matching verified process
   components.
4. Write positive Schema V3 tests for independently verified `grounding`, `control`,
   `reasoning_unit`, `observability`, and `group_theoretic` component names.
5. Run the focused tests and verify failures come from current presence-based reward logic.
6. Add optional `epistemic_process` parameters to `compute_abstention_reward()` and
   `classify_case_v3()`.
7. Award `H * optimizer_score()` only when `thought_align` is true and at least one verified
   component exists. Preserve case classification based on `thought_align` for compatibility.
8. Replace Schema V3 presence and coverage rewards with verified component lookup. Retain the
   original overlays unchanged in the serialized diagnostic record.
9. Apply the same behavior to the packaged `rg_tracer` compatibility copy.
10. Run the focused abstention and Schema V3 tests.

## Task 4: Remove response-word bonuses from the legacy orchestrator

**Files:**

- Modify: `gepa_mindfulness/training/pipeline.py`
- Modify: `tests/test_honesty_rewards.py`
- Modify: `tests/test_training_cli.py`

1. Replace `test_honesty_increases_reward` with a failing style-invariance test whose response text
   changes from plain prose to uncertainty vocabulary without changing verified inputs.
2. Add a failing process-sensitivity test that passes a verified assessment to `_compute_reward()`.
3. Run the tests and verify the uncertainty-marker bonus causes the invariance failure.
4. Change `_honesty_bonus()` to accept an `EpistemicProcessAssessment | None` and return only its
   verified score multiplied by the configured process weight. Keep old config keys as documented
   compatibility aliases.
5. Add `epistemic_process` to `_compute_reward()` and forward it to abstention scoring.
6. Preserve deception fingerprint logging and the existing equality test proving deception
   diagnostics do not affect reward.
7. Run `python -m pytest tests/test_honesty_rewards.py tests/test_training_cli.py -q`.

## Task 5: Require provenance for every nonzero reward-integrity component

**Files:**

- Modify: `gepa_mindfulness/core/reward_integrity.py`
- Modify: `gepa_mindfulness/training/reward_pipeline.py`
- Modify: `gepa_mindfulness/training/trajectory.py`
- Modify: `tests/test_reward_integrity_rewards.py`
- Modify: `tests/test_rl_trajectory.py`

1. Add failing tests for positive nonzero components with no provenance, observable provenance,
   and trusted-evaluator provenance.
2. Add a failing JSON round-trip test for a trusted evaluator contract stored with a trajectory.
3. Run the tests and verify current code accepts unsupported positive values.
4. Add `reward_component_provenance: Mapping[str, RewardProvenance]` to `Trajectory`, preserving
   the existing evidence mapping during migration.
5. Require one valid provenance record for every nonzero `RewardObservation` component. For the
   observable route, preserve the existing authorized-reference subset checks. For the evaluator
   route, require exact evaluator identifiers and versions.
6. Serialize and restore provenance without inventing absent values.
7. Keep zero components valid without provenance.
8. Run reward-integrity, trajectory, pipeline, and serialization tests.

## Task 6: Document the reward contract and commit PR-0

**Files:**

- Modify: `docs/thought_alignment.md`
- Modify: `gepa_mindfulness/core/README.md`
- Modify: `gepa_mindfulness/schema_v3/README.md`
- Create: `docs/epistemic_process_rewards.md`

1. Document the Epistemic Process Reward Rule, compatibility aliases, allowed provenance routes,
   optimizer eligibility, default/max weights, and known Goodhart risks.
2. Add a component table covering `r_thought`, `r_grounding`, `r_control`, `r_reasoning_unit`,
   `r_observability`, and `r_group_theoretic`.
3. Run targeted tests from Tasks 1 through 5.
4. Run `python -m black --check --line-length 100` on modified Python files.
5. Run `python -m ruff check` on modified Python files.
6. Run `python -m mypy gepa_mindfulness/core gepa_mindfulness/schema_v3`.
7. Run `git diff --check` and inspect the complete stage diff.
8. Commit with message `feat: ground epistemic process rewards`.
