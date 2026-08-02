# Phase 1 Trace-Reference Compatibility Remediation Plan

**Goal:** Restore the pre-overlay `Trajectory.trace_references` string API and JSON format while
keeping typed observable evidence as the only authority for reward-integrity scoring.

## Global Constraints

- Use strict TDD and record the failing and passing commands.
- Preserve legacy construction with `trace_references=("trace-1",)`.
- Preserve legacy JSON with `"trace_references": ["trace-1"]` and byte-compatible field shape.
- Legacy string trace IDs are diagnostic provenance only and never authorize negative rewards.
- Typed `EvidenceReference` values remain enum-classified and only observable kinds may authorize
  reward-integrity evidence.
- Preserve all Phase 1 safety, curriculum, packaging, adapter, and default-off behavior.
- Python source lines are at most 100 characters; do not run `bd` or modify `.beads/*`.

### Task 1: Separate legacy trace IDs from typed reward evidence

**Files:**
- Modify: `gepa_mindfulness/training/trajectory.py`
- Modify: `gepa_mindfulness/training/contracts.py`
- Modify: `gepa_mindfulness/training/reward_pipeline.py`
- Modify: `tests/test_rl_trajectory.py`
- Modify: `tests/test_reward_integrity_rewards.py`

**Interfaces:**
- `Trajectory.trace_references` remains `tuple[str, ...]` in construction and serialization.
- Add keyword-only `Trajectory.evidence_references: tuple[EvidenceReference, ...] = ()`.
- `reward_component_evidence` remains typed and must be a subset of
  `Trajectory.evidence_references`; legacy trace IDs do not satisfy this check.
- `RewardRequest.observable_references` must be a subset of
  `Trajectory.evidence_references`, never `trace_references`.

- [ ] Write regression tests that construct and round-trip a legacy string trace reference.
- [ ] Write a regression that restores the prior JSON representation containing string
  `trace_references` and no `evidence_references`.
- [ ] Write a regression proving a legacy trace ID cannot authorize a negative component.
- [ ] Write a regression proving typed observable evidence still authorizes the component and
  round-trips under the separate `evidence_references` JSON field.
- [ ] Run focused tests and observe the compatibility failures before implementation.
- [ ] Implement the split without coercing legacy strings into typed evidence.
- [ ] Run trajectory/reward tests, all Phase 1 tests, the full suite, and static checks.
- [ ] Commit as `fix: preserve legacy trace references`.
