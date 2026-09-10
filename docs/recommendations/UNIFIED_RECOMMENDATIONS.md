# Unified V5 Recommendations

[`registry.yaml`](registry.yaml) is the single authored recommendation registry. This reader
summarizes current maturity and links each record to repository evidence. Stable research IDs are
pending Task 5 primary-source metadata verification; they can motivate or support a design choice,
but do not prove this architecture.

## P0

### REC-001 — Verified epistemic-process reward rule.

**Status:** Implemented.

- Repository evidence: [`epistemic_process.py`](../../gepa_mindfulness/core/epistemic_process.py), [`rewards.py`](../../gepa_mindfulness/core/rewards.py).
- Acceptance tests: [`test_epistemic_process_rewards.py`](../../tests/test_epistemic_process_rewards.py), [`test_reward_integrity_rewards.py`](../../tests/test_reward_integrity_rewards.py).
- Research: [`REF-HEART`](#research-reference-ids-pending-task-5).

### REC-002 — Action-bound epistemic commitments.

**Status:** Accepted.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py).
- Planned acceptance checks: `tests/test_logging_schema.py`, `tests/test_v5_records.py`.
- Research: [`REF-WMLLM`](#research-reference-ids-pending-task-5), [`REF-DWM`](#research-reference-ids-pending-task-5).

### REC-003 — One canonical 17-case V5 manifest.

**Status:** Implemented.

- Repository evidence: [`17_case_manifest.yaml`](../../evaluation/cases/17_case_manifest.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Acceptance tests: [`test_v5_case_registry.py`](../../tests/test_v5_case_registry.py), [`test_v5_documentation_consistency.py`](../../tests/test_v5_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

### REC-004 — Versioned recommendation/decision registry.

**Status:** Implemented.

- Repository evidence: [`recommendations.py`](../../evaluation/recommendations.py), [`registry.yaml`](registry.yaml).
- Acceptance tests: [`test_recommendation_registry.py`](../../tests/test_recommendation_registry.py), [`test_recommendation_documentation_consistency.py`](../../tests/test_recommendation_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

### REC-005 — Case × robustness stripe × repeat evaluation.

**Status:** Accepted.

- Repository evidence: [`robustness_stripes.yaml`](../../evaluation/cases/robustness_stripes.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Planned acceptance checks: `tests/test_v5_runner.py`, `tests/test_v5_records.py`.
- Research: [`REF-CONSISTENCY`](#research-reference-ids-pending-task-5).

## P1

### REC-006 — World/artifact state != evidence/belief state.

**Status:** Accepted.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py).
- Planned acceptance checks: `tests/test_verification_state.py`.
- Research: [`REF-EDGEMEM`](#research-reference-ids-pending-task-5).

### REC-007 — Structured failure graph.

**Status:** Accepted.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py).
- Planned acceptance checks: `tests/test_failure_graph.py`.
- Research: [`REF-AGENTSCOPE`](#research-reference-ids-pending-task-5).

### REC-008 — Runtime Planner/Executor/Verifier authority separation.

**Status:** Accepted.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py).
- Planned acceptance checks: `tests/test_runtime_governance.py`.
- Research: [`REF-HEART`](#research-reference-ids-pending-task-5).

### REC-009 — Verified skill lifecycle.

**Status:** Accepted.

- Repository evidence: [`reward_pipeline.py`](../../gepa_mindfulness/training/reward_pipeline.py).
- Planned acceptance checks: `tests/test_skill_lifecycle.py`.
- Research: [`REF-SEGOS`](#research-reference-ids-pending-task-5), [`REF-SKILLGLOW`](#research-reference-ids-pending-task-5), [`REF-REPOTOSKILL`](#research-reference-ids-pending-task-5).

### REC-010 — Online experience collection; offline harness/skill evolution.

**Status:** Accepted.

- Repository evidence: [`reward_pipeline.py`](../../gepa_mindfulness/training/reward_pipeline.py).
- Planned acceptance checks: `tests/test_learning_surfaces.py`, `tests/test_skill_lifecycle.py`.
- Research: [`REF-COEVOLVE`](#research-reference-ids-pending-task-5), [`REF-HOH`](#research-reference-ids-pending-task-5).

## P2

### REC-011 — Multiple competing hypotheses + information-gain inquiry.

**Status:** Experimental.

- Repository evidence: [`robustness_stripes.yaml`](../../evaluation/cases/robustness_stripes.yaml).
- Planned acceptance checks: `tests/test_experimental_overlays.py`.
- Research: [`REF-PEARL`](#research-reference-ids-pending-task-5).

### REC-012 — Small adaptive multi-agent topology codebook.

**Status:** Experimental.

- Repository evidence: [`reward_pipeline.py`](../../gepa_mindfulness/training/reward_pipeline.py).
- Planned acceptance checks: `tests/test_experimental_overlays.py`.
- Research: [`REF-MASKILLS`](#research-reference-ids-pending-task-5).

### REC-013 — Declarative global/focus/local orchestration scope.

**Status:** Experimental.

- Repository evidence: [`reward_pipeline.py`](../../gepa_mindfulness/training/reward_pipeline.py).
- Planned acceptance checks: `tests/test_experimental_overlays.py`.
- Research: [`REF-AGENTSCOPE`](#research-reference-ids-pending-task-5).

### REC-014 — Mechanistic/circuit audit of actual model changes.

**Status:** Experimental.

- Repository evidence: [`rewards.py`](../../gepa_mindfulness/core/rewards.py).
- Planned acceptance checks: `tests/test_experimental_overlays.py`.
- Research: [`REF-SAE`](#research-reference-ids-pending-task-5).

## Research reference IDs pending Task 5

Task 5 will retrieve primary metadata and source links for the stable IDs listed above. Until then,
the IDs identify intended traceability records only and make no bibliographic or empirical claim.
