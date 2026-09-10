# Unified V5 Recommendations

`evaluation/recommendations/registry.yaml` is the canonical authored registry. This reader groups
the current recommendations by priority and maturity. A research reference ID names a future
traceability entry; Task 5 must verify that metadata from primary sources. Research can motivate or
support a repository design choice, but does not prove this architecture.

## P0

### Implemented

- [REC-001](../../evaluation/recommendations/registry.yaml) — Verified epistemic-process reward
  rule. Code: [`epistemic_process.py`](../../gepa_mindfulness/core/epistemic_process.py).
  Tests: [`test_epistemic_process_rewards.py`](../../tests/test_epistemic_process_rewards.py).
  Research: [`REF-HEART`](#research-reference-ids-pending-task-5).
- [REC-003](../../evaluation/recommendations/registry.yaml) — One canonical 17-case V5 manifest.
  Code: [`registry.py`](../../evaluation/cases/registry.py). Tests:
  [`test_v5_case_registry.py`](../../tests/test_v5_case_registry.py).
- [REC-004](../../evaluation/recommendations/registry.yaml) — Versioned recommendation/decision
  registry. Code: [`recommendations.py`](../../evaluation/recommendations.py). Tests:
  [`test_recommendation_registry.py`](../../tests/test_recommendation_registry.py).

### Accepted

- [REC-002](../../evaluation/recommendations/registry.yaml) — Action-bound epistemic commitments.
  Planned tests: `tests/test_logging_schema.py`, `tests/test_v5_records.py`. Research:
  [`REF-WMLLM`](#research-reference-ids-pending-task-5),
  [`REF-DWM`](#research-reference-ids-pending-task-5).
- [REC-005](../../evaluation/recommendations/registry.yaml) — Case × robustness stripe × repeat
  evaluation. Existing registry: [`robustness_stripes.yaml`](../../evaluation/cases/robustness_stripes.yaml).
  Planned tests: `tests/test_v5_runner.py`, `tests/test_v5_records.py`. Research:
  [`REF-CONSISTENCY`](#research-reference-ids-pending-task-5).

## P1 — Accepted

- [REC-006](../../evaluation/recommendations/registry.yaml) — World/artifact state != evidence/belief
  state. Planned test: `tests/test_verification_state.py`. Research:
  [`REF-EDGEMEM`](#research-reference-ids-pending-task-5).
- [REC-007](../../evaluation/recommendations/registry.yaml) — Structured failure graph. Planned
  test: `tests/test_failure_graph.py`. Research:
  [`REF-AGENTSCOPE`](#research-reference-ids-pending-task-5).
- [REC-008](../../evaluation/recommendations/registry.yaml) — Runtime Planner/Executor/Verifier
  authority separation. Planned test: `tests/test_runtime_governance.py`. Research:
  [`REF-HEART`](#research-reference-ids-pending-task-5).
- [REC-009](../../evaluation/recommendations/registry.yaml) — Verified skill lifecycle. Planned
  test: `tests/test_skill_lifecycle.py`. Research:
  [`REF-SEGOS`](#research-reference-ids-pending-task-5),
  [`REF-SKILLGLOW`](#research-reference-ids-pending-task-5),
  [`REF-REPOTOSKILL`](#research-reference-ids-pending-task-5).
- [REC-010](../../evaluation/recommendations/registry.yaml) — Online experience collection; offline
  harness/skill evolution. Planned tests: `tests/test_learning_surfaces.py`,
  `tests/test_skill_lifecycle.py`. Research:
  [`REF-COEVOLVE`](#research-reference-ids-pending-task-5),
  [`REF-HOH`](#research-reference-ids-pending-task-5).

## P2 — Experimental

- [REC-011](../../evaluation/recommendations/registry.yaml) — Multiple competing hypotheses +
  information-gain inquiry. Planned test: `tests/test_experimental_overlays.py`. Research:
  [`REF-PEARL`](#research-reference-ids-pending-task-5).
- [REC-012](../../evaluation/recommendations/registry.yaml) — Small adaptive multi-agent topology
  codebook. Planned test: `tests/test_experimental_overlays.py`. Research:
  [`REF-MASKILLS`](#research-reference-ids-pending-task-5).
- [REC-013](../../evaluation/recommendations/registry.yaml) — Declarative global/focus/local
  orchestration scope. Planned test: `tests/test_experimental_overlays.py`. Research:
  [`REF-AGENTSCOPE`](#research-reference-ids-pending-task-5).
- [REC-014](../../evaluation/recommendations/registry.yaml) — Mechanistic/circuit audit of actual
  model changes. Planned test: `tests/test_experimental_overlays.py`. Research:
  [`REF-SAE`](#research-reference-ids-pending-task-5).

## Research reference IDs pending Task 5

Task 5 will create verified metadata and source links for the stable IDs listed above. Until then,
these IDs identify intended traceability records only; they do not make a bibliographic or empirical
claim.
