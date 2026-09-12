# Unified V5 Recommendations

[`registry.yaml`](registry.yaml) is the single authored recommendation registry. This reader
summarizes current maturity and links each record to repository evidence. Stable research IDs are
documented in [`RESEARCH_TRACEABILITY.md`](RESEARCH_TRACEABILITY.md); they can motivate or support
a design choice, but do not establish this architecture as an empirical result.

## P0

### P0 — Implemented

#### REC-001 — Verified epistemic-process reward rule.

- Repository evidence: [`epistemic_process.py`](../../gepa_mindfulness/core/epistemic_process.py), [`rewards.py`](../../gepa_mindfulness/core/rewards.py).
- Acceptance tests: [`test_epistemic_process_rewards.py`](../../tests/test_epistemic_process_rewards.py), [`test_reward_integrity_rewards.py`](../../tests/test_reward_integrity_rewards.py).
- Research: [`REF-HEART`](RESEARCH_TRACEABILITY.md#ref-heart).

#### REC-003 — One canonical 17-case V5 manifest.

- Repository evidence: [`17_case_manifest.yaml`](../../evaluation/cases/17_case_manifest.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Acceptance tests: [`test_v5_case_registry.py`](../../tests/test_v5_case_registry.py), [`test_v5_documentation_consistency.py`](../../tests/test_v5_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

#### REC-004 — Versioned recommendation/decision registry.

- Repository evidence: [`recommendations.py`](../../evaluation/recommendations.py), [`registry.yaml`](registry.yaml).
- Acceptance tests: [`test_recommendation_registry.py`](../../tests/test_recommendation_registry.py), [`test_recommendation_documentation_consistency.py`](../../tests/test_recommendation_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

### P0 — Accepted

#### REC-002 — Action-bound epistemic commitments.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py).
- Planned acceptance checks: `tests/test_logging_schema.py`, `tests/test_v5_records.py`.
- Research: [`REF-WMLLM`](RESEARCH_TRACEABILITY.md#ref-wmllm), [`REF-DWM`](RESEARCH_TRACEABILITY.md#ref-dwm), [`REF-SHEAVES`](RESEARCH_TRACEABILITY.md#ref-sheaves), [`REF-HERO`](RESEARCH_TRACEABILITY.md#ref-hero).

#### REC-005 — Case × robustness stripe × repeat evaluation.

- Repository evidence: [`robustness_stripes.yaml`](../../evaluation/cases/robustness_stripes.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Planned acceptance checks: `tests/test_v5_runner.py`, `tests/test_v5_records.py`.
- Research: [`REF-CONSISTENCY`](RESEARCH_TRACEABILITY.md#ref-consistency), [`REF-LEXICAL-PERTURB`](RESEARCH_TRACEABILITY.md#ref-lexical-perturb), [`REF-TOKENIZER-BETRAYAL`](RESEARCH_TRACEABILITY.md#ref-tokenizer-betrayal).

## P1

### P1 — Implemented

#### REC-006 — World/artifact state != evidence/belief state.

- Repository evidence: [`state.py`](../../gepa_mindfulness/verification/state.py).
- Limit: evidence identifiers are not dereferenced or issuer-authenticated by this record layer.
- Acceptance checks: [`test_world_evidence_state.py`](../../tests/test_world_evidence_state.py).
- Research: [`REF-EDGEMEM`](RESEARCH_TRACEABILITY.md#ref-edgemem), [`REF-GRAPHMEM`](RESEARCH_TRACEABILITY.md#ref-graphmem).

#### REC-007 — Structured failure graph.

- Repository evidence: [`failure_graph.py`](../../gepa_mindfulness/verification/failure_graph.py).
- Limit: verifier identifiers are preserved but are not authenticated by the graph layer.
- Acceptance checks: [`test_failure_graph.py`](../../tests/test_failure_graph.py).
- Research: [`REF-AGENTSCOPE`](RESEARCH_TRACEABILITY.md#ref-agentscope).

#### REC-008 — Runtime Planner/Executor/Verifier authority separation.

- Repository evidence: [`interfaces.py`](../../gepa_mindfulness/verification/interfaces.py), [`runtime_governance.py`](../../gepa_mindfulness/verification/runtime_governance.py).
- Limit: authority is process-local and enrollment authenticates neither principals nor evidence
  issuers.
- Acceptance checks: [`test_runtime_authority.py`](../../tests/test_runtime_authority.py),
  [`test_verifier_interfaces.py`](../../tests/test_verifier_interfaces.py).
- Research: [`REF-HEART`](RESEARCH_TRACEABILITY.md#ref-heart), [`REF-BIOMETRIC-MEM`](RESEARCH_TRACEABILITY.md#ref-biometric-mem).

#### REC-009 — Verified skill lifecycle.

- Repository evidence: [`skill_lifecycle.py`](../../gepa_mindfulness/skill_lifecycle.py).
- Architecture boundary: [`controlled_evolution.md`](../controlled_evolution.md).
- Limit: lifecycle authority is confined to one protected SQLite catalog, authority domain, and
  pinned evaluation authority. A committed record does not install or deploy a skill.
- Acceptance checks: [`test_verified_skill_lifecycle.py`](../../tests/test_verified_skill_lifecycle.py),
  [`test_verified_skill_lifecycle_review.py`](../../tests/test_verified_skill_lifecycle_review.py).
- Research: [`REF-SEGOS`](RESEARCH_TRACEABILITY.md#ref-segos), [`REF-SKILLGLOW`](RESEARCH_TRACEABILITY.md#ref-skillglow), [`REF-REPOTOSKILL`](RESEARCH_TRACEABILITY.md#ref-repotoskill), [`REF-DSR`](RESEARCH_TRACEABILITY.md#ref-dsr).

#### REC-010 — Online experience collection; offline harness/skill evolution.

- Repository evidence: [`learning_surfaces.py`](../../gepa_mindfulness/learning_surfaces.py), [`skill_lifecycle.py`](../../gepa_mindfulness/skill_lifecycle.py), [`coevolution.py`](../../gepa_mindfulness/coevolution.py).
- Architecture boundary: [`controlled_evolution.md`](../controlled_evolution.md).
- Limit: acceptance decisions are audit-only, and each validated decision can be consumed once.
  Consumption does not execute, install, deploy, or roll back a model, harness, or skill. A
  decision has no universal authority across catalogs.
- Acceptance checks: [`test_learning_surfaces.py`](../../tests/test_learning_surfaces.py),
  [`test_offline_evolution_epochs.py`](../../tests/test_offline_evolution_epochs.py),
  [`test_verified_skill_lifecycle.py`](../../tests/test_verified_skill_lifecycle.py),
  [`test_verified_skill_lifecycle_review.py`](../../tests/test_verified_skill_lifecycle_review.py),
  [`test_model_harness_coevolution.py`](../../tests/test_model_harness_coevolution.py).
- Research: [`REF-COEVOLVE`](RESEARCH_TRACEABILITY.md#ref-coevolve), [`REF-HOH`](RESEARCH_TRACEABILITY.md#ref-hoh).

## P2

### P2 — Experimental

#### REC-011 — Multiple competing hypotheses + information-gain inquiry.

- Repository evidence: [`experimental_overlays.py`](../../evaluation/experimental_overlays.py), [`experimental_records.py`](../../evaluation/experimental_records.py), [`experimental_v5_overlays.md`](../experimental_v5_overlays.md).
- Acceptance checks: [`test_experimental_overlay_registry.py`](../../tests/test_experimental_overlay_registry.py), [`test_experimental_overlay_flags.py`](../../tests/test_experimental_overlay_flags.py), [`test_experimental_overlay_records.py`](../../tests/test_experimental_overlay_records.py).
- Research: [`REF-PEARL`](RESEARCH_TRACEABILITY.md#ref-pearl).

#### REC-012 — Small adaptive multi-agent topology codebook.

- Repository evidence: [`experimental_overlays.py`](../../evaluation/experimental_overlays.py), [`experimental_records.py`](../../evaluation/experimental_records.py), [`experimental_v5_overlays.md`](../experimental_v5_overlays.md).
- Acceptance checks: [`test_experimental_overlay_registry.py`](../../tests/test_experimental_overlay_registry.py), [`test_experimental_overlay_flags.py`](../../tests/test_experimental_overlay_flags.py), [`test_experimental_overlay_records.py`](../../tests/test_experimental_overlay_records.py).
- Research: [`REF-MASKILLS`](RESEARCH_TRACEABILITY.md#ref-maskills).

#### REC-013 — Declarative global/focus/local orchestration scope.

- Repository evidence: [`experimental_overlays.py`](../../evaluation/experimental_overlays.py), [`experimental_records.py`](../../evaluation/experimental_records.py), [`experimental_v5_overlays.md`](../experimental_v5_overlays.md).
- Acceptance checks: [`test_experimental_overlay_registry.py`](../../tests/test_experimental_overlay_registry.py), [`test_experimental_overlay_flags.py`](../../tests/test_experimental_overlay_flags.py), [`test_experimental_overlay_records.py`](../../tests/test_experimental_overlay_records.py).
- Research: [`REF-AGENTSCOPE`](RESEARCH_TRACEABILITY.md#ref-agentscope).

#### REC-014 — Mechanistic/circuit audit of actual model changes.

- Repository evidence: [`experimental_overlays.py`](../../evaluation/experimental_overlays.py), [`experimental_records.py`](../../evaluation/experimental_records.py), [`experimental_v5_overlays.md`](../experimental_v5_overlays.md).
- Acceptance checks: [`test_experimental_overlay_registry.py`](../../tests/test_experimental_overlay_registry.py), [`test_experimental_overlay_flags.py`](../../tests/test_experimental_overlay_flags.py), [`test_experimental_overlay_records.py`](../../tests/test_experimental_overlay_records.py).
- Research: [`REF-SAE`](RESEARCH_TRACEABILITY.md#ref-sae).

## Research traceability

See [`RESEARCH_TRACEABILITY.md`](RESEARCH_TRACEABILITY.md) for primary metadata, source links,
repository inferences, and maturity limits for every stable research ID.
