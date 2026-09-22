# Unified V5 Recommendations

[`registry.yaml`](registry.yaml) is the single authored recommendation registry. This reader
summarizes current maturity and links each record to repository evidence. Stable research IDs are
documented in [`RESEARCH_TRACEABILITY.md`](RESEARCH_TRACEABILITY.md); they can motivate or support
a design choice, but do not establish this architecture as an empirical result.

## P0

### P0 — Implemented

#### REC-001 — Verified epistemic-process reward rule.

- Consolidation evidence: [confidence.py](../../src/mindful_trace_gepa/confidence.py), [test_calibration_and_pipeline.py](../../tests/factuality_observability/test_calibration_and_pipeline.py).

September 2026 consolidation: the [reward contract](../epistemic_process_rewards.md) preserves
legacy verified-process fields, adds confidence provenance and self-serving scrutiny diagnostics,
and keeps trace appearance outside optimizer targets.

- Repository evidence: [`confidence.py`](../../src/mindful_trace_gepa/confidence.py), [`epistemic_process.py`](../../gepa_mindfulness/core/epistemic_process.py), [`rewards.py`](../../gepa_mindfulness/core/rewards.py).
- Acceptance tests: [`test_epistemic_process_rewards.py`](../../tests/test_epistemic_process_rewards.py), [`test_reward_integrity_rewards.py`](../../tests/test_reward_integrity_rewards.py).
- Research: [`REF-HEART`](RESEARCH_TRACEABILITY.md#ref-heart).

#### REC-002 — Action-bound epistemic commitments.

- Repository evidence: [`logging_schema.py`](../../src/mindful_trace_gepa/logging_schema.py), [`v5_records.py`](../../evaluation/v5_records.py).
- Acceptance tests: [`test_action_bound_logging.py`](../../tests/test_action_bound_logging.py), [`test_v5_provenance.py`](../../tests/test_v5_provenance.py).
- Research: [`REF-WMLLM`](RESEARCH_TRACEABILITY.md#ref-wmllm), [`REF-DWM`](RESEARCH_TRACEABILITY.md#ref-dwm), [`REF-SHEAVES`](RESEARCH_TRACEABILITY.md#ref-sheaves), [`REF-HERO`](RESEARCH_TRACEABILITY.md#ref-hero).

#### REC-003 — One canonical 17-case V5 manifest.

- Repository evidence: [`17_case_manifest.yaml`](../../evaluation/cases/17_case_manifest.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Acceptance tests: [`test_v5_case_registry.py`](../../tests/test_v5_case_registry.py), [`test_v5_documentation_consistency.py`](../../tests/test_v5_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

#### REC-004 — Versioned recommendation/decision registry.

- Repository evidence: [`recommendations.py`](../../evaluation/recommendations.py), [`registry.yaml`](registry.yaml).
- Acceptance tests: [`test_recommendation_registry.py`](../../tests/test_recommendation_registry.py), [`test_recommendation_documentation_consistency.py`](../../tests/test_recommendation_documentation_consistency.py).
- Research: No Task 5 reference is assigned.

#### REC-005 — Case × robustness stripe × repeat evaluation.

- Consolidation evidence: [common.py](../../evaluation/suites/common.py), [test_v5_consolidation.py](../../tests/test_v5_consolidation.py), [test_evaluator_matched_errors.py](../../tests/test_evaluator_matched_errors.py).

The [V5 integration guide](../17_CASE_FRAMEWORK.md#consolidated-results-and-failure-lifecycle)
now specifies subtype selection, action-event subtype binding, decomposed assessment and
repair-before-reinforce. The [invariant inventory](../../research/invariants.yaml) records
enforcement scope. No manifest changes or additional canonical cases were introduced.

- Repository evidence: [`common.py`](../../evaluation/suites/common.py), [`v5_runner.py`](../../evaluation/v5_runner.py), [`v5_records.py`](../../evaluation/v5_records.py).
- Canonical inputs: [`robustness_stripes.yaml`](../../evaluation/cases/robustness_stripes.yaml), [`registry.py`](../../evaluation/cases/registry.py).
- Acceptance tests: [`test_v5_cell_planner.py`](../../tests/test_v5_cell_planner.py), [`test_v5_evaluation_record.py`](../../tests/test_v5_evaluation_record.py), [`test_v5_repeat_metrics.py`](../../tests/test_v5_repeat_metrics.py).
- Research: [`REF-CONSISTENCY`](RESEARCH_TRACEABILITY.md#ref-consistency), [`REF-LEXICAL-PERTURB`](RESEARCH_TRACEABILITY.md#ref-lexical-perturb), [`REF-TOKENIZER-BETRAYAL`](RESEARCH_TRACEABILITY.md#ref-tokenizer-betrayal).

## P1

### P1 — Implemented

#### REC-006 — World/artifact state != evidence/belief state.

- Consolidation evidence: [test_governed_evidence_commit.py](../../tests/test_governed_evidence_commit.py).

The [host commit adapter](../VERIFICATION_AND_RUNTIME_AUTHORITY.md) now binds exact proposed
evidence updates to existing WRITE grants and independent source-action verification. Equivalent
claim candidates retain separate sources and remain provisional before host commitment.

- Repository evidence: [`state.py`](../../gepa_mindfulness/verification/state.py).
- Limit: evidence identifiers are not dereferenced or issuer-authenticated by this record layer.
- Acceptance checks: [`test_world_evidence_state.py`](../../tests/test_world_evidence_state.py).
- Research: [`REF-EDGEMEM`](RESEARCH_TRACEABILITY.md#ref-edgemem), [`REF-GRAPHMEM`](RESEARCH_TRACEABILITY.md#ref-graphmem).

#### REC-007 — Structured failure graph.

- Consolidation evidence: [failure_atlas.py](../../evaluation/failure_atlas.py), [test_v5_failure_atlas.py](../../tests/test_v5_failure_atlas.py).

[`failure_atlas.py`](../../evaluation/failure_atlas.py) indexes V5 episode failures across runs
while existing failure graphs localize within-trajectory causes. Atlas repair keeps the original
failure and independently verified same-target regression record. Equivalence preserves
provenance and family-balanced repair candidates exclude hidden evaluation.

- Repository evidence: [`failure_atlas.py`](../../evaluation/failure_atlas.py), [`failure_graph.py`](../../gepa_mindfulness/verification/failure_graph.py).
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

- Consolidation evidence: [eligibility.py](../../gepa_mindfulness/training/eligibility.py), [test_training_eligibility.py](../../tests/test_training_eligibility.py), [test_synthetic_v5_provenance.py](../../tests/test_synthetic_v5_provenance.py).

[`eligibility.py`](../../gepa_mindfulness/training/eligibility.py) consolidates explicit TRAIN,
DEVELOPMENT, REGRESSION and HIDDEN_EVAL input policy across GEPA/RL and synthetic conversion.
This complements existing protected evaluation receipts. Labels do not detect stripped or
externally leaked holdout content; private catalog integration remains a host responsibility.

- Repository evidence: [`eligibility.py`](../../gepa_mindfulness/training/eligibility.py), [`learning_surfaces.py`](../../gepa_mindfulness/learning_surfaces.py), [`skill_lifecycle.py`](../../gepa_mindfulness/skill_lifecycle.py), [`coevolution.py`](../../gepa_mindfulness/coevolution.py).
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

#### REC-015 — Internal-state semantic continuity and epistemic continuity audit.

Experimental and disabled by default. Extends the existing trajectory, semantic, memory, and
action-bound interfaces. Dependencies are REC-002, REC-005, REC-006, REC-010, REC-011 and REC-014.
Synthetic matched controls verify record behavior; empirical effectiveness remains unestablished.

- Repository evidence: [`internal_state_trajectory.py`](../../modules/semantic_intent_robustness/internal_state_trajectory.py), [`continuity_audit.py`](../../modules/semantic_intent_robustness/continuity_audit.py), [`README.md`](../../modules/semantic_intent_robustness/README.md).
- Acceptance checks: [`test_sot_state_continuity.py`](../../tests/test_sot_state_continuity.py), [`test_epistemic_continuity.py`](../../tests/test_epistemic_continuity.py), [`test_motivated_forgetting.py`](../../tests/test_motivated_forgetting.py), [`test_continuity_evaluation.py`](../../tests/test_continuity_evaluation.py).
- Research: [`REF-SOT`](RESEARCH_TRACEABILITY.md#ref-sot).

REC-016 through REC-019 remain experimental research overlays. They retain exactly 17 canonical
cases (IDs 1-17). Diagnostic signals do not independently authorize external actions, training,
deployment, or repair success. These additions neither require nor directly reward private
chain-of-thought; the existing verified epistemic-process reward contract remains unchanged.

#### REC-016 — EvoFlint semantic-laundering quality-diversity search.

Experimental and disabled by default. A bounded archive can retain distinct semantic strategies and evidence-linked generation insights without treating novelty or generated judgments as behavioral authority.

Apply bounded quality-diversity search to semantic-laundering transformations within the fixed 17-case framework. Require verified semantic preservation, immutable lineage, hidden-evaluation exclusion, and independently verified behavioral failures before FailureAtlas admission.

- Repository evidence: [`evolutionary_atlas.py`](../../modules/semantic_intent_robustness/evolutionary_atlas.py), [`research_overlays.md`](../research_overlays.md).
- Acceptance checks: [`test_evolutionary_semantic_atlas.py`](../../tests/test_evolutionary_semantic_atlas.py).
- Research: [`REF-EVOFLINT`](RESEARCH_TRACEABILITY.md#ref-evoflint).
- Limitations: The paper does not study GEPA or this repository's semantic-laundering curriculum. The implementation uses harmless synthetic transformations and structured feature novelty; it does not reproduce the paper's attack generator or empirical results.

#### REC-017 — Round-trip structural communication audit.

Experimental and disabled by default. Communication audits distinguish changed meaning and channel failure from verified preservation with a changed policy judgment, avoiding false semantic-laundering successes.

Audit public structured objects through serialization and extraction. Keep exact equality, verified semantics, heuristic similarity, non-equivalence, and unknown outcomes separate; attribute individual stage faults only when independent stage evidence supports attribution.

- Repository evidence: [`serialization_roundtrip.py`](../../evaluation/serialization_roundtrip.py), [`research_overlays.md`](../research_overlays.md).
- Acceptance checks: [`test_serialization_roundtrip.py`](../../tests/test_serialization_roundtrip.py).
- Research: [`REF-COMM-BOTTLENECK`](RESEARCH_TRACEABILITY.md#ref-comm-bottleneck).
- Limitations: The paper does not evaluate alignment laundering. The implementation verifies a bounded propositional fragment and a synthetic JSON codec; arbitrary natural-language equivalence requires a host-supplied verifier and independently authenticated stage evidence.

#### REC-018 — Formal reasoning / LogicTrack audit.

Experimental and disabled by default. Formal audits expose invalid public inference steps while preserving separate premise grounding and behavioral verification. Solver outcomes alone do not authorize actions or receive optimizer reward.

Check explicit public reasoning objects with a bounded solver adapter and optional bounded retries. Keep premise grounding, formal validity, factual correctness, calibration, and behavioral outcomes separate without importing the paper's reasoning-text reward.

- Repository evidence: [`formal_reasoning.py`](../../gepa_mindfulness/verification/formal_reasoning.py), [`research_overlays.md`](../research_overlays.md).
- Acceptance checks: [`test_formal_reasoning_audit.py`](../../tests/test_formal_reasoning_audit.py).
- Research: [`REF-LOGICTRACK`](RESEARCH_TRACEABILITY.md#ref-logictrack).
- Limitations: Formal validity does not establish factual truth, authentic evidence, or a faithful natural-language translation. The reference solver supports a limited propositional fragment; host-authenticated grounding and existing training-eligibility controls remain separate.

#### REC-019 — Latent-to-language transition audit.

Experimental and disabled by default. Separating internal-state movement from public language and action changes prevents latent metrics from being interpreted as behavioral success or deployment authority.

Measure comparable latent, language, and policy/action deltas independently. Report latent-language decoupling or language change without a matching measured latent signal while retaining origin, comparability, and unavailable-state information.

- Repository evidence: [`latent_language_transition.py`](../../modules/semantic_intent_robustness/latent_language_transition.py), [`research_overlays.md`](../research_overlays.md).
- Acceptance checks: [`test_latent_language_transition.py`](../../tests/test_latent_language_transition.py).
- Research: [`REF-LATENT-LANGUAGE-GAP`](RESEARCH_TRACEABILITY.md#ref-latent-language-gap).
- Limitations: These diagnostics do not establish intent, deception, causal use of a representation, successful steering, or alignment. Transfer ratios depend on measurement normalization. Black-box behavioral evaluation remains available without internal-state access.

## Research traceability

See [`RESEARCH_TRACEABILITY.md`](RESEARCH_TRACEABILITY.md) for primary metadata, source links,
repository inferences, and maturity limits for every stable research ID.
