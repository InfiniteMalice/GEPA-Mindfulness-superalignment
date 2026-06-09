# Structured Logging Contract

New structured events can be wrapped in `EventEnvelope` from `mindful_trace_gepa.logging_schema`. The envelope carries `schema_version`, `event_id`, `event_type`, `timestamp`, optional linkage IDs such as `run_id`, `rollout_id`, `trace_id`, `sample_id`, `conversation_id`, `checkpoint_id`, `checkpoint_step`, `model_id`, `model_checkpoint_hash`, `dataset_id`, `policy_version`, `config_hash`, and a JSON-compatible `payload`.

Legacy trace rows still load. Viewer code normalizes old rows by treating their `stage` as the event type and leaves missing optional fields empty.

Supported event types include reasoning checkpoints, reward breakdowns, token telemetry, semantic assessments, principle robustness assessments, memory write/retrieval assessments, memory laundering reports, CPT pairwise examples and training metrics, SSR units/resolve attempts/repair reports, deception probes, attribution references, review events, repair events, objective specifications, validator-capture assessments, proxy-objective assessments, novelty assessments, objective-posterior updates, robust-objective decisions, proxy-breakdown reports, and objective-validation interrupts.

Telemetry honesty:

- lightweight token logs are labeled `telemetry_mode: synthetic`;
- measured tokenizer log probabilities can set `telemetry_available: true` and name the backend;
- synthetic token confidence must not be presented as measured model confidence;
- unavailable circuit telemetry is `null` with `telemetry_status`, not measured zero;
- large attribution graphs should be referenced externally rather than embedded in JSONL rows.

Objective robustness events may include `objective_id`, `design_context_reference`,
`deployment_context_reference`, `proxy_features`, `proxy_likelihood`,
`proxy_correlation_confidence`, `optimization_pressure`, `novel_state_detected`,
`distribution_shift_detected`, `novelty_score`, `shift_score`,
`plausible_objective_count`, `posterior_confidence`, `catastrophic_downside_possible`,
`selected_action`, `preserves_optionality`, `reversible`, `clarification_required`,
`interrupt_required`, `review_required`, `semantic_assessment_reference`,
`memory_boundary_reference`, `value_decomposition_reference`,
`deception_fingerprint_reference`, and `attribution_graph_reference`.

Use stable IDs and compact summaries, hashes, or references for objective robustness rows. Do not
embed large attribution graphs in normal JSONL rows. `ObjectiveValidationInterrupt` is an advisory
control signal for review priority; it is not an execution shortcut and does not authorize
irreversible action.

Monitoring and peer review metadata may include `review_status`, `reviewer_id`, `reviewer_disagreement`, `drift_flag`, `repair_event`, `supersedes_event_id`, and `notes`. These fields are evidence for review, not automatic reward penalties or new 17-case categories.
