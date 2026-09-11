# Structured Logging Contract

New structured events can be wrapped in `EventEnvelope` from
`mindful_trace_gepa.logging_schema`. An envelope carries `schema_version`, `event_id`, `event_type`,
`timestamp`, and a JSON-compatible `payload`.

Existing optional linkage fields are `run_id`, `rollout_id`, `trace_id`, `sample_id`,
`conversation_id`, `checkpoint_id`, `checkpoint_step`, `model_id`, `model_checkpoint_hash`,
`dataset_id`, `policy_version`, and `config_hash`. Action-bound optional linkage fields are `action_id`,
`parent_event_ids`, `evidence_refs`, `model_version`, `harness_version`, `case_version`, `case_id`,
`stripe_id`, `repeat_id`, `seed`, `authorization_scope`, `verifier_refs`, `valid_from`, `valid_until`,
and `superseded_by`.

Legacy trace rows still load. Viewer code normalizes old rows by treating their `stage` as the event type and leaves missing optional fields empty.

Supported event types include reasoning checkpoints, reward breakdowns, token telemetry, semantic
assessments, principle robustness assessments, memory write/retrieval assessments, memory laundering
reports, CPT pairwise examples and training metrics, SSR units/resolve attempts/repair reports, deception
probes, attribution references, review events, repair events, objective specifications, validator-capture
assessments, proxy-objective assessments, novelty assessments, objective-posterior updates,
robust-objective decisions, proxy-breakdown reports, and objective-validation interrupts. Action-bound
event types are prediction commits, proposed and executed actions, observed outcomes, verification results,
epistemic assessments, and case assessments.

Action-bound metadata is optional for backward compatibility. Supplied linkage IDs, versions, scopes, and
reference strings must be nonblank. Reference collections are snapshotted as immutable tuples; they serialize
as JSON lists. Supplied `case_id` values are built-in integers from 0 through 17, `repeat_id` values are
nonnegative built-in integers, and `seed` values are built-in integers. Validity bounds are ISO-8601 datetimes
with an explicit UTC offset; this avoids ambiguous comparisons between naive and aware datetimes.

Raw evidence and raw action or outcome events are append-only. A derived assessment may identify a later
replacement through `superseded_by`, but that relationship does not rewrite or delete its source event or
evidence. This schema records linkage metadata only; event ordering and which event types may be superseded
are enforced by the separate sequence validator introduced in a later change.

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
