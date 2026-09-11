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

## Action-bound envelope construction

`EventEnvelope` keeps every linkage field optional at construction time so non-action-bound event
types and historical rows retain their existing construction and normalization behavior. When
`event_type` is action-bound, `payload` must be a JSON-compatible mapping. Construction deep-copies
the mapping into immutable internal containers. `EventEnvelope.to_dict()` returns a fresh ordinary
JSON mapping and fresh nested JSON arrays; changing that returned data does not change the event.

Supplied linkage IDs, versions, scopes, and reference strings must be nonblank. Reference
collections are snapshotted as immutable tuples and serialize as JSON arrays. A supplied `case_id`
must be a built-in integer from 0 through 17. A supplied `repeat_id` must be nonnegative. Every
integer in an action-bound payload, and every supplied `repeat_id` or `seed`, must be a built-in
integer in the inclusive range `-9007199254740991` through `9007199254740991`. Construction raises
`ValueError` outside that serialization-safe JSON range instead of deferring failure or precision
loss to a JSON serializer.

Validity bounds use the lexical subset
`YYYY-MM-DDTHH:MM:SS[.fraction](Z|+HH:MM|-HH:MM)`. The optional fraction contains one through six
digits. Numeric offset hours range from `00` through `23`; numeric offset minutes range from `00`
through `59`. Uppercase `Z` represents UTC. Calendar dates and times must also be valid, and leap
seconds are not supported. Both bounds include an explicit offset, so ordering compares aware
datetimes without truncating accepted fractional seconds.

## Action-bound sequence validation

Call the public `validate_action_bound_sequence()` function to enforce sequence-level requirements.
The validator ignores non-action-bound event types. For every action-bound event, the validator
requires nonblank `run_id`, `model_version`, and `harness_version` values. The pair
`(run_id, repeat_id)` identifies one evaluation unit, and one unit must retain the same model and
harness versions.

The validator enforces these causal links:

- An `action_proposed` event cites exactly one earlier `prediction_commit` event.
- An `action_executed` event cites exactly one earlier matching `action_proposed` event.
- An `outcome_observed` event cites exactly one earlier matching `action_executed` event.
- A `verification_result` event cites exactly one earlier matching `outcome_observed` event.
- An `epistemic_assessment` event cites one or more earlier `verification_result` events.
- A `case_assessment` event cites one or more earlier `epistemic_assessment` events.

All parents of one derived assessment must belong to the same evaluation unit and resolve to one
`action_id`. A verification or assessment may omit its envelope `action_id`; when supplied, the
`action_id` must match the action resolved from its causal ancestry. Parent references must be
distinct and point backward to existing events. Envelope IDs and typed semantic IDs must not be
reused.

Raw evidence and raw action or outcome events are append-only. Only an epistemic or case assessment
may identify a later same-type replacement in the same evaluation unit and action ancestry through
`superseded_by`. The replacement relationship does not rewrite or delete its source event or
evidence.

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
