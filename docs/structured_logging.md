# Structured Logging Contract

New structured events can be wrapped in `EventEnvelope` from `mindful_trace_gepa.logging_schema`. The envelope carries `schema_version`, `event_id`, `event_type`, `timestamp`, optional linkage IDs such as `run_id`, `rollout_id`, `trace_id`, `sample_id`, `conversation_id`, `checkpoint_id`, `checkpoint_step`, `model_id`, `model_checkpoint_hash`, `dataset_id`, `policy_version`, `config_hash`, and a JSON-compatible `payload`.

Legacy trace rows still load. Viewer code normalizes old rows by treating their `stage` as the event type and leaves missing optional fields empty.

Supported event types include reasoning checkpoints, reward breakdowns, token telemetry, semantic assessments, principle robustness assessments, memory write/retrieval assessments, memory laundering reports, CPT pairwise examples and training metrics, SSR units/resolve attempts/repair reports, deception probes, attribution references, review events, and repair events.

Telemetry honesty:

- lightweight token logs are labeled `telemetry_mode: synthetic`;
- measured tokenizer log probabilities can set `telemetry_available: true` and name the backend;
- synthetic token confidence must not be presented as measured model confidence;
- unavailable circuit telemetry is `null` with `telemetry_status`, not measured zero;
- large attribution graphs should be referenced externally rather than embedded in JSONL rows.

Monitoring and peer review metadata may include `review_status`, `reviewer_id`, `reviewer_disagreement`, `drift_flag`, `repair_event`, `supersedes_event_id`, and `notes`. These fields are evidence for review, not automatic reward penalties or new 17-case categories.
