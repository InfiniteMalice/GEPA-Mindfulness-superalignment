# Source based metacognitive monitoring

`verification.metacognitive_monitoring.monitor_episode` is disabled by default and does not
inspect its input while disabled. Pass an exact `MonitoringInput` and `enabled=True` to derive
source-linked assumption, shared-dependency, contradiction and task-evidence diagnostics.

The engine preserves original statuses and history. Unverified, observed, inferred, stale or
unavailable premises are not treated as independently supported assumptions. Contradicted
applicability conditions produce scope warnings without making the premise universally false.
Superseded premises resolve through the existing evidence graph while originals remain retained.
Conflicting live hypothesis assessments survive; a newer or more confident assessor does not win.

Shared-dependency analysis identifies both explicit common assumptions and common evidence
references. It reports shared provenance without claiming independence, falsehood or deceptive
intent. `shared_analysis=False` provides the planned mechanism ablation and records that
dependency analysis was not assessed.

Optional task claim IDs let the host identify materially ambiguous instructions, consequential
stakes and the requested factual commitments. Supported ambiguity and stakes declarations lead
to a clarification proposal; missing support for an answer leads to an evidence request.
Neither diagnostic rewrites a claim. The host owns the semantics and authentication of these
declarations; naming a claim does not prove it is true.

Separate world/model/monitor uncertainty is retained from the original epistemic estimate.
Missing uncertainty remains null. Absence of continuity, calibration, repetition, strategy or
evidence-use telemetry is explicitly recorded in `unassessed_domains`, and the assessment remains
partial. A partial assessment with no findings means no problem was found in the inspected
records; it does not certify the uninspected domains. The event-bound PEO adapter will run the
existing specialized diagnostics after validating their causal episode.

All findings remain diagnostic and unverified. The engine provides no execution, storage, reward,
truth-promotion or network operation. A healthy supported input with independent alternatives
returns no intervention finding. Finding IDs are deterministic bounded hashes of structured
source identities, so long or separator-containing source IDs cannot overflow or alias them.
Accepted dense episodes retain at most 512 findings and 512 shared dependencies in the diagnostic
view. `omitted_finding_count` and `omitted_dependency_count` report excess records exactly;
affected domains remain unassessed and the result remains partial. Consequential ambiguity and
contradiction findings receive priority. Original source records are retained without pruning.

The assessment wrapper now exports `metacognitive-assessment-v2` to version the omission-count
fields. The original `metacognitive-v1` wrapper remains readable and upgrades with zero omission
counts. Nested source/finding contracts retain their original v1 schemas. Unknown fields and
unsupported versions still fail closed.

Run `python -m pytest -q tests/test_metacognitive_monitoring.py tests/test_metacognitive_records.py`
for the source-based engine and record contracts. CPU fixture tests establish software behavior
only; no trained-model alignment effect is measured.
