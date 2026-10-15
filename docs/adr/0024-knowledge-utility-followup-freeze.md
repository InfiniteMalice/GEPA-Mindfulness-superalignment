# ADR 0024: Offline task-review acceptance and packet freeze

Status: Accepted for offline implementation; design, plan and native execution approved on
2026-10-14. Operational task-review acceptance and offline packet freeze approved and completed
on 2026-10-15. The subsequent registration binds that exact frozen artifact for offline scoring.
Live execution remains unauthorized and requires a separate decision on concrete execution inputs.

## Context

ADR 0023 pins the corrected draft v3 packet and reports 48 pending task reviews. The retained
review has a blind first pass followed by an unblinded recheck of two corrections. Turning that
evidence into accepted administrative state must preserve the original null and scientific
limitations, rather than infer authenticated independence or execution authority.

## Decision

Add a fixed evidence profile, pure acceptance/freeze construction, and a bounded local CLI.
The candidate binds 14 original evidence files and all 48 task assessments. A separate decision
records explicit maintainer acceptance and freeze authorization, including all retained limitations.
The tool validates the declaration and exact bindings without claiming human authentication.

Freeze creates a new packet version and preserves every source field except the specified schema,
status, freeze flag and packet digest, plus an embedded freeze record. Source templates remain
historical draft templates. Existing artifacts are never overwritten. Live execution remains false.

An initially empty registry admits frozen packets to scoring only after a separately reviewed
registration of their exact packet/source/candidate/decision digests. Registered inputs emit
report-v2 with accepted administrative state and unauthenticated task-review provenance. Draft
v3 continues to emit report-v1 unchanged. Metric arithmetic and two-reviewer response adjudication
do not change. The [protocol](../experiments/knowledge_utility_followup.md) defines the interfaces.

## Consequences and verification

The fixed profile intentionally supports one retained evidence set. Formatting changes, new tasks,
different reviews or altered findings require reviewed profile changes. Hashes bind content;
they do not establish identity, blinding, independence, contamination status or authorization.
Human approval remains an operator responsibility with a retained reference.

Tests cover strict decisions, complete limitations, bounded explicit input reads, source preservation,
reproducibility, exclusive publication, registered transformation validation, draft byte compatibility,
frozen report state and cross-packet binding failures. Synthetic approvals never populate the runtime
registry. Local preparation checks the retained evidence, and the full suite checks compatibility.
All results remain DEVELOPMENT, unverified, unpaired and ineligible to confer training authority.

## Operational registration on 2026-10-15

After presentation of the concrete candidate and its ten limitations, the maintainer explicitly
accepted the evidence and authorized offline freeze. The operator validated all 14 evidence files
and verified that freeze changed only the permitted fields. The subsequent registry entry binds
the packet, source, candidate and decision digests listed in the
[protocol](../experiments/knowledge_utility_followup.md#frozen-scoring-and-registration).
The retained freeze-record fixture supports exact reconstruction and production-registry tests;
synthetic decisions remain unregistered. Registration changes admission to offline scoring,
without changing metric arithmetic, draft compatibility, or execution authority.
