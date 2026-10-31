# Inspectable evidence workspace implementation plan

Goal: implement the optional Librarian-inspired inspectable workspace prototype as a bounded,
read-only hierarchy over existing ArtifactSnapshot inputs, with original evidence recovery.
User master sections15/22/23 and standing ready-PR/review/merge instructions authorize execution.
Use the approved quality-gate fallback; repo-quality-gate remains unavailable. No bd/.beads edits.
PR812 final-head CI is pending; do not publish this dependent increment until PR10 merges.

Design: a workspace is a host-only index of existing source/derived item IDs, never a filesystem,
new evidence store, truth checker, training executor or admission policy. Preserve the complete
original snapshot outside the index. Synthetic content must already be a DerivedInterpretation,
with its existing unverified status and input lineage. The generator cannot certify its text.
Raw sources always appear once; active summaries are optional. Parent links may organize only
existing direct derivation inputs under an active summary. Shared provenance remains a DAG,
so parent placement is separate from complete source-reference links. Deleting a summary means
omitting it from a new index; neither raw sources nor archived snapshots can be deleted here.

Approaches considered: exporting arbitrary on-disk trees adds path/execution/ACL surfaces;
a latent-memory model is unavailable and would not be reproduced by text compression. A pure
index plus the existing authorized retrieval preserves inspectability with no new dependencies.
This choice is the implementer's constrained adaptation, not an upstream RL reproduction.

Interfaces: WorkspaceNode(node_id,item_id,parent_id=None), EvidenceWorkspace(snapshot_digest,
nodes,source_refs), prepare_evidence_workspace(snapshot,nodes,source_refs,enabled=False), and
retrieve_evidence_workspace(workspace,snapshot,topology,query,authorize=None,enabled=False).
Both functions fail before input processing when disabled. All inputs are revalidated/detached;
workspace/source bindings and capacity are checked before invoking the existing ACL callback.
The retrieval result is an existing ArtifactRetrieval plus a producer-facing catalog made only
from its permitted handles/content. No original private IDs, labels, denied ancestry or ACL
metadata enters the catalog. Withheld parents cannot hide allowed raw children: those become
roots. Derived items retain unverified/historical/untrusted channels; no evidence promotion.
References recover original permitted source handles, quotes, locations and observation times.

Implementation: add gepa_mindfulness/verification/evidence_workspace.py and deterministic tests
in tests/test_evidence_workspace.py. Tests first fail for absent module, then establish immutable
versioned serialization, raw coverage, input/parent/lineage binding, capacity/default-off before
callbacks, denied/unresolved/changed access, private ID omission, stale/unverified channels,
source fallback and source recovery. Use existing artifact_fixtures without moving unrelated tests.
Then document the source/adaptation, unsupported RL/latent backends and held-out Study F protocol;
add a disabled example, registry/traceability source record, guide and source-note wheel resources.
Run related artifact/memory/eligibility and documentation/source suites, Black/Python3.10/import
checks and real wheel smoke. One fresh final independent review is required by executing-plans;
no implementer delegation. Publish ready only after PR10 merge, babysit CI/reviews, fix verified
findings minimally and merge with the existing authority. No API/model spending or deployment.

Review focus: catalogs must not expose denied source IDs/labels; summary omission cannot erase
raw evidence; workspace digests do not authenticate truth; ancestor ACL changes must invalidate
previous retrievals; no view can silently become optimizer evidence. Retain all17 cases/case0,
CASE×STRIPE×REPEAT and the Constitution unchanged. Experimental/Diagnostic software only.

Pre-flight: preparation and retrieval share EvidenceWorkspace; retrieval must call the same
preparation validator before callbacks. Ruling: preserve original snapshot as host-owned input
and return a separate catalog, rather than copying private text into a new store. If wrong,
a consumer could mistake an index for an authenticated source; docs and tests pin the boundary.

Task evidence: absent-module RED, then 17 passing workspace tests. Three source-label/index-size
regressions observed RED and fixed; long-ID/deep-ancestry expansion observed RED and now reserves
all archived item rows before callbacks. Related artifact/memory suite: 107 passed in 2.98s.
Ruling: label both the catalog and each row, preserving strongest original source admission;
otherwise copying the intended producer view could strip the only shared training guard.
Ruling: extend existing evaluation/recommendations.py REFERENCE_IDS with the newly verified
citation, alongside its registry and exact inventory test. The closed bibliography intentionally
rejected the appended source (35 related failures); do not loosen its sequence validator.

Final related acceptance: 205 passed in 12.12s across nine artifact/memory/workspace/source/docs
modules; standalone workspace suite 27 passed in 1.49s. Existing bibliography ID and arXiv
sequences extended in lockstep; strict ordering and all prior tests retained. Six changed Python
files parse as Python3.10 and follow Black/100-column/import organization. Actual wheel contains
34 byte-equal resources and both changed runtime files; isolated import/default-off/19-registry/
new citation/resource smoke passed. Rebuild after the final constructor-docstring clarification
so the published wheel evidence matches exact source. No model/API/training experiment run.

Final independent whole-branch review: /root/review_evidence_workspace found no actionable
findings in all 18 files; 110 independent tests passed plus selective ancestor-revocation and
optimizer-admission probes. All 18 upload hashes matched; reviewer made no edits. Final wheel
SHA256 e6009972bdd6f2a528f4e97898abf810f137fa6ef83ed5c3b017539598f33676;
34 required resources and 2 changed runtime files match exact source, isolated smoke passed.
No second review or model experiment needed/claimed. PR10 CI must pass and merge before this
increment is published; final-head remote CI remains the integration gate. Local Torch/Ruff
unavailable. No deferred actionable review findings, no protected policy/case/authority changes.
