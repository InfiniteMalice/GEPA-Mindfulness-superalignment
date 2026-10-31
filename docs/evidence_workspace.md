# Inspectable evidence workspace

Maturity: **Experimental** CPU preparation/retrieval and **Diagnostic** output. Disabled by
default. No research effectiveness or hardware qualification is established. Native Librarian
RL construction and MEMORILLA latent memory are **Unsupported** here.

This optional prototype extends [existing evidence memory](evidence_memory.md) and
[artifact provenance/topology](evidence_topology.md). A host-owned `ArtifactSnapshot` remains
intact; an `EvidenceWorkspace` adds placements over existing source and derived item IDs.
It is an in-memory index, with no filesystem writes or model execution. A generator may propose
an index and `DerivedInterpretation` records. Neither those records nor source digests certify
truth. The existing constructor requires derived claims to remain unverified with complete
input bindings, evidence references and original observation times.

`WorkspaceNode(node_id, item_id, parent_id=None)` contains private host coordinates. Each raw
source must appear exactly once. A synthetic node may be omitted in a new index while the
original source collection and archived interpretation remain intact. A parent must refer to
an existing derived item that directly lists the child among its inputs. Shared provenance
remains separate from single-parent placement; source references recover all raw ancestors.
The index allows at most 256 nodes, 16 levels, 64 observable construction references and
262144 serialized UTF-8 bytes. Bounds describe a small prototype, not a large-corpus benchmark.

Use `prepare_evidence_workspace(snapshot, nodes, source_refs, enabled=True)` to validate and
bind the index. `source_refs` records construction provenance, without authenticating it.
Use `retrieve_evidence_workspace(workspace, snapshot, topology, query, authorize=host_acl,
enabled=True)` for every read. The implementation revalidates the whole index/snapshot binding
and existing topology before delegating to `retrieve_artifact_evidence` in artifact-index mode.
No additional truth, access, reward or admission authority is introduced. There is no persistent
cache of authorization decisions. Unknown, denied, revoked and unavailable inputs retain the
existing fail-closed behavior. Stale material is withheld for current-evidence queries.

Only the result's `producer_view` is for an answering agent. The complete result is a private
host audit retaining the existing retrieval record. The catalog uses fresh opaque handles,
permitted text, locations, original times, channels and raw-source handles. Private node labels,
source IDs, ACL descriptors and denied ancestry are absent. An allowed raw source remains
available at the root when its synthetic parent is withheld. Omitting an active summary also
removes it from the catalog. A returned `candidate_evidence` channel remains a candidate for
independent verification; the workspace does not upgrade summaries or factual claims.

The result, catalog and individual rows retain non-training labels; REGRESSION/HIDDEN_EVAL
restrictions from raw sources survive projection and derived ancestry. Stripping provenance
cannot be detected by the shared guard and is prohibited. Catalog text is untrusted data;
embedded instructions have no authority. Known output expansion is reserved before host ACL
callbacks using `(item_count + 8) * serialized_input_bytes + 262144`, under a 4 MiB ceiling.
This conservative reservation includes repeated source IDs for every archived interpretation;
an index can be rejected even when a particular authorization outcome would produce less text.

The [disabled example](../configs/evidence_workspace.example.json) is an operator declaration,
not an automatic loader or runtime authorization. The host supplies the typed snapshot, topology,
query, construction provenance and current ACL callback. No credentials, downloads, API spending,
training, evidence promotion, runtime action or deployment occur in these functions.

## Source and adaptation

[REF-LIBRARIAN-AGENTS](recommendations/RESEARCH_TRACEABILITY.md#ref-librarian-agents): Singh et al.,
[Librarian Agents](https://arxiv.org/abs/2610.22273), v1 dated 2026-10-15. Primary sections 2.1–2.4
and Appendix A.1/A.7–A.11 checked 2026-10-31; paper CC BY 4.0. The author's linked repository
[teresaz05/librarian-agents](https://github.com/teresaz05/librarian-agents) was checked at revision
`27c9d86391b5da9c04e88b920076fa3867782bb9`; root LICENSE is MIT, Git blob SHA-1
`7cc8db2603a370f335fd5422a0ed114285558dfc`. No upstream code is copied or imported.

This prototype adapts reusable source-linked organization only. It does not implement the
paper's GRPO-trained builder, content executor, answerability probes or answering model.
The [authored method note](recommendations/source_artifacts/librarian-2610.22273v1-method.txt)
is a retained local summary, not publisher HTML/PDF bytes. Its digest is recorded in the
[integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md).

## Reproducible evaluation

Use the existing [Study F plan/capture/audit](research_ablation_suite.md) with matched corpora,
answerer, independent evaluator, query and construction budgets, seeds and source revisions.
Declare unavailable latent and hybrid runtime arms explicitly. Keep all future evaluation
queries and answer keys out of construction, data selection and checkpoint selection.
Use independent held-out corpus families; repeated questions within one corpus are dependent.

Compare existing structured retrieval, the workspace and any genuinely configured optional
backend. Record exact source recovery, unsupported synthetic claims, contradictions, original
source fidelity, calibration, answer completion, actual construction/query tokens and latency.
Count failed construction and unanswered queries in denominators. Compare matched existing
PEO/DeLoop histories with necessary rechecks and changing observations; lower repeat rate alone
does not establish success. Apply protected 17-case and constitutional regressions before any
existing governed admission. No canonical case, reward eligibility or manifest is changed.

The deterministic tests establish software boundaries only. No answering/training experiment
has run; no accuracy, efficiency, retention, anti-loop or alignment gain is claimed.
