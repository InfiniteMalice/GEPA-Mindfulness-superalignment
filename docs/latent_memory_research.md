# Latent-memory research qualification

PR 11g records **Diagnostic** source qualification for MEMORILLA. Native latent-memory
execution, training, hybrid answering and comparative effectiveness remain **Unsupported**.
This stage adds no backend, model, dependency, runtime configuration loader or optimizer path.
The existing [evidence workspace](evidence_workspace.md) remains its own text/index prototype.

## Checked method and artifacts

[Fallahpour et al., Memorilla: Latent Semantic Memory for LLMs](https://arxiv.org/abs/2610.21375v1),
v1 dated 2026-10-15, paper CC BY 4.0. A frozen encoder stores document embeddings;
query-conditioned trainable slots read the collection and project into a frozen decoder's
input space. Answer-token supervision trains memory. The agentic study trains memory with
GRPO alongside retained text history; it does not establish that a text cache reproduces the method.
Sections 2.1–2.6, 6 and Appendices B.1–B.2, F, G–G.1 were checked 2026-11-02.

The author-linked [snap-stanford/memorilla](https://github.com/snap-stanford/memorilla/tree/b915c1315870518b8e55be34d3427fc8f8654119)
was checked at `b915c1315870518b8e55be34d3427fc8f8654119`:

| Checked artifact | Git blob SHA-1 | Qualification relevance |
| --- | --- | --- |
| `README.md` | `f665f23316347e16469301c9e548d04f0a725df3` | Author release and documented runtime prerequisites. |
| `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | Apache-2.0 code license; dataset/model terms remain separate. |
| `memorilla/memory.py` | `1fa9a3269f01e8ddc93006537632d43ee3c841d8` | Query-conditioned slots, document attention, projection and saved configuration. |
| `memorilla/llm.py` | `5148f3c9a5d4567dafaa7851f06c2eb5653f7467` | Decoder embedding injection and model/tokenizer loading. |
| `memorilla/embeddings.py` | `42425e19f23e6eb460ee7f8612bac497f5d58916` | Collection/shard layout and encoder/download side effects. |
| `memorilla/data.py` | `decf58fa063b353d09b9ee79db1458d759868f7d` | Collection/row lookup, prompt construction and answer labels. |
| `train.py` | `76f200a525973471f8a6f75efba315b7e3fd45ff` | Frozen decoder, memory optimization, warm starts and GPU trainer. |
| `pyproject.toml` | `08271f249d5c9f677ba0076e0deb43ae06b039c0` | Python >=3.11 and pinned GPU-oriented dependencies. |
| `docs/evaluation.md` | `d48e6edd52a29a3ba05d4bd4007ec43d6a3c4326` | Baseline prompts/context limits differ; do not infer matched GEPA comparisons. |

The retained [authored method note](recommendations/source_artifacts/memorilla-2610.21375v1-method.txt)
is not publisher bytes. Its byte count/SHA-256 are in the
[integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md); the stable citation is
[REF-MEMORILLA](recommendations/RESEARCH_TRACEABILITY.md#ref-memorilla).
No upstream code, model or dataset was copied, imported or executed. The release has training
and evaluation code; availability of particular remote model/data files was not qualified here.

## External runtime qualification

The checked release requires Python >=3.11; GEPA also supports Python 3.10. Its declared
Torch 2.7.0, vLLM 0.9.1, Transformers 4.53.3 and Lightning 2.6.1 stack is not added to GEPA's
dependencies. A separately configured runtime is needed before any native experiment. No
operator GPU/storage/training budget, trained memory checkpoint or compatible encoder/decoder
is configured for this stage. Source qualification is not hardware qualification.

An independently authorized host must retain exact code/dependency/environment versions,
encoder/decoder/tokenizer revisions, memory checkpoint and configuration digests, dtype,
attention implementation, shapes, slot count, source corpus and resource limits. Use actual
saved configuration; do not infer missing settings from the state-dict fallback. Record
missing/unexpected warm-start keys and distinguish intentional adaptation from reproduction.
Check the applicable model/data terms separately from the code license.

Upstream loading can enable remote model code; absent data/embeddings can trigger downloads
or GPU encoding. These calls have not been attached to GEPA. Any future host must admit and
pin executable artifacts and enforce its existing runtime and spending authorization before
those side effects. An operator declaration or generated proposal is never an execution grant.

## Evidence and access boundary

Keep raw sources independently accessible using existing
[artifact evidence](evidence_topology.md) and [memory boundaries](evidence_memory.md).
Bind every ordered source chunk and version to the encoder, embedding and collection artifact.
Upstream collection IDs and equal row counts do not prove source content identity; construction
uses the first row of each collection ID, so conflicting rows require independent rejection.

Before a latent read, the host must authorize the complete contributing source collection.
After additions, corrections, revocations or access changes, invalidate or rebuild affected
embeddings and derived memory as appropriate. Attention weights and generated citations do
not prove which sources influenced an output. Do not expose denied source names, labels,
content, embeddings or cached latent influence to the answering model.

Latent output and generated summaries remain unverified proposals. Recover exact permitted
raw sources and independently check support/contradiction before any evidence admission or
optimizer-facing claim. Retrieval alone also does not verify truth. Reuse existing host
verification, reward eligibility and provenance; introduce no alternate truth or access authority.
Preserve abstention, clarification, redirection and independent review when recovery or support
is unresolved. This document does not implement those future latent-backend controls.

## Reproducible Study F protocol

Use the existing [Study F preparation/capture/audit](research_ablation_suite.md) and immutable
manifest, task lineage, evaluator contract, metric and resampling records. Until qualified,
declare latent and hybrid conditions `unsupported` with a reason; retain them in the roster.
The existing `configured` availability means an operator supplied a runtime declaration,
not that the method is faithful, effective or authorized. No new condition IDs are introduced.

For a qualified external experiment, preregister structured retrieval, workspace, latent and
hybrid arms with the same corpora, held-out queries, answerer, seeds/repeats and relevant
budgets. Record actual prompts, tokenization, truncation, context/generation limits, sampling,
source/checkpoint identities and costs. Keep construction, training and selection independent
of held-out queries/answers. An oracle union of successful answers is not an executable hybrid.

Measure exact source recovery, unsupported recall, contradictions, calibration, abstention,
completion and actual construction/query resources separately. Failed, unanswered and missing
observations remain in declared denominators; unrun measurements remain missing, not zero.
Use corpus/trajectory-level dependence and existing paired uncertainty reporting. Verify
claim support independently of the answerer and keep raw metrics separate from research scores.

Negative controls include swapped source/embedding bindings, conflicting collection rows,
unavailable/revoked sources, stale cached memories, unsupported synthetic facts and disagreement
between independent evaluators. For existing PEO/DeLoop histories include completed actions,
answered clarifications, useful rechecks and changed observations. Measure repeat avoidance,
source recovery and completion together, including premature termination. Better recall alone
does not establish better action selection.

The existing constitutional and 17-case protected suites remain required before governed
admission. No case definition, manifest, reward eligibility, verifier, evidence policy,
authorization, human review or promotion mechanism changes. Source/packaging tests establish
software traceability only; no latent-memory, hybrid, anti-loop or alignment gain is claimed.
