# Research reproducibility and source custody

Maturity: **Experimental** optional software; exported records are **Diagnostic**, **DEVELOPMENT**,
not optimizer input and confer no authority. Trained-model efficacy is **Unqualified**. Ordinary
inference has no dependency on these paths. No existing contract or migration is replaced.

## Typed API and trust boundary

`gepa_mindfulness.verification.research_archive` reuses existing ArtifactObservation, ArtifactRecord
and ArtifactSnapshot. `ResearchSource` binds artifact/version, bytes digest/size, original references,
actor and timestamp. `ResearchCatalog` contains versioned original observations and no accepted
claims. `SourceAdmissionPacket` binds archive identity and before/source state. `ArchiveHostContext`
provides a trusted evaluator and host custody callback; records and callbacks supplied by a model
are never credentials.

`propose_research_source(..., payload: bytes, clock, enabled=True)` constructs a proposal only.
`ResearchArchive(existing_host_root, store_id, ..., enabled=True)` requires POSIX no-follow,
directory-relative I/O. `append_scope(before, source)` binds logical store, physical root, before/
after catalogs and source. `append(...)` independently checks exact bytes, host custody and existing
action-bound one-use WRITE authorization before exclusive publication. `catalog` returns a detached
snapshot. `read(catalog, artifact_id, version, trusted_catalog_digest=..., enabled=True)` rechecks
immutable metadata and original source bytes. Reopening requires the exact trusted current catalog
pin. Older pinned catalogs remain readable; no version is overwritten. Source limit is 32 MiB;
record envelopes retain the existing 4 MiB/depth-32 bound. Partial I/O retains orphans and never
advances the in-memory catalog. Host serializes writers and persists the latest pin; this prototype
is not a concurrent transactional database or a production credential issuer. Root provisioning,
host callbacks and local storage permissions remain host responsibilities.

`evaluation.phase2_reporting` adds versioned `SourceSelector` (bounded JSON pointer, codec and
exact hash), `ModelExperimentManifest` (code/model/harness/dataset versions, seeds, condition),
`PairedSource`, `ProtectedRegression`, `ResearchComparisonProtocol`, `ResearchVerificationPacket`
and `ResearchHostContext`. `report_research_comparison(...)` derives values from pinned archive
bytes, rejects ambiguous/nonfinite JSON and candidate-mismatched protected receipts, authenticates
detached custody/protocol/rubric packets, then reuses existing `estimate_paired`. Model self-review,
truthy non-Boolean callbacks and mutation do not authenticate evidence. Aliased source selections
cannot masquerade as distinct observations. The host must verify protocol timing, rubric validity,
cluster assumptions and actual independence; role names or multiple identical model instances do
not establish independence.

Intervals use declared dependency clusters and a seeded resampling policy. Repeated captures from
one case are one cluster. Missing pairs, unknown dependencies, missing authentication or too few
clusters yield no interval. Sign-adjusted improvement intervals preserve lower-is-better metrics.
An authenticated protected *model* failure withholds improvement interpretation even with a
positive observed delta. Software passes cannot substitute for model capability regression. All
reports retain general alignment `NOT_ESTABLISHED` and promotion `NOT_AUTHORIZED`. The existing
controlled-evolution accepted stores remain the owners of promotion decisions.

`finite_reconstruction_trial(...)` reloads an archived bundled finite fixture, checks its cached
qualification independently, reconstructs a compact checkpoint, reruns the checker, records exact
fidelity/source hashes, serialized bytes and wall time. Both checkpoints retain references to the
original immutable archive. These are checkpoint sizes, not peak memory or transformer bounds.

## Offline reproduction

Run from a full Git checkout with the documented development dependencies. No network, model
weights download, remote security endpoint or runtime grant is used by the historical audit:

```bash
python -m evaluation.phase2_reproducibility --enabled --repository-root . \
  --pins configs/phase2_source_pins.json --output /tmp/new-phase2-audit.json
python -m docs.experiments.phase2_pr9_resource_probe --enabled \
  --archive-root /tmp/new-phase2-archive > /tmp/new-phase2-resource.json
python -m pytest -q tests/test_research_archive.py tests/test_phase2_reporting.py \
  tests/test_phase2_reproducibility.py
```

Choose unused output paths. The audit output and resource root refuse overwrite. The resource
probe uses explicitly synthetic source custody and host grants scoped to its own new archive;
these authorize no production admission. Historical `audit_stages(root, pins, enabled=True)` reads
bounded exact Git blobs without shell/filter/network execution. Git must support `--no-lazy-fetch` (Git 2.45+); missing objects fail without transport,
with lazy fetch disabled and all protocols denied for the audit subprocess. Full commit pins and manifest or
inline source digests identify stage versions. Checks run against each historical commit, not
later changed README/config files. Hash equality establishes retained byte identity, not the truth
of authored summaries or independent rerunning of historical experiments. Caller-owned pins need
trusted custody. The two early stages lacked JSON manifests; their original logs and selected
sources are pinned directly rather than inventing historical artifacts.

The [integrated result](experiments/phase2_results.md), [delivery map](phase2_delivery.md),
[source audit](phase2_research_sources.md), [architecture decision](adr/0026-cva-causal-interpretability.md)
and per-stage validation reports preserve software/research distinctions. Optional random tiny
GPT-2/Llama studies exercise actual hooks, interventions and KL; constructed dictionaries are not
trained SAEs. Locked semantic fidelity, trained model reliability, architecture calibration and
intervention-informed parameter updates remain unrun. Unknowns, nulls and falsifications remain
in original raw artifacts and must not be discarded by a comparative summary.

## Connector-published checkout

For the final connector-published delivery, first follow [published source provenance](phase2_publication.md) to verify and import the bundled original Git history. The historical audit pins intentionally retain original research commit IDs; a fresh clone needs that local provenance import. Runtime code, original measurements and ordinary inference are unchanged. Frozen publication-status statements describe their original audit time.
