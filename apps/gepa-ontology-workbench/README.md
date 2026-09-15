# GEPA Ontology Governed Workbench

GEPA is a local-first review surface for the Mindfulness Constitution ontology. It separates normative concepts from operational evidence, exposes the ontology's protected kernel and invariants, validates proposed additions without mutating canonical data, and generates explicitly noncanonical context or training bundles.

## Governance boundaries

- Canonical ontology data is read-only in the workbench.
- The protected normative kernel cannot be revised incidentally through a proposal.
- Operational evidence may update assessments, but it does not redefine normative concepts.
- Training exports require curated behavior examples and never claim learned or deployed behavior.
- Generated bundles include the ontology version and verified SHA-256 reference hash, but remain noncanonical artifacts.

## Workbench modes

- **Explore** filters concepts by layer, type, and registered relation predicate while preserving separate concept and relation selection.
- **Assess** shows support and opposition independently, retains uncertainty, and identifies correlated evidence groups.
- **Invariants** presents the 22 ordered invariants and protected-kernel membership.
- **Improve** checks proposed ontology additions against explicit node-type and relation registries before enabling governed exports.

## Local development

Prerequisite: Node.js `>=22.13.0`.

From the repository root:

```bash
cd apps/gepa-ontology-workbench
npm install
npm run dev
```

No publishing, deployment, or access-policy change is part of the local workflow.

## Validation commands

```bash
npm test
npm run lint
npm run build
npm audit --omit=dev
npm audit # informational; currently expected to exit nonzero
```

`npm audit --omit=dev` is the production dependency gate and must exit with status `0`. The full
`npm audit` command is informational and currently exits nonzero because of six known
development-dependency findings. The verification report records the exact paths and mitigations.

## Export behavior

After a proposal passes semantic checks and the canonical digest is verified, the workbench can
generate JSON, YAML, or Markdown previews. Each bundle records the requested target IDs separately
and includes the node definition for every relation or assessment endpoint. Training bundles
accept exactly one requested target until a governed multi-target attribution schema exists.

Copy and download failures are recoverable and do not discard the preview. Object URLs created
for downloads are always revoked.

## Evidence derivation and runtime authority

The context exporter keeps a one-hop semantic neighborhood and permits semantic cycles. Optional
`EvidenceItem.derivedFrom` links identify parent evidence records. The exporter validates those
links as a separate provenance DAG and retains ancestor evidence outside the immediate neighborhood
in `provenanceEvidence`. Cycles, missing parents, and conflicting evidence IDs block export.
Inputs without derivation links preserve the legacy bundle shape. The current sample data contains
unstructured source citations; the exporter does not infer derivation edges from those citations.

Proposal checks and similarity warnings do not promote a claim or mutate canonical data. The
Python host commit adapter and its trust limits are documented in
[Verification and runtime authority](../../docs/VERIFICATION_AND_RUNTIME_AUTHORITY.md).
The workbench has no backend commit endpoint and does not authenticate external evidence.

`app/lib/bundles.test.ts` tests graph separation, ancestor retention, and export rejection.
