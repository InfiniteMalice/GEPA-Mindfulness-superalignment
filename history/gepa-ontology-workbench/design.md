# GEPA Ontology Governed Workbench — Design Specification

**Status:** Approved for implementation
**Target ontology:** GEPA Mindfulness Ontology v1.0.0
**Date:** 2026-08-10

## Purpose

Build a human-readable and LLM-readable semantic workbench for the GEPA Mindfulness Ontology. The workbench must help contributors explore the ontology, inspect evidence without confusing it for truth, and produce bounded context or training bundles for a governed self-improvement cycle.

The workbench is not a second ontology engine. It is a focused interface over the ontology design. Canonical concepts remain authoritative and immutable in the interface. Runtime evidence, assessments, and proposed additions remain separate from canonical definitions.

## Product principles

1. Preserve all 22 core ontology invariants from section 39 of the source specification.
2. Keep the normative layer, operational layer, and their bridge visually and semantically distinct.
3. Treat evidence, metrics, rewards, and training targets as proxies rather than normative identity.
4. Preserve support and opposition independently; do not hide contradiction through premature scalarization.
5. Reject unnecessary concepts by checking existing IDs, aliases, definitions, and semantic neighbors first.
6. Never add a proposed concept automatically to the canonical ontology.
7. Keep v1 compact: no graph database, model runtime, training orchestration, authentication, or durable collaborative backend.

## Audience and success criteria

The primary audience is mixed: human researchers and contributors, plus LLM agents consuming structured context.

The workbench succeeds when either audience can:

- understand a concept in plain language;
- identify its canonical ID, layer, type, relations, provenance, and maturity;
- distinguish definitions from observations and proxies;
- inspect evidence for and against a proposition;
- identify unresolved tensions and forbidden inferences;
- see how operational systems evaluate, train for, or mitigate normative targets;
- assemble a deterministic, bounded context or training bundle;
- test whether a proposed addition is genuinely new and invariant-compatible.

## Scope

### Included in v1

- A single-page ontology workbench with Explore, Assess, Improve, and Invariants modes.
- Search across concept names, canonical IDs, types, relations, evaluators, and invariants.
- A compact interactive map separating normative and operational entities through a governed bridge.
- Detailed concept inspection with definition, relations, provenance, evidence, maturity, and gaps.
- Paraconsistent assessment examples that preserve independent support and opposition.
- A guarded improvement flow that previews deterministic Markdown, JSON, and YAML bundles.
- Duplicate, alias, overlap, provenance, layer-boundary, protected-kernel, and invariant checks for proposals.
- Representative ontology data covering the protected kernel, imperatives, contemplative values, key virtues, failure modes, operational mappings, relation families, maturity facets, and all 22 invariants.
- Responsive, keyboard-accessible interaction.

### Explicitly excluded from v1

- Editing or mutating canonical ontology files.
- Automatically accepting or merging proposals.
- Executing training jobs, evaluator runs, or model updates.
- Persisting shared proposals or evidence to a hosted database.
- User accounts, permissions, uploads, or external connectors.
- Reimplementing the repository's Python ontology API, CLI, inference engine, or training stack.
- Claiming that a passing validation result proves alignment or correctness.

## Architecture

The site is a self-contained single-page application with a deterministic in-memory ontology model. It contains four bounded interface modules:

### Explore

Explore presents the ontology as three visibly distinct regions:

- **Normative layer:** what ought to hold, including imperatives, values, virtues, constraints, harms, agency concepts, epistemic concepts, and failures.
- **Governed bridge:** qualified relations such as `evaluates`, `trains_for`, `provides_evidence_for`, and `mitigates`.
- **Operational layer:** what exists or is observed, including evaluators, cases, datasets, reward components, trace events, modules, training methods, runtime controls, and evidence artifacts.

Selecting a concept opens a detail panel. Search and filters update both the map and concept list without changing the source model.

### Assess

Assess demonstrates structured evidence rather than Boolean truth. Each assessment may contain support, opposition, contradiction, uncertainty, provenance, dependency groups, and a named scalarization policy when scalarization is explicitly requested.

Correlated evidence is visibly grouped. Absence of evidence never renders as evidence of absence.

### Improve

Improve is a proposal and export workbench, not a canonical editor. The flow is:

1. Select an existing target or enter a proposed addition.
2. Attach a concise definition, type, layer, relations, provenance, and uncertainty.
3. Run deterministic semantic checks.
4. Review blockers, warnings, existing equivalents, and unresolved tensions.
5. Preview and download a governed context or training bundle.

Valid proposals retain lifecycle `proposed`. No interface action changes canonical v1.

### Invariants

Invariants exposes all 22 core invariants as a permanent review surface. Each invariant includes a plain-language explanation, machine-readable key, and examples of permitted or forbidden interpretations.

## Data model

The client model uses small, explicit records:

- `OntologyNode`: ID, label, layer, type, definition, lifecycle, protected status, aliases, provenance, and maturity facets.
- `OntologyRelation`: source, predicate, target, relation family, epistemic status, provenance, and optional context.
- `Assessment`: subject, target, support, opposition, contradiction, uncertainty, provenance, dependency group, and optional named policy.
- `Invariant`: numeric order, stable key, statement, explanation, and guarded inference patterns.
- `Proposal`: candidate node or relation plus validation results; never part of the canonical node collection.
- `ContextBundle`: ontology version and hash, selected nodes, relevant relations, evidence, policies, invariants, tensions, and provenance.

The model preserves distinctions among probability, confidence, heuristic score, belief weight, and normalized metric.

## Proposal validation

Validation runs locally and deterministically. Results are either blockers or warnings.

### Blockers

- Exact canonical ID collision.
- Exact or high-confidence alias collision.
- Unsupported node type, relation, domain, or range.
- Cross-layer identity assertion, such as a reward component being a normative value.
- Claim that evidence, evaluation, training, or runtime enforcement establishes normative identity.
- Protected-kernel contradiction presented as an incidental operational update.
- Missing required identity, definition, type, layer, or provenance fields.
- Proposal export that omits ontology version or relevant invariants.

### Warnings

- Strong semantic overlap with an existing concept.
- Weak provenance or contested epistemic status.
- Uncertain quantity without an interval or qualitative uncertainty.
- Correlated evidence that may be double-counted.
- Causal language stronger than the supplied evidence.
- New categorical classification that discards available continuous information.
- Missing operational mapping or maturity evidence.

The interface explains each result and points to the relevant invariant. A blocked proposal can still be inspected, but it cannot be exported as invariant-compatible.

## Bundle generation

Users can export two closely related artifacts:

- **Context bundle:** a bounded semantic subgraph for an LLM or human reviewer.
- **Training bundle:** the same semantic core plus learning objective, positive and negative examples, evaluator targets, unresolved tensions, and explicit forbidden inferences.

Bundles are deterministic for the same inputs and ordering. Every bundle includes:

- ontology version and hash;
- generation timestamp;
- target concepts and canonical IDs;
- relevant direct relations;
- applicable invariants and forbidden inferences;
- evidence with provenance and dependency groups;
- uncertainty and unresolved tensions;
- named policies used in aggregation or scalarization;
- lifecycle and authority labels.

The bundle states that it is an input to evaluation or training, not evidence of learned or deployed behavior.

## Interface design

The approved visual direction is a restrained research instrument:

- dark mineral background and warm ivory text;
- muted jade for normative entities;
- cool blue for operational entities;
- amber for bridge relations, tensions, and warnings;
- coral only for blocking failures;
- serif typography for conceptual definitions and clean sans-serif typography for controls and metadata.

The desktop workbench uses a map-and-detail split view. Mobile collapses to a searchable list followed by the selected concept detail. Dense material uses progressive disclosure rather than dashboard clutter.

The opening viewport centers the ontology's essential distinction: “what ought to hold” versus “what is observed,” connected by a bridge labeled “Evidence informs. It does not define.”

## Interaction behavior

- Search provides immediate results and supports canonical IDs.
- Tabs switch between Explore, Assess, Improve, and Invariants without navigation loss.
- Map nodes, concept rows, and relations are keyboard focusable and have accessible labels.
- Relation selection changes the detail context without silently changing the primary concept.
- Proposal validation runs on request and displays a stable result summary.
- Bundle previews are readable before download.
- Clipboard and download actions report success or failure without losing the current selection.
- Reduced-motion preferences disable decorative transitions.

## Error handling

- Empty search and empty result states provide direct recovery actions.
- Invalid IDs, quantities, or relations show field-level explanations.
- Broken internal references render as validation failures rather than disappearing.
- Contradictory evidence remains visible with an explanation.
- Oversized context selections require narrowing rather than silently truncating important material.
- Export errors leave the preview intact for copying.
- Unexpected client errors produce a small recoverable message and do not modify source data.

## Verification

Implementation verification must cover:

- successful production compilation;
- search by label, ID, type, relation, and invariant;
- layer and relation filtering;
- concept selection and relation traversal;
- independent support and opposition rendering;
- correlated-evidence warnings;
- exact duplicate, alias, and semantic-overlap checks;
- protected-kernel and cross-layer blockers;
- all 22 invariant rules appearing in the review surface;
- deterministic context and training bundle generation;
- rejection of blocked proposal export;
- keyboard navigation, focus visibility, labels, reduced motion, and responsive layout.

Representative fixtures include Goal Fixation, Validator Capture, conflicting imperatives, correlated evidence, partial reversibility, proxy degradation under distribution shift, and correct-but-deceptive behavior.

## Implementation boundary

The first release is intentionally a governed reference and bundle-authoring interface. It demonstrates the semantic contract needed by a later repository integration while avoiding overlap with the existing ontology runtime. Future integration can replace the embedded reference dataset with generated `graph.json` and connect exported bundles to repository workflows without changing the interface's conceptual boundaries.
