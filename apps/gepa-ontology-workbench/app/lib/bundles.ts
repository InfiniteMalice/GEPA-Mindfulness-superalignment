import type {
  Assessment,
  ContextBundle,
  EvidenceItem,
  Invariant,
  OntologyNode,
  OntologyRelation,
  TrainingFixture,
} from "./ontology-types";

export interface BuildBundleInput {
  kind: "context" | "training";
  targetIds: readonly string[];
  generatedAt: string;
  nodes: readonly OntologyNode[];
  relations: readonly OntologyRelation[];
  assessments: readonly Assessment[];
  invariants: readonly Invariant[];
  ontologyVersion: string;
  ontologyHash: string;
  trainingFixtures: readonly TrainingFixture[];
}

const yamlScalar = (value: unknown) => {
  if (value === null || value === undefined) return "null";
  return typeof value === "string" ? JSON.stringify(value) : String(value);
};
const requiredInvariantKeys = new Set([
  "coequal-imperatives", "revisable-goals", "metrics-are-proxies", "no-evidence-identity",
  "independent-support-opposition", "contradiction-is-information", "uncertainty-persists", "no-manufactured-certainty",
  "correlated-evidence", "absence-not-absence", "delayed-discretization", "retained-profiles",
  "named-scalarization", "partial-ordering", "contextual-proxy-reliability", "training-intent-not-success",
  "training-success-not-deployment", "runtime-enforcement-not-virtue", "normative-contradiction-not-inconsistency",
  "historical-reproducibility", "runtime-evidence-authority", "corrigible-governance",
]);

const toYaml = (value: unknown, indent = 0): string => {
  const space = " ".repeat(indent);
  if (Array.isArray(value)) {
    return value.length === 0 ? `${space}[]` : value.map((item) => {
      if (item && typeof item === "object") return `${space}-\n${toYaml(item, indent + 2)}`;
      return `${space}- ${yamlScalar(item)}`;
    }).join("\n");
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .map(([key, item]) => {
        if (item && typeof item === "object") return `${space}${key}:\n${toYaml(item, indent + 2)}`;
        return `${space}${key}: ${yamlScalar(item)}`;
      }).join("\n");
  }
  return `${space}${yamlScalar(value)}`;
};

const toMarkdown = (bundle: ContextBundle) => [
  "# GEPA Ontology Context Bundle",
  "",
  `- Kind: ${bundle.kind}`,
  `- Ontology version: ${bundle.ontologyVersion}`,
  `- Ontology hash: ${bundle.ontologyHash}`,
  `- Generated at: ${bundle.generatedAt}`,
  `- Authority: ${bundle.authority}`,
  "",
  "## Requested targets",
  ...bundle.requestedTargetIds.map((id) => `- ${id}`),
  "",
  "## Nodes",
  ...bundle.nodes.flatMap((node) => [
    `- **${node.label}** (${node.id}): ${node.definition}`,
    `  - Layer/type: ${node.layer}/${node.type}; lifecycle: ${node.lifecycle}; protected: ${node.protected}`,
    `  - Aliases: ${node.aliases.join(", ") || "none"}`,
    `  - Provenance: ${node.provenance.join("; ")}`,
    `  - Maturity: ${JSON.stringify(node.maturity)}`,
  ]),
  "",
  "## Relations",
  ...bundle.relations.flatMap((relation) => [
    `- ${relation.id}: ${relation.source} --${relation.predicate}--> ${relation.target} (${relation.family}; ${relation.epistemicStatus})`,
    `  - Provenance: ${relation.provenance.join("; ")}`,
    ...(relation.context ? [`  - Context: ${relation.context}`] : []),
  ]),
  "",
  "## Assessments",
  ...bundle.assessments.flatMap((assessment) => [
    `- ${assessment.id}: ${assessment.subject} -> ${assessment.target}`,
    `  - Support: ${JSON.stringify(assessment.support)}`,
    `  - Opposition: ${JSON.stringify(assessment.opposition)}`,
    `  - Contradiction: ${assessment.contradiction}`,
    `  - Provenance: ${assessment.provenance.join("; ")}`,
    `  - Evidence: ${assessment.evidence.map((evidence) => `${evidence.label} [${evidence.dependencyGroup ?? "independent"}]`).join("; ") || "none"}`,
    `  - Dependency group: ${assessment.dependencyGroup ?? "none"}`,
    `  - Scalarization policy: ${assessment.scalarizationPolicy ?? "none"}`,
  ]),
  "",
  "## Invariants",
  ...bundle.invariants.map((invariant) => `- ${invariant.order}. ${invariant.key}: ${invariant.statement}\n  - Explanation: ${invariant.explanation}\n  - Forbidden inference: ${invariant.forbiddenInference}`),
  ...(bundle.provenanceEvidence ? [
    "", "## Evidence derivation",
    ...bundle.provenanceEvidence.flatMap((evidence) => [
      `- ${evidence.id}: ${evidence.label}`,
      `  - Provenance: ${evidence.provenance.join("; ")}`,
      `  - Derived from: ${evidence.derivedFrom?.join(", ") || "source observation"}`,
    ]),
  ] : []),
  "",
  "## Unresolved tensions",
  ...bundle.unresolvedTensions.map((tension) => `- ${tension}`),
  "",
  "## Policies",
  ...bundle.policies.map((policy) => `- ${policy}`),
  ...(bundle.training ? [
    "",
    "## Training",
    `- objective: ${bundle.training.objective}`,
    "- positiveExamples:",
    ...bundle.training.positiveExamples.flatMap((example) => [
      `  - behavior: ${example.behavior}`,
      `    expectedOutcome: ${example.expectedOutcome}`,
      `    rationale: ${example.rationale}`,
    ]),
    "- negativeExamples:",
    ...bundle.training.negativeExamples.flatMap((example) => [
      `  - behavior: ${example.behavior}`,
      `    expectedOutcome: ${example.expectedOutcome}`,
      `    rationale: ${example.rationale}`,
    ]),
    `- evaluatorTargets: ${bundle.training.evaluatorTargets.join(", ") || "none"}`,
    "- forbiddenInferences:",
    ...bundle.training.forbiddenInferences.map((inference) => `  - ${inference}`),
  ] : []),
].join("\n");

export function buildContextBundle(input: BuildBundleInput): ContextBundle {
  const suppliedInvariantKeys = new Set(input.invariants.map((invariant) => invariant.key));
  const missingInvariantKeys = [...requiredInvariantKeys].filter((key) => !suppliedInvariantKeys.has(key));
  const extraInvariantKeys = [...suppliedInvariantKeys].filter((key) => !requiredInvariantKeys.has(key));
  if (input.invariants.length !== requiredInvariantKeys.size || missingInvariantKeys.length > 0 || extraInvariantKeys.length > 0) {
    throw new Error(`Context bundle requires a complete set of 22 invariant keys. Missing: ${missingInvariantKeys.join(", ") || "none"}; extra: ${extraInvariantKeys.join(", ") || "none"}.`);
  }
  const targetIds = new Set(input.targetIds);
  const knownNodeIds = new Set(input.nodes.map((node) => node.id));
  for (const targetId of targetIds) {
    if (!knownNodeIds.has(targetId)) throw new Error(`Unknown context target: ${targetId}.`);
  }
  const relations = input.relations.filter((relation) => targetIds.has(relation.source) || targetIds.has(relation.target));
  const assessments = input.assessments.filter((assessment) => targetIds.has(assessment.subject) || targetIds.has(assessment.target));
  const provenanceEvidence = collectProvenanceEvidence(input.assessments, assessments);
  const referencedNodeIds = new Set([
    ...targetIds,
    ...relations.flatMap((relation) => [relation.source, relation.target]),
    ...assessments.flatMap((assessment) => [assessment.subject, assessment.target]),
  ]);
  const nodes = input.nodes.filter((node) => referencedNodeIds.has(node.id));
  const unresolvedTensions = assessments
    .filter((assessment) => assessment.contradiction > 0)
    .map((assessment) => `${assessment.id}: contradiction ${assessment.contradiction}`);
  const policies = [...new Set(assessments.map((assessment) => assessment.scalarizationPolicy).filter((policy): policy is string => Boolean(policy)))];
  const training = input.kind === "training" ? buildTrainingContent(input, relations) : undefined;

  return {
    kind: input.kind,
    ontologyVersion: input.ontologyVersion,
    ontologyHash: input.ontologyHash,
    generatedAt: input.generatedAt,
    authority: "generated_noncanonical_bundle",
    requestedTargetIds: [...targetIds],
    nodes,
    relations,
    assessments,
    ...(provenanceEvidence ? { provenanceEvidence } : {}),
    invariants: input.invariants,
    unresolvedTensions,
    policies,
    ...(training ? { training } : {}),
  };
}

function collectProvenanceEvidence(
  allAssessments: readonly Assessment[], selectedAssessments: readonly Assessment[],
): readonly EvidenceItem[] | undefined {
  const evidenceById = new Map<string, EvidenceItem>();
  const identity = (item: EvidenceItem) => JSON.stringify([
    item.label, item.provenance, item.dependencyGroup, item.derivedFrom ?? [],
  ]);
  for (const item of allAssessments.flatMap((assessment) => assessment.evidence)) {
    const previous = evidenceById.get(item.id);
    if (previous && identity(previous) !== identity(item)) {
      throw new Error(`Conflicting evidence identity: ${item.id}.`);
    }
    evidenceById.set(item.id, item);
  }
  if (![...evidenceById.values()].some((item) => item.derivedFrom !== undefined)) return undefined;

  // Only evidence derivation edges participate here; semantic relation cycles remain valid.
  const finished = new Set<string>();
  const visiting = new Set<string>();
  const ordered: EvidenceItem[] = [];
  const visit = (id: string) => {
    if (finished.has(id)) return;
    if (visiting.has(id)) throw new Error(`Evidence provenance cycle at: ${id}.`);
    const item = evidenceById.get(id);
    if (!item) throw new Error(`Unknown provenance parent: ${id}.`);
    visiting.add(id);
    for (const parent of item.derivedFrom ?? []) visit(parent);
    visiting.delete(id);
    finished.add(id);
    ordered.push(item);
  };
  for (const id of evidenceById.keys()) visit(id);

  const included = new Set<string>();
  const include = (id: string) => {
    if (included.has(id)) return;
    included.add(id);
    for (const parent of evidenceById.get(id)?.derivedFrom ?? []) include(parent);
  };
  for (const assessment of selectedAssessments) {
    for (const item of assessment.evidence) include(item.id);
  }
  return ordered.filter((item) => included.has(item.id));
}

function buildTrainingContent(input: BuildBundleInput, relations: readonly OntologyRelation[]) {
  if (input.targetIds.length !== 1) {
    throw new Error("Training bundles require exactly one requested target.");
  }
  const [targetId] = input.targetIds;
  const fixture = input.trainingFixtures.find((candidate) => candidate.targetId === targetId);
  if (!fixture || fixture.positiveExamples.length === 0 || fixture.negativeExamples.length === 0) {
    throw new Error(`No curated training examples are available for target: ${targetId}.`);
  }
  const targetIds = new Set(input.targetIds);
  const relatedEvaluators = relations
    .filter((relation) => targetIds.has(relation.target) && ["evaluates", "tests"].includes(relation.predicate))
    .map((relation) => relation.source);

  return {
    objective: fixture.objective,
    positiveExamples: fixture.positiveExamples,
    negativeExamples: fixture.negativeExamples,
    evaluatorTargets: [...new Set([...fixture.evaluatorTargets, ...relatedEvaluators])],
    forbiddenInferences: [...new Set([
      ...fixture.forbiddenInferences,
      ...input.invariants.map((invariant) => invariant.forbiddenInference),
    ])],
  };
}

export function serializeBundle(bundle: ContextBundle, format: "json" | "yaml" | "markdown"): string {
  if (format === "json") return JSON.stringify(bundle, null, 2);
  if (format === "yaml") return toYaml(bundle);
  return toMarkdown(bundle);
}
