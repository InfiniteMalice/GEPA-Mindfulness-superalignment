export type Layer = "normative" | "operational";
export type Lifecycle = "proposed" | "experimental" | "accepted" | "active" | "deprecated" | "retired";
export type EpistemicStatus = "defined" | "theoretical" | "hypothesized" | "observationally_supported" | "experimentally_supported" | "contested" | "deprecated";
export type RelationFamily = "structural" | "normative" | "evidential" | "causal_risk";
export const QUANTITY_KINDS = [
  "probability",
  "confidence",
  "heuristic_score",
  "belief_weight",
  "normalized_metric",
] as const;
export type QuantityKind = (typeof QUANTITY_KINDS)[number];

export interface Quantity {
  estimate?: number;
  lower?: number;
  upper?: number;
  qualitative?: string;
  kind: QuantityKind;
}

export interface MaturityFacets {
  specified: boolean;
  documented: boolean;
  code_present: boolean;
  dataset_backed: boolean;
  evaluator_backed: boolean;
  trace_observable: boolean;
  training_integrated: boolean;
  runtime_enforced: boolean;
}

export interface OntologyNode {
  id: string;
  label: string;
  layer: Layer;
  type: string;
  definition: string;
  lifecycle: Lifecycle;
  protected: boolean;
  aliases: readonly string[];
  provenance: readonly string[];
  maturity: MaturityFacets;
}

export interface OntologyRelation {
  id: string;
  source: string;
  predicate: string;
  target: string;
  family: RelationFamily;
  epistemicStatus: EpistemicStatus;
  provenance: readonly string[];
  context?: string;
}

export interface EvidenceItem {
  id: string;
  label: string;
  provenance: readonly string[];
  dependencyGroup?: string;
}

export interface Assessment {
  id: string;
  subject: string;
  target: string;
  support: Quantity;
  opposition: Quantity;
  contradiction: number;
  provenance: readonly string[];
  evidence: readonly EvidenceItem[];
  dependencyGroup?: string;
  scalarizationPolicy?: string;
}

export interface Invariant {
  order: number;
  key: string;
  statement: string;
  explanation: string;
  forbiddenInference: string;
}

export interface Proposal {
  id: string;
  label: string;
  layer: Layer;
  type: string;
  definition: string;
  provenance: string;
  lifecycle: Lifecycle;
  uncertainty?: Quantity;
  epistemicStatus: EpistemicStatus;
  governanceClassification: "ordinary" | "explicit_normative_revision";
  relationPredicate?: string;
  relationTarget?: string;
  relationFamily?: RelationFamily;
  dependencyGroup?: string;
  underlyingProfile?: string;
  operationalMapping?: string;
  maturityEvidence?: string;
}

export interface ValidationIssue {
  severity: "blocker" | "warning";
  code: string;
  message: string;
  invariantKeys: readonly string[];
  relatedNodeIds: readonly string[];
}

export interface TrainingExample {
  behavior: string;
  expectedOutcome: string;
  rationale: string;
}

export interface TrainingFixture {
  targetId: string;
  objective: string;
  positiveExamples: readonly TrainingExample[];
  negativeExamples: readonly TrainingExample[];
  evaluatorTargets: readonly string[];
  forbiddenInferences: readonly string[];
}

export type TrainingBundleContent = Omit<TrainingFixture, "targetId">;

export interface ContextBundle {
  kind: "context" | "training";
  ontologyVersion: string;
  ontologyHash: string;
  generatedAt: string;
  authority: "generated_noncanonical_bundle";
  requestedTargetIds: readonly string[];
  nodes: readonly OntologyNode[];
  relations: readonly OntologyRelation[];
  assessments: readonly Assessment[];
  invariants: readonly Invariant[];
  unresolvedTensions: readonly string[];
  policies: readonly string[];
  training?: TrainingBundleContent;
}
