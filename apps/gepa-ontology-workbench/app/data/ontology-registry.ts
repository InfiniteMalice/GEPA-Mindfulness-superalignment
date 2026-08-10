import type { Layer, RelationFamily } from "../lib/ontology-types";

interface NodeTypeDefinition {
  readonly layers: readonly Layer[];
}

export const NODE_TYPE_REGISTRY = {
  imperative: { layers: ["normative"] },
  contemplative_value: { layers: ["normative"] },
  virtue: { layers: ["normative"] },
  agency_concept: { layers: ["normative"] },
  epistemic_concept: { layers: ["normative"] },
  action_concept: { layers: ["normative"] },
  constraint: { layers: ["normative"] },
  harm: { layers: ["normative"] },
  failure_mode: { layers: ["normative"] },
  derived_category: { layers: ["normative", "operational"] },
  behavioral_case: { layers: ["operational"] },
  assessment: { layers: ["operational"] },
  observation: { layers: ["operational"] },
  evaluator: { layers: ["operational"] },
  probe: { layers: ["operational"] },
  metric: { layers: ["operational"] },
  reward_component: { layers: ["operational"] },
  dataset: { layers: ["operational"] },
  dataset_example: { layers: ["operational"] },
  module: { layers: ["operational"] },
  pipeline: { layers: ["operational"] },
  training_method: { layers: ["operational"] },
  training_stage: { layers: ["operational"] },
  policy_decision: { layers: ["operational"] },
  control_signal: { layers: ["operational"] },
  trace_event: { layers: ["operational"] },
  evidence_artifact: { layers: ["operational"] },
  artifact: { layers: ["operational"] },
  model: { layers: ["operational"] },
  model_checkpoint: { layers: ["operational"] },
  runtime_backend: { layers: ["operational"] },
  runtime_gate: { layers: ["operational"] },
  configuration: { layers: ["operational"] },
} as const satisfies Record<string, NodeTypeDefinition>;

export const RELATION_FAMILY_REGISTRY = [
  "structural",
  "normative",
  "evidential",
  "causal_risk",
] as const satisfies readonly RelationFamily[];

export interface RelationDefinition {
  readonly family: RelationFamily;
  readonly domainLayers: readonly Layer[];
  readonly rangeLayers: readonly Layer[];
}

const bothLayers = ["normative", "operational"] as const;
const normativeLayer = ["normative"] as const;
const operationalLayer = ["operational"] as const;

export const RELATION_REGISTRY = {
  is_a: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  part_of: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  instance_of: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  has_dimension: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  derived_from: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  alias_of: { family: "structural", domainLayers: bothLayers, rangeLayers: bothLayers },
  supports: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  protects: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  constrains: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  requires: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  balances: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  in_tension_with: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  violates: { family: "normative", domainLayers: normativeLayer, rangeLayers: normativeLayer },
  mitigates: { family: "normative", domainLayers: bothLayers, rangeLayers: normativeLayer },
  operationalizes: { family: "evidential", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  evaluates: { family: "evidential", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  tests: { family: "evidential", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  measures: { family: "evidential", domainLayers: operationalLayer, rangeLayers: bothLayers },
  estimates: { family: "evidential", domainLayers: operationalLayer, rangeLayers: bothLayers },
  observes: { family: "evidential", domainLayers: operationalLayer, rangeLayers: bothLayers },
  provides_evidence_for: { family: "evidential", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  provides_evidence_against: { family: "evidential", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  records_evidence_about: { family: "evidential", domainLayers: operationalLayer, rangeLayers: bothLayers },
  trains_for: { family: "normative", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  constrains_runtime_for: { family: "normative", domainLayers: operationalLayer, rangeLayers: normativeLayer },
  causes: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
  contributes_to: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
  increases_risk_of: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
  decreases_risk_of: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
  enables: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
  amplifies: { family: "causal_risk", domainLayers: bothLayers, rangeLayers: normativeLayer },
} as const satisfies Record<string, RelationDefinition>;
