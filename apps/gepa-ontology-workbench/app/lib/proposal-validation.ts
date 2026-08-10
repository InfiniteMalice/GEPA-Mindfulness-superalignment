import {
  NODE_TYPE_REGISTRY,
  RELATION_FAMILY_REGISTRY,
  RELATION_REGISTRY,
} from "../data/ontology-registry";
import type {
  Layer,
  OntologyNode,
  OntologyRelation,
  Proposal,
  Quantity,
  ValidationIssue,
} from "./ontology-types";
import { QUANTITY_KINDS } from "./ontology-types";

const normalize = (value: string) => value.toLowerCase().replace(/[_:-]+/g, " ").replace(/\s+/g, " ").trim();
const registryKey = (value: string) => normalize(value.replace(/([a-z0-9])([A-Z])/g, "$1_$2")).replaceAll(" ", "_");
const words = (value: string) => new Set(normalize(value).split(" ").filter(Boolean));
const forbiddenPredicates = new Set(["achieves", "proves", "internalizes", "runtime_enforces"]);
const identityPredicates = new Set(["is_a", "instance_of", "alias_of"]);
const proxyTypes = new Set(["reward_component", "metric", "evaluator", "observation", "control_signal", "probe"]);
const quantityKinds = new Set<string>(QUANTITY_KINDS);

const issue = (
  severity: ValidationIssue["severity"],
  code: string,
  message: string,
  invariantKeys: readonly string[] = [],
  relatedNodeIds: readonly string[] = [],
): ValidationIssue => ({ severity, code, message, invariantKeys, relatedNodeIds });

const jaccard = (left: Set<string>, right: Set<string>) => {
  const intersection = [...left].filter((word) => right.has(word)).length;
  const union = new Set([...left, ...right]).size;
  return union === 0 ? 0 : intersection / union;
};

const hasModeledUncertainty = (quantity: Quantity | undefined) => Boolean(
  quantity
  && (quantity.estimate !== undefined
    || (quantity.lower !== undefined && quantity.upper !== undefined)
    || quantity.qualitative?.trim()),
);

const quantityValuesAreValid = (quantity: Quantity) => {
  const values = [quantity.estimate, quantity.lower, quantity.upper].filter((value): value is number => value !== undefined);
  return values.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
    && !(quantity.lower !== undefined && quantity.upper !== undefined && quantity.lower > quantity.upper);
};

const isAllowedLayer = (allowed: readonly Layer[], layer: Layer) => allowed.includes(layer);

export function validateProposal(
  proposal: Proposal,
  nodes: readonly OntologyNode[],
  relations: readonly OntologyRelation[],
): readonly ValidationIssue[] {
  void relations;
  const issues: ValidationIssue[] = [];
  const requiredText = ["id", "label", "definition", "type", "provenance"] as const;
  for (const field of requiredText) {
    if (!proposal[field]?.trim()) issues.push(issue("blocker", `missing-${field}`, `Proposal ${field} is required.`));
  }
  if (!proposal.layer) issues.push(issue("blocker", "missing-layer", "Proposal layer is required."));
  if (proposal.lifecycle !== "proposed") {
    issues.push(issue("blocker", "lifecycle-not-proposed", "Proposal lifecycle must remain proposed.", ["corrigible-governance"]));
  }
  if (!hasModeledUncertainty(proposal.uncertainty)) {
    issues.push(issue("blocker", "missing-uncertainty", "Proposal uncertainty must include an estimate, interval, or qualitative value.", ["uncertainty-persists"]));
  } else if (proposal.uncertainty) {
    if (!quantityKinds.has(proposal.uncertainty.kind) || !quantityValuesAreValid(proposal.uncertainty)) {
      issues.push(issue("blocker", "invalid-uncertainty", "Proposal uncertainty must use a supported quantity kind and values from 0 to 1.", ["uncertainty-persists"]));
    }
    const hasInterval = proposal.uncertainty.lower !== undefined && proposal.uncertainty.upper !== undefined;
    if (proposal.uncertainty.estimate !== undefined && !hasInterval && !proposal.uncertainty.qualitative?.trim()) {
      issues.push(issue("warning", "unsupported-certainty", "Point uncertainty estimate lacks an interval or qualitative uncertainty.", ["no-manufactured-certainty"]));
    }
  }

  const typeKey = registryKey(proposal.type);
  const nodeType = NODE_TYPE_REGISTRY[typeKey as keyof typeof NODE_TYPE_REGISTRY];
  if (!nodeType) {
    issues.push(issue("blocker", "unsupported-node-type", "Proposal node type is not defined by the ontology registry."));
  } else if (!isAllowedLayer(nodeType.layers, proposal.layer)) {
    issues.push(issue("blocker", "unsupported-type-for-layer", "Proposal node type is not allowed in the selected layer.", ["no-evidence-identity"]));
  }

  const normalizedId = normalize(proposal.id);
  const duplicate = nodes.find((node) => normalize(node.id) === normalizedId);
  if (duplicate) {
    issues.push(issue("blocker", "duplicate-id", "Proposal ID already exists.", [], [duplicate.id]));
    if (duplicate.protected && proposal.governanceClassification !== "explicit_normative_revision") {
      issues.push(issue("blocker", "protected-kernel-revision", "Protected-kernel changes require explicit normative revision governance.", ["corrigible-governance"], [duplicate.id]));
    }
  }

  const overlappingIdentity = nodes.find((node) => [node.label, ...node.aliases]
    .some((identity) => normalize(identity) === normalize(proposal.label)));
  if (overlappingIdentity) {
    issues.push(issue("blocker", "alias-overlap", "Proposal label overlaps an existing label or alias.", [], [overlappingIdentity.id]));
  }

  const proposalWords = words(`${proposal.label} ${proposal.definition}`);
  const semanticMatches = nodes.filter((node) => jaccard(proposalWords, words(`${node.label} ${node.definition}`)) >= 0.72);
  if (semanticMatches.length > 0) {
    issues.push(issue("warning", "semantic-overlap", "Proposal is semantically similar to existing ontology content.", [], semanticMatches.map((node) => node.id)));
  }

  if (proposal.layer === "normative" && proxyTypes.has(typeKey)) {
    issues.push(issue("blocker", "cross-layer-identity", "Operational proxy types cannot define normative identity.", ["metrics-are-proxies", "no-evidence-identity"]));
  }

  const predicate = proposal.relationPredicate?.trim();
  const targetId = proposal.relationTarget?.trim();
  const hasPredicate = Boolean(predicate);
  const hasTarget = Boolean(targetId);
  const hasFamily = Boolean(proposal.relationFamily);
  const hasIncompleteRelation = (hasPredicate || hasTarget || hasFamily)
    && !(hasPredicate && hasTarget && hasFamily);
  if (hasIncompleteRelation) {
    issues.push(issue("blocker", "incomplete-relation", "Relation predicate, target, and family must be supplied together."));
  }

  if (proposal.relationFamily && !RELATION_FAMILY_REGISTRY.includes(proposal.relationFamily)) {
    issues.push(issue("blocker", "unknown-relation-family", "Relation family is not defined by the ontology registry."));
  }

  if (predicate) {
    if (forbiddenPredicates.has(predicate)) {
      issues.push(issue("blocker", "forbidden-bridge-inference", "Relation predicate makes a forbidden bridge inference.", ["training-intent-not-success", "runtime-enforcement-not-virtue"]));
    }
    const relationDefinition = RELATION_REGISTRY[predicate as keyof typeof RELATION_REGISTRY];
    if (!relationDefinition) {
      issues.push(issue("blocker", "unknown-relation-predicate", "Relation predicate is not defined by the ontology registry."));
    } else {
      if (proposal.relationFamily && proposal.relationFamily !== relationDefinition.family) {
        issues.push(issue("blocker", "relation-family-mismatch", "Relation family does not match the registered predicate family."));
      }
      if (!isAllowedLayer(relationDefinition.domainLayers, proposal.layer)) {
        issues.push(issue("blocker", "relation-domain-mismatch", "Proposal layer is outside the registered relation domain."));
      }

      const target = nodes.find((node) => node.id === targetId);
      if (target && !isAllowedLayer(relationDefinition.rangeLayers, target.layer)) {
        issues.push(issue("blocker", "relation-range-mismatch", "Relation target layer is outside the registered relation range."));
      }
      if (target && target.layer !== proposal.layer && identityPredicates.has(predicate)) {
        issues.push(issue("blocker", "forbidden-bridge-identity", "Cross-layer structural identity assertions are forbidden.", ["no-evidence-identity"]));
      }
      if (target?.protected
        && proposal.layer === "normative"
        && ["structural", "normative"].includes(relationDefinition.family)
        && proposal.governanceClassification !== "explicit_normative_revision") {
        issues.push(issue("blocker", "protected-kernel-revision", "Normative or structural relations to protected-kernel concepts require explicit normative revision governance.", ["corrigible-governance"], [target.id]));
      }
      if (relationDefinition.family === "causal_risk" && !/(experiment|study|observ(?:ed|ation(?:al(?:ly)?)?)|hypothes(?:is|ized))/i.test(proposal.provenance)) {
        issues.push(issue("warning", "causal-provenance", "Causal relation lacks explicit experimental, observational, study, or hypothesis provenance."));
      }
    }
  }

  if (targetId && !nodes.some((node) => node.id === targetId)) {
    issues.push(issue("blocker", "unresolved-relation-target", "Relation target does not resolve to an ontology node."));
  }

  if (proposal.provenance.trim() && (proposal.provenance.trim().length < 12 || !proposal.provenance.includes(":"))) {
    issues.push(issue("warning", "weak-provenance", "Proposal provenance is too weak to resolve to a named source."));
  }
  if (proposal.epistemicStatus === "contested") {
    issues.push(issue("warning", "contested-epistemic-status", "Proposal epistemic status is contested."));
  }
  if (proposal.dependencyGroup?.trim()) {
    issues.push(issue("warning", "correlated-evidence", "Proposal evidence declares a dependency group and must not be counted as independent confirmation.", ["correlated-evidence"]));
  }
  if (typeKey === "derived_category" && !proposal.underlyingProfile?.trim()) {
    issues.push(issue("warning", "discretization-loss", "Derived categorical proposal does not retain its underlying continuous profile.", ["delayed-discretization", "retained-profiles"]));
  }
  if (!proposal.operationalMapping?.trim()) {
    issues.push(issue("warning", "missing-operational-mapping", "Proposal lacks an operational mapping."));
  }
  if (!proposal.maturityEvidence?.trim()) {
    issues.push(issue("warning", "missing-maturity-evidence", "Proposal lacks implementation maturity evidence."));
  }

  return issues;
}
