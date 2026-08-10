import { describe, expect, it } from "vitest";
import { ontologyNodes, ontologyRelations } from "../data/ontology";
import type { Proposal } from "./ontology-types";
import { validateProposal } from "./proposal-validation";

const valid: Proposal = {
  id: "op:novel_calibration_evaluator",
  label: "Novel Calibration Evaluator",
  layer: "operational",
  type: "evaluator",
  definition: "Measures whether a system communicates calibration under novel uncertainty.",
  provenance: "experiment: calibration pilot",
  lifecycle: "proposed",
  uncertainty: { kind: "confidence", qualitative: "probably_high" },
  epistemicStatus: "experimentally_supported",
  governanceClassification: "ordinary",
  operationalMapping: "eval:novel_calibration",
  maturityEvidence: "dataset:calibration-pilot",
};

const codes = (issues: readonly { code: string }[]) => issues.map((issue) => issue.code);

describe("validateProposal", () => {
  it("blocks duplicate and overlapping ontology identities", () => {
    expect(codes(validateProposal({ ...valid, id: "failure:goal_fixation" }, ontologyNodes, ontologyRelations))).toContain("duplicate-id");
    expect(codes(validateProposal({ ...valid, label: "Increase Knowledge" }, ontologyNodes, ontologyRelations))).toContain("alias-overlap");
    expect(codes(validateProposal({ ...valid, layer: "normative", type: "RewardComponent" }, ontologyNodes, ontologyRelations))).toContain("cross-layer-identity");
  });

  it("blocks exact legacy aliases but not partial-word overlap", () => {
    expect(codes(validateProposal({ ...valid, label: "Increase Knowledge" }, ontologyNodes, ontologyRelations))).toContain("alias-overlap");
    expect(codes(validateProposal({ ...valid, label: "Goal" }, ontologyNodes, ontologyRelations))).not.toContain("alias-overlap");
  });

  it("requires complete, grounded proposals", () => {
    expect(codes(validateProposal({ ...valid, provenance: "" }, ontologyNodes, ontologyRelations))).toContain("missing-provenance");
    expect(codes(validateProposal({ ...valid, relationPredicate: "achieves", relationTarget: "norm:honesty" }, ontologyNodes, ontologyRelations))).toContain("forbidden-bridge-inference");
    expect(codes(validateProposal({ ...valid, relationPredicate: "unknown", relationTarget: "missing:target" }, ontologyNodes, ontologyRelations))).toEqual(expect.arrayContaining(["unknown-relation-predicate", "unresolved-relation-target"]));
  });

  it("accepts a distinct operational evaluator without blockers", () => {
    expect(validateProposal(valid, ontologyNodes, ontologyRelations).filter((issue) => issue.severity === "blocker")).toHaveLength(0);
  });

  it("requires proposed lifecycle and modeled uncertainty", () => {
    expect(codes(validateProposal({ ...valid, lifecycle: "active" } as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("lifecycle-not-proposed");
    expect(codes(validateProposal({ ...valid, uncertainty: undefined } as unknown as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("missing-uncertainty");
    expect(codes(validateProposal({
      ...valid,
      uncertainty: { kind: "confidence", estimate: 0.99 },
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("unsupported-certainty");
    expect(codes(validateProposal({
      ...valid,
      uncertainty: { kind: "confidence", lower: 0.8, upper: 0.2 },
    }, ontologyNodes, ontologyRelations))).toContain("invalid-uncertainty");
    expect(codes(validateProposal({
      ...valid,
      uncertainty: { kind: "unsupported", qualitative: "low confidence" },
    } as unknown as Proposal, ontologyNodes, ontologyRelations))).toContain("invalid-uncertainty");
  });

  it("uses explicit node-type and layer registries", () => {
    expect(codes(validateProposal({ ...valid, type: "imaginary_type" }, ontologyNodes, ontologyRelations)))
      .toContain("unsupported-node-type");
    expect(codes(validateProposal({ ...valid, layer: "normative", type: "evaluator" }, ontologyNodes, ontologyRelations)))
      .toEqual(expect.arrayContaining(["unsupported-type-for-layer", "cross-layer-identity"]));
  });

  it("requires paired relation fields and validates registry family, domain, and range", () => {
    const incompleteCodes = codes(validateProposal(
      { ...valid, relationPredicate: "tests" } as Proposal,
      ontologyNodes,
      ontologyRelations,
    ));
    expect(incompleteCodes.filter((code) => code === "incomplete-relation")).toHaveLength(1);
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "tests",
      relationTarget: "failure:goal_fixation",
      relationFamily: "unsupported",
    } as unknown as Proposal, ontologyNodes, ontologyRelations))).toContain("unknown-relation-family");
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "tests",
      relationTarget: "failure:goal_fixation",
      relationFamily: "evidential",
    } as Proposal, ontologyNodes, ontologyRelations))).not.toContain("unknown-relation-predicate");
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "tests",
      relationTarget: "failure:goal_fixation",
      relationFamily: "normative",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("relation-family-mismatch");
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "requires",
      relationTarget: "norm:honesty",
      relationFamily: "normative",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("relation-domain-mismatch");
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "tests",
      relationTarget: "op:reward_component",
      relationFamily: "evidential",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("relation-range-mismatch");
  });

  it("includes canonical relations absent from representative sample edges", () => {
    for (const predicate of ["provides_evidence_against", "tests", "operationalizes"]) {
      expect(codes(validateProposal({
        ...valid,
        relationPredicate: predicate,
        relationTarget: "failure:goal_fixation",
        relationFamily: "evidential",
      } as Proposal, ontologyNodes, ontologyRelations))).not.toContain("unknown-relation-predicate");
    }
  });

  it("blocks cross-layer identity and incidental protected-kernel revision", () => {
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "is_a",
      relationTarget: "norm:honesty",
      relationFamily: "structural",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("forbidden-bridge-identity");
    expect(codes(validateProposal({
      ...valid,
      id: "norm:corrigibility_exception",
      label: "Corrigibility Exception",
      layer: "normative",
      type: "virtue",
      relationPredicate: "violates",
      relationTarget: "norm:corrigibility",
      relationFamily: "normative",
      governanceClassification: "ordinary",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("protected-kernel-revision");
    expect(codes(validateProposal({
      ...valid,
      id: "norm:corrigibility_constraint",
      label: "Corrigibility Constraint",
      layer: "normative",
      type: "constraint",
      relationPredicate: "constrains",
      relationTarget: "norm:corrigibility",
      relationFamily: "normative",
      governanceClassification: "ordinary",
    }, ontologyNodes, ontologyRelations))).toContain("protected-kernel-revision");
    expect(codes(validateProposal({
      ...valid,
      id: "norm:corrigibility_constraint",
      label: "Corrigibility Constraint",
      layer: "normative",
      type: "constraint",
      relationPredicate: "constrains",
      relationTarget: "norm:corrigibility",
      relationFamily: "normative",
      governanceClassification: "explicit_normative_revision",
    }, ontologyNodes, ontologyRelations))).not.toContain("protected-kernel-revision");
  });

  it("emits each designed evidence, representation, and maturity warning", () => {
    expect(codes(validateProposal({ ...valid, provenance: "note" }, ontologyNodes, ontologyRelations)))
      .toContain("weak-provenance");
    expect(codes(validateProposal({ ...valid, epistemicStatus: "contested" } as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("contested-epistemic-status");
    expect(codes(validateProposal({ ...valid, dependencyGroup: "trace:482" } as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("correlated-evidence");
    expect(codes(validateProposal({
      ...valid,
      relationPredicate: "causes",
      relationTarget: "failure:goal_fixation",
      relationFamily: "causal_risk",
      provenance: "note: conceptual mapping",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("causal-provenance");
    expect(codes(validateProposal({
      ...valid,
      layer: "normative",
      type: "derived_category",
      underlyingProfile: "",
    } as Proposal, ontologyNodes, ontologyRelations))).toContain("discretization-loss");
    expect(codes(validateProposal({ ...valid, operationalMapping: "" } as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("missing-operational-mapping");
    expect(codes(validateProposal({ ...valid, maturityEvidence: "" } as Proposal, ontologyNodes, ontologyRelations)))
      .toContain("missing-maturity-evidence");
  });

  it("accepts the documented observational and hypothesis provenance forms", () => {
    for (const provenance of ["observation: trace review", "hypothesized: drift mechanism"]) {
      expect(codes(validateProposal({
        ...valid,
        relationPredicate: "causes",
        relationTarget: "failure:goal_fixation",
        relationFamily: "causal_risk",
        provenance,
      }, ontologyNodes, ontologyRelations))).not.toContain("causal-provenance");
    }
  });
});
