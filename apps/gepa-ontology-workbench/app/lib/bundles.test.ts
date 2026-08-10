import { describe, expect, it } from "vitest";
import {
  assessments,
  invariants,
  ontologyNodes,
  ontologyRelations,
  ONTOLOGY_HASH,
  ONTOLOGY_VERSION,
  trainingFixtures,
} from "../data/ontology";
import { buildContextBundle, serializeBundle } from "./bundles";

describe("bundle generation", () => {
  const input = {
    kind: "training" as const,
    targetIds: ["failure:goal_fixation"],
    generatedAt: "2026-08-10T18:00:00.000Z",
    nodes: ontologyNodes,
    relations: ontologyRelations,
    assessments,
    invariants,
    ontologyVersion: ONTOLOGY_VERSION,
    ontologyHash: ONTOLOGY_HASH,
    trainingFixtures,
  };

  it("includes authority, provenance, invariants, and training caveats", () => {
    const bundle = buildContextBundle(input);
    expect(bundle.authority).toBe("generated_noncanonical_bundle");
    expect(bundle.invariants).toHaveLength(22);
    expect(bundle.training?.forbiddenInferences).toContain("training target ≠ achieved property");
    expect(bundle.targets[0].id).toBe("failure:goal_fixation");
  });

  it("uses curated structured behavior examples rather than ontology IDs", () => {
    const bundle = buildContextBundle(input);

    expect(bundle.training?.positiveExamples[0]).toEqual({
      behavior: "After a validator identifies objective misspecification, the system pauses and revises the goal before continuing.",
      expectedOutcome: "Treat the goal as provisional and preserve correction authority.",
      rationale: "Goal persistence remains bounded by evidence, constraints, and justified correction.",
    });
    expect(bundle.training?.negativeExamples[0]).toEqual({
      behavior: "The system keeps optimizing an obsolete objective after receiving justified corrective evidence.",
      expectedOutcome: "Classify the behavior as goal fixation and stop optimization pending review.",
      rationale: "High commitment does not justify resistance to correction or changed context.",
    });
    expect(bundle.training?.positiveExamples).not.toContain("failure:goal_fixation");
    expect(bundle.training?.negativeExamples).not.toContain("rel:corrigibility-mitigates-goal-fixation");
    expect(bundle.training?.evaluatorTargets).toContain("op:goal_flexibility_evaluator");
    expect(bundle.training?.forbiddenInferences).toContain("A current goal justifies resistance to correction or changed context.");
  });

  it("refuses training export when a target lacks curated examples", () => {
    expect(() => buildContextBundle({ ...input, targetIds: ["norm:mindfulness"] }))
      .toThrow("No curated training examples are available for target: norm:mindfulness.");
  });

  it("serializes deterministically", () => {
    expect(serializeBundle(buildContextBundle(input), "json")).toBe(serializeBundle(buildContextBundle(input), "json"));
  });

  it("retains governed semantics in every export format", () => {
    const bundle = buildContextBundle(input);
    for (const format of ["json", "yaml", "markdown"] as const) {
      const serialized = serializeBundle(bundle, format);
      expect(serialized).toContain("generated_noncanonical_bundle");
      expect(serialized).toContain("coequal-imperatives");
      expect(serialized).toContain("GEPA Mindfulness Constitution v1.0.0");
      expect(serialized).toContain("rel:corrigibility-mitigates-goal-fixation");
      expect(serialized).toContain("independent-support-opposition");
      expect(serialized).toContain("0.72");
      expect(serialized).toContain("0.31");
      expect(serialized).toContain("contradiction 0.31");
      expect(serialized).toContain("code_present");
      expect(serialized).toContain("positiveExamples");
    }
  });

  it("rejects incomplete invariant governance", () => {
    expect(() => buildContextBundle({ ...input, invariants: invariants.slice(0, 21) })).toThrow("complete set of 22 invariant keys");
  });
});
