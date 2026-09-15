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
    expect(bundle.requestedTargetIds).toEqual(["failure:goal_fixation"]);
    expect(bundle.nodes.map((node) => node.id)).toEqual(expect.arrayContaining([
      "failure:goal_fixation",
      "norm:corrigibility",
      "op:goal_flexibility_evaluator",
    ]));
    const includedNodeIds = new Set(bundle.nodes.map((node) => node.id));
    for (const relation of bundle.relations) {
      expect(includedNodeIds.has(relation.source)).toBe(true);
      expect(includedNodeIds.has(relation.target)).toBe(true);
    }
    for (const assessment of bundle.assessments) {
      expect(includedNodeIds.has(assessment.subject)).toBe(true);
      expect(includedNodeIds.has(assessment.target)).toBe(true);
    }
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

  it("refuses multi-target training exports until attribution has a governed schema", () => {
    expect(() => buildContextBundle({
      ...input,
      targetIds: ["failure:goal_fixation", "failure:goal_fixation"],
    })).toThrow("Training bundles require exactly one requested target.");
  });

  it("serializes deterministically", () => {
    expect(serializeBundle(buildContextBundle(input), "json")).toBe(serializeBundle(buildContextBundle(input), "json"));
  });

  it("keeps YAML JSON-compatible for null and undefined values", () => {
    const bundle = buildContextBundle(input);
    const extendedBundle = { ...bundle, nullableExtension: null } as typeof bundle & {
      nullableExtension: null;
    };
    const yaml = serializeBundle(extendedBundle, "yaml");

    expect(yaml).toContain("nullableExtension: null");
    expect(yaml).not.toContain("undefined");
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

  it("allows semantic cycles while retaining a bounded context neighborhood", () => {
    const nodes = ontologyNodes.slice(0, 3);
    const relations = nodes.map((node, index) => ({
      ...ontologyRelations[0], id: `cycle:${index}`, source: node.id,
      target: nodes[(index + 1) % nodes.length].id,
    }));
    const bundle = buildContextBundle({
      ...input, kind: "context", targetIds: [nodes[0].id], nodes, relations, assessments: [],
    });
    expect(bundle.nodes).toHaveLength(3);
    expect(bundle.relations).toHaveLength(2);
    expect(bundle.authority).toBe("generated_noncanonical_bundle");
  });

  it("retains provenance ancestors outside the immediate semantic neighborhood", () => {
    const [first, second, third] = ontologyNodes;
    const source = { id: "source", label: "Observed source", provenance: ["record:source"] };
    const derivative = {
      id: "derivative", label: "Translation", provenance: ["transform:translation"],
      derivedFrom: ["source"],
    };
    const local = { ...assessments[0], subject: first.id, target: second.id, evidence: [derivative] };
    const remote = { ...assessments[0], id: "remote", subject: third.id, target: third.id, evidence: [source] };
    const bundle = buildContextBundle({
      ...input, kind: "context", targetIds: [first.id], relations: [], assessments: [local, remote],
    });
    expect(bundle.assessments).toEqual([local]);
    expect(bundle.provenanceEvidence).toEqual([source, derivative]);
    expect(serializeBundle(bundle, "markdown")).toContain("transform:translation");
    expect(serializeBundle(bundle, "markdown")).toContain("Derived from: source");
  });

  it("rejects cycles in evidence derivation without banning semantic cycles", () => {
    const evidence = [
      { id: "a", label: "A", provenance: ["record:a"], derivedFrom: ["b"] },
      { id: "b", label: "B", provenance: ["record:b"], derivedFrom: ["a"] },
    ];
    expect(() => buildContextBundle({
      ...input, assessments: [{ ...assessments[0], evidence }],
    })).toThrow("provenance cycle");
  });

  it("rejects dangling provenance and conflicting evidence identities", () => {
    const evidence = [{ id: "a", label: "A", provenance: ["record:a"], derivedFrom: ["missing"] }];
    expect(() => buildContextBundle({
      ...input, assessments: [{ ...assessments[0], evidence }],
    })).toThrow("Unknown provenance parent");
    expect(() => buildContextBundle({
      ...input, assessments: [{ ...assessments[0], evidence: [
        evidence[0], { ...evidence[0], provenance: ["different:source"] },
      ] }],
    })).toThrow("Conflicting evidence identity");
  });

  it("rejects missing context targets instead of exporting an empty authority wrapper", () => {
    expect(() => buildContextBundle({ ...input, kind: "context", targetIds: ["unknown"] }))
      .toThrow("Unknown context target");
  });
});
