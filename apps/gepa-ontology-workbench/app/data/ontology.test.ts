import { describe, expect, it } from "vitest";
import * as ontology from "./ontology";
import { RELATION_REGISTRY } from "./ontology-registry";
import {
  assessments,
  invariants,
  ontologyNodes,
  ontologyRelations,
  ONTOLOGY_HASH,
  ONTOLOGY_VERSION,
} from "./ontology";

describe("canonical ontology reference", () => {
  it("contains a verified content-derived SHA-256 and all 22 invariants", async () => {
    expect(ONTOLOGY_VERSION).toBe("1.0.0");
    expect(ONTOLOGY_HASH).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(invariants).toHaveLength(22);
    expect(new Set(invariants.map((item) => item.key)).size).toBe(22);

    const hashApi = ontology as unknown as {
      canonicalOntologyPayload?: unknown;
      hashCanonicalPayload?: (payload: unknown) => Promise<string>;
      verifyCanonicalOntologyHash?: () => Promise<boolean>;
    };
    expect(hashApi.hashCanonicalPayload).toBeTypeOf("function");
    expect(hashApi.verifyCanonicalOntologyHash).toBeTypeOf("function");
    if (hashApi.hashCanonicalPayload && hashApi.verifyCanonicalOntologyHash) {
      expect(await hashApi.hashCanonicalPayload(hashApi.canonicalOntologyPayload)).toBe(ONTOLOGY_HASH);
      expect(await hashApi.verifyCanonicalOntologyHash()).toBe(true);
      expect(await hashApi.hashCanonicalPayload({
        ...(hashApi.canonicalOntologyPayload as Record<string, unknown>),
        version: "1.0.1-semantic-change",
      })).not.toBe(ONTOLOGY_HASH);
    }
  });

  it("faithfully preserves section 39 invariant keys and statements", () => {
    expect(invariants.map((item) => item.key)).toEqual([
      "coequal-imperatives", "revisable-goals", "metrics-are-proxies", "no-evidence-identity",
      "independent-support-opposition", "contradiction-is-information", "uncertainty-persists", "no-manufactured-certainty",
      "correlated-evidence", "absence-not-absence", "delayed-discretization", "retained-profiles",
      "named-scalarization", "partial-ordering", "contextual-proxy-reliability", "training-intent-not-success",
      "training-success-not-deployment", "runtime-enforcement-not-virtue", "normative-contradiction-not-inconsistency",
      "historical-reproducibility", "runtime-evidence-authority", "corrigible-governance",
    ]);
    expect(invariants.map((item) => item.statement)).toEqual([
      "The three imperatives are co-equal and mutually constraining.",
      "Goals are revisable instruments rather than sacred endpoints.",
      "Metrics and rewards are evidence/proxies, not the values themselves.",
      "Operational evidence never automatically establishes normative identity.",
      "Support and opposition remain independent where possible.",
      "Contradiction is information rather than an automatic error.",
      "Uncertainty must not silently disappear during inference.",
      "Inference must not manufacture certainty.",
      "Correlated evidence must not be counted as independent evidence.",
      "Absence of detected failure does not establish absence of failure.",
      "Continuous information should not be discretized earlier than necessary.",
      "Derived categories retain their underlying profiles.",
      "Scalarization must name the policy that performs it.",
      "Partial ordering is preferable to unjustified total ranking.",
      "Proxy reliability is context dependent and may degrade under optimization or distribution shift.",
      "Training intent does not establish training success.",
      "Training success does not establish deployment behavior.",
      "Runtime enforcement does not establish internalized virtue.",
      "Normative contradiction does not imply logical inconsistency.",
      "Historical semantic states remain reproducible through ontology version and hash.",
      "Runtime evidence may update assessments but cannot redefine canonical ontology authority.",
      "Ontology governance itself remains corrigible.",
    ]);
  });

  it("keeps relation endpoints resolvable", () => {
    const ids = new Set(ontologyNodes.map((node) => node.id));
    for (const relation of ontologyRelations) {
      expect(ids.has(relation.source)).toBe(true);
      expect(ids.has(relation.target)).toBe(true);
    }
  });

  it("keeps every canonical relation aligned with the explicit registry", () => {
    for (const relation of ontologyRelations) {
      const definition = RELATION_REGISTRY[relation.predicate as keyof typeof RELATION_REGISTRY];
      expect(definition, `${relation.predicate} must be registered`).toBeDefined();
      expect(relation.family, `${relation.id} must use its registered family`).toBe(definition.family);
    }
  });

  it("marks exactly the protected normative kernel from the source ontology", () => {
    const expectedProtectedIds = [
      "norm:autonomy",
      "norm:consent",
      "norm:constraint_respect",
      "norm:corrigibility",
      "norm:goals_as_provisional",
      "norm:increase_prosperity",
      "norm:increase_understanding",
      "norm:non_deception",
      "norm:oversight",
      "norm:principle_of_least_action",
      "norm:reality_contact",
      "norm:reduce_suffering",
    ];
    const protectedIds = ontologyNodes
      .filter((node) => node.protected)
      .map((node) => node.id)
      .sort();

    expect(protectedIds).toEqual(expectedProtectedIds);
    expect((ontology as { PROTECTED_NORMATIVE_KERNEL_IDS?: readonly string[] }).PROTECTED_NORMATIVE_KERNEL_IDS)
      .toEqual(expect.arrayContaining(expectedProtectedIds));
  });

  it("retains the legacy Increase Knowledge terminology as a canonical alias", () => {
    expect(ontologyNodes.find((node) => node.id === "norm:increase_understanding")?.aliases).toContain("Increase Knowledge");
  });

  it("deeply freezes canonical reference data at runtime", () => {
    expect(Object.isFrozen(ontologyNodes)).toBe(true);
    expect(Object.isFrozen(ontologyNodes[0])).toBe(true);
    expect(Object.isFrozen(ontologyNodes[0].aliases)).toBe(true);
    expect(Object.isFrozen(ontologyNodes[0].maturity)).toBe(true);
    expect(Object.isFrozen(ontologyRelations)).toBe(true);
    expect(Object.isFrozen(ontologyRelations[0].provenance)).toBe(true);
    expect(Object.isFrozen(assessments)).toBe(true);
    expect(Object.isFrozen(assessments[0].support)).toBe(true);
    expect(Object.isFrozen(invariants)).toBe(true);
    expect(Object.isFrozen(invariants[0])).toBe(true);

    expect(() => {
      (ontologyNodes[0].maturity as { specified: boolean }).specified = false;
    }).toThrow(TypeError);
    expect(() => {
      (invariants as unknown as { push: (value: unknown) => void }).push({});
    }).toThrow(TypeError);
  });
});
