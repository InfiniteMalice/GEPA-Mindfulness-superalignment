import { describe, expect, it } from "vitest";
import { ontologyNodes, ontologyRelations } from "../data/ontology";
import { searchNodes } from "./search";

describe("searchNodes", () => {
  it("finds labels, IDs, aliases, types, and connected predicates", () => {
    expect(searchNodes("goal fixation", ontologyNodes, ontologyRelations)[0].id).toBe("failure:goal_fixation");
    expect(searchNodes("failure:goal_fixation", ontologyNodes, ontologyRelations)[0].id).toBe("failure:goal_fixation");
    expect(searchNodes("increase knowledge", ontologyNodes, ontologyRelations)[0].id).toBe("norm:increase_understanding");
    expect(searchNodes("evaluates", ontologyNodes, ontologyRelations).some((node) => node.id === "op:goal_flexibility_evaluator")).toBe(true);
  });

  it("returns the canonical order for an empty query", () => {
    expect(searchNodes("", ontologyNodes, ontologyRelations)).toEqual(ontologyNodes);
  });
});
