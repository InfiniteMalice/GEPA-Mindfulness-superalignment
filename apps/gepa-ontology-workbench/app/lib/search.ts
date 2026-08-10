import type { OntologyNode, OntologyRelation } from "./ontology-types";

const normalize = (value: string) => value.toLowerCase().replace(/[_:-]+/g, " ").replace(/\s+/g, " ").trim();

export function searchNodes(
  query: string,
  nodes: readonly OntologyNode[],
  relations: readonly OntologyRelation[],
): readonly OntologyNode[] {
  const needle = normalize(query);
  if (!needle) return nodes;

  const predicatesByNode = new Map<string, string[]>();
  for (const relation of relations) {
    for (const id of [relation.source, relation.target]) {
      predicatesByNode.set(id, [...(predicatesByNode.get(id) ?? []), relation.predicate]);
    }
  }

  return nodes
    .map((node, index) => {
      const fields = [node.label, node.id, node.type, node.definition, ...node.aliases, ...(predicatesByNode.get(node.id) ?? [])];
      const normalized = fields.map(normalize);
      const exact = normalized.some((field) => field === needle) ? 0 : 1;
      const prefix = normalized.some((field) => field.startsWith(needle)) ? 0 : 1;
      const contains = normalized.some((field) => field.includes(needle))
        || needle.split(" ").every((word) => normalized.some((field) => field.includes(word)));
      return { node, index, exact, prefix, contains };
    })
    .filter((item) => item.contains)
    .sort((a, b) => a.exact - b.exact || a.prefix - b.prefix || a.index - b.index)
    .map((item) => item.node);
}
