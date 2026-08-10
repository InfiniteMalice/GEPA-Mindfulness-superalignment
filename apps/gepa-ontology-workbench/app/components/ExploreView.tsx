"use client";

import { useMemo, useState } from "react";
import { ontologyNodes, ontologyRelations } from "../data/ontology";
import { searchNodes } from "../lib/search";

export interface ExploreViewProps {
  query: string;
  selectedId: string;
  selectedRelationId?: string | null;
  onSelect: (id: string) => void;
  onRelationSelect?: (id: string) => void;
  onClearSearch?: () => void;
}

type LayerFilter = "all" | "normative" | "operational";

export function ExploreView({
  query,
  selectedId,
  selectedRelationId,
  onSelect,
  onRelationSelect,
  onClearSearch,
}: ExploreViewProps) {
  const [layer, setLayer] = useState<LayerFilter>("all");
  const [predicate, setPredicate] = useState("all");
  const predicates = useMemo(() => [...new Set(ontologyRelations.map((relation) => relation.predicate))].sort(), []);
  const matchingRelations = useMemo(
    () => ontologyRelations.filter((relation) => predicate === "all" || relation.predicate === predicate),
    [predicate],
  );
  const connectedIds = useMemo(
    () => new Set(matchingRelations.flatMap((relation) => [relation.source, relation.target])),
    [matchingRelations],
  );
  const results = useMemo(
    () => searchNodes(query, ontologyNodes, ontologyRelations).filter((node) => (
      (layer === "all" || node.layer === layer)
      && (predicate === "all" || connectedIds.has(node.id))
    )),
    [connectedIds, layer, predicate, query],
  );
  const normative = results.filter((node) => node.layer === "normative");
  const operational = results.filter((node) => node.layer === "operational");

  return (
    <section className="explore-view" aria-labelledby="explore-heading">
      <div className="explore-intro">
        <p className="eyebrow">Ontology explorer</p>
        <h1 id="explore-heading">What ought to hold <span aria-hidden="true">→</span> what is observed</h1>
        <p>Trace the boundary between protected commitments and operational evidence without collapsing one into the other.</p>
      </div>

      <div className="explore-controls">
        <div className="filter-group" aria-label="Filter concepts by layer">
          {(["all", "normative", "operational"] as const).map((option) => (
            <button
              type="button"
              key={option}
              aria-pressed={layer === option}
              className={layer === option ? "is-active" : undefined}
              onClick={() => setLayer(option)}
            >
              {option}
            </button>
          ))}
        </div>
        <label className="relation-filter">
          <span>Relation predicate</span>
          <select value={predicate} onChange={(event) => setPredicate(event.target.value)}>
            <option value="all">All predicates</option>
            {predicates.map((option) => <option value={option} key={option}>{option}</option>)}
          </select>
        </label>
        <p>{results.length} concepts in view</p>
      </div>

      {results.length === 0 ? (
        <div className="empty-results" role="status">
          <p>No concepts match “{query}”.</p>
          <button type="button" onClick={onClearSearch}>Clear search</button>
        </div>
      ) : (
        <div className="ontology-map">
          <ConceptColumn label="Normative" description="What ought to hold" nodes={normative} selectedId={selectedId} onSelect={onSelect} />
          <div className="evidence-bridge">
            <p>Evidence informs.<br /><strong>It does not define.</strong></p>
            <div className="relation-chips" aria-label="Relation traversal">
              {matchingRelations.map((relation) => {
                const source = ontologyNodes.find((node) => node.id === relation.source)?.label ?? relation.source;
                const target = ontologyNodes.find((node) => node.id === relation.target)?.label ?? relation.target;
                const label = `${source} ${relation.predicate} ${target}`;
                return <button
                  type="button"
                  key={relation.id}
                  aria-label={`Select relation ${label}`}
                  aria-pressed={selectedRelationId === relation.id}
                  className={selectedRelationId === relation.id ? "is-selected" : undefined}
                  onClick={() => onRelationSelect?.(relation.id)}
                >{label}</button>;
              })}
            </div>
          </div>
          <ConceptColumn label="Operational" description="What is observed and tested" nodes={operational} selectedId={selectedId} onSelect={onSelect} />
        </div>
      )}
    </section>
  );
}

function ConceptColumn({ label, description, nodes, selectedId, onSelect }: {
  label: "Normative" | "Operational";
  description: string;
  nodes: readonly (typeof ontologyNodes)[number][];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const layerClass = label.toLowerCase();
  return (
    <section className={`concept-column ${layerClass}`} aria-label={`${label} concepts`}>
      <header>
        <p className="layer-label">{label}</p>
        <p>{description}</p>
      </header>
      <div className="concept-list">
        {nodes.map((node) => (
          <button
            type="button"
            key={node.id}
            aria-current={selectedId === node.id ? "true" : undefined}
            className={selectedId === node.id ? "is-selected" : undefined}
            onClick={() => onSelect(node.id)}
          >
            <span>{node.label}</span>
            <small>{node.type.replaceAll("_", " ")}</small>
          </button>
        ))}
      </div>
    </section>
  );
}
