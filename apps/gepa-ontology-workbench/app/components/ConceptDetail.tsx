"use client";

import { useState } from "react";
import type { Assessment, Invariant, OntologyNode, OntologyRelation, Quantity } from "../lib/ontology-types";

export interface ConceptDetailProps {
  node: OntologyNode;
  relations: readonly OntologyRelation[];
  assessments: readonly Assessment[];
  invariants: readonly Invariant[];
  nodes: readonly OntologyNode[];
  selectedRelation?: OntologyRelation;
}

const guardrailFor = (node: OntologyNode, invariants: readonly Invariant[]) => {
  const key = node.id === "failure:goal_fixation" ? "revisable-goals"
    : node.id === "failure:reward_hacking" || node.id === "op:reward_component" ? "metrics-are-proxies"
      : node.layer === "operational" ? "no-evidence-identity"
        : "runtime-evidence-authority";
  return invariants.find((invariant) => invariant.key === key) ?? invariants[0];
};

export function ConceptDetail({ node, relations, assessments, invariants, nodes, selectedRelation }: ConceptDetailProps) {
  const [copyMessage, setCopyMessage] = useState("");
  const relationGroups = new Map<string, OntologyRelation[]>();
  for (const relation of relations) {
    relationGroups.set(relation.family, [...(relationGroups.get(relation.family) ?? []), relation]);
  }
  const guardrail = guardrailFor(node, invariants);

  const copyId = async () => {
    try {
      if (!navigator.clipboard) throw new Error("Clipboard access is unavailable.");
      await navigator.clipboard.writeText(node.id);
      setCopyMessage("Canonical ID copied.");
    } catch {
      setCopyMessage("Could not copy the canonical ID.");
    }
  };

  return (
    <article className="concept-detail">
      <div className="detail-topline">
        <span className={`layer-label ${node.layer}`}>{node.layer}</span>
        <span>type: {node.type.replaceAll("_", " ")}</span>
        <span>{node.lifecycle}</span>
        {node.protected && <span className="protected-badge">Protected</span>}
      </div>
      <h2>{node.label}</h2>
      <div className="canonical-id">
        <code>{node.id}</code>
        <button type="button" onClick={copyId}>Copy ID</button>
      </div>
      {copyMessage && <p className="copy-message" role="status">{copyMessage}</p>}
      <p className="definition">{node.definition}</p>

      {selectedRelation && <section className="selected-relation-context" aria-label="Selected relation context">
        <p className="eyebrow">Selected relation context</p>
        <p>{nodes.find((candidate) => candidate.id === selectedRelation.source)?.label ?? selectedRelation.source}{" "}
          <code>{selectedRelation.predicate}</code>{" "}
          {nodes.find((candidate) => candidate.id === selectedRelation.target)?.label ?? selectedRelation.target}</p>
        <small>Primary concept remains {node.label}. Relation traversal does not replace the concept selection.</small>
      </section>}

      <DetailSection title="Assessment signal">
        {assessments.length === 0 ? <p className="muted">No direct assessment has been recorded.</p> : assessments.map((assessment) => (
          <div className="assessment" key={assessment.id}>
            <p>{assessment.id.replace("assessment:", "")}</p>
            <QuantityBar label="Support" quantity={assessment.support} tone="support" />
            <QuantityBar label="Opposition" quantity={assessment.opposition} tone="opposition" />
          </div>
        ))}
      </DetailSection>

      <DetailSection title="Direct relations">
        {relations.length === 0 ? <p className="muted">No direct relations.</p> : [...relationGroups].map(([family, group]) => (
          <div className="relation-group" key={family}>
            <p className="relation-family">{family.replaceAll("_", " ")}</p>
            {group.map((relation) => {
              const source = nodes.find((candidate) => candidate.id === relation.source)?.label ?? relation.source;
              const target = nodes.find((candidate) => candidate.id === relation.target)?.label ?? relation.target;
              return <p key={relation.id}><code>{source}</code> <span aria-hidden="true">-&gt;</span> <code>{relation.predicate}</code> <span aria-hidden="true">-&gt;</span> <code>{target}</code></p>;
            })}
          </div>
        ))}
      </DetailSection>

      <DetailSection title="Provenance">
        <ul className="provenance-list">
          {[...new Set([
            node.provenance,
            ...relations.map((relation) => relation.provenance),
            ...assessments.map((assessment) => assessment.provenance),
            ...assessments.flatMap((assessment) => assessment.evidence.map((evidence) => evidence.provenance)),
          ].flat())].map((source) => <li key={source}>{source}</li>)}
        </ul>
      </DetailSection>

      <DetailSection title="Maturity facets">
        <div className="maturity-grid">
          {Object.entries(node.maturity).map(([facet, available]) => (
            <div className={available ? "is-true" : "is-false"} key={facet} aria-label={`${facet.replaceAll("_", " ")}: ${available ? "true" : "false"}`}>
              <span aria-hidden="true">{available ? "yes" : "no"}</span>
              {facet.replaceAll("_", " ")}
              <span className="sr-only">: {available ? "true" : "false"}</span>
            </div>
          ))}
        </div>
      </DetailSection>

      <section className="guardrail" aria-labelledby="guardrail-title">
        <p className="eyebrow">Applicable invariant guardrail</p>
        <h3 id="guardrail-title">{guardrail.statement}</h3>
        <p>{guardrail.forbiddenInference}</p>
      </section>
    </article>
  );
}

function DetailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="detail-section"><h3>{title}</h3>{children}</section>;
}

function QuantityBar({ label, quantity, tone }: { label: string; quantity: Quantity; tone: "support" | "opposition" }) {
  const uncertainty = quantity.lower === undefined || quantity.upper === undefined
    ? "uncertainty not specified"
    : `uncertainty ${Math.round(quantity.lower * 100)}-${Math.round(quantity.upper * 100)}%`;
  if (quantity.estimate === undefined) {
    return (
      <div className={`quantity-bar ${tone} unavailable`}>
        <div><span>{label}</span><strong>Unavailable</strong></div>
        <small>{quantity.kind.replaceAll("_", " ")} - {uncertainty}</small>
      </div>
    );
  }
  const estimate = quantity.estimate;
  return (
    <div className={`quantity-bar ${tone}`}>
      <div><span>{label}</span><strong>{Math.round(estimate * 100)}%</strong></div>
      <div className="bar-track" aria-label={`${label}: ${Math.round(estimate * 100)}%, ${quantity.kind}, ${uncertainty}`}><span style={{ width: `${Math.max(0, Math.min(100, estimate * 100))}%` }} /></div>
      <small>{quantity.kind.replaceAll("_", " ")} - {uncertainty}</small>
    </div>
  );
}
