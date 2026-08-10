"use client";

import type { Assessment, MaturityFacets, OntologyNode, Quantity } from "../lib/ontology-types";

export interface AssessViewProps {
  assessments: readonly Assessment[];
  nodes: readonly OntologyNode[];
}

const labelFor = (id: string, nodes: readonly OntologyNode[]) => nodes.find((node) => node.id === id)?.label ?? id;
const percentage = (value: number) => `${Math.round(value * 100)}%`;
const maturityLabel = (facet: keyof MaturityFacets) => facet.replaceAll("_", " ");

const quantityPresentation = (quantity: Quantity) => {
  if (quantity.estimate !== undefined) return percentage(quantity.estimate);
  if (quantity.lower !== undefined && quantity.upper !== undefined) return `${percentage(quantity.lower)}–${percentage(quantity.upper)}`;
  if (quantity.qualitative?.trim()) return quantity.qualitative;
  return "Unavailable";
};

const quantityWarnings = (quantity: Quantity) => {
  const warnings: string[] = [];
  const hasRange = quantity.lower !== undefined && quantity.upper !== undefined;
  if ((!hasRange && quantity.estimate === undefined && !quantity.qualitative?.trim()) || /uncalibrated/i.test(quantity.qualitative ?? "")) {
    warnings.push("Evidence quality warning: quantity is unavailable or uncalibrated.");
  }
  if (quantity.estimate !== undefined && !hasRange && !quantity.qualitative?.trim()) {
    warnings.push("Point estimate has no interval or qualitative uncertainty.");
  }
  return warnings;
};

export function AssessView({ assessments, nodes }: AssessViewProps) {
  return (
    <section className="assess-view" aria-labelledby="assess-heading">
      <header className="mode-intro">
        <p className="eyebrow">Evidence review</p>
        <h1 id="assess-heading">Assess without collapsing disagreement</h1>
        <p>Support, opposition, and contradiction remain distinct evidence dimensions.</p>
      </header>
      <div className="assessment-cards">
        {assessments.map((assessment) => {
          const subject = nodes.find((node) => node.id === assessment.subject);
          const target = nodes.find((node) => node.id === assessment.target);
          const evidenceByDependency = assessment.evidence.reduce<Map<string, typeof assessment.evidence>>((groups, evidence) => {
            if (!evidence.dependencyGroup) return groups;
            groups.set(evidence.dependencyGroup, [...(groups.get(evidence.dependencyGroup) ?? []), evidence]);
            return groups;
          }, new Map());
          const correlatedEvidence = [...evidenceByDependency]
            .filter(([, evidence]) => evidence.length > 1)
            .map(([dependencyGroup, evidence]) => `${evidence.map((item) => item.label).join(" and ")} share dependency group ${dependencyGroup}. Correlated evidence is not independent confirmation.`);
          const qualityWarnings = [...quantityWarnings(assessment.support), ...quantityWarnings(assessment.opposition)];
          const maturityGaps = [subject, target].flatMap((node) => node
            ? (Object.entries(node.maturity) as [keyof MaturityFacets, boolean][])
              .filter(([, available]) => !available)
              .map(([facet]) => `Maturity gap: ${node.label} — ${maturityLabel(facet)} is false.`)
            : []);
          return (
            <article className="assessment-card" key={assessment.id} aria-labelledby={`${assessment.id}-heading`}>
              <p className="assessment-id">{assessment.id.replace("assessment:", "")}</p>
              <h2 id={`${assessment.id}-heading`}>{labelFor(assessment.subject, nodes)} <span aria-hidden="true">→</span> {labelFor(assessment.target, nodes)}</h2>
              <div className="evidence-quality-warnings" aria-label="Evidence quality warnings">
                {correlatedEvidence.map((warning) => <p className="evidence-warning" role="note" key={warning}>{warning}</p>)}
                {assessment.provenance.length === 0 && <p className="evidence-warning" role="note">Missing provenance.</p>}
                {qualityWarnings.map((warning) => <p className="evidence-warning" role="note" key={warning}>{warning}</p>)}
                {maturityGaps.map((warning) => <p className="evidence-warning" role="note" key={warning}>{warning}</p>)}
              </div>
              <div className="assessment-quantities">
                <QuantitySummary label="Support" quantity={assessment.support} tone="support" />
                <QuantitySummary label="Opposition" quantity={assessment.opposition} tone="opposition" />
              </div>
              <dl className="assessment-facts">
                <div><dt>Computed contradiction</dt><dd>Contradiction: {percentage(assessment.contradiction)}</dd></div>
                <div><dt>Dependency groups</dt><dd>{[...new Set(assessment.evidence.map((evidence) => evidence.dependencyGroup).filter(Boolean))].join("; ") || "No dependency group recorded."}</dd></div>
                <div><dt>Unresolved tension</dt><dd>{assessment.contradiction > 0 ? "Present — retain both support and opposition for review." : "None recorded."}</dd></div>
                <div><dt>Scalarization policy</dt><dd>{assessment.scalarizationPolicy ?? "No scalarization applied."}</dd></div>
                <div><dt>Provenance</dt><dd>{assessment.provenance.length > 0 ? assessment.provenance.join("; ") : "Missing provenance."}</dd></div>
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function QuantitySummary({ label, quantity, tone }: { label: string; quantity: Quantity; tone: "support" | "opposition" }) {
  const interval = quantity.lower === undefined || quantity.upper === undefined
    ? quantity.qualitative?.trim() || "Uncertainty interval not specified"
    : `${percentage(quantity.lower)}–${percentage(quantity.upper)}`;
  const presentation = quantityPresentation(quantity);
  return (
    <section className={`assessment-quantity ${tone}`} aria-label={`${label}: ${presentation}, ${quantity.kind}`}>
      <div><h3>{label}</h3><strong>{presentation}</strong></div>
      <p>{quantity.kind.replaceAll("_", " ")} · uncertainty {interval}</p>
    </section>
  );
}
