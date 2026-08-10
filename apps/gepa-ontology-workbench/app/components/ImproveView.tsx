"use client";

import { useMemo, useState } from "react";
import { ONTOLOGY_HASH, ONTOLOGY_VERSION, trainingFixtures } from "../data/ontology";
import { buildContextBundle, serializeBundle } from "../lib/bundles";
import type {
  Assessment,
  EpistemicStatus,
  Invariant,
  Layer,
  OntologyNode,
  OntologyRelation,
  Proposal,
  RelationFamily,
  ValidationIssue,
} from "../lib/ontology-types";
import { validateProposal } from "../lib/proposal-validation";

type BundleKind = "context" | "training";
type BundleFormat = "json" | "yaml" | "markdown";

export interface ImproveViewProps {
  selectedId: string;
  nodes: readonly OntologyNode[];
  relations: readonly OntologyRelation[];
  assessments: readonly Assessment[];
  invariants: readonly Invariant[];
  onOpenInvariant: (key: string) => void;
}

const initialProposal: Proposal = {
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

const formatExtension: Record<BundleFormat, string> = { json: "json", yaml: "yaml", markdown: "md" };

export function ImproveView({ selectedId, nodes, relations, assessments, invariants, onOpenInvariant }: ImproveViewProps) {
  const [proposal, setProposal] = useState<Proposal>(initialProposal);
  const [issues, setIssues] = useState<readonly ValidationIssue[] | null>(null);
  const [bundleKind, setBundleKind] = useState<BundleKind>("context");
  const [format, setFormat] = useState<BundleFormat>("json");
  const [preview, setPreview] = useState("");
  const [status, setStatus] = useState("");
  const blockers = useMemo(() => (issues ?? []).filter((issue) => issue.severity === "blocker"), [issues]);
  const warnings = useMemo(() => (issues ?? []).filter((issue) => issue.severity === "warning"), [issues]);
  const relatedNodeLabels = useMemo(() => new Map(nodes.map((node) => [node.id, node.label])), [nodes]);
  const selectedNode = nodes.find((node) => node.id === selectedId);
  const hasCuratedTraining = trainingFixtures.some((fixture) => fixture.targetId === selectedId);
  const validationAllowsExport = issues !== null && blockers.length === 0;
  const canGenerate = validationAllowsExport && (bundleKind === "context" || hasCuratedTraining);

  const update = <K extends keyof Proposal>(field: K, value: Proposal[K]) => {
    setProposal((current) => ({ ...current, [field]: value }));
    setIssues(null);
    setPreview("");
    setStatus("");
  };

  const runChecks = () => {
    setIssues(validateProposal(proposal, nodes, relations));
    setPreview("");
    setStatus("");
  };

  const generateBundle = () => {
    try {
      const generatedAt = new Date().toISOString();
      const bundle = buildContextBundle({
        kind: bundleKind,
        targetIds: [selectedId],
        generatedAt,
        nodes,
        relations,
        assessments,
        invariants,
        ontologyVersion: ONTOLOGY_VERSION,
        ontologyHash: ONTOLOGY_HASH,
        trainingFixtures,
      });
      setPreview(serializeBundle(bundle, format));
      setStatus("Governed bundle generated.");
    } catch (error) {
      setPreview("");
      setStatus(`Could not generate the governed bundle: ${error instanceof Error ? error.message : "Unexpected generation error."}`);
    }
  };

  const copyBundle = async () => {
    try {
      if (!preview || !navigator.clipboard) throw new Error("Clipboard access is unavailable.");
      await navigator.clipboard.writeText(preview);
      setStatus("Bundle copied to clipboard.");
    } catch {
      setStatus("Could not copy the bundle.");
    }
  };

  const downloadBundle = () => {
    if (!preview) return;
    let objectUrl: string | undefined;

    try {
      const blob = new Blob([preview], { type: format === "json" ? "application/json" : "text/plain;charset=utf-8" });
      objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `gepa-governed-bundle.${formatExtension[format]}`;
      link.click();
      setStatus("Bundle download started.");
    } catch {
      setStatus("Could not download the bundle.");
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    }
  };

  return (
    <section className="improve-view" aria-labelledby="improve-heading">
      <header className="mode-intro">
        <p className="eyebrow">Governed proposal</p>
        <h1 id="improve-heading">Propose before you export</h1>
        <p>The canonical ontology remains immutable here. This form tests a proposal and exports bounded context around the selected target.</p>
      </header>
      <form className="proposal-form" onSubmit={(event) => { event.preventDefault(); runChecks(); }}>
        <fieldset>
          <legend>Proposal details</legend>
          <FormField label="Canonical ID" value={proposal.id} onChange={(value) => update("id", value)} />
          <FormField label="Label" value={proposal.label} onChange={(value) => update("label", value)} />
          <label>Layer<select value={proposal.layer} onChange={(event) => update("layer", event.target.value as Layer)}><option value="operational">Operational</option><option value="normative">Normative</option></select></label>
          <FormField label="Type" value={proposal.type} onChange={(value) => update("type", value)} />
          <label>Lifecycle<input value={proposal.lifecycle} readOnly aria-readonly="true" /></label>
          <label>Epistemic status<select value={proposal.epistemicStatus} onChange={(event) => update("epistemicStatus", event.target.value as EpistemicStatus)}><option value="defined">Defined</option><option value="theoretical">Theoretical</option><option value="hypothesized">Hypothesized</option><option value="observationally_supported">Observationally supported</option><option value="experimentally_supported">Experimentally supported</option><option value="contested">Contested</option></select></label>
          <label>Governance classification<select value={proposal.governanceClassification} onChange={(event) => update("governanceClassification", event.target.value as Proposal["governanceClassification"])}><option value="ordinary">Ordinary proposal</option><option value="explicit_normative_revision">Explicit normative revision</option></select></label>
          <FormField label="Qualitative uncertainty" value={proposal.uncertainty?.qualitative ?? ""} onChange={(value) => update("uncertainty", { kind: proposal.uncertainty?.kind ?? "confidence", qualitative: value || undefined })} />
          <label className="wide-field">Definition<textarea value={proposal.definition} onChange={(event) => update("definition", event.target.value)} rows={4} /></label>
          <label className="wide-field">Provenance<textarea value={proposal.provenance} onChange={(event) => update("provenance", event.target.value)} rows={2} /></label>
          <FormField label="Relation predicate (optional)" value={proposal.relationPredicate ?? ""} onChange={(value) => update("relationPredicate", value || undefined)} />
          <FormField label="Relation target (optional)" value={proposal.relationTarget ?? ""} onChange={(value) => update("relationTarget", value || undefined)} />
          <label>Relation family (optional)<select value={proposal.relationFamily ?? ""} onChange={(event) => update("relationFamily", (event.target.value || undefined) as RelationFamily | undefined)}><option value="">No relation family</option><option value="structural">Structural</option><option value="normative">Normative</option><option value="evidential">Evidential</option><option value="causal_risk">Causal/risk</option></select></label>
          <FormField label="Evidence dependency group (optional)" value={proposal.dependencyGroup ?? ""} onChange={(value) => update("dependencyGroup", value || undefined)} />
          <FormField label="Underlying continuous profile (optional)" value={proposal.underlyingProfile ?? ""} onChange={(value) => update("underlyingProfile", value || undefined)} />
          <FormField label="Operational mapping" value={proposal.operationalMapping ?? ""} onChange={(value) => update("operationalMapping", value || undefined)} />
          <FormField label="Maturity evidence" value={proposal.maturityEvidence ?? ""} onChange={(value) => update("maturityEvidence", value || undefined)} />
        </fieldset>
        <button className="primary-action" type="submit">Run semantic checks</button>
      </form>

      {issues !== null && <section className="validation-results" aria-live="polite" aria-labelledby="validation-heading">
        <h2 id="validation-heading">Semantic checks</h2>
        {issues.length === 0 ? <p className="validation-success">No blockers or warnings found for this proposal.</p> : <>
          <IssueList title="Blockers" issues={blockers} relatedNodeLabels={relatedNodeLabels} onOpenInvariant={onOpenInvariant} />
          <IssueList title="Warnings" issues={warnings} relatedNodeLabels={relatedNodeLabels} onOpenInvariant={onOpenInvariant} />
        </>}
      </section>}

      <section className="bundle-panel" aria-labelledby="bundle-heading">
        <div className="bundle-heading"><div><p className="eyebrow">Export boundary</p><h2 id="bundle-heading">Governed bundle <span className="noncanonical-badge">Generated · noncanonical</span></h2></div><p>Selected target: <code>{selectedId}</code></p></div>
        <div className="bundle-controls">
          <label>Bundle purpose<select value={bundleKind} onChange={(event) => { setBundleKind(event.target.value as BundleKind); setPreview(""); setStatus(""); }}><option value="context">Context</option><option value="training">Training</option></select></label>
          <label>Format<select value={format} onChange={(event) => { setFormat(event.target.value as BundleFormat); setPreview(""); setStatus(""); }}><option value="json">JSON</option><option value="yaml">YAML</option><option value="markdown">Markdown</option></select></label>
          <button className="primary-action" type="button" disabled={!canGenerate} onClick={generateBundle}>Generate governed bundle</button>
        </div>
        {!canGenerate && <p className="bundle-gate" role="status">
          {validationAllowsExport && bundleKind === "training" && !hasCuratedTraining
            ? `Training export is unavailable because ${selectedNode?.label ?? selectedId} has no curated behavior examples.`
            : "Run semantic checks with no blockers before generating a bundle."}
        </p>}
        <p className="bundle-disclaimer">This bundle may inform evaluation or training. It does not establish learned or deployed behavior.</p>
        {preview && <div className="bundle-preview"><label>Read-only bundle preview<textarea readOnly value={preview} rows={14} /></label><div className="bundle-actions"><button type="button" onClick={copyBundle}>Copy bundle</button><button type="button" onClick={downloadBundle}>Download bundle</button></div></div>}
      </section>
      {status && <p className="bundle-status" role={status.startsWith("Could not") ? "alert" : "status"}>{status}</p>}
    </section>
  );
}

function FormField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label>{label}<input value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function IssueList({ title, issues, relatedNodeLabels, onOpenInvariant }: { title: string; issues: readonly ValidationIssue[]; relatedNodeLabels: ReadonlyMap<string, string>; onOpenInvariant: (key: string) => void }) {
  if (issues.length === 0) return null;
  return <section className={`issue-list ${title.toLowerCase()}`} aria-label={title}><h3>{title}</h3><ul>{issues.map((issue) => {
    return <li key={issue.code}><strong>{issue.message}</strong>{issue.relatedNodeIds.length > 0 && <p>Related existing nodes: {issue.relatedNodeIds.map((id) => relatedNodeLabels.get(id) ?? id).join(", ")}</p>}<p>Relevant invariants: {issue.invariantKeys.length === 0 ? "No invariant reference" : issue.invariantKeys.map((key) => <button className="invariant-reference" key={key} type="button" onClick={() => onOpenInvariant(key)}>{key}</button>)}</p></li>;
  })}</ul></section>;
}
