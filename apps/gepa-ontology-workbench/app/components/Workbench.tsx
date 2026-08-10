"use client";

import { useEffect, useState } from "react";
import {
  assessments,
  invariants,
  ONTOLOGY_HASH,
  ONTOLOGY_VERSION,
  ontologyNodes,
  ontologyRelations,
  verifyCanonicalOntologyHash,
} from "../data/ontology";
import { ConceptDetail } from "./ConceptDetail";
import { AssessView } from "./AssessView";
import { ExploreView } from "./ExploreView";
import { ImproveView } from "./ImproveView";
import { InvariantsView } from "./InvariantsView";

type WorkbenchTab = "explore" | "assess" | "improve" | "invariants";

const tabs: readonly { id: WorkbenchTab; label: string }[] = [
  { id: "explore", label: "Explore" },
  { id: "assess", label: "Assess" },
  { id: "improve", label: "Improve" },
  { id: "invariants", label: "Invariants" },
];

export function Workbench() {
  const [tab, setTab] = useState<WorkbenchTab>("explore");
  const [selectedId, setSelectedId] = useState("failure:goal_fixation");
  const [query, setQuery] = useState("");
  const [selectedRelationId, setSelectedRelationId] = useState<string | null>(null);
  const [requestedInvariantKey, setRequestedInvariantKey] = useState<string | null>(null);
  const [canonicalDigestStatus, setCanonicalDigestStatus] = useState<"checking" | "verified" | "unverified">("checking");
  const selectedNode = ontologyNodes.find((node) => node.id === selectedId) ?? ontologyNodes[0];
  const directRelations = ontologyRelations.filter(
    (relation) => relation.source === selectedNode.id || relation.target === selectedNode.id,
  );
  const matchingAssessments = assessments.filter(
    (assessment) => assessment.subject === selectedNode.id || assessment.target === selectedNode.id,
  );
  const selectedRelation = ontologyRelations.find((relation) => relation.id === selectedRelationId);

  useEffect(() => {
    let mounted = true;
    verifyCanonicalOntologyHash()
      .then((verified) => {
        if (mounted) setCanonicalDigestStatus(verified ? "verified" : "unverified");
      })
      .catch(() => {
        if (mounted) setCanonicalDigestStatus("unverified");
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <main className="workbench-shell">
      <header className="workbench-header">
        <a className="gepa-mark" href="#workbench" aria-label="GEPA Ontology Workbench home">
          <span aria-hidden="true">G/</span>
          <span>GEPA <em>Mindfulness</em></span>
        </a>
        <div className="ontology-version" title={ONTOLOGY_HASH}>Ontology v{ONTOLOGY_VERSION}</div>
        <label className="global-search">
          <span className="sr-only">Search the ontology</span>
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search concepts, aliases, relations…"
          />
        </label>
        <p
          className={`canonical-status ${canonicalDigestStatus}`}
          role="status"
          aria-label="Canonical ontology digest status"
        >
          <span aria-hidden="true">●</span>{" "}
          {canonicalDigestStatus === "verified"
            ? "Verified canonical"
            : canonicalDigestStatus === "checking"
              ? "Verifying canonical digest"
              : "Canonical digest unverified"}
        </p>
      </header>

      <nav className="workbench-tabs" aria-label="Workbench modes">
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={tab === item.id}
            className={tab === item.id ? "is-active" : undefined}
            onClick={() => { setTab(item.id); if (item.id !== "invariants") setRequestedInvariantKey(null); }}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <section id="workbench" className="workbench-content" aria-label={`${tab} workbench`}>
        <div className="workbench-main">
          {tab === "explore" && <ExploreView
            query={query}
            selectedId={selectedNode.id}
            selectedRelationId={selectedRelationId}
            onSelect={(id) => { setSelectedId(id); setSelectedRelationId(null); }}
            onRelationSelect={setSelectedRelationId}
            onClearSearch={() => setQuery("")}
          />}
          {tab === "assess" && <AssessView assessments={assessments} nodes={ontologyNodes} />}
          {tab === "improve" && <ImproveView
            selectedId={selectedNode.id}
            nodes={ontologyNodes}
            relations={ontologyRelations}
            assessments={assessments}
            invariants={invariants}
            canonicalDigestStatus={canonicalDigestStatus}
            onOpenInvariant={(key) => { setRequestedInvariantKey(key); setTab("invariants"); }}
          />}
          {tab === "invariants" && <InvariantsView key={requestedInvariantKey ?? "all"} invariants={invariants} requestedInvariantKey={requestedInvariantKey} />}
        </div>
        <aside className="concept-detail-panel" aria-label="Selected concept detail">
          <ConceptDetail
            node={selectedNode}
            relations={directRelations}
            assessments={matchingAssessments}
            invariants={invariants}
            nodes={ontologyNodes}
            selectedRelation={selectedRelation}
          />
        </aside>
      </section>
      <ImprovementCycle />
    </main>
  );
}

function ImprovementCycle() {
  const steps = [
    ["Select", "bounded target"],
    ["Observe", "evidence + provenance"],
    ["Validate", "22 invariants"],
    ["Propose", "never auto-merge"],
    ["Export", "JSON/YAML/Markdown"],
  ] as const;
  return <section className="improvement-cycle" aria-labelledby="cycle-heading"><p className="eyebrow">Improvement cycle</p><h2 id="cycle-heading" className="sr-only">Governed improvement cycle</h2><ol>{steps.map(([step, boundary]) => <li key={step}><strong>{step}</strong><span>{boundary}</span></li>)}</ol></section>;
}
