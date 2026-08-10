"use client";

import { useMemo, useState } from "react";
import type { Invariant } from "../lib/ontology-types";

export interface InvariantsViewProps {
  invariants: readonly Invariant[];
  requestedInvariantKey?: string | null;
}

const kernelInvariantKeys = new Set([
  "coequal-imperatives",
  "no-evidence-identity",
  "runtime-evidence-authority",
  "corrigible-governance",
]);

export function InvariantsView({ invariants, requestedInvariantKey }: InvariantsViewProps) {
  const [query, setQuery] = useState(() => requestedInvariantKey ?? "");
  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return invariants;
    return invariants.filter((invariant) => [
      invariant.order,
      invariant.key,
      invariant.statement,
      invariant.explanation,
      invariant.forbiddenInference,
    ].join(" ").toLowerCase().includes(needle));
  }, [invariants, query]);

  return (
    <section className="invariants-view" aria-labelledby="invariants-heading">
      <header className="mode-intro">
        <p className="eyebrow">Governance review</p>
        <h1 id="invariants-heading">The 22 invariants</h1>
        <p>Every proposed change remains legible against the canonical governance boundary.</p>
      </header>
      <label className="invariant-search">
        <span>Search invariants</span>
        <input autoFocus={Boolean(requestedInvariantKey)} type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Number, key, statement, explanation, or forbidden inference" />
      </label>
      {results.length === 0 ? (
        <div className="empty-results" role="status">
          <p>No invariants match “{query}”.</p>
          <button type="button" onClick={() => setQuery("")}>Clear search</button>
        </div>
      ) : (
        <ol className="invariant-list" start={1} aria-label={`${results.length} matching invariants`}>
          {results.map((invariant) => (
            <li id={`invariant-${invariant.key}`} key={invariant.key} value={invariant.order}>
              <article className="invariant-card">
                <header>
                  <p>{invariant.order}. <code>{invariant.key}</code></p>
                  {kernelInvariantKeys.has(invariant.key) && <span className="kernel-badge">Kernel</span>}
                </header>
                <h2>{invariant.statement}</h2>
                <p>{invariant.explanation}</p>
                <p className="forbidden-inference"><strong>Forbidden inference</strong>{invariant.forbiddenInference}</p>
              </article>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
