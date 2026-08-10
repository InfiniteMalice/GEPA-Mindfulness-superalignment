# GEPA Ontology Governed Workbench Final Fix Report

Date: 2026-08-10

Branch: `feature/gepa-ontology-workbench-pr`

Status: Complete; source worktree preserved locally. A separate private Sites deployment was
performed after verification; no public publishing or access-policy change was made.

## Governing inputs

The fix wave was reviewed against:

- `history/gepa-ontology-workbench/design.md`
- the attached GEPA Mindfulness Ontology source
- the unresolved Codex and CodeRabbit review threads on PR #750
- the repository quality gate and documentation precision checks

The approved design and ontology source governed any ambiguity in the implementation plan.

## Findings resolved

1. **Real canonical SHA-256:** Added recursive key normalization with locale-independent code-unit ordering and Web Crypto SHA-256. The canonical payload covers version, protected-kernel registry, nodes, relations, explicit registries, and all invariants. The verified digest is `sha256:5de6b2de53cc333181fb6966b5f8a95f0fa51e565c4b65426cdd51e7f304aada`. The UI starts in a checking state and uses "Verified canonical" only after recomputation succeeds.
2. **Exact protected normative kernel:** Added the exact 12-member registry and missing Goals as Provisional, Non-Deception, and Principle of Least Action concepts; corrected protected flags and Autonomy's node type. Proposal validation blocks incidental protected-node revisions.
3. **Complete validation registry and rules:** Added explicit node-type, relation-family, and predicate definitions with layer domains/ranges. Proposal checks now cover required fields, lifecycle, modeled uncertainty, supported types, relation pairing, family/domain/range, unresolved targets, duplicates, aliases, semantic overlap, cross-layer identities, forbidden bridges, protected revisions, weak and causal provenance, unsupported certainty, evidence correlation, discretization loss, and maturity gaps. No allowed predicate is inferred from sample edges.
4. **Relation filtering and traversal:** Added a registered-predicate selector and keyboard-operable native relation buttons. Relation context is tracked separately, so traversal never silently changes the primary concept.
5. **Safe training bundles:** Added structured curated behavior examples with expected outcomes and rationales, evaluator targets, and forbidden inferences. Training export refuses targets without complete curated positive and negative examples; context export remains available.
6. **Evidence fidelity:** Missing estimates render as `Unavailable` without an evidential bar. Assessments model individual evidence items and dependency groups; the shipped Deception Probe and Trace Evidence pair now raises an explicit correlated-evidence warning.
7. **Dependency posture:** Upgraded compatible framework and toolchain packages, regenerated the lockfile, removed the original production advisories, and retained Sites build compatibility. Exact residual development-only findings are documented below.
8. **Starter cleanup and documentation:** Deleted `tests/rendered-html.test.mjs`, verified there are no starter-preview references in the product surface, renamed the package, and replaced the generic starter README with GEPA-specific boundaries and commands.
9. **Recoverable downloads:** Wrapped Blob creation, object-URL creation, link construction, and click in `try/catch/finally`. A failed download reports an alert, preserves the preview, and revokes any URL that was created.
10. **Formatting and CSS cleanup:** Consolidated the Copy ID button's 44-pixel touch-target declaration and expanded naturally touched ontology records and interfaces for reviewability.

## Dependency changes

| Package | Before | After |
| --- | ---: | ---: |
| `next` / `eslint-config-next` | 16.2.6 | 16.3.0 |
| `react` / `react-dom` / `react-server-dom-webpack` | 19.2.6 | 19.2.8 |
| `@cloudflare/vite-plugin` | 1.37.1 | 1.51.2 |
| `wrangler` | 4.92.0 | 4.120.1 |
| `vite` | 8.0.13 | 8.2.1 |
| `vitest` | 4.0.18 | 4.1.10 |
| `@vitejs/plugin-react` | 6.0.2 | 6.0.5 |
| `@vitejs/plugin-rsc` | 0.5.26 | 0.5.34 |

The initial audit reported 19 vulnerabilities: 1 critical, 11 high, 6 moderate, and 1 low. Compatible upgrades and `npm audit fix` reduced the production tree to zero vulnerabilities and the full development tree to 6 vulnerabilities: 2 high and 4 moderate.

### Residual upstream development findings

- `drizzle-kit@0.31.10 -> @esbuild-kit/esm-loader -> @esbuild-kit/core-utils -> esbuild<=0.24.2`: GHSA-67mh-4wv8-2f99, a development-server request/read issue. `0.31.10` is the latest stable `drizzle-kit`; npm only proposes a breaking downgrade to `0.18.1`. Mitigation: keep database generation local and do not expose this development tooling to untrusted networks.
- `vinext@0.0.50 -> image-size`: GHSA-w3rx-r6r6-pgpr and GHSA-5p2g-fcmc-qvqq,
  denial-of-service risks in ICNS/JXL/HEIF parsing. `0.2.1` is the latest non-beta `vinext`
  release, and it still depends on vulnerable `image-size@2.0.2`; npm's `latest` tag currently
  points to `1.0.0-beta.5`, which has the same dependency. Mitigation: the workbench does not
  accept or process user-supplied images, and this package remains outside the production
  dependency audit. Reevaluate the pinned version when a fixed `vinext` release is available.

`npm audit fix --force` was intentionally not used because its proposed downgrades are breaking and would weaken the validated Sites toolchain. These residuals should be reevaluated when fixed stable `drizzle-kit` or `vinext` releases are available.

## Test-driven regression coverage

New regression cases cover:

- digest format, successful verification, mutation sensitivity, deep freezing, exact kernel membership, registry completeness, and relation/registry agreement;
- every new proposal blocker and warning category;
- structured training fixtures, evaluator targets, forbidden inferences, and missing-fixture refusal;
- predicate filtering, keyboard traversal, and stable primary selection;
- unavailable quantities and correlated evidence;
- digest-gated UI copy, export gates, generation failures, clipboard behavior, and exception-safe downloads.

Each behavior was introduced through a focused failing test followed by the smallest implementation change and a focused green rerun.

## Final verification evidence

All commands ran from the feature worktree after the final canonical digest was regenerated.

| Command | Result |
| --- | --- |
| `npm.cmd test` | PASS: 6 files, 46 tests |
| `npm.cmd run lint` | PASS: exit 0, no diagnostics |
| `npm.cmd run build` | PASS: all five vinext/Vite 8.2.1 environments built |
| `npm.cmd exec vinext check` | PASS: 100% compatible; 6 supported, 0 partial, 0 issues |
| `npm.cmd audit --omit=dev` | PASS: 0 vulnerabilities |
| `npm.cmd audit` | Expected nonzero: 6 development findings, 2 high and 4 moderate, limited to the two paths above |
| focused ontology/hash suite | PASS: 7 tests |
| mechanical governance check | PASS: 22 invariants, exact 12-member kernel, real SHA-256 format, no stale preview artifacts, GEPA README |
| `git diff --check` | PASS after final plan whitespace cleanup |

The production build emits two future Vite native-config warnings for the existing JSON import and extensionless Sites plugin import in `vite.config.ts`; they do not affect the current build and are outside this fix wave.

## PR integration

PR #750 began with integration commit `bf6273f`. The review fixes are consolidated in the next
commit on `feature/gepa-ontology-workbench-pr`; `git log` is the canonical commit record.

## Scope and remaining concerns

No public publishing, authentication, durable storage, access-policy change, or canonical
auto-mutation was performed. The only remaining concerns are the two documented upstream
development-tool dependency paths and the nonblocking future Vite configuration warnings. The
production audit is clean.
