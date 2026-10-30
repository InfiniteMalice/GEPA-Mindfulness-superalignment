# PR 7 — immutable dataset research

Parent: `be838eec75ecda6562e4528c9fb0635c7eb479f8`; ready PR workflow and merge authority persist.
The maintainer approved the repo-quality-gate fallback because that skill is unavailable.
Use the fallback spec/design/implementation/tests/independent-review/evidence sequence and
the executing-plans/TDD workflows. No Beads operations or protected-source changes.

## Task specification and design

Implement a default-off, CPU-only AutoSFT-inspired dataset research preparation pipeline.
Reuse `DatasetManifest`, its ancestry/partition validator, rich synthetic-case validation,
`SystemConfig`, `CurationPolicy`, `GenerationReview`, and detached host authentication.
Preserve source rows in an append-only pool; recipes select/order/count immutable rows.
Derived revisions bind the previous pool, recipe, fixed checkpoint/trainer/budget contract,
operation, role provenance, evidence, and independent review. Reject stale inputs,
unknown ancestry, held-out origins, identity reuse, misleading structural labels, budget
overflow, reviewer self-certification, and incomplete/mutated host responses.

Support rebalancing, pruning, repair, replacement, targeted synthesis, curriculum ordering,
and robustness augmentation as explicit, validated recipe transformations. The paper's six
families are descriptive post-hoc labels; this typed operation API is a GEPA adaptation.
Never run model APIs/training or make preparation a training admission authority.
Reports and proposed datasets remain DEVELOPMENT, optimizer_input=false, confers_authority=false.

Independent roles are captured as versioned external-runtime provenance, not implemented
as a new orchestration architecture. No candidate-provided verifier code is executed.
An independently configured host authenticates the complete pool, recipe, review, policy,
held-out separation, token counts, and role independence on every preparation call.

## Primary-source evidence

Browser access on 2026-10-30 succeeded for arXiv:2610.22194v1, dated 2026-10-15,
CC BY 4.0. Read sections 2.1–2.2, 4.1–4.3, B.2–B.4, D; the paper links
Thinking-Space/AutoSFT. The upstream repository is a placeholder, so no upstream code
is copied and no licensed code reproduction is claimed. Record the exact checked rendition
and its digest separately from an authored method summary. The implementation adapts
the fixed-base-checkpoint loop and separated roles, adding stricter independent host review
and no held-out feedback. Full multi-agent policy/training reproduction is Unsupported.

## Dependency order and acceptance

1. Write failing tests for records, immutable operations, source/partition boundaries,
   review bindings/independence, matched budgets, deterministic output, and round-trip schemas.
   Expected initial failure: missing dataset-research module.
2. Implement bounded records and pure preparation, reusing existing validation/authentication.
   Expected: targeted tests pass and original input bytes remain unchanged.
3. Add disabled example configuration, governance/retraining protocol, citations, source digest,
   applicability/compatibility notes and wheel resources. Run focused compatibility suites,
   formatting, source/link checks, and actual wheel byte/smoke verification.
4. One fresh independent whole-change review required by executing-plans; verify findings
   against code, RED→GREEN fixes where needed. Record software evidence separately from
   unrun training experiments; publish a ready PR, monitor checks/reviews, merge when clean.

Pre-flight: records feed preparation; both use the existing manifest and review schemas.
Ruling: use an isolated source snapshot and remote Git objects because this environment
does not expose a local Git checkout. Publication must preserve every unrelated base blob.
Ruling: preparation remains non-authoritative; no existing general training-dataset admission
API exists to attach safely. The external host must independently admit an exact derived
dataset through its configured training runtime. Claiming automatic admission would create
a competing authority. Full research effectiveness remains unmeasured.

## Review focus

Check whole-payload authentication, replay/stale bindings, transitive partition contamination,
nonfinite/boolean integers, callback mutation/exceptions, pre-callback complete report bounds,
append-only row/version identity, operation semantics, fixed-checkpoint/budget matching,
and the distinction between a reviewed research proposal and training admission.

## Execution evidence (before final review fixes)

New tests first failed at import because no dataset-research module existed. Implementation
then passed 34 tests; additional report-capacity/contract/authority probes brought this to39.
One test expectation was corrected to include constructor-time rejection of nested hidden
metadata; the source boundary had already rejected it before preparation.
The initial compatibility run identified missing reference allowlist/reciprocal-reader
attachments. Those attachments were completed without weakening checks. The corrected
24-file run passed493tests,22skips in82.16s. Subprocess tests require an absolute repository
root in PYTHONPATH in this snapshot environment. Six Python files passed Black, Python3.10
syntax and100-column checks. A no-isolation wheel built successfully;25resource bytes matched
source, with isolated wheel import/record/default-off/documentation smoke.

One fresh final reviewer is examining the whole change; two reproduced findings concern
historical version reuse and synthesis ordering. Fixes and final evidence are not yet recorded.
No external training, paid generation, empirical improvement or live admission is claimed.

## Final independent review and fix pass

Fresh reviewer reported two Important findings and no Critical/Minor findings. Seven new
regressions failed against the reviewed runtime: historical version reuse (one), hidden
reordering under four data-edit labels, and changed replacement sampling counts (two).
The single fix pass retains/binds immutable prior version lineage, substitutes repaired rows
in place with counts preserved, and appends synthesis/augmentation after existing blocks.
All46dataset tests now pass (15.17s); final expanded compatibility/wheel checks are running.
Reviewer independently ran39dataset tests and93reference/link tests, plus25invalid-number
and several contamination/contract probes. No second review dispatch is needed.

Final: fixed historical version reuse — multi-revision regression RED→GREEN.
Final: fixed operation reordering and replacement count leakage — six parameterized
regressions RED→GREEN. No deferred minors. External host authentication, semantic labels,
tokenization and training compliance remain operator-owned; these limits are documented.

## Final acceptance evidence

Expanded29-file compatibility suite:740passed,22skipped in87.47s, including runtime
authority, training eligibility, V5 CLI and17-case clarification/abstention contracts.
No test was weakened. Six Python files pass Black, Python3.10 syntax and100columns.
Rebuilt no-isolation wheel SHA-256cc42052ffd74a2f65d4442e0d28b1ba4ef382d385b9108009113896f13659995:
25resource files and both runtime modules match source bytes; isolated wheel module,
reference loader, record round-trip, default-off and documentation smoke pass.
Full local pytest is not claimed: Torch is unavailable in this environment. Remote full CI
is a separate required gate before merge. Protected base source/manifest/authority blobs
remain unchanged; publication overlays only18intended files on the exact merged base tree.

Final review findings are resolved by the seven RED→GREEN regressions and green expanded
suite. No deferred minors. Independent host/semantic/training behavior is unqualified,
not an unreviewed claim of implemented authority. Source paper version/license/rendition
and digest are linked in existing bibliography/matrix; no upstream code/dependencies copied.
Training/admission and empirical AutoSFT effectiveness remain Unsupported.
