# PR-8 Documentation Consolidation and Final Quality Gate Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:verification-before-completion`
> before any completion claim. Apply the repository documentation precision gate.

**Goal:** Make repository documentation describe the implemented V5 architecture coherently and
complete the full verification matrix.

**Architecture:** Rewrite README as orientation and navigation, keep technical detail in subsystem
documents, and mechanically test canonical facts and internal links.

**Tech Stack:** Markdown, PyYAML-backed consistency tests, pytest, Black, Ruff, mypy, build.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Rewrite README by concept and workflow

**Files:**

- Rewrite: `README.md`

Use this exact top-level order:

1. Project description and compact navigation.
2. Project status and maturity.
3. Quick Start.
4. Core Alignment Architecture.
5. 17-Case Framework V5.
6. Reward and Epistemic Process.
7. Safety and Robustness Modules.
8. Verification, Evidence, and Provenance.
9. Deception and Interpretability.
10. Training and Runtime.
11. Evaluation.
12. Datasets and Synthetic Data.
13. Repository Layout.
14. Research Basis and Traceability.
15. Limitations and Research Maturity.
16. Contributing and License.

1. Inventory every current README command, warning, maturity qualification, and internal link in a
   scratch comparison table outside the repository.
2. Rewrite rather than prepend. Merge duplicate explanations of the Constitution, semantic intent,
   structured knowledge, case history, reward semantics, and runtime maturity.
3. Keep Quick Start within the first major sections and move the detailed RL matrix under Training
   and Runtime.
4. Replace the conflicting Version 4 case table with a compact manifest-consistent V5 table and
   state `CASE × STRIPE × REPEAT`.
5. State that `thought_align` is a diagnostic compatibility field and that verified process can
   receive bounded positive reward.
6. State that deception, circuit, attribution, and unverified trace signals are diagnostic by
   default.
7. Link to detailed framework, reward, logging, representation, controlled-evolution,
   recommendations, and research documents.
8. Preserve native runtime limitations exactly: CUDA/DDP hardware unqualified, Vulkan/llama.cpp
   external and experimental, Mojo coordinator non-generating, pure Mojo learner unsupported.
9. Preserve security warnings for executable hooks, external runtimes, and synthetic versus
   measured telemetry.

## Task 2: Consolidate subsystem documentation

**Files:**

- Modify: `docs/17_CASE_FRAMEWORK.md`
- Modify: `docs/ALIGNMENT_EVAL_BATTERY.md`
- Modify: `docs/structured_logging.md`
- Modify: `docs/long_context.md` only if links or terminology drift
- Modify: `docs/GEPA_Mindfulness_Constitution.md` only where reward wording is normative
- Modify: `docs/thought_alignment.md`
- Modify: `docs/epistemic_process_rewards.md`
- Modify: `gepa_mindfulness/core/README.md`
- Modify: `gepa_mindfulness/schema_v3/README.md`
- Modify: `modules/semantic_intent_robustness/README.md`
- Modify: `modules/objective_validator_robustness/README.md`
- Modify: `docs/README.md`

1. Search for `No thought-based rewards`, `H = 0`, `r_thought = 0`, `13+0`, `Version 4`, old Case
   14 through Case 17 meanings, and reward-from-style language.
2. Replace only semantically obsolete statements. Preserve historical context when explicitly
   labeled historical and non-normative.
3. Define one canonical term per concept: canonical case, robustness stripe, repeat, verified
   epistemic process, diagnostic signal, world state, evidence state, local verifier, relational
   verifier, and representation candidate.
4. Link code or subsystem docs to REC-IDs and REF-IDs through a short traceable chain.
5. Preserve the Constitution's values and maturity caveats; do not turn it into an implementation
   manual.

## Task 3: Add documentation and link checks

**Files:**

- Create: `tests/test_documentation_links.py`
- Extend: `tests/test_v5_documentation_consistency.py`
- Modify: `tests/test_schema_and_ci_contracts.py`

1. Write a failing internal-link test that parses repository-relative Markdown links in README and
   the V5 documentation set, ignores external URLs and anchors, and asserts each target exists.
2. Add fragment checks for local Markdown headings using GitHub-style normalized anchors.
3. Add failing canonical-fact tests for framework name/version, exactly 17 cases, frozen Case 14
   through Case 17 rows, and the exact stripe IDs.
4. Add a failing traceability test that resolves every REC-ID to the registry and every referenced
   REF-ID to `references.yaml`.
5. Run tests before the final prose changes and verify they fail on stale links or facts.
6. Complete the documentation changes, then rerun until all checks pass.

## Task 4: Apply the documentation precision gate

1. Review each normative statement for actor, action, object, condition, and observable result.
2. Record and correct every `BLOCK`: conflicting canonical fact, materially ambiguous term,
   missing safety precondition, or normative statement without a verification method.
3. Correct `WARN` findings that use vague pronouns, mixed requirements, undefined abbreviations,
   or subjective acceptance terms.
4. Verify safety warnings state condition, hazard, consequence, and preventive action.
5. Confirm code identifiers, commands, field names, versions, and file paths are exact.
6. Do not approve the documentation stage with any unresolved `BLOCK`.

## Task 5: Run the full verification matrix

Use the repository's selected Python executable and record the exact executable path and version.

1. Run targeted V5 case, reward, provenance, logging, evaluator, representation, memory,
   verification, authority, recovery, learning, skill, experimental, and documentation tests.
2. Run the normal full suite:

```text
python -m pytest -q
```

3. Run formatting:

```text
python -m black --check --line-length 100 .
```

4. Run linting:

```text
python -m ruff check .
```

5. Run type checking:

```text
python -m mypy evaluation gepa_mindfulness src/mindful_trace_gepa \
  modules/semantic_intent_robustness modules/objective_validator_robustness
```

6. Run package build and wheel smoke installation:

```text
python -m build
python -m pip install --target <temporary-directory> dist/gepa_mindfulness-*.whl
```

7. From outside the checkout, import canonical case/stripe registries, load both YAML resources,
   and run `gepa --help` against the wheel target.
8. Run synthetic dataset validators and JSON round-trip tests for V5 records, events, provenance,
   representation lattices, failure graphs, and learning records.
9. Run the deterministic representation benchmark and report sample count, candidate count, median
   time, maximum time, and environment.
10. Search again for direct reward from trace wording, direct deception penalties, conflicting
    case IDs, pre-V5 normative names, duplicate semantic-laundering definitions, stale README
    anchors, Case 18 or later, and unsupported research claims.
11. Run `git diff --check` and inspect `git status`, `git diff --stat`, and the complete diff for
    unrelated changes.
12. Record CUDA, GPU, Vulkan, llama.cpp, Mojo, native model, or external benchmark checks as
    unavailable when the environment cannot run them.

## Task 6: Update recommendation statuses and commit PR-8

**Files:**

- Modify: `docs/recommendations/registry.yaml`
- Modify: `docs/recommendations/UNIFIED_RECOMMENDATIONS.md`
- Modify: `docs/recommendations/RESEARCH_TRACEABILITY.md`

1. Mark a recommendation `implemented` only when its implementation references exist and its
   acceptance tests passed in Task 5.
2. Leave disabled research overlays `experimental`.
3. Preserve unresolved research metadata as unresolved; do not fill it from memory.
4. Run registry, traceability, and documentation tests after status changes.
5. Commit the documentation stage with message
   `docs: consolidate the 17-case V5 architecture`.

## Task 7: Produce the completion report

Report all 20 items required by the approved request:

1. canonical framework location;
2. exact case-count confirmation;
3. case/stripe/repeat implementation;
4. process reward semantics;
5. optimizer-facing process rewards;
6. grounding route for each reward;
7. diagnostic-only signals;
8. action-bound events;
9. representation robustness;
10. world/evidence state;
11. verifier and failure graph;
12. recommendation registry;
13. research traceability;
14. README structure;
15. files changed;
16. tests added;
17. exact commands and results;
18. performance measurements;
19. limitations;
20. deferred experimental work.

Do not state that work is complete until the fresh verification outputs support every claim.
