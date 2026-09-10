# PR-1 V5 Manifest and Traceability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` for
> loaders and consistency checks. Browse only primary paper sources for bibliographic metadata.

**Goal:** Establish canonical V5 machine metadata and a traceable architectural decision registry.

**Architecture:** YAML files hold authored canonical facts. Frozen Python records validate and
expose them. Code compatibility maps derive from the manifest, while tests compare authored docs
mechanically against the same source.

**Tech Stack:** PyYAML, `importlib.resources`, dataclasses, pytest, Markdown.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Create the canonical case and stripe manifests

**Files:**

- Create: `evaluation/cases/__init__.py`
- Create: `evaluation/cases/17_case_manifest.yaml`
- Create: `evaluation/cases/robustness_stripes.yaml`
- Create: `evaluation/cases/registry.py`
- Modify: `pyproject.toml`
- Test: `tests/test_v5_case_registry.py`

**Case manifest shape:**

```yaml
framework_name: GEPA Mindfulness 17-Case Framework V5
framework_version: 17case-v5
canonical_case_count: 17
cases:
  - id: 1
    key: correct_high_confidence_aligned_answer
    title: Correct high-confidence grounded answer
    expected_epistemic_behavior: Answer correctly with calibrated high confidence.
    confidence_semantics: high
    stakes_semantics: not_applicable
    compatibility:
      legacy_versions: [v1, v2, v3, v4]
```

Use established machine keys for Cases 1 through 17. Use the frozen V5 titles and expected
behaviors for Cases 14 through 17. Do not include Case 0 in `cases`.

**Stripe manifest shape:**

```yaml
registry_version: 17case-v5
stripes:
  - id: NONE
    title: No robustness perturbation
    allowed_subtypes: []
```

1. Write failing tests for version, exact count, exact ID sequence, unique keys, required fields,
   frozen Case 14 through Case 17 values, and the exact 11 stripe IDs.
2. Add a package-resource test that loads both YAML files without relying on the current directory.
3. Run tests and verify failure because the package does not exist.
4. Implement frozen `CanonicalCase`, `RobustnessStripe`, and `V5Registry` records.
5. Implement `load_case_manifest()` and `load_stripe_registry()` with exact-field validation and
   actionable errors for missing, duplicate, or invalid values.
6. Add `evaluation = ["cases/*.yaml"]` to `[tool.setuptools.package-data]`.
7. Run `python -m pytest tests/test_v5_case_registry.py -q`.

## Task 2: Derive compatibility maps and eliminate case drift

**Files:**

- Modify: `gepa_mindfulness/core/clarifying_abstention.py`
- Modify: `gepa_mindfulness/schema_v3/case_v3.py`
- Modify: `gepa_mindfulness/schema_v3/validators.py`
- Modify: `gepa_mindfulness/schema_v3/__init__.py`
- Modify: `reasoning-generalization-tracer/src/rg_tracer/schema_v3/case_v3.py`
- Modify: `reasoning-generalization-tracer/src/rg_tracer/schema_v3/validators.py`
- Modify: `tests/test_17_case_clarifying_abstention.py`
- Modify: `tests/test_schema_and_ci_contracts.py`
- Modify: `tests/test_schema_v3.py`

1. Add failing tests asserting `CASE_NAMES == {case.id: case.key for case in manifest.cases}` and
   both compatibility copies agree.
2. Add failing validator tests proving canonical IDs 14 through 17 are accepted and Case 18 is
   rejected. Preserve Case 0 as fallback.
3. Run the focused tests and verify current Schema V3 validators reject 14 through 17.
4. Build `CASE_NAMES`, `ORIGINAL_CASE_IDS`, and `FRAMEWORK_CASE_IDS` from the validated registry.
5. Build `APPENDED_AMBIGUITY_CASES` from manifest compatibility fields plus the existing typed
   abstention and ambiguity enums.
6. Update both Schema V3 validators to accept 0 through 17 and describe 0 as non-canonical.
7. Run the focused tests and both package import-contract tests.

## Task 3: Add machine-readable documented-case consistency

**Files:**

- Modify: `docs/17_CASE_FRAMEWORK.md`
- Test: `tests/test_v5_documentation_consistency.py`

1. Give the canonical case table stable HTML markers:

```markdown
<!-- canonical-cases:start -->
| ID | Machine key | Human title |
...
<!-- canonical-cases:end -->
```

2. Write a failing test that parses table rows between the markers and compares IDs, keys, and
   titles to the manifest.
3. Run the test and verify failure against the existing two-table structure.
4. Rewrite the framework document around V5, exactly 17 base cases, robustness stripes, repeats,
   and overlay-not-case semantics. Preserve IDK versus ambiguity abstention guidance.
5. Run the documentation consistency test.

## Task 4: Create the recommendation registry

**Files:**

- Create: `docs/recommendations/registry.yaml`
- Create: `docs/recommendations/UNIFIED_RECOMMENDATIONS.md`
- Create: `evaluation/recommendations.py`
- Test: `tests/test_recommendation_registry.py`

**Python interface:**

```python
@dataclass(frozen=True)
class Recommendation:
    recommendation_id: str
    title: str
    priority: str
    status: str
    rationale: str
    targets: tuple[str, ...]
    supersedes: tuple[str, ...]
    dependencies: tuple[str, ...]
    research_refs: tuple[str, ...]
    repo_refs: tuple[str, ...]
    acceptance_tests: tuple[str, ...]
    implementation_refs: tuple[str, ...]
```

1. Write failing tests for exact IDs REC-001 through REC-014, allowed priorities and statuses,
   unique IDs, dependency resolution, supersession resolution, and non-empty acceptance methods.
2. Seed all 14 recommendations with statuses reflecting repository reality: implemented only when
   code and tests exist; experimental for disabled overlays; accepted for approved but incomplete
   architecture.
3. Implement a strict YAML loader. Reject unknown fields so registry drift is visible.
4. Write `UNIFIED_RECOMMENDATIONS.md` as a concise reader grouped by priority and status. Link each
   recommendation to code, tests, and research IDs without claiming research proves the design.
5. Run `python -m pytest tests/test_recommendation_registry.py -q`.

## Task 5: Verify primary research metadata and create traceability

**Files:**

- Create: `docs/recommendations/RESEARCH_TRACEABILITY.md`
- Create: `docs/recommendations/references.yaml`
- Test: `tests/test_research_traceability.py`

1. Query the canonical arXiv abstract or export API for every identifier listed in the approved
   specification. Do not use search snippets as bibliographic authority.
2. Record exact title, ordered authors, year, arXiv identifier, DOI and venue status when the
   primary metadata supplies them, and canonical paper URL.
3. If a supplied ID does not resolve, record `metadata_status: unresolved` and omit unsupported
   authors, DOI, venue, and findings. Preserve the requested stable REF-ID.
4. Write failing tests requiring all requested REF-IDs, unique arXiv IDs, resolved REC-ID links,
   valid metadata-status values, and local repository notes.
5. Implement a simple YAML validation helper in `evaluation/recommendations.py` and run the tests.
6. Write `RESEARCH_TRACEABILITY.md` with three explicitly labeled fields per entry:
   `Source demonstrates`, `Repository inference`, and `Maturity`.
7. Use only restrained connection verbs such as motivates, supports, reports, inspires, or
   suggests. Do not state that a paper proves the unified architecture.

## Task 6: Verify and commit PR-1

**Files:**

- Create: `docs/adr/0001-17-case-v5-unified-architecture.md`

1. Create the `docs/adr` directory and record the accepted decision, context, alternatives,
   consequences, reward-grounding boundary, exactly-17-case invariant, and experimental-overlay
   boundary. Link the approved design spec without replacing it.
2. State that the ADR describes repository policy and design intent, not empirical proof.
3. Add the ADR to the documentation link checks planned for PR-8.
4. Run case registry, Schema V3, clarifying abstention, recommendation, and research tests.
5. Run `python -m build`, install the wheel into a temporary target, and load both V5 manifests
   through `importlib.resources` from outside the repository.
6. Run Black, Ruff, and mypy on changed Python paths.
7. Run `git diff --check`.
8. Search for conflicting Case 14 through Case 17 definitions and references to Case 18 or later.
9. Inspect the complete stage diff and commit with message
   `feat: establish canonical 17-case V5 registry`.
