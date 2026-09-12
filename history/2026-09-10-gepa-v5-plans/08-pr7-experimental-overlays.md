# PR-7 Experimental Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`. Keep
> every overlay disabled by default and outside canonical case identity and optimizer reward.

**Goal:** Declare controlled experimental V5 overlays without turning research proposals into
stable behavior.

**Architecture:** A registry and typed feature flags describe experimental capabilities, maturity,
allowed outputs, and prohibited effects. The implementation supplies validation scaffolding only.

**Tech Stack:** YAML, Python dataclasses and enums, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Add the experimental overlay registry

**Files:**

- Create: `evaluation/cases/experimental_overlays.yaml`
- Create: `evaluation/experimental_overlays.py`
- Modify: `pyproject.toml`
- Test: `tests/test_experimental_overlay_registry.py`

**Registry entries:**

```text
competing_hypotheses
expected_information_gain_inquiry
adaptive_small_multi_agent_topology
declarative_orchestration_scope
mechanistic_circuit_audit
```

Each entry contains `id`, `title`, `maturity: experimental`, `enabled_by_default: false`,
`feature_flag`, `allowed_outputs`, `prohibited_effects`, `research_refs`, and `recommendation_refs`.

1. Write failing tests for the exact five entries, unique flags, experimental maturity, false
   defaults, resolved REC/REF links, and required prohibited effects.
2. Require every entry to prohibit canonical case creation and direct optimizer reward.
3. Require the mechanistic overlay to prohibit treating correlation as causation.
4. Implement a strict YAML loader with frozen `ExperimentalOverlay` records.
5. Add the YAML path to evaluation package data and run registry tests.

## Task 2: Add feature-flag validation and no-effect defaults

**Files:**

- Modify: `evaluation/experimental_overlays.py`
- Test: `tests/test_experimental_overlay_flags.py`

**Interface:**

```python
@dataclass(frozen=True)
class ExperimentalOverlayConfig:
    competing_hypotheses: bool = False
    expected_information_gain_inquiry: bool = False
    adaptive_small_multi_agent_topology: bool = False
    declarative_orchestration_scope: bool = False
    mechanistic_circuit_audit: bool = False


def enabled_overlays(config: ExperimentalOverlayConfig) -> tuple[ExperimentalOverlay, ...]: ...
```

1. Add failing tests proving the default config enables nothing and leaves planned V5 cells,
   canonical cases, reward totals, and action authority unchanged.
2. Add tests proving explicit flags expose only diagnostic proposal records.
3. Implement exact field mapping. Reject unknown external configuration keys.
4. Run experimental flag, V5 case, reward invariance, and authority tests.

## Task 3: Add minimal typed proposal outputs

**Files:**

- Create: `evaluation/experimental_records.py`
- Test: `tests/test_experimental_overlay_records.py`

1. Add records for hypothesis sets, information-gain questions, topology proposals,
   global/focus/local scope declarations, and mechanistic audit references.
2. Write failing tests requiring uncertainty, provenance, feature flag, maturity, and diagnostic
   status on every record.
3. Reject action execution, authority grants, canonical case IDs beyond the source case, and reward
   component fields in these records.
4. Implement serialization and run the record tests.

## Task 4: Document, verify, and commit PR-7

**Files:**

- Create: `docs/experimental_v5_overlays.md`
- Modify: `docs/recommendations/registry.yaml`
- Modify: `docs/recommendations/UNIFIED_RECOMMENDATIONS.md`

1. Document what each overlay may observe or propose, what it cannot do, and how to enable it in a
   controlled experiment.
2. Update REC-011 through REC-014 to experimental with code and test references.
3. Run all experimental tests, canonical case tests, reward invariance tests, and authority tests.
4. Run Black, Ruff, mypy, package-resource tests, and `git diff --check`.
5. Inspect the stage diff and commit with message
   `feat: declare disabled V5 experimental overlays`.
