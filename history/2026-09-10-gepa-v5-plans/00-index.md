# GEPA Mindfulness V5 Implementation Plan Index

> **For agentic workers:** Use `superpowers:executing-plans` for inline execution. Use
> `superpowers:subagent-driven-development` only when the maintainer explicitly requests
> subagents. Execute numbered steps in order and preserve each plan's commit boundary.

**Goal:** Implement the approved `17case-v5` architecture as nine independently reviewable stages.

**Architecture:** The stages establish reward grounding and canonical metadata first, then add
events, evaluation, representation robustness, governance, controlled learning, experimental
declarations, and final documentation. Later stages consume typed interfaces from earlier stages.

**Tech Stack:** Python 3.10+, dataclasses, enums, protocols, PyYAML, pytest, Ruff, Black, mypy.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Global constraints

- Framework version is exactly `17case-v5`.
- The canonical manifest contains exactly 17 cases with IDs 1 through 17.
- Case 0 is a non-canonical fallback.
- Cases 14 through 17 retain the frozen meanings in the specification.
- Generated reasoning wording and diagnostic signals never directly change optimizer fitness.
- Every nonzero epistemic-process reward has observable evidence or a trusted evaluator contract.
- Raw input and raw evidence remain immutable; derived records use provenance and supersession.
- Existing logging and semantic-intent frameworks are extended, not replaced.
- Python lines are at most 100 characters.
- No new runtime dependency is added without maintainer approval.
- Do not invoke `bd` or edit `.beads/*` unless the maintainer explicitly requests it.

## Execution order

1. `01-pr0-verified-epistemic-rewards.md`
2. `02-pr1-v5-manifest-recommendations-traceability.md`
3. `03-pr2-action-bound-events.md`
4. `04-pr3-v5-evaluator.md`
5. `05-pr4-representation-robustness.md`
6. `06-pr5-verification-and-authority.md`
7. `07-pr6-learning-and-skills.md`
8. `08-pr7-experimental-overlays.md`
9. `09-pr8-documentation-consolidation.md`

Run each plan's targeted checks before its local commit. Run the full quality gate in Plan 09.
