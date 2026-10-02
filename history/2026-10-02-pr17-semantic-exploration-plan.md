# Semantic exploration implementation plan

Use Superpowers executing-plans inline; one fresh whole-branch reviewer at the end.
Spec: `history/2026-10-02-pr17-semantic-exploration-spec.md`.
Goal: bounded, non-authoritative evidence-search proposals under explicit host budgets.
Base: `2c89b34629071141d044d61fb8c77a704e9576d6` (merged PR-16).

## Global constraints

Python >=3.10, 100-column Python, no new dependency, no beads automation. Exactly 17 cases,
19 recommendations, defaults/reward/training admission/authority unchanged. All new telemetry
is experimental and non-TRAIN. Use the existing checkout on its isolated feature branch.

## Review Focus

- No unknown uncertainty/gain/reversibility may become an optimistic proposal.
- High uncertainty alone cannot bypass distance, stakes, reversibility or budget constraints.
- Exact primitive checks and nested snapshots must not invoke hostile serializers or accept mutation.
- Selected legacy questions must preserve valid Unicode/whitespace and remain bounded and non-TRAIN.
- Gain protocols, compute units and candidate identity must remain interpretable and deterministic.

### Task 1: Propose bounded semantic exploration

Files: create `gepa_mindfulness/verification/semantic_exploration.py`,
`tests/test_semantic_exploration.py`, `docs/semantic_exploration.md`,
`docs/adr/0018-semantic-exploration.md`. Update packaged docs, four recommendation files,
evaluation/recommendations.py and registry/traceability tests.
Interfaces and exact thresholds: follow the linked spec's Contracts and Selection sections.

1. Write tests for matched selection changes, all 15 descriptors, monitor deferral, every rejection,
   deterministic ties, bounds, existing record deserialization/admission, hostile nested records,
   strict numbers/enums/flags and public projection. Run focused pytest: expected missing-module RED.
2. Implement records and selector; reuse existing hypothesis validation helpers, question record,
   config and eligibility. Run focused tests/coverage: expected GREEN and >=80% module coverage.
3. Add guide with executable example, ADR, source/result/inference/maturity mapping. Register
   REF-NIGHT-SCIENCE and reciprocal REC-011 links for existing LADDER/REASONING-TOPOLOGY.
   Run registry/traceability/reader tests: expected 69 refs, 19 recommendations, 17 cases.
4. Run Ruff/Black, scoped mypy, Python3.10 syntax/100-column checks. Build wheel/sdist, installed-wheel
   module/doc/registry/example/CLI/no-Torch smoke, then full Torch-enabled suite: expected pass.
5. Commit, task-done with `../venv/Scripts/python.exe -m pytest tests/test_semantic_exploration.py -q`,
   generate review-package. Dispatch one gpt-6-astra/high fresh read-only reviewer using Review Focus.
6. Regrade all findings; one Important/Critical RED/GREEN fix pass and final suite as needed.
   Record every declined-to-judge ruling/cost. Push feature branch, open and attach draft PR-17.

## Rulings and evidence

Pre-flight: one cohesive task; no cross-task interfaces.
Ruling: Established next-PR authorization covers inline design/implementation/draft publication.
Cost: reversible draft may need revision; no merge.
Ruling: Host supplies candidates, normalized gain protocol, costs and public text. Cost: selection
contracts do not authenticate, calibrate or generate good candidates.
Ruling: Unknown monitor uncertainty defers inquiry; known high monitor uncertainty prioritizes monitor
inquiry. Cost: conservative deferral may need a host-directed diagnostic before this selector is useful.
Ruling: Distance caps and ranking are explicit local heuristics, not reproduced paper algorithms.
Cost: behavioral usefulness remains unmeasured and defaults need empirical tuning.

## Validation evidence

- Initial missing-module RED; 46 focused tests GREEN after implementation and correcting fixture enum names.
- Zero monitor uncertainty at a zero host threshold regression RED, then GREEN after requiring positive
  monitor uncertainty before prioritizing its inquiry. Added nested-record and repeated-call boundaries.
- 49 focused tests pass with 100% statement coverage of the new module (164 statements).
- 124 focused/registry/research/reader tests pass. Registry checks caught shared YAML-list duplicate
  notes and reader evidence-link omissions; both were corrected without changing existing source claims.
- Ruff, Black (628 files), new-module plus CI-scoped mypy, Python3.10 syntax and 100-column checks pass.
- Four primary sources were consulted; canonical cybersecurity Reasoning Topology Matters identity
  retained, one Night Science entry added. 69 references, 19 recommendations, 17 canonical cases.
- Wheel/sdist build and installed-wheel exact module/guide/registry/example/CLI/no-Torch smoke pass.
- Full Torch-enabled suite: 4625 passed, 18 skipped, 16 warnings in 227.68 seconds.
