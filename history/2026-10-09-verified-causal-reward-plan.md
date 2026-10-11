# Verified Causal Reward Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the preserved native execution method. Read the approved design before this plan. Repository AGENTS.md prohibits Markdown task tracking and unauthorized beads use, so this plan uses numbered instructions instead of the skill template's checkboxes. Do not invoke `bd` or edit `.beads/*`.

**Goal:** Add an explicitly invoked, default-disabled adapter that applies only independently authenticated, admitted, nonduplicated causal penalties and blocks severe-event optimizer use.

**Architecture:** Wrap the existing `RewardPipeline` without changing existing providers or call sites. Separate exact input records from eligibility/authentication/scoring; enabled results require an explicit optimizer accessor. Keep diagnostic records, training admission and action authorization unchanged.

**Tech Stack:** Existing Python 3.10-compatible code, standard-library dataclasses/JSON/hashlib/math, pytest, Ruff, Black, mypy and setuptools. No new dependency or model service.

**Spec:** `history/2026-10-09-verified-causal-reward-design.md`; review copy `outputs/PR5-proposed-design.md`.

**Status:** Written design and this implementation plan approved on 2026-10-09. Native execution is carried forward from the earlier PRs. On 2026-10-10 the maintainer removed the live experiment packet/failure-example prerequisite for implementation. The approved native plan may proceed with authored tests and disabled defaults. Live penalty application retains its independent evidence and activation gates; no implementation task has started.

## Global constraints

- Preserve exactly 17 canonical cases and fallback Case 0; do not modify registry, stripe or reward semantics.
- Defaults: `enabled=False`, `weight=0.0`, `max_adjustment=0.0`.
- Disabled and zero-weight calls return the exact existing `RewardResult` object, with one provider call and no causal callbacks.
- Enabled deductions are `min(weight, max_adjustment) * failure`, with `failure` in `{0, 1}`; zero deduction assigns the baseline scalar directly.
- Keep `CausalPair` non-TRAIN and preserve every existing recursive eligibility restriction, including failed V5 outcomes.
- Preserve the exclusive observable-evidence and trusted-evaluator provenance routes. Neither serialized status nor a declared evaluator authenticates itself.
- One target per call, two arms, at most 64 factor keys per arm, 256 source-reference entries, 32 accounting layers and 16 experiment authorization references.
- IDs: at most 256 characters. Reasons: at most 4,096 characters. Complete new canonical input envelope: at most 4 MiB UTF-8 and nesting depth at most 32, root depth 1. No truncation.
- Python source lines at most 100 characters; required import sections, public Google-style docstrings, exact schemas, detached validation and finite-number checks.
- No trainer wiring, automatic deployment, paid model calls, generated-code execution, new persistent store or automatic merge.
- Planned expected test outcomes below are acceptance criteria, not completed results.

## Review focus

1. A provider mutates nested request data: reject enabled scoring; preserve the caller's request. Task 3 tests this explicitly.
2. A callback changes a nested source while returning True: withhold, retain original audit facts, and never expose an optimizer value. Task 3 owns this test.
3. A severe verdict accompanies missing accounting or ineligible provenance: retain the authenticated severe event and block the accessor. Tasks 3 and 4 exercise the combinations.
4. The same failure is renamed across metrics, pairs or evidence aliases: local event identity ignores presentation labels; host accounting rejects alternate source IDs. Tasks 2 and 4 test both boundaries.
5. A disabled caller supplies invalid optional causal data or a signed-zero baseline: ignore optional data and preserve the exact baseline object/value. Task 3 verifies this compatibility path.

## Files and responsibilities

| Path | Responsibility |
| --- | --- |
| `gepa_mindfulness/training/causal_reward_records.py` | Public records, exact serialization, bounds, complete request/context digests and strict construction. |
| `gepa_mindfulness/training/causal_reward.py` | Admission, event identity, host callback boundaries, pipeline composition and live optimizer accessor. |
| `tests/causal_reward_fixtures.py` | Shared authored training-origin fixtures and recording host/provider doubles; never production experiment evidence. |
| `tests/test_causal_reward_records.py` | Wire schemas, immutable snapshots, references, coordinates, numerics and input bounds. |
| `tests/test_causal_reward_admission.py` | Eligibility, attribution, overlap completeness and stable event identity. |
| `tests/test_causal_reward.py` | Public adapter, detached authentication, exact no-op behavior and severe access control. |
| `tests/test_causal_reward_integration.py` | Real provider/overlay integration, source restrictions, decision table and replay. |
| `docs/causal_reward_adapter.md` | API/host contracts and executable offline example. |
| `docs/epistemic_process_rewards.md` | Link the optional adapter without changing existing provenance policy. |
| `docs/adr/0021-causal-alignment-diagnostic-extension.md` | Record the actual PR-5 approvals, independent-evidence disposition and limits. |
| `pyproject.toml`, `tests/test_documentation_links.py` | Include the guide in the wheel and local-link validation. |

## Execution prerequisites and separate live-use gate

1. Read AGENTS.md, CODING_STANDARDS.md, the corrected design and this plan. Use the repository quality gate and retained native execution method. Preserve unrelated files and existing approvals.
2. Use an isolated feature branch based on current merged work, including PR-6 (#788). The corrected documents on `codex/causal-reward-gate-corrections` supersede the older PR-4-based planning copies on `codex/verified-causal-reward-pr5`. Integrate the corrected documents before implementation; do not execute the stale plan or force-push old branches.
3. Record the current evidence honestly: the pilot is diagnostic and does not authorize a real causal deduction. Missing live failure evidence does not stop Task 1. Use explicitly authored fixtures to test software behavior; fixture callbacks are not independent evidence.
4. Before any live nonzero deduction or optimizer use, satisfy design section 10: independently authenticated causal verdict, admitted source lineage, exact provider/overlay accounting for that causal failure and separately authorized experiment settings. Missing or rejected live evidence withholds live use, not software implementation.
5. Re-run the baseline command preserved in the policy-review packet on the actual implementation checkout. Record actual commands and results in the execution ledger rather than claiming historical tests validate new code.

Historical baseline: 480 passed, 6 warnings in 12.53 seconds. Independent live-use evidence remains unestablished. This documentation correction changes neither those observations nor the default-disabled policy.

## Shared interface decisions

These decisions resolve constructor and joining choices from the approved design. Use keyword-only constructors for new records. Every public wire record has `to_dict() -> dict[str, object]` and `from_dict(data: object) -> SelfType`, with `SelfType` written as the concrete class for Python 3.10. Use `<class-name-in-kebab-case>-v1` schema tags. Reject extra fields, missing required fields and noncanonical nested round trips. Snapshot before validating or calling a host function.

Use `JsonObject = dict[str, object]` for detached callback payloads. Store retained source payloads internally as canonical JSON strings, exposed only as freshly decoded copies. A constructor never keeps a caller-owned mutable dictionary. Existing typed records retain their public interfaces; do not use private reward serializers or create a new generic serialization framework.

### Record fields

| Record | Fields and types |
| --- | --- |
| `CausalRewardPolicy` | `policy_id: str`, `policy_version: str`, `provider_id: str`, `provider_version: str`, `provider_config_digest: str`, `enabled: bool=False`, `weight: float=0.0`, `max_adjustment: float=0.0`, `experiment_refs: tuple[EvidenceReference, ...]=()`. Positive weight requires positive cap and nonempty external-record experiment references. |
| `CausalRewardContext` | `before: RewardRequest`, `target: RewardRequest`, `before_cell: V5EvaluationCell`, `target_cell: V5EvaluationCell`, `before_case_mapping: tuple[str, str]`, `target_case_mapping: tuple[str, str]`, `before_family_id: str`, `target_family_id: str`, `before_lineage: tuple[str, ...]`, `target_lineage: tuple[str, ...]`, `before_eligibility: str \| None`, `target_eligibility: str \| None`, `before_factors: tuple[tuple[str, str], ...]`, `target_factors: tuple[tuple[str, str], ...]`, `change: FactorChange`, `seed_policy: str`, `sources: tuple[tuple[EvidenceReference, str], ...]`, `target_event_refs: tuple[EvidenceReference, ...]`. Source strings contain complete JSON objects, including source eligibility. |
| `CausalRewardVerdict` | `context_digest: str`, `policy_digest: str`, `evaluator: TrustedEvaluatorContract`, `status: str`, `human_required: bool`, `relevance: str`, all eight `bool \| None` facts from design section 5, `reason: str`, `evidence_refs: tuple[EvidenceReference, ...]`, `provenance: RewardProvenance`. |
| `CausalRewardAccounting` | `context_digest: str`, `verdict_digest: str`, `policy_digest: str`, `baseline_total: float`, `provider_id: str`, `provider_version: str`, `provider_config_digest: str`, `layers: tuple[tuple[str, str], ...]`, `evidence_refs: tuple[EvidenceReference, ...]`. Layer entries are unique names and `not_counted`, `already_counted` or `unknown` for the distinct verified causal failure. An incorrect-answer-only reduction is not causal overlap; host authorization verifies layer semantics. |
| `CausalRewardUseRequest` | `context: CausalRewardContext`, `verdict: CausalRewardVerdict`, `accounting: CausalRewardAccounting`, `policy: CausalRewardPolicy`, `baseline_total: float`, `integrity_snapshot: str \| None`, `event_key: str`. Integrity snapshot is canonical JSON of the actual complete breakdown. |

Case mappings are `(literal_trajectory_case_id, canonical_manifest_case_key)`. Check the first value against the trajectory and the second against the validated cell's manifest entry; do not guess case aliases. Null legacy case IDs cannot establish this enabled mapping. Fallback Case 0 retains existing default behavior; it is not given a canonical cell or enabled causal deduction.

Context constructors accept missing/non-TRAIN eligibility labels as auditable data, then admission rejects them. Malformed schema types still raise errors. This distinction permits a valid severe verdict to survive admission failure. Source-family IDs and lineage must be nonempty; the host authenticates their meaning and completeness. Bound lineage collections within the shared envelope limit and cap each at 256 unique IDs.

Both arms' declared fact maps share keys; literal differences equal the one `FactorChange`. More than one literal change is structurally invalid. A nominal single change independently found to be semantically compound has `single_factor_verified=False` and receives `not_attributable`; it is not promoted to a single-variable penalty.

The target event references are nonempty, include observable action or output evidence, and resolve to the target request. Outcome evidence may also be an external record. The verifier authenticates that they identify the actual target action/outcome. Before-only references cannot populate this target event set. All source references used anywhere in verdict/accounting resolve through the context's exact source index and the owning request's allowed references.

### Public and internal signatures

In `causal_reward_records.py` expose:

```python
def request_digest(request: RewardRequest) -> str: ...
def reward_record_digest(record: object) -> str: ...
def validate_input_envelope(payload: object) -> None: ...
```

`reward_record_digest` accepts only the exact new serializable record types and uses their complete canonical wire representation. `request_digest` covers the exact canonical trajectory and allowed references. Validate nesting/size before unbounded recursive JSON encoding; reject cycles, unsupported values and nonfinite floats. Existing digest helper output remains SHA-256 of canonical JSON, not authentication.

In `causal_reward.py` define:

```python
def _admission_reasons(context: CausalRewardContext) -> tuple[str, ...]: ...
def _event_key(context: CausalRewardContext) -> str: ...
def _failure(verdict: CausalRewardVerdict) -> bool: ...
def _snapshot_integrity(result: RewardResult) -> str | None: ...

class CausalRewardAdapter:
    def __init__(
        self, pipeline: RewardPipeline, *, policy: CausalRewardPolicy,
        authenticate_verdict: Callable[[JsonObject], bool] | None = None,
        authorize_use: Callable[[JsonObject], bool] | None = None,
    ) -> None: ...

    def score(
        self, target_request: RewardRequest, *,
        observation: RewardObservation | None = None,
        context: CausalRewardContext | None = None,
        verdict: CausalRewardVerdict | None = None,
        accounting: CausalRewardAccounting | None = None,
    ) -> RewardResult | CausalRewardDecision: ...
```

`CausalRewardDecision` is produced by an internal adapter factory and exported as an immutable result. Public fields are `base_result: RewardResult`, `baseline_total: float`, `deduction: float`, `diagnostic_total: float`, `disposition: str`, `reasons: tuple[str, ...]`, `severe_status: str`, `audit: str`. Audit is canonical JSON retaining exact wire bindings, evidence/provenance and independent gate outcomes. `severe_status` is `verified_violation`, `verified_clear`, or `unknown`. Its methods are `optimizer_reward() -> float` and `to_dict() -> dict[str, object]`; there is no `from_dict`, `.total`, or numeric coercion. Live admission is private and excluded from restoreable state. Export the audit-only marker `training_eligibility="DEVELOPMENT"` on the report so it cannot be laundered into training provenance.

Required accounting layers include `provider` and, when the actual pipeline enables its integrity overlay, `reward_integrity`. Other provider-internal layers may be named and must be authenticated as exhaustive. Missing required layers become unknown accounting. Extra layers are not automatically trusted. A provider/configuration mismatch against policy or an accounting digest/scalar mismatch is an invalid binding and raises `ValueError`; an incomplete but well-formed layer roster withholds. `authorize_use` resolves the policy identity to the actual provider configuration; the adapter cannot authenticate arbitrary provider internals by introspection.

## Task 1: Exact reward records and detached snapshots

**Files:** Create `causal_reward_records.py`, `tests/causal_reward_fixtures.py`, and `tests/test_causal_reward_records.py` at the paths above.

**Interfaces:** Consume existing request/trajectory, V5 cell, FactorChange, evidence and reward-provenance types. Produce all five input records and three record helper signatures above. The fixture `make_context() -> CausalRewardContext` returns two authored, explicitly TRAIN-origin request snapshots and sources. `make_policy() -> CausalRewardPolicy` defaults to disabled; `make_verdict(context, policy, **facts)` and `make_accounting(context, policy, verdict, baseline_total, **overrides)` return exact typed authored fixtures. Fixture docstrings state that these are not gate evidence.

1. Write `test_wire_round_trip_is_exact_and_detached`, asserting `restored.to_dict() == original.to_dict()` for each record and that modifying a returned dictionary does not change the original. Add cases for extra/missing keys, noncanonical nested trajectory payloads, invalid references, wrong evaluator/provenance route and invalid policy numerics.
2. Run `python -m pytest tests/test_causal_reward_records.py -q`. Confirm failure is from absent records or the specified validation behavior before implementing.
3. Implement the exact constructors and helpers. Use per-class explicit serializers and immutable snapshots; preserve complete existing trajectory serialization. Apply enum/value sets from the design and exact schema tags. Validate V5 cells with existing constructors and case mappings against the manifest. Do not import or relax `CausalPair`.
4. Add `test_limits_fail_without_truncation`: accept boundary values and reject 65 factor keys, 257 source entries, 33 layers, 17 experiment refs, 257-character IDs, 4,097-character reasons, depth 33 and encoded size above `4 * 1024 * 1024`. Add cycles, duplicate keys/IDs, booleans and NaN/Infinity. Validate complete envelope limits as well as standalone record limits.
5. Add `test_arm_identities_and_declared_change_are_exact`: distinct target IDs; same policy/model/backend/harness/repeat; shared-seed equality; independent-seed preservation; valid case-changing pairs; one actual factor difference; mismatched case mapping rejected. An enabled unknown/legacy case cannot be coerced into a canonical case.
6. Run the record tests, then Ruff and Black on the changed Python files. Commit only this task's files as `feat(reward): add exact causal reward evidence records` after checks pass.

Core round-trip assertion, parameterized across the five input record types:

```python
original = record.to_dict()
restored = type(record).from_dict(original)
assert restored.to_dict() == original
returned = restored.to_dict()
returned["schema_version"] = "changed"
assert restored.to_dict() == original
```

## Task 2: Eligibility, event identity and failure predicates

**Files:** Create `causal_reward.py` and `tests/test_causal_reward_admission.py`; extend fixtures only as needed.

**Interfaces:** Consume Task 1 records. Produce `_admission_reasons`, `_event_key`, `_failure` and `_snapshot_integrity` for Task 3. These are implementation helpers, not alternate optimizer entry points.

1. Write `test_nested_restriction_defeats_outer_train`: an outer TRAIN declaration with an inner HIDDEN_EVAL or failed V5 source returns admission rejection. Parameterize DEVELOPMENT, REGRESSION, absent declarations, malformed retained provenance and incomplete generated-data review. An intact admitted fixture returns no admission reason.
2. Run `python -m pytest tests/test_causal_reward_admission.py -q` and verify the expected missing-helper or policy failure.
3. Implement `_admission_reasons` using unchanged `require_training_eligible` on both complete request payloads and source records, plus explicit declaration checks for this interface. Preserve structured rejection details in the later audit while returning stable high-level reason codes. Never remove a record or infer eligibility from a parent.
4. Write `test_failure_requires_independent_incorrectness`: True only for the two exact design predicates with `single_factor_verified=True`, `before_correct=True`, `after_correct=False`. False for appropriate updates, correct targets, equal acceptable actions, unknown facts, consistently wrong baseline and semantic compound interventions. Test the three exact contradiction rules from the design at record validation or the semantic boundary, consistently raising `ValueError`.
5. Implement `_failure` with explicit predicates. Write `test_event_key_ignores_presentation_labels`: same complete target and sorted target event references yield equal keys across changed before-arm/family labels and review reasons; different actual target content or target evidence changes the key. A host-alias rejection is tested later, because hashes cannot infer semantic equivalence of external IDs.
6. Implement integrity snapshot serialization from every field of the actual `RewardIntegrityBreakdown`, including provenance and self-serving-justification diagnostics. Do not drop unknown fields silently or rely on a private serializer. Test a populated breakdown's deterministic round trip to canonical JSON and inclusion of each evidence/provenance field.
7. Run Task 1 and Task 2 tests plus `tests/test_training_eligibility.py` and `tests/test_reward_integrity_rewards.py`. Commit as `feat(reward): enforce causal admission and event accounting boundaries`.

Admission assertions for the intact and nested-restriction cases:

```python
assert _admission_reasons(make_context()) == ()
assert "admission_rejected" in _admission_reasons(restricted_context)
assert restricted_context.to_dict() == retained_restricted_payload
```

Here `restricted_context` is the same exact context with one source JSON object containing `{"training_eligibility": "TRAIN", "source_record": {"training_eligibility": "HIDDEN_EVAL"}}`; preserve all source identities and matching request references. `retained_restricted_payload` is captured immediately before admission checking.

## Task 3: Authenticated scoring and guarded optimizer result

**Files:** Extend `causal_reward.py`; create `tests/test_causal_reward.py`; extend fixtures with a recording pipeline/provider and callbacks.

**Interfaces:** Consume Tasks 1–2. Produce the exact public `CausalRewardAdapter` and `CausalRewardDecision` API above. Both host callbacks receive fresh plain dictionaries; acceptance requires `result is True` and an unchanged full payload after the callback.

1. Write `test_disabled_returns_same_result_without_optional_validation`: use a recording pipeline returning a sentinel `RewardResult`. Assert object identity, one call, observation forwarded unchanged, and zero callback calls even with invalid optional causal arguments. Repeat for weight zero and signed-zero baseline; verify sign with `math.copysign`.
2. Run `python -m pytest tests/test_causal_reward.py -q` and verify the expected failure. Implement only the disabled path, constructor snapshot and minimal decision scaffolding needed for the next tests.
3. Write `test_enabled_authentication_is_detached_and_exact`: reject callbacks returning 1, truthy objects, False, None, exceptions or True after nested mutation. Original context/policy/verdict remains unchanged. Verify private reasoning cannot become support via either provenance route. Bind callbacks to the exact current target and policy.
4. Implement enabled scoring in the design's seven-step order. Snapshot the actual baseline total after one pipeline call; detect provider mutation of the detached target and reject. Preserve existing observation validation in the pipeline. For missing context/verdict/accounting, return a withheld enabled decision rather than inventing zero-valued evidence. Do not call a callback whose required payload is incomplete.
5. Write `test_baseline_accounting_uses_actual_pipeline_result`: reject mismatched scalar/digests/provider descriptors; incomplete required layers yield `accounting_unknown`; overlap with the same causal failure yields `already_counted`; authorize against the exact active integrity snapshot. The callback cannot substitute a different baseline.
6. Write `test_severe_remains_visible_when_other_gates_fail`: authenticated severe True with missing accounting and rejected nested source admission remains `verified_violation`, baseline numeric unchanged, and `optimizer_reward()` raises. Severe None or unauthenticated verdict also blocks. Authenticated severe False alone cannot overcome another rejected gate.
7. Finish the decision factory and stable reason ordering. Preserve every gate result in audit, but export no live capability. `disposition` is `applied` when a nonzero deduction occurs, `unchanged` when the accessor may return a zero-deduction result, and `withheld` otherwise. Apply no deduction until every required gate passes. Suppress `severe_unknown` only when authenticated severe status is known; do not report a claimed True as verified before authentication.
8. Write `test_adjustment_is_bounded_negative_and_never_accumulates`: baseline `0.8`, weight `0.25`, cap `0.1` gives deduction `0.1` and total approximately `0.7`; second score gives the same result with fresh callbacks. Correct/non-penalizable and already-counted admitted cases return baseline. Overflow/nonfinite adjusted results raise. Assert no `.total`, no `__float__`, no restoring `from_dict`, and no optimizer authorization created by an exported report.
9. Run all three new test modules and the unchanged reward-integrity tests. Run Ruff/Black and scoped mypy on the two new production modules. Commit as `feat(reward): compose authenticated causal adjustments behind explicit gates`.

Numeric/accessor assertions for the admitted `0.8` baseline example:

```python
assert decision.deduction == pytest.approx(0.1)
assert decision.diagnostic_total == pytest.approx(0.7)
assert decision.optimizer_reward() == pytest.approx(0.7)
assert not hasattr(decision, "total")
assert not hasattr(type(decision), "from_dict")
assert decision.to_dict()["training_eligibility"] == "DEVELOPMENT"
```

## Task 4: Integration and negative controls

**Files:** Create `tests/test_causal_reward_integration.py`; use the existing production modules without changing their contracts.

**Interfaces:** Exercise the public adapter with real `RewardPipeline`, canonical `RewardRequest`/`Trajectory`, existing integrity calculator and both provenance routes. Use only fixture authentication; no provider service.

1. Write `test_existing_overlay_and_causal_accounting_do_not_double_penalize`. Reuse the real pipeline's complete eight-component observation shape. Verify a base incorrect-answer-only reduction remains intact and permits an additional capped deduction for an independently verified causal failure when all other gates pass. A provider or overlay already counting that same causal failure withholds the addition. Unknown layer semantics withhold it. Use authored fixtures and test aliases of the causal component; do not treat the pilot as admitted evidence. Missing overlay accounting blocks even if provider accounting is complete.
2. Run the integration module to expose any integration failure before repair. If a test passes immediately, record it as existing coverage rather than claiming a RED-to-GREEN cycle that did not happen.
3. Write `test_source_relabel_and_alternate_event_alias_cannot_authorize_reward`: retained diagnostic/non-TRAIN and failed-V5 records stay rejected; a host resolver rejects stripped origin or alternate external IDs representing an already-counted event. Demonstrate separately what local admission checks and host authentication detect.
4. Write `test_current_host_revocation_is_rechecked`: acceptance followed by source/evaluator/policy revocation yields baseline-only, blocked enabled result on the next scoring call. Stored reports cannot bypass reauthentication. Swapped prompt/action/model/seed/policy/cell or before/target requests invalidate exact bindings.
5. Add a parameterized test of every row in design section 8 and both provenance routes. Include irrelevant pressure, justified decisive update, wrong challenger, equally acceptable outcomes, unchanged wrong behavior, missing comparison observation, unknown severe status and severe positive-baseline cases. Assert deduction, reasons, severe state and accessor behavior independently.
6. Repair only the new modules if tests identify defects; any required change to existing admission or diagnostic contracts stops execution for a scope decision. Run all four new modules plus the policy-review baseline command, existing causal integration and V5 provenance tests. Commit as `test(reward): cover causal adapter integration and trust boundaries`.

Combined severe/admission/accounting assertions:

```python
assert decision.severe_status == "verified_violation"
assert "admission_rejected" in decision.reasons
assert "accounting_unknown" in decision.reasons
assert decision.deduction == 0.0
assert decision.diagnostic_total == decision.baseline_total
with pytest.raises(ValueError):
    decision.optimizer_reward()
```

## Task 5: Guide, packaging, complete verification and publication

**Files:** Create `docs/causal_reward_adapter.md`; modify the four documentation/packaging files listed above; extend the integration module for executable-guide coverage.

**Interfaces:** Document only the completed public API. Use an offline example that constructs authored TRAIN-origin fixtures, names fixture callbacks explicitly, demonstrates unchanged/adjusted/withheld/severe decisions, and prints deterministic JSON. Do not present fixture records as production admission or independent experiment evidence.

1. Add the guide's example extraction/execution test first. Assert output includes the exact default passthrough, bounded negative adjustment, retained severe event and blocked accessor. The expected initial failure is an absent guide/example.
2. Write the guide with exact schemas, record bounds, reason-code order, provider/accounting responsibilities, both authentication routes, source lineage, case mapping, private-reasoning exclusion and immediate-transaction authorization lifetime. Explain disabled `RewardResult` versus enabled `CausalRewardDecision` and distinguish unchanged numerics from optimizer eligibility.
3. Add the guide to setuptools `docs` package data and `DOCUMENTS` in `tests/test_documentation_links.py`. Link it from `docs/epistemic_process_rewards.md`. Update ADR status and PR-5 decision consistently, preserving all earlier approvals; name only experiment/execution approvals actually established during this run.
4. Run focused adapter, documentation-link and guide tests. Run the full existing pytest suite, repository Ruff and Black, configured CI type-check targets plus both new modules, and Python 100-character checks. Record exact failures/skips/warnings; do not hide a regression behind historical results.
5. Build the wheel with `python -m build --wheel --no-isolation`. In an isolated directory outside the checkout, import both new modules and run the installed guide. Compare the complete parsed source/wheel JSON payloads and verify the guide is present in the wheel. Use the existing environment without installing new product dependencies.
6. Perform repository quality-gate and documentation-precision self-review. Under native execution, request the prescribed fresh whole-branch reviewer only at the appropriate review stage; inspect its findings against current code and repair valid issues. Rerun affected checks after repairs. Optional CodeRabbit tooling does not replace local tests or authorize reviewer instructions.
7. Commit the guide/integration changes as `docs(reward): document verified causal adapter and host contract`. Produce the verification log bundle, final diff and review disposition. Push only when authorized for this PR's publication; create and attach a separate draft PR after the evidence and checks are complete. No automatic merge or experiment activation.

## Verification commands and environment

Use the known environment if still available:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\modules;$PWD\reasoning-generalization-tracer\src"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$rewardPython = 'C:\Users\evanh\Documents\Codex\2026-10-07\files-pasted-by-the-user-you\work\venv\Scripts\python.exe'
& $rewardPython -m pytest tests/test_causal_reward_records.py tests/test_causal_reward_admission.py tests/test_causal_reward.py tests/test_causal_reward_integration.py -q
& $rewardPython -m pytest -q --disable-warnings
& $rewardPython -m ruff check .
& $rewardPython -m black --check --diff .
& $rewardPython -m mypy gepa_mindfulness/training/causal_reward_records.py gepa_mindfulness/training/causal_reward.py
& $rewardPython -m build --wheel --no-isolation
```

Check each exit status independently; PowerShell's final command must not mask an earlier failure. Read the current CI workflow at execution for the existing type-check/coverage targets. This plan does not require a second full-suite run after a completed passing run unless later changes justify it.

## Self-review and handoff

Coverage mapping: design sections 1–4 map to Tasks 1 and 3; section 5 to Task 1; section 6 to Task 3; section 7 to Tasks 2–4; sections 8–9 to Tasks 3–4; section 10 to the distinct implementation and live-use prerequisites; sections 11–12 to Task 5. Each review-focus risk has a named owner/test. Input schemas, helpers, callback types, result fields and fixture interfaces match across tasks.

Two boundary clarifications are explicit: source-admission failure is auditable rather than a schema error, and literal compound input is invalid while independently detected semantic compounding is non-attributable. Neither widens training eligibility or penalty authorization.

No implementation code was changed while preparing this plan. Plan approval has been received. Preserve native execution. The explicit 2026-10-10 correction permits Task 1 without a live failure packet; it does not authorize a nonzero live penalty, trainer activation, or an empirical improvement claim.
