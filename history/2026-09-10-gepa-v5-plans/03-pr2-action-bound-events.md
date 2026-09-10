# PR-2 Action-Bound Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` during
> implementation and preserve backward compatibility for legacy event rows.

**Goal:** Record prediction, action, outcome, verification, and epistemic assessment as linked,
immutable structured events.

**Architecture:** Extend `EventEnvelope` with optional linkage metadata and add focused frozen
payload records. A sequence validator enforces prediction-before-action and immutable version
identity without introducing a second logging framework.

**Tech Stack:** Python dataclasses, enums, JSON-compatible mappings, pytest.

**Spec:** `history/2026-09-10-gepa-v5-unified-architecture-design.md`

## Task 1: Extend event types and envelope linkage

**Files:**

- Modify: `src/mindful_trace_gepa/logging_schema.py`
- Modify: `docs/structured_logging.md`
- Test: `tests/test_action_bound_logging.py`
- Modify: `tests/test_structured_logging_contract.py`

**New enum values:**

```python
PREDICTION_COMMIT = "prediction_commit"
ACTION_PROPOSED = "action_proposed"
ACTION_EXECUTED = "action_executed"
OUTCOME_OBSERVED = "outcome_observed"
VERIFICATION_RESULT = "verification_result"
EPISTEMIC_ASSESSMENT = "epistemic_assessment"
CASE_ASSESSMENT = "case_assessment"
```

**New optional envelope fields:**

```python
action_id: str | None = None
parent_event_ids: tuple[str, ...] = ()
evidence_refs: tuple[str, ...] = ()
model_version: str | None = None
harness_version: str | None = None
case_version: str | None = None
case_id: int | None = None
stripe_id: str | None = None
repeat_id: int | None = None
seed: int | None = None
authorization_scope: str | None = None
verifier_refs: tuple[str, ...] = ()
valid_from: str | None = None
valid_until: str | None = None
superseded_by: str | None = None
```

1. Write failing round-trip tests for every new event type and linkage field.
2. Add failing validation tests for blank IDs, negative repeat IDs, canonical case bounds, and
   `valid_until` earlier than `valid_from`.
3. Add a legacy-row test proving old event rows normalize without new fields.
4. Run tests and verify failures against the current envelope.
5. Add fields with immutable tuple coercion in `__post_init__`. Preserve `to_dict()` omission of
   `None`; serialize empty tuples as empty lists only when explicitly retained by `asdict`.
6. Make `make_event_envelope()` accept the new IDs without changing existing call sites.
7. Update structured logging documentation with raw-evidence append-only and derived-event
   supersession rules.
8. Run the focused tests.

## Task 2: Add typed action-bound payloads

**Files:**

- Create: `src/mindful_trace_gepa/action_bound_events.py`
- Modify: `src/mindful_trace_gepa/__init__.py`
- Test: `tests/test_action_bound_events.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PredictionCommit:
    prediction_commit_id: str
    predicted_outcome: object
    confidence: float
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    action_class: str
    reversible: bool
    authorization_scope: str
    prediction_commit_id: str


@dataclass(frozen=True)
class OutcomeObservation:
    observation_id: str
    action_id: str
    actual_outcome: object
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class VerificationResult:
    verifier_id: str
    verifier_version: str
    observation_id: str
    verified: bool
    verifier_refs: tuple[str, ...]
```

1. Write failing tests for blank identifiers, out-of-range confidence, missing prediction link,
   missing authorization scope, and missing evidence on an observed outcome.
2. Add a test proving a frozen `PredictionCommit` raises `FrozenInstanceError` on mutation.
3. Run tests and verify module-not-found failure.
4. Implement the records with JSON-compatible `to_dict()` methods and strict validation.
5. Add `make_*_event()` helpers that wrap payloads in the existing `EventEnvelope`.
6. Run `python -m pytest tests/test_action_bound_events.py -q`.

## Task 3: Enforce event sequence and coupled version identity

**Files:**

- Create: `src/mindful_trace_gepa/event_sequence.py`
- Test: `tests/test_action_bound_event_sequence.py`

**Interface:**

```python
@dataclass(frozen=True)
class EvaluatedSystemVersion:
    model_version: str
    harness_version: str


def validate_action_bound_sequence(events: Sequence[EventEnvelope]) -> None:
    ...
```

1. Write a valid literal fixture containing prediction commit, proposed action, executed action,
   observed outcome, verification result, epistemic assessment, and case assessment.
2. Write failing tests for action before prediction, outcome before execution, verifier without
   observation, missing parent links, prediction rewrite under the same ID, and model or harness
   version drift inside one run/repeat.
3. Run tests and verify failures because the validator is absent.
4. Implement one-pass indexing by event ID and semantic payload IDs. Reject duplicates and forward
   references that violate the required temporal order.
5. Permit derived assessments to use `superseded_by`; forbid supersession on prediction commits,
   executed actions, and raw outcome observations.
6. Run the sequence tests and structured logging regression tests.

## Task 4: Update trainer logging references

**Files:**

- Modify: `src/mindful_trace_gepa/logging_schema.py`
- Modify: `tests/test_structured_logging_extensions.py`
- Modify: `tests/test_training_cli_logging.py`

1. Add failing tests for optional prediction, action, observation, verification, epistemic, and
   case assessment references in `trainer_metric_optional_fields()`.
2. Add the exact allowed keys and verify unknown keys remain excluded.
3. Run structured logging and training logging tests.

## Task 5: Verify and commit PR-2

1. Run all action-bound and structured logging tests.
2. Run viewer tests to prove new envelopes remain viewable through legacy normalization.
3. Run Black, Ruff, and mypy on `src/mindful_trace_gepa` and modified tests.
4. Run a JSON serialization smoke test for a complete event sequence.
5. Run `git diff --check`, inspect the stage diff, and commit with message
   `feat: add action-bound epistemic events`.
