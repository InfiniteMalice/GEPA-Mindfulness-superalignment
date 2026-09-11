"""Trust-boundary tests for V5 records and PR-2 action-bound event sequences."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

import evaluation
from evaluation import (
    BehaviorRecord,
    CaseIdentity,
    DiagnosticRecord,
    EpistemicRecord,
    OutcomeRecord,
    RobustnessIdentity,
    ScoreRecord,
    SystemIdentity,
    V5EvaluationRecord,
    validate_v5_record_provenance,
)
from mindful_trace_gepa.logging_schema import EventEnvelope


def _record(
    *,
    passed: bool = True,
    epistemic_process: float = 0.75,
) -> V5EvaluationRecord:
    """Return one literal Case 14 record whose references name the trusted fixture."""

    return V5EvaluationRecord(
        case=CaseIdentity(
            case_id=14,
            case_version="17case-v5",
            case_key="correct_high_stakes_clarifying_abstention",
            case_title="Correct high-stakes clarifying abstention",
        ),
        robustness=RobustnessIdentity(stripe_id="TOOL_ERROR", subtype=None),
        system=SystemIdentity(
            repeat_id=2,
            seed=4_242,
            model_version="mindful-model-2026-09-10",
            harness_version="v5-harness-1.0.0",
        ),
        epistemics=EpistemicRecord(
            prediction_ref="event:prediction-14-2",
            evidence_refs=("evidence:request-14", "evidence:tool-failure-14"),
            verifier_refs=("event:verification-result-14-2",),
            confidence=0.82,
        ),
        behavior=BehaviorRecord(
            action_refs=("event:action-proposed-14-2",),
            abstained=True,
            requested_clarification=True,
        ),
        outcome=OutcomeRecord(
            observation_refs=("event:outcome-observed-14-2",),
            verifier_refs=("event:verification-result-14-2",),
            passed=passed,
        ),
        scores=ScoreRecord(
            correctness=float(passed),
            calibration=0.82,
            abstention=1.0,
            epistemic_process=epistemic_process,
            total=0.8925 if passed else 0.25,
        ),
        diagnostics=DiagnosticRecord(
            trace_summary="The requested tool failed, so the model asked a targeted question.",
            deception_signal=0.13,
            mechanistic_signal=0.44,
        ),
    )


def _base_metadata() -> dict[str, object]:
    """Return the exact cell identity required on every action-bound fixture event."""

    return {
        "run_id": "run:case-14-tool-error-repeat-2",
        "repeat_id": 2,
        "model_version": "mindful-model-2026-09-10",
        "harness_version": "v5-harness-1.0.0",
        "case_version": "17case-v5",
        "case_id": 14,
        "stripe_id": "TOOL_ERROR",
        "seed": 4_242,
    }


def _event(
    *,
    event_id: str,
    event_type: str,
    timestamp: str,
    payload: dict[str, object],
    **metadata: object,
) -> EventEnvelope:
    """Construct one literal PR-2 envelope without using production event helpers."""

    return EventEnvelope(
        schema_version="1.0",
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        payload=payload,
        **cast(Any, {**_base_metadata(), **metadata}),
    )


def _verified_sequence(
    *,
    verified: bool = True,
    outcome_passed: bool = True,
    epistemic_assessment: str = "verified",
    case_assessment: str | None = None,
) -> tuple[EventEnvelope, ...]:
    """Return a complete same-cell prediction-to-assessment event sequence."""

    prediction = _event(
        event_id="event:prediction-14-2",
        event_type="prediction_commit",
        timestamp="2026-09-10T12:00:00Z",
        evidence_refs=("evidence:request-14", "evidence:tool-failure-14"),
        payload={
            "prediction_commit_id": "prediction-14-2",
            "predicted_outcome": {"behavior": "clarify"},
            "confidence": 0.82,
            "evidence_refs": ["evidence:request-14", "evidence:tool-failure-14"],
        },
    )
    proposed = _event(
        event_id="event:action-proposed-14-2",
        event_type="action_proposed",
        timestamp="2026-09-10T12:00:01Z",
        parent_event_ids=(prediction.event_id,),
        action_id="action-14-2",
        authorization_scope="evaluation-only",
        payload={
            "action_id": "action-14-2",
            "action_class": "request_clarification",
            "reversible": True,
            "authorization_scope": "evaluation-only",
            "prediction_commit_id": "prediction-14-2",
        },
    )
    executed = replace(
        proposed,
        event_id="event:action-executed-14-2",
        event_type="action_executed",
        timestamp="2026-09-10T12:00:02Z",
        parent_event_ids=(proposed.event_id,),
    )
    observation = _event(
        event_id="event:outcome-observed-14-2",
        event_type="outcome_observed",
        timestamp="2026-09-10T12:00:03Z",
        parent_event_ids=(executed.event_id,),
        action_id="action-14-2",
        evidence_refs=("evidence:observed-clarification-14",),
        payload={
            "observation_id": "observation-14-2",
            "action_id": "action-14-2",
            "actual_outcome": {"passed": outcome_passed},
            "evidence_refs": ["evidence:observed-clarification-14"],
        },
    )
    verification = _event(
        event_id="event:verification-result-14-2",
        event_type="verification_result",
        timestamp="2026-09-10T12:00:04Z",
        parent_event_ids=(observation.event_id,),
        action_id="action-14-2",
        verifier_refs=("verifier:tool-error-contract-v1",),
        payload={
            "verifier_id": "tool-error-contract",
            "verifier_version": "v1",
            "observation_id": "observation-14-2",
            "verified": verified,
            "verifier_refs": ["verifier:tool-error-contract-v1"],
        },
    )
    epistemic = _event(
        event_id="event:epistemic-assessment-14-2",
        event_type="epistemic_assessment",
        timestamp="2026-09-10T12:00:05Z",
        parent_event_ids=(verification.event_id,),
        action_id="action-14-2",
        payload={"assessment": epistemic_assessment},
    )
    resolved_case_assessment = case_assessment
    if resolved_case_assessment is None:
        resolved_case_assessment = "pass" if outcome_passed else "fail"
    case = _event(
        event_id="event:case-assessment-14-2",
        event_type="case_assessment",
        timestamp="2026-09-10T12:00:06Z",
        parent_event_ids=(epistemic.event_id,),
        action_id="action-14-2",
        payload={"assessment": resolved_case_assessment},
    )
    return prediction, proposed, executed, observation, verification, epistemic, case


def test_validated_same_cell_sequence_authorizes_optimizer_scores() -> None:
    """Removing sequence validation must not leave positive optimizer credit available."""

    record = _record()
    events = _verified_sequence()

    verified = validate_v5_record_provenance(record, events)

    assert verified.run_id == "run:case-14-tool-error-repeat-2"
    assert verified.optimizer_scores() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }
    assert record.optimizer_scores(events) == verified.optimizer_scores()


def test_external_record_alone_cannot_mint_optimizer_scores() -> None:
    """Deserialized score claims must not authorize themselves without PR-2 provenance."""

    untrusted = V5EvaluationRecord.from_dict(_record().to_dict())

    with pytest.raises(TypeError):
        untrusted.optimizer_scores()  # type: ignore[call-arg]


def test_public_api_exposes_only_a_nonconstructible_verified_result_contract() -> None:
    """An exported implementation type must not offer an unchecked authorization mint."""

    verified = validate_v5_record_provenance(_record(), _verified_sequence())
    result_contract = evaluation.V5ProvenanceResult

    assert not hasattr(evaluation, "VerifiedV5Evaluation")
    assert "VerifiedV5Evaluation" not in evaluation.__all__
    assert isinstance(verified, result_contract)
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        result_contract()  # type: ignore[misc]


def test_runtime_verified_result_constructor_always_revalidates_provenance() -> None:
    """Obtaining the private runtime type must not reveal an unchecked constructor or factory."""

    events = _verified_sequence()
    verified = validate_v5_record_provenance(_record(), events)
    runtime_type = cast(Any, type(verified))
    record_payload = cast(dict[str, Any], _record().to_dict())
    record_payload["outcome"]["passed"] = False
    counterfeit = V5EvaluationRecord.from_dict(record_payload)

    assert not hasattr(runtime_type, "_from_validated")
    with pytest.raises(ValueError, match="observed outcome.*passed"):
        runtime_type(counterfeit, events)


def test_empty_positive_provenance_and_passing_refs_fail_closed() -> None:
    """Empty strings arrays must not stand in for linked evidence and verifier events."""

    payload = cast(dict[str, Any], _record().to_dict())
    payload["epistemics"]["evidence_refs"] = []
    payload["epistemics"]["verifier_refs"] = []
    payload["behavior"]["action_refs"] = []
    payload["outcome"]["observation_refs"] = []
    payload["outcome"]["verifier_refs"] = []
    counterfeit = V5EvaluationRecord.from_dict(payload)

    with pytest.raises(ValueError, match="evidence_refs"):
        counterfeit.optimizer_scores(_verified_sequence())


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("epistemics", "prediction_ref", "event:verification-result-14-2", "prediction_commit"),
        ("epistemics", "prediction_ref", "event:missing-prediction", "unresolved"),
        ("epistemics", "evidence_refs", ["evidence:invented"], "evidence_refs"),
        ("epistemics", "verifier_refs", ["event:missing-verifier"], "unresolved"),
        ("behavior", "action_refs", ["event:prediction-14-2"], "action"),
        ("outcome", "observation_refs", ["event:action-proposed-14-2"], "outcome_observed"),
        ("outcome", "verifier_refs", ["event:epistemic-assessment-14-2"], "verification"),
    ],
)
def test_record_references_must_resolve_to_their_exact_event_roles(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    """A nonempty but detached or wrong-type reference must never satisfy provenance."""

    payload = cast(dict[str, Any], _record().to_dict())
    payload[section][field] = value
    record = V5EvaluationRecord.from_dict(payload)

    with pytest.raises(ValueError, match=message):
        record.optimizer_scores(_verified_sequence())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat_id", 3),
        ("model_version", "different-model"),
        ("harness_version", "different-harness"),
        ("case_version", "different-case-version"),
        ("case_id", 15),
        ("stripe_id", "NONE"),
        ("seed", 4_243),
    ],
)
def test_event_sequence_identity_must_equal_the_record_cell(field: str, value: object) -> None:
    """A self-consistent sequence for another cell must not authorize this record."""

    events = tuple(replace(event, **{field: value}) for event in _verified_sequence())

    with pytest.raises(ValueError, match=field):
        _record().optimizer_scores(events)


def test_passing_outcome_and_positive_process_reject_unverified_results() -> None:
    """A verification event with ``verified=False`` cannot authorize pass or process credit."""

    with pytest.raises(ValueError, match="verified=True"):
        _record().optimizer_scores(_verified_sequence(verified=False))


@pytest.mark.parametrize(
    ("event_passed", "record_passed"),
    [(False, True), (True, False)],
)
def test_record_outcome_claim_must_match_independent_observed_truth(
    event_passed: bool,
    record_passed: bool,
) -> None:
    """Changing only an untrusted record claim must not contradict observed outcome truth."""

    event_truth = _verified_sequence(outcome_passed=event_passed)
    record_payload = cast(dict[str, Any], _record(passed=event_passed).to_dict())
    record_payload["outcome"]["passed"] = record_passed
    untrusted_record = V5EvaluationRecord.from_dict(record_payload)

    with pytest.raises(ValueError, match="observed outcome.*passed"):
        untrusted_record.optimizer_scores(event_truth)


@pytest.mark.parametrize(
    "actual_outcome",
    [
        {"passed": 1},
        {"passed": True, "detail": "clarified"},
        {"result": True},
    ],
)
def test_observed_outcome_requires_exact_typed_v5_pass_mapping(
    actual_outcome: dict[str, object],
) -> None:
    """Truthy aliases, missing fields, and extra fields must not become outcome truth."""

    events = list(_verified_sequence())
    observation_payload = dict(events[3].payload)
    observation_payload["actual_outcome"] = actual_outcome
    events[3] = replace(events[3], payload=observation_payload)

    with pytest.raises(ValueError, match="observed outcome.*passed"):
        _record().optimizer_scores(events)


@pytest.mark.parametrize(
    ("event_passed", "case_assessment"),
    [(True, "fail"), (False, "pass")],
)
def test_record_outcome_claim_must_match_independent_active_case_assessment(
    event_passed: bool,
    case_assessment: str,
) -> None:
    """A verified observation cannot substitute for the matching case-assessment result."""

    event_truth = _verified_sequence(
        outcome_passed=event_passed,
        case_assessment=case_assessment,
    )
    record_payload = cast(dict[str, Any], _record(passed=not event_passed).to_dict())
    record_payload["outcome"]["passed"] = event_passed
    untrusted_record = V5EvaluationRecord.from_dict(record_payload)

    with pytest.raises(ValueError, match="case_assessment.*passed"):
        untrusted_record.optimizer_scores(event_truth)


def test_superseded_passing_case_assessment_cannot_authorize_outcome() -> None:
    """A stale pass result must not override its active failing successor."""

    events = list(_verified_sequence())
    successor_id = "event:case-assessment-successor-14-2"
    events[6] = replace(events[6], superseded_by=successor_id)
    events.append(
        replace(
            events[6],
            event_id=successor_id,
            timestamp="2026-09-10T12:00:07Z",
            payload={"assessment": "fail"},
            superseded_by=None,
        )
    )

    with pytest.raises(ValueError, match="active case_assessment.*passed"):
        _record().optimizer_scores(events)


@pytest.mark.parametrize(
    "payload",
    [
        {"assessment": True},
        {"assessment": "passed"},
        {"assessment": "pass", "score": 1.0},
    ],
)
def test_active_case_assessment_requires_exact_typed_v5_payload(
    payload: dict[str, object],
) -> None:
    """Untyped, ambiguous, or expanded case payloads cannot authorize outcome truth."""

    events = list(_verified_sequence())
    events[6] = replace(events[6], payload=payload)

    with pytest.raises(ValueError, match="case_assessment.*assessment"):
        _record().optimizer_scores(events)


def test_positive_process_requires_an_epistemic_assessment_route() -> None:
    """A verified observation alone must not mint positive epistemic-process credit."""

    events = _verified_sequence()[:5]

    with pytest.raises(ValueError, match="epistemic_assessment"):
        _record().optimizer_scores(events)


def _superseded_epistemic_sequence(
    *,
    stale_assessment: str,
    successor_assessment: str,
) -> tuple[EventEnvelope, ...]:
    """Return one independently authored epistemic replacement followed by its case result."""

    events = list(_verified_sequence(epistemic_assessment=stale_assessment))
    successor_id = "event:epistemic-assessment-successor-14-2"
    stale = replace(events[5], superseded_by=successor_id)
    successor = replace(
        events[5],
        event_id=successor_id,
        timestamp="2026-09-10T12:00:06Z",
        payload={"assessment": successor_assessment},
    )
    case = replace(
        events[6],
        timestamp="2026-09-10T12:00:07Z",
        parent_event_ids=(successor.event_id,),
    )
    return *events[:5], stale, successor, case


def test_superseded_qualifying_epistemic_assessment_cannot_authorize_process_credit() -> None:
    """A stale verified assessment must not override its active unverified successor."""

    events = _superseded_epistemic_sequence(
        stale_assessment="verified",
        successor_assessment="unverified",
    )

    with pytest.raises(ValueError, match="active epistemic_assessment.*verified"):
        _record().optimizer_scores(events)


def test_active_qualifying_epistemic_successor_authorizes_process_credit() -> None:
    """A valid active verified successor must replace an earlier unverified assessment."""

    events = _superseded_epistemic_sequence(
        stale_assessment="unverified",
        successor_assessment="verified",
    )

    assert _record().optimizer_scores(events)["epistemic_process"] == 0.75


@pytest.mark.parametrize(
    "payload",
    [
        {"assessment": "verified", "score": 1.0},
        {"assessment": True},
        {"assessment": "plausible"},
    ],
)
def test_active_epistemic_assessment_requires_exact_typed_v5_payload(
    payload: dict[str, object],
) -> None:
    """Untyped, ambiguous, or expanded assessment payloads cannot authorize process credit."""

    events = list(_verified_sequence())
    events[5] = replace(events[5], payload=payload)

    with pytest.raises(ValueError, match="epistemic_assessment.*assessment"):
        _record().optimizer_scores(events)


def test_failed_zero_process_record_accepts_resolved_negative_verification() -> None:
    """A failing record may retain an audited negative verification without positive credit."""

    record = _record(passed=False, epistemic_process=0.0)
    events = _verified_sequence(
        outcome_passed=False,
        epistemic_assessment="unverified",
    )

    assert record.optimizer_scores(events) == {
        "correctness": 0.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.0,
        "total": 0.25,
    }


def test_verified_wrapper_is_detached_from_later_record_event_and_diagnostic_mutation() -> None:
    """Caller mutation after validation must not change the immutable verified score snapshot."""

    record = _record()
    events = list(_verified_sequence())
    verified = validate_v5_record_provenance(record, events)

    object.__setattr__(record.scores, "total", 0.0)
    object.__setattr__(record.diagnostics, "deception_signal", 99.0)
    object.__setattr__(events[0], "case_id", 1)

    assert verified.optimizer_scores() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }


def test_verified_optimizer_output_does_not_dispatch_to_a_score_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing a serializer must not inject diagnostic fields into verified output."""

    verified = validate_v5_record_provenance(_record(), _verified_sequence())
    monkeypatch.setattr(
        ScoreRecord,
        "to_dict",
        lambda self: {"diagnostics": ["counterfeit-score-serializer"]},
    )

    assert verified.optimizer_scores() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }


def test_optimizer_use_revalidates_mutated_score_values() -> None:
    """A frozen-object bypass must fail before an invalid score reaches optimizer output."""

    record = _record()
    object.__setattr__(record.scores, "epistemic_process", 2.0)

    with pytest.raises(ValueError, match="epistemic_process"):
        record.optimizer_scores(_verified_sequence())
