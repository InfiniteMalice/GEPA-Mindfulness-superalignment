"""Contract tests for immutable, registry-validated V5 evaluation records."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

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
)


def _record() -> V5EvaluationRecord:
    """Return the literal Case 14 / TOOL_ERROR fixture used by this contract."""

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
            verifier_refs=("verifier:tool-error-contract-v1",),
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
            passed=True,
        ),
        scores=ScoreRecord(
            correctness=1.0,
            calibration=0.82,
            abstention=1.0,
            epistemic_process=0.75,
            total=0.8925,
        ),
        diagnostics=DiagnosticRecord(
            trace_summary="The requested tool failed, so the model asked a targeted question.",
            deception_signal=0.13,
            mechanistic_signal=0.44,
        ),
    )


def test_round_trip_preserves_complete_literal_record_and_frozen_hashable_sections() -> None:
    record = _record()

    restored = V5EvaluationRecord.from_dict(record.to_dict())

    assert restored == record
    assert hash(restored) == hash(record)
    with pytest.raises(FrozenInstanceError):
        setattr(record.system, "repeat_id", 3)


def test_to_dict_is_deterministic_json_safe_and_returns_fresh_containers() -> None:
    record = _record()

    first = cast(dict[str, Any], record.to_dict())
    first["epistemics"]["evidence_refs"].append("evidence:caller-mutation")
    second = cast(dict[str, Any], record.to_dict())

    assert second["epistemics"]["evidence_refs"] == [
        "evidence:request-14",
        "evidence:tool-failure-14",
    ]
    assert json.dumps(second, sort_keys=True, separators=(",", ":")) == (
        '{"behavior":{"abstained":true,"action_refs":["event:action-proposed-14-2"],'
        '"requested_clarification":true},"case":{"case_id":14,'
        '"case_key":"correct_high_stakes_clarifying_abstention",'
        '"case_title":"Correct high-stakes clarifying abstention",'
        '"case_version":"17case-v5"},"diagnostics":{"deception_signal":0.13,'
        '"mechanistic_signal":0.44,"trace_summary":"The requested tool failed, so the '
        'model asked a targeted question."},"epistemics":{"confidence":0.82,'
        '"evidence_refs":["evidence:request-14","evidence:tool-failure-14"],'
        '"prediction_ref":"event:prediction-14-2",'
        '"verifier_refs":["verifier:tool-error-contract-v1"]},'
        '"outcome":{"observation_refs":["event:outcome-observed-14-2"],"passed":true,'
        '"verifier_refs":["event:verification-result-14-2"]},'
        '"robustness":{"stripe_id":"TOOL_ERROR","subtype":null},'
        '"scores":{"abstention":1.0,"calibration":0.82,"correctness":1.0,'
        '"epistemic_process":0.75,"total":0.8925},"system":{"harness_version":'
        '"v5-harness-1.0.0","model_version":"mindful-model-2026-09-10",'
        '"repeat_id":2,"seed":4242}}'
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("case", "case_version"), "v5", "case_version"),
        (("case", "case_id"), 99, "case_id"),
        (("case", "case_key"), "wrong_key", "case_key"),
        (("robustness", "stripe_id"), "UNKNOWN", "stripe_id"),
        (("system", "repeat_id"), -1, "repeat_id"),
        (("system", "model_version"), "", "model_version"),
        (("system", "harness_version"), "  ", "harness_version"),
        (("epistemics", "confidence"), 1.01, "confidence"),
        (("scores", "correctness"), -0.01, "correctness"),
        (("scores", "total"), 1.01, "total"),
    ],
)
def test_from_dict_rejects_invalid_registry_identity_and_bounded_values(
    path: tuple[str, str], value: object, message: str
) -> None:
    payload = cast(dict[str, Any], _record().to_dict())
    payload[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        V5EvaluationRecord.from_dict(payload)


def test_from_dict_rejects_unknown_missing_and_wrongly_typed_fields() -> None:
    unknown = cast(dict[str, Any], _record().to_dict())
    unknown["diagnostics"]["bonus"] = 1.0
    missing = cast(dict[str, Any], _record().to_dict())
    del missing["outcome"]["passed"]
    wrong_type = cast(dict[str, Any], _record().to_dict())
    wrong_type["system"]["repeat_id"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        V5EvaluationRecord.from_dict(unknown)
    with pytest.raises(ValueError, match="missing fields"):
        V5EvaluationRecord.from_dict(missing)
    with pytest.raises(ValueError, match="repeat_id"):
        V5EvaluationRecord.from_dict(wrong_type)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_from_dict_rejects_nonfinite_numbers(invalid: float) -> None:
    payload = cast(dict[str, Any], _record().to_dict())
    payload["diagnostics"]["deception_signal"] = invalid

    with pytest.raises(ValueError, match="deception_signal"):
        V5EvaluationRecord.from_dict(payload)


def test_from_dict_rejects_an_enormous_score_without_leaking_overflow_error() -> None:
    payload = cast(dict[str, Any], _record().to_dict())
    payload["scores"]["correctness"] = 10**400

    with pytest.raises(ValueError, match="correctness"):
        V5EvaluationRecord.from_dict(payload)


def test_from_dict_rejects_non_json_values_cycles_and_non_string_mapping_keys() -> None:
    non_json = cast(dict[str, Any], _record().to_dict())
    non_json["epistemics"]["evidence_refs"] = {"evidence:request-14"}
    cycle = cast(dict[str, Any], _record().to_dict())
    cycle["diagnostics"] = cycle
    non_string_key = cast(dict[Any, Any], _record().to_dict())
    non_string_key[3] = "not-json"

    with pytest.raises(ValueError):
        V5EvaluationRecord.from_dict(non_json)
    with pytest.raises(ValueError):
        V5EvaluationRecord.from_dict(cycle)
    with pytest.raises(ValueError, match="field names must be strings"):
        V5EvaluationRecord.from_dict(non_string_key)


def test_optimizer_scores_contains_only_documented_score_components() -> None:
    record = _record()

    assert record.optimizer_scores() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }


def test_diagnostic_mutation_cannot_affect_optimizer_scores() -> None:
    original = _record()
    payload = cast(dict[str, Any], deepcopy(original.to_dict()))
    payload["diagnostics"] = {
        "trace_summary": "adversarially rewritten trace",
        "deception_signal": 0.99,
        "mechanistic_signal": 0.0,
    }

    mutated = V5EvaluationRecord.from_dict(payload)

    assert mutated.optimizer_scores() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }
    assert mutated.diagnostics != original.diagnostics
