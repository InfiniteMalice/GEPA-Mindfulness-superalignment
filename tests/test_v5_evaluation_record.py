"""Contract tests for immutable, registry-validated V5 evaluation records."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
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
    v5_records,
)
from evaluation.cases.registry import RobustnessStripeRegistry


class _SubclassCaseIdentity(CaseIdentity):
    """Deliberately non-canonical case section used at the root trust boundary."""


class _SubclassRobustnessIdentity(RobustnessIdentity):
    """Deliberately non-canonical robustness section used at the root trust boundary."""


class _SubclassSystemIdentity(SystemIdentity):
    """Deliberately non-canonical system section used at the root trust boundary."""


class _SubclassEpistemicRecord(EpistemicRecord):
    """Deliberately non-canonical epistemic section used at the root trust boundary."""


class _SubclassBehaviorRecord(BehaviorRecord):
    """Deliberately non-canonical behavior section used at the root trust boundary."""


class _SubclassOutcomeRecord(OutcomeRecord):
    """Deliberately non-canonical outcome section used at the root trust boundary."""


class _MaliciousScoreRecord(ScoreRecord):
    """A score subclass that carries mutable diagnostics and overrides serialization."""

    mutable_state: dict[str, list[str]]

    def __init__(self) -> None:
        super().__init__(
            correctness=1.0,
            calibration=0.82,
            abstention=1.0,
            epistemic_process=0.75,
            total=0.8925,
        )
        object.__setattr__(self, "mutable_state", {"diagnostics": ["leaked-trace"]})

    def to_dict(self) -> dict[str, object]:
        """Attempt to add diagnostic data to an optimizer-facing serialization."""

        return {**super().to_dict(), **self.mutable_state}


class _SubclassDiagnosticRecord(DiagnosticRecord):
    """Deliberately non-canonical diagnostic section used at the root trust boundary."""


class _StringSubclass(str):
    """A string subclass that must not cross exact V5 scalar boundaries."""


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


@pytest.mark.parametrize(
    ("field_name", "section", "expected_type"),
    [
        (
            "case",
            _SubclassCaseIdentity(
                case_id=14,
                case_version="17case-v5",
                case_key="correct_high_stakes_clarifying_abstention",
                case_title="Correct high-stakes clarifying abstention",
            ),
            "CaseIdentity",
        ),
        (
            "robustness",
            _SubclassRobustnessIdentity(stripe_id="TOOL_ERROR", subtype=None),
            "RobustnessIdentity",
        ),
        (
            "system",
            _SubclassSystemIdentity(
                repeat_id=2,
                seed=4_242,
                model_version="mindful-model-2026-09-10",
                harness_version="v5-harness-1.0.0",
            ),
            "SystemIdentity",
        ),
        (
            "epistemics",
            _SubclassEpistemicRecord(
                prediction_ref="event:prediction-14-2",
                evidence_refs=("evidence:request-14", "evidence:tool-failure-14"),
                verifier_refs=("verifier:tool-error-contract-v1",),
                confidence=0.82,
            ),
            "EpistemicRecord",
        ),
        (
            "behavior",
            _SubclassBehaviorRecord(
                action_refs=("event:action-proposed-14-2",),
                abstained=True,
                requested_clarification=True,
            ),
            "BehaviorRecord",
        ),
        (
            "outcome",
            _SubclassOutcomeRecord(
                observation_refs=("event:outcome-observed-14-2",),
                verifier_refs=("event:verification-result-14-2",),
                passed=True,
            ),
            "OutcomeRecord",
        ),
        ("scores", _MaliciousScoreRecord(), "ScoreRecord"),
        (
            "diagnostics",
            _SubclassDiagnosticRecord(
                trace_summary="The requested tool failed, so the model asked a targeted question.",
                deception_signal=0.13,
                mechanistic_signal=0.44,
            ),
            "DiagnosticRecord",
        ),
    ],
)
def test_record_rejects_subclasses_for_every_nested_section(
    field_name: str, section: object, expected_type: str
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be an exact {expected_type}"):
        cast(Any, replace)(_record(), **{field_name: section})


def test_root_record_snapshots_every_caller_owned_nested_section() -> None:
    """Mutating a source section after root construction must not rewrite the root record."""

    source = _record()
    sections = {
        "case": source.case,
        "robustness": source.robustness,
        "system": source.system,
        "epistemics": source.epistemics,
        "behavior": source.behavior,
        "outcome": source.outcome,
        "scores": source.scores,
        "diagnostics": source.diagnostics,
    }
    record = V5EvaluationRecord(**cast(Any, sections))
    before = record.to_dict()

    assert all(getattr(record, name) is not section for name, section in sections.items())
    object.__setattr__(sections["case"], "case_title", "caller rewrite")
    object.__setattr__(sections["robustness"], "stripe_id", "NONE")
    object.__setattr__(sections["system"], "seed", 99)
    object.__setattr__(sections["epistemics"], "confidence", 0.0)
    object.__setattr__(sections["behavior"], "abstained", False)
    object.__setattr__(sections["outcome"], "passed", False)
    object.__setattr__(sections["scores"], "total", 0.0)
    object.__setattr__(sections["diagnostics"], "deception_signal", 99.0)

    assert record.to_dict() == before


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("case", "case_version", _StringSubclass("17case-v5")),
        (
            "case",
            "case_key",
            _StringSubclass("correct_high_stakes_clarifying_abstention"),
        ),
        (
            "case",
            "case_title",
            _StringSubclass("Correct high-stakes clarifying abstention"),
        ),
        ("robustness", "stripe_id", _StringSubclass("TOOL_ERROR")),
        ("system", "model_version", _StringSubclass("mindful-model-2026-09-10")),
        ("system", "harness_version", _StringSubclass("v5-harness-1.0.0")),
        ("epistemics", "prediction_ref", _StringSubclass("event:prediction-14-2")),
        ("epistemics", "evidence_refs", [_StringSubclass("evidence:request-14")]),
        (
            "epistemics",
            "verifier_refs",
            [_StringSubclass("event:verification-result-14-2")],
        ),
        ("behavior", "action_refs", [_StringSubclass("event:action-proposed-14-2")]),
        ("outcome", "observation_refs", [_StringSubclass("event:outcome-observed-14-2")]),
        (
            "outcome",
            "verifier_refs",
            [_StringSubclass("event:verification-result-14-2")],
        ),
        (
            "diagnostics",
            "trace_summary",
            _StringSubclass("The requested tool failed, so the model asked a targeted question."),
        ),
    ],
)
def test_from_dict_rejects_string_subclasses_for_every_record_identifier(
    section: str,
    field: str,
    value: object,
) -> None:
    """A value-equal string subclass must not bypass V5 identity validation."""

    payload = cast(dict[str, Any], _record().to_dict())
    payload[section][field] = value

    with pytest.raises(ValueError, match=field):
        V5EvaluationRecord.from_dict(payload)


def test_record_subtype_rejects_a_string_subclass_even_when_value_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subtype membership must not allow a value-equal custom string implementation."""

    registry = v5_records.load_stripe_registry()
    stripes = tuple(
        replace(stripe, allowed_subtypes=("retry",)) if stripe.id == "TOOL_ERROR" else stripe
        for stripe in registry.stripes
    )
    monkeypatch.setattr(
        v5_records,
        "load_stripe_registry",
        lambda: RobustnessStripeRegistry(registry.registry_version, stripes),
    )
    v5_records._canonical_stripe_map.cache_clear()
    try:
        with pytest.raises(ValueError, match="subtype"):
            RobustnessIdentity("TOOL_ERROR", _StringSubclass("retry"))
    finally:
        v5_records._canonical_stripe_map.cache_clear()


def test_record_registry_identity_loaders_are_called_once_per_cached_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing many record sections must not repeatedly parse the packaged YAML files."""

    case_loader = v5_records.load_case_manifest
    stripe_loader = v5_records.load_stripe_registry
    calls = {"case": 0, "stripe": 0}

    def counted_case_loader():
        calls["case"] += 1
        return case_loader()

    def counted_stripe_loader():
        calls["stripe"] += 1
        return stripe_loader()

    monkeypatch.setattr(v5_records, "load_case_manifest", counted_case_loader)
    monkeypatch.setattr(v5_records, "load_stripe_registry", counted_stripe_loader)
    v5_records._canonical_case_map.cache_clear()
    v5_records._canonical_stripe_map.cache_clear()
    try:
        for _ in range(5):
            _record()
        assert calls == {"case": 1, "stripe": 1}
        with pytest.raises(TypeError):
            cast(Any, v5_records._canonical_case_map())[14] = ("rewrite", "rewrite")
        with pytest.raises(TypeError):
            cast(Any, v5_records._canonical_stripe_map())["TOOL_ERROR"] = ()
    finally:
        v5_records._canonical_case_map.cache_clear()
        v5_records._canonical_stripe_map.cache_clear()


def test_malicious_score_subclass_cannot_inject_mutable_diagnostics() -> None:
    malicious = _MaliciousScoreRecord()

    assert malicious.to_dict() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
        "diagnostics": ["leaked-trace"],
    }
    with pytest.raises(ValueError, match=r"scores must be an exact ScoreRecord"):
        replace(_record(), scores=malicious)


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


def test_optimizer_scores_requires_action_bound_provenance() -> None:
    record = _record()

    with pytest.raises(TypeError):
        record.optimizer_scores()  # type: ignore[call-arg]


def test_score_section_serialization_has_only_documented_fields() -> None:
    record = _record()

    assert record.scores.to_dict() == {
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

    assert mutated.scores.to_dict() == {
        "correctness": 1.0,
        "calibration": 0.82,
        "abstention": 1.0,
        "epistemic_process": 0.75,
        "total": 0.8925,
    }
    assert mutated.diagnostics != original.diagnostics
