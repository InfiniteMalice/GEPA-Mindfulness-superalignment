"""Experimental alignment-study contracts; every record remains an inert diagnostic."""

# Standard library
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, cast

# Local
from evaluation.causal_records import CausalCapture, CausalPair, PromptTurn, canonical_json
from evaluation.cva_benchmark import BenchmarkCapture, BenchmarkResponse, CaptureVerificationInput
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.interpret.native_causal_records import CausalCommit, run_measurement_digest
from gepa_mindfulness.interpret.native_records import NativeInput
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_records,
    _record,
    array,
    digest,
    references,
    text,
    texts,
)
from gepa_mindfulness.verification.diagnostic_records import restore_refs
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from synthetic_data.cva_fixtures import SecurityFixture, _known_fixture

FAMILIES = (
    "grounded_honesty",
    "abstention",
    "clarification",
    "semantic_laundering",
    "dual_path_specificity",
    "constitutional_conflict",
    "constraint_verified_alignment",
)
CONDITIONS = (
    "baseline",
    "behaviorally_trained",
    "diagnostic_instrumented",
    "intervention_informed",
)


def legacy(value: Any, kind: type) -> Any:
    if type(value) is not kind:
        raise ValueError(f"expected exact {kind.__name__}")
    payload = cast(Any, value).to_dict()
    validate_input_envelope(payload)
    return kind.from_dict(payload)


def _system(value: Any) -> SystemIdentity:
    if type(value) is not dict or set(value) != {f.name for f in fields(SystemIdentity)}:
        raise ValueError("incorrect system identity fields")
    return SystemIdentity(**value)


def _turns(values: tuple[PromptTurn, ...]) -> None:
    for turn in array(values, 12, required=True):
        legacy(turn, PromptTurn)
        text(turn.content, "turn content", 16384)


def _factors(values: tuple[tuple[str, str], ...]) -> None:
    for row in array(values, 16, required=True):
        array(row, 2, required=True)
        if len(row) != 2:
            raise ValueError("factor requires name and value")
        text(row[0], "factor name")
        text(row[1], "factor value", 32768)
        try:
            payload = json.loads(row[1])
            validate_input_envelope(payload)
        except (ValueError, TypeError, RecursionError) as error:
            raise ValueError("invalid factor JSON") from error
        if canonical_json(payload) != row[1]:
            raise ValueError("factor must be canonical JSON")
    if len(dict(values)) != len(values):
        raise ValueError("duplicate factor names")


def _read_factors(values: Any) -> tuple:
    return tuple(_read_array(row) for row in _read_array(values))


@dataclass(frozen=True)
class StudyRoles(CVARecord):
    discovery: str
    prediction: str
    executor: str
    verifier: str
    skeptic: str
    registrar: str

    schema_version = "alignment-study-roles-v1"

    def __post_init__(self) -> None:
        for f in fields(self):
            text(getattr(self, f.name), "role actor")


@dataclass(frozen=True)
class StudyTemplate(CVARecord):
    template_id: str
    family: str
    before_turns: tuple[PromptTurn, ...]
    after_turns: tuple[PromptTurn, ...]
    before_factors: tuple[tuple[str, str], ...]
    after_factors: tuple[tuple[str, str], ...]
    before_expected: tuple[str, ...]
    after_expected: tuple[str, ...]
    claimed_equivalence: bool
    source_refs: tuple[EvidenceReference, ...]
    cva_before: SecurityFixture | None = None
    cva_after: SecurityFixture | None = None

    schema_version = "alignment-study-template-v1"
    restorers = {
        "before_turns": lambda v: tuple(PromptTurn.from_dict(t) for t in _read_array(v)),
        "after_turns": lambda v: tuple(PromptTurn.from_dict(t) for t in _read_array(v)),
        "before_factors": _read_factors,
        "after_factors": _read_factors,
        "before_expected": _read_array,
        "after_expected": _read_array,
        "source_refs": restore_refs,
        "cva_before": lambda v: None if v is None else SecurityFixture.from_dict(v),
        "cva_after": lambda v: None if v is None else SecurityFixture.from_dict(v),
    }

    def __post_init__(self) -> None:
        text(self.template_id, "template_id")
        if type(self.family) is not str or self.family not in FAMILIES:
            raise ValueError("unknown study family")
        _turns(self.before_turns)
        _turns(self.after_turns)
        _factors(self.before_factors)
        _factors(self.after_factors)
        if dict(self.before_factors).keys() != dict(self.after_factors).keys():
            raise ValueError("factor rosters differ")
        if dict(self.before_factors) == dict(self.after_factors):
            raise ValueError("a controlled comparison requires declared changes")
        texts(self.before_expected, required=True, maximum=8)
        texts(self.after_expected, required=True, maximum=8)
        if type(self.claimed_equivalence) is not bool:
            raise ValueError("equivalence is a declared boolean, not an established fact")
        references(self.source_refs)
        if (self.cva_before is None) != (self.cva_after is None):
            raise ValueError("CVA study requires both original fixtures")
        if self.cva_before is not None:
            assert self.cva_after is not None
            _known_fixture(self.cva_before)
            _known_fixture(self.cva_after)
            if self.family != "constraint_verified_alignment":
                raise ValueError("CVA fixtures require their research family")


@dataclass(frozen=True)
class StudyItem(CVARecord):
    template_id: str
    family: str
    pair: CausalPair
    cva_before: SecurityFixture | None = None
    cva_after: SecurityFixture | None = None

    schema_version = "alignment-study-item-v1"
    restorers = {
        "pair": CausalPair.from_dict,
        "cva_before": lambda v: None if v is None else SecurityFixture.from_dict(v),
        "cva_after": lambda v: None if v is None else SecurityFixture.from_dict(v),
    }

    def __post_init__(self) -> None:
        text(self.template_id, "template_id")
        if type(self.family) is not str or self.family not in FAMILIES:
            raise ValueError("unknown study family")
        pair = legacy(self.pair, CausalPair)
        if pair.training_eligibility.value != "DEVELOPMENT":
            raise ValueError("this public study path requires DEVELOPMENT data")
        if (self.cva_before is None) != (self.cva_after is None):
            raise ValueError("CVA study requires both fixtures")
        if self.cva_before is not None:
            assert self.cva_after is not None
            _known_fixture(self.cva_before)
            _known_fixture(self.cva_after)
            if self.family != "constraint_verified_alignment":
                raise ValueError("CVA family mismatch")


@dataclass(frozen=True)
class AlignmentStudyPlan(CVARecord):
    plan_id: str
    items: tuple[StudyItem, ...]
    roles: StudyRoles
    condition: str
    physical_model_digest: str | None = None
    candidate_ref: EvidenceReference | None = None

    schema_version = "alignment-study-plan-v1"
    restorers = {
        "items": lambda v: _read_records(v, StudyItem),
        "roles": StudyRoles.from_dict,
        "candidate_ref": lambda v: None if v is None else EvidenceReference.from_dict(v),
    }

    def __post_init__(self) -> None:
        text(self.plan_id, "plan_id")
        items = tuple(_record(i, StudyItem) for i in array(self.items, 64, required=True))
        if len({i.pair.pair_id for i in items}) != len(items):
            raise ValueError("duplicate study pair")
        versions = {
            canonical_json(v.system.to_dict()) for i in items for v in (i.pair.before, i.pair.after)
        }
        if len(versions) > 128:
            raise ValueError("study roster exceeds bounded identity budget")
        ids = [v.variant_id for i in items for v in (i.pair.before, i.pair.after)]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate study variant")
        _record(self.roles, StudyRoles)
        if type(self.condition) is not str or self.condition not in CONDITIONS:
            raise ValueError("unknown study condition")
        if self.physical_model_digest is not None:
            digest(self.physical_model_digest)
        if self.condition == "diagnostic_instrumented" and self.physical_model_digest is None:
            raise ValueError("diagnostic condition requires its physical frozen model")
        candidate = self.condition in ("behaviorally_trained", "intervention_informed")
        if candidate != (self.candidate_ref is not None):
            raise ValueError("candidate conditions require explicit candidate provenance")
        if self.candidate_ref is not None:
            references((self.candidate_ref,))


@dataclass(frozen=True)
class PublicStudyInput(CVARecord):
    variant_id: str
    pair_digest: str
    prompt_digest: str
    system: SystemIdentity
    turns: tuple[PromptTurn, ...]

    schema_version = "alignment-public-input-v1"
    restorers = {
        "system": _system,
        "turns": lambda v: tuple(PromptTurn.from_dict(t) for t in _read_array(v)),
    }

    def __post_init__(self) -> None:
        text(self.variant_id, "variant_id")
        digest(self.pair_digest)
        digest(self.prompt_digest)
        if type(self.system) is not SystemIdentity:
            raise ValueError("exact system identity required")
        self.system.__post_init__()
        _turns(self.turns)
        from evaluation.causal_records import content_digest

        if content_digest([t.to_dict() for t in self.turns]) != self.prompt_digest:
            raise ValueError("public prompt digest mismatch")


@dataclass(frozen=True)
class PublicBehavior(CVARecord):
    actions: tuple[str, ...]
    public_output: str
    evidence_refs: tuple[EvidenceReference, ...]
    native_input: NativeInput | None = None
    physical_model_digest: str | None = None
    tokenizer_id: str | None = None
    tokenizer_version: str | None = None
    cva_capture: BenchmarkCapture | None = None

    schema_version = "alignment-public-behavior-v1"
    restorers = {
        "actions": _read_array,
        "evidence_refs": restore_refs,
        "native_input": lambda v: None if v is None else NativeInput.from_dict(v),
        "cva_capture": lambda v: None if v is None else BenchmarkCapture.from_dict(v),
    }

    def __post_init__(self) -> None:
        # Repeated public actions can be meaningful (e.g. repeated clarification).
        for action in array(self.actions, 32, required=True):
            text(action, "action", 2048)
        text(self.public_output, "public_output", 65536)
        references(self.evidence_refs)
        if self.cva_capture is not None:
            capture = _record(self.cva_capture, BenchmarkCapture)
            if capture.evidence_refs != self.evidence_refs:
                raise ValueError("CVA and public behavior require the same original evidence")
            try:
                parsed = json.loads(self.public_output)
                response = BenchmarkResponse.from_dict(parsed)
            except (ValueError, TypeError, RecursionError) as error:
                raise ValueError(
                    "structured CVA output must preserve its original JSON response"
                ) from error
            if response != capture.response:
                raise ValueError("CVA response differs from the original public output")
        native = self.native_input is not None
        if any(
            (value is not None) != native
            for value in (self.physical_model_digest, self.tokenizer_id, self.tokenizer_version)
        ):
            raise ValueError("native provenance must be complete or absent")
        if native:
            _record(self.native_input, NativeInput)
            digest(self.physical_model_digest)
            text(self.tokenizer_id, "tokenizer_id")
            text(self.tokenizer_version, "tokenizer_version")


@dataclass(frozen=True)
class StudyBehaviorBatch(CVARecord):
    plan_digest: str
    captures: tuple[CausalCapture, ...]
    outputs: tuple[PublicBehavior | None, ...]
    started_at: str
    completed_at: str

    schema_version = "alignment-behavior-batch-v1"
    restorers = {
        "captures": lambda v: tuple(CausalCapture.from_dict(c) for c in _read_array(v)),
        "outputs": lambda v: tuple(
            None if o is None else PublicBehavior.from_dict(o) for o in _read_array(v)
        ),
    }

    def __post_init__(self) -> None:
        digest(self.plan_digest)
        captures = tuple(legacy(c, CausalCapture) for c in array(self.captures, 128, required=True))
        array(self.outputs, 128, required=True)
        if len(captures) != len(self.outputs):
            raise ValueError("capture/output roster mismatch")
        if len({(c.pair_digest, c.variant_id) for c in captures}) != len(captures):
            raise ValueError("duplicate behavior capture")
        for capture, output in zip(captures, self.outputs):
            if (capture.status == "censored") != (output is None):
                raise ValueError("censoring must retain absent output")
            if output is not None:
                result = _record(output, PublicBehavior)
                if (
                    capture.actions != result.actions
                    or capture.evidence_refs != result.evidence_refs
                ):
                    raise ValueError("capture differs from its original public output")
        start = parse_rfc3339_datetime(self.started_at, "started_at")
        if parse_rfc3339_datetime(self.completed_at, "completed_at") < start:
            raise ValueError("behavior completion precedes start")


@dataclass(frozen=True)
class StudyCVAObservation(CVARecord):
    plan_digest: str
    item_digest: str
    behavior_digest: str
    behavior_completed_at: str
    public_input: PublicStudyInput
    public_behavior: PublicBehavior
    behavior_capture: CausalCapture
    benchmark: CaptureVerificationInput

    schema_version = "alignment-cva-observation-v1"
    restorers = {
        "public_input": PublicStudyInput.from_dict,
        "public_behavior": PublicBehavior.from_dict,
        "behavior_capture": CausalCapture.from_dict,
        "benchmark": CaptureVerificationInput.from_dict,
    }

    def __post_init__(self) -> None:
        for value in (self.plan_digest, self.item_digest, self.behavior_digest):
            digest(value)
        parse_rfc3339_datetime(self.behavior_completed_at, "behavior_completed_at")
        view = _record(self.public_input, PublicStudyInput)
        output = _record(self.public_behavior, PublicBehavior)
        capture = legacy(self.behavior_capture, CausalCapture)
        packet = _record(self.benchmark, CaptureVerificationInput)
        if (
            capture.status != "observed"
            or output.cva_capture != packet.capture
            or capture.evidence_refs != output.evidence_refs
            or capture.actions != output.actions
            or capture.pair_digest != view.pair_digest
            or capture.variant_id != view.variant_id
            or capture.prompt_digest != view.prompt_digest
            or capture.system != view.system
            or packet.slot.variant.prompt_digest != view.prompt_digest
            or packet.slot.variant.system != view.system
        ):
            raise ValueError("CVA observation must bind the exact original study capture")


@dataclass(frozen=True)
class StudyInterventionPacket(CVARecord):
    plan: AlignmentStudyPlan
    item: StudyItem
    behavior: StudyBehaviorBatch
    commit: CausalCommit

    schema_version = "alignment-intervention-packet-v1"
    restorers = {
        "plan": AlignmentStudyPlan.from_dict,
        "item": StudyItem.from_dict,
        "behavior": StudyBehaviorBatch.from_dict,
        "commit": CausalCommit.from_dict,
    }

    def __post_init__(self) -> None:
        _record(self.plan, AlignmentStudyPlan)
        _record(self.item, StudyItem)
        _record(self.behavior, StudyBehaviorBatch)
        _record(self.commit, CausalCommit)


@dataclass(frozen=True)
class StudyMechanismRecord(CVARecord):
    plan_digest: str
    item_digest: str
    behavior_digest: str
    report_json: str

    schema_version = "alignment-mechanism-record-v1"

    def __post_init__(self) -> None:
        for value in (self.plan_digest, self.item_digest, self.behavior_digest):
            digest(value)
        text(self.report_json, "causal report", 4 * 1024 * 1024)
        try:
            report = json.loads(self.report_json)
            validate_input_envelope(report)
        except (ValueError, RecursionError, TypeError) as error:
            raise ValueError("invalid causal report JSON") from error
        if type(report) is not dict or canonical_json(report) != self.report_json:
            raise ValueError("causal report requires canonical JSON object")
        CausalCommit.from_dict(report.get("commit"))
        if report.get("measurement_digest") != run_measurement_digest(report):
            raise ValueError("causal measurement digest mismatch")
        if (
            report.get("evidence_status") != "UNRESOLVED"
            or report.get("optimizer_input") is not False
        ):
            raise ValueError("raw study instrumentation cannot self-promote evidence")
