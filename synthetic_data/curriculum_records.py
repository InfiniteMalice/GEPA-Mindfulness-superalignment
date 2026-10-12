"""Bounded immutable inputs for offline curriculum diagnostics, never admission."""

# Standard library
from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

# Third-party
# Local
from evaluation.causal_records import CausalPair, canonical_json, content_digest
from evaluation.improvement_records import SystemConfig
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.learning_surfaces import LessonProposal
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

JsonObject = dict[str, Any]
Refs = tuple[EvidenceReference, ...]
IDs = tuple[str, ...]
_MAX_BYTES = 4 * 1024 * 1024
R = TypeVar("R", bound="CurriculumRecord")


def _pairs(items: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode_curriculum_json(text: str) -> Any:
    """Decode bounded JSON, preserving syntax errors separately from invalid values."""
    if type(text) is not str or len(text) > _MAX_BYTES:
        raise ValueError("invalid JSON text size")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 32:
                raise ValueError("embedded input nesting exceeds 32")
        elif char in "]}":
            depth -= 1
    try:
        value = json.loads(text, object_pairs_hook=_pairs)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("invalid bounded JSON") from error
    validate_input_envelope(value)
    return value


def validate_curriculum_envelope(value: object) -> None:
    """Bound the complete wire envelope and every embedded JSON layer before use."""
    validate_input_envelope(value)
    pending = [(value, 1)]
    total = 0
    while pending:
        item, depth = pending.pop()
        if depth > 32:
            raise ValueError("decoded nesting exceeds 32")
        if type(item) is dict:
            total += 2 + max(0, len(item) - 1)
            for key, child in item.items():
                total += len(canonical_json(key).encode("utf-8")) + 1
                pending.append((child, depth + 1))
        elif type(item) is list:
            if len(item) > 1024:
                raise ValueError("array exceeds 1024 entries")
            total += 2 + max(0, len(item) - 1)
            pending.extend((child, depth + 1) for child in item)
        else:
            if type(item) is int and abs(item) > 2**53 - 1:
                raise ValueError("integer exceeds serialization-safe range")
            total += len(canonical_json(item).encode("utf-8"))
            if type(item) is str and item.lstrip().startswith(("{", "[", '"')):
                # Bracket-prefixed source text must be bounded even when it is not JSON.
                try:
                    decoded = decode_curriculum_json(item)
                except json.JSONDecodeError:
                    if len(item) > 16384:
                        raise
                else:
                    pending.append((decoded, depth))
        if total > _MAX_BYTES:
            raise ValueError("decoded input exceeds 4 MiB")


def read_curriculum_object(value: str) -> JsonObject:
    """Decode an exact canonical object after lexical depth and decoded size checks."""
    decoded = decode_curriculum_json(value)
    validate_curriculum_envelope(decoded)
    if type(decoded) is not dict or canonical_json(decoded) != value:
        raise ValueError("source requires a canonical JSON object")
    return decoded


def utc_time(value: str) -> datetime:
    """Parse the contract's exact second-resolution UTC timestamp."""
    if type(value) is not str or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value):
        raise ValueError("timestamp requires exact UTC seconds")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _digest(value: str) -> None:
    if not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("digest requires lowercase SHA-256")


def _encode(value: Any) -> Any:
    if type(value) is tuple:
        return [_encode(item) for item in value]
    if type(value) is TrustedEvaluatorContract:
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if type(value) in (*_RECORDS, EvidenceReference, LessonProposal, SystemConfig, CausalPair):
        return value.to_dict()
    return value


def _hints(kind: type) -> dict[str, Any]:
    return get_type_hints(kind)


def _typed(value: Any, kind: Any, restore: bool = False) -> Any:
    origin, args = get_origin(kind), get_args(kind)
    if origin is types.UnionType:
        for option in args:
            try:
                return _typed(value, option, restore)
            except ValueError:
                pass
        raise ValueError("invalid optional field")
    if origin is tuple:
        if type(value) is not (list if restore else tuple) or len(value) > 1024:
            raise ValueError("invalid bounded array")
        return tuple(_typed(item, args[0], restore) for item in value)
    if kind in (str, int, bool, type(None)):
        if type(value) is not kind:
            raise ValueError("incorrect scalar type")
        if kind is str and not value.strip():
            raise ValueError("text must be nonblank")
        if kind is int and not 0 <= value <= 2**53 - 1:
            raise ValueError("invalid nonnegative integer")
        return value
    if kind not in (
        *_RECORDS,
        EvidenceReference,
        LessonProposal,
        SystemConfig,
        CausalPair,
        TrustedEvaluatorContract,
    ):
        raise ValueError("unsupported record type")
    if restore:
        if kind is TrustedEvaluatorContract:
            if type(value) is not dict or set(value) != {f.name for f in fields(kind)}:
                raise ValueError("invalid evaluator fields")
            if any(type(v) is not str or not v.strip() for v in value.values()):
                raise ValueError("invalid evaluator identity")
            value = kind(**value)
        else:
            value = kind.from_dict(value)
    if type(value) is not kind:
        raise ValueError("incorrect record type")
    if kind is EvidenceReference and not value.is_observable:
        raise ValueError("private evidence is not permitted")
    if not restore:
        # Public serializers omit private routing state and revalidate nested records.
        if kind is TrustedEvaluatorContract:
            value.__post_init__()
        else:
            wire = value.to_dict()
            validate_curriculum_envelope(wire)
            kind.from_dict(wire)
    return value


@dataclass(frozen=True, kw_only=True)
class CurriculumRecord:
    """Exact allowlisted fields; nested domain records use their public serializers."""

    def __post_init__(self) -> None:
        if type(self) not in _RECORDS:
            raise ValueError("unsupported curriculum record")
        hints = _hints(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            _typed(value, hints[field.name])
            if type(value) is str:
                if field.name.endswith("_json"):
                    read_curriculum_object(value)
                elif len(value) > (16384 if field.name in ("reason", "goal") else 256):
                    raise ValueError(f"{field.name} exceeds text limit")
                if field.name.endswith("digest"):
                    _digest(value)
            if type(value) is tuple and all(type(v) is str for v in value):
                if len(value) != len(set(value)) or any(len(v) > 256 for v in value):
                    raise ValueError(f"invalid {field.name} identities")
            if field.name == "evidence_refs":
                if not value or len({r.reference_id for r in value}) != len(value):
                    raise ValueError("evidence references must be nonempty and unique")
                if type(self) in (LessonReview, GenerationReview, ReplayEvidence, MilestoneReview):
                    if not any(r.source_kind is EvidenceSourceKind.EXTERNAL_RECORD for r in value):
                        raise ValueError("review/replay requires external evidence")
        self._validate()
        validate_curriculum_envelope(self._wire())

    def _validate(self) -> None:
        pass

    def _wire(self) -> JsonObject:
        name = re.sub(r"(?<!^)(?=[A-Z])", "-", type(self).__name__).lower()
        return {
            "schema_version": f"curriculum-{name}-v1",
            **{f.name: _encode(getattr(self, f.name)) for f in fields(self)},
        }

    def to_dict(self) -> JsonObject:
        """Return a detached, revalidated wire object."""
        self.__post_init__()
        return self._wire()

    @classmethod
    def from_dict(cls: type[R], data: object) -> R:
        """Reject unknown schemas and fields before restoring allowlisted nested types."""
        validate_curriculum_envelope(data)
        names = {f.name for f in fields(cls)}
        if type(data) is not dict or set(data) != names | {"schema_version"}:
            raise ValueError("unexpected or missing record fields")
        name = re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()
        if data["schema_version"] != f"curriculum-{name}-v1":
            raise ValueError("unsupported schema version")
        hints = _hints(cls)
        return cls(**{n: _typed(data[n], hints[n], True) for n in names})


def _choice(value: str, choices: str) -> None:
    if value not in choices.split():
        raise ValueError("unsupported status or policy")


@dataclass(frozen=True, kw_only=True)
class CurationPolicy(CurriculumRecord):
    policy_id: str
    version: str
    evaluator: TrustedEvaluatorContract


@dataclass(frozen=True, kw_only=True)
class EpisodeSource(CurriculumRecord):
    episode_id: str
    case_ids: IDs
    family_id: str
    parent_episode_ids: IDs
    system: SystemConfig
    terminal_status: str
    source_json: str
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(
            self.terminal_status, "completed_success completed_failure pending cancelled incomplete"
        )
        if not self.case_ids:
            raise ValueError("episode requires source case identities")


@dataclass(frozen=True, kw_only=True)
class BankEntry(CurriculumRecord):
    lesson: LessonProposal
    version: str
    source_episode_ids: IDs
    source_json: str


@dataclass(frozen=True, kw_only=True)
class BankSnapshot(CurriculumRecord):
    bank_id: str
    version: str
    entries: tuple[BankEntry, ...]

    def _validate(self) -> None:
        unique(self.entries, "lesson", key=lambda entry: entry.lesson.lesson_id)


@dataclass(frozen=True, kw_only=True)
class EvidenceCheck(CurriculumRecord):
    name: str
    status: str
    reason: str
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.status, "pass fail unknown")


@dataclass(frozen=True, kw_only=True)
class ReplayEvidence(CurriculumRecord):
    spec_json: str
    system: SystemConfig
    seed_policy: str
    seeds: tuple[int | None, ...]
    results_json: str
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.seed_policy, "shared independent unknown")


@dataclass(frozen=True, kw_only=True)
class LessonReview(CurriculumRecord):
    lesson_digest: str
    bank_digest: str
    source_digests: IDs
    policy_digest: str
    checks: tuple[EvidenceCheck, ...]
    replay: ReplayEvidence | None
    valid_from: str
    valid_until: str
    status: str
    human_required: bool
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.status, "verified disputed unresolved")
        if utc_time(self.valid_until) <= utc_time(self.valid_from):
            raise ValueError("review interval must increase")
        unique(self.checks, "name")
        for digest in self.source_digests:
            _digest(digest)


@dataclass(frozen=True, kw_only=True)
class UtilitySlot(CurriculumRecord):
    slot_id: str
    task_id: str
    content_digest: str
    family_id: str
    condition: str
    repeat_id: int
    seed: int | None
    system: SystemConfig
    rubric_digest: str
    budget_json: str

    def _validate(self) -> None:
        _choice(self.condition, "none lesson_only bank_only bank_plus_lesson")


@dataclass(frozen=True, kw_only=True)
class UtilityObservation(CurriculumRecord):
    slot: UtilitySlot
    status: str
    success: bool | None
    source_json: str
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.status, "observed failed censored missing")
        if (self.status == "observed") != (self.success is not None):
            raise ValueError("only observed outcomes carry success")


@dataclass(frozen=True, kw_only=True)
class UtilityStudy(CurriculumRecord):
    study_id: str
    lesson_digest: str
    bank_digest: str
    seed_policy: str
    expected_slots: tuple[UtilitySlot, ...]
    observations: tuple[UtilityObservation, ...]

    def _validate(self) -> None:
        _choice(self.seed_policy, "shared independent unknown")
        unique(self.expected_slots, "slot_id")
        unique(self.observations, "slot", key=lambda item: item.slot.slot_id)
        expected = {slot.slot_id: slot for slot in self.expected_slots}
        if any(expected.get(obs.slot.slot_id) != obs.slot for obs in self.observations):
            raise ValueError("observation differs from declared slot")


@dataclass(frozen=True, kw_only=True)
class MilestoneReview(CurriculumRecord):
    milestone_id: str
    episode_digest: str
    step_id: str
    goal: str
    before_json: str
    after_json: str
    step_completed: bool | None
    goal_progress: bool | None
    epistemic_update: bool | None
    final_success: bool | None
    goal_authorized: bool | None
    constraint_violation: bool | None
    severe_event: bool | None
    status: str
    human_required: bool
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.status, "verified disputed unresolved")


@dataclass(frozen=True, kw_only=True)
class BoundaryCandidate(CurriculumRecord):
    candidate_id: str
    pair: CausalPair
    source_json: str
    row_list: str
    row_id_key: str
    row_id: str
    case_ids: IDs
    parent_case_ids: IDs
    generator_id: str
    generator_config_json: str
    coverage_tags: IDs

    def _validate(self) -> None:
        for tag in self.coverage_tags:
            _choice(tag, "invariance sensitivity benign_assistance legitimate_refusal")


@dataclass(frozen=True, kw_only=True)
class GenerationReview(CurriculumRecord):
    candidate_digest: str
    policy_digest: str
    status: str
    human_required: bool
    checks: tuple[EvidenceCheck, ...]
    evidence_refs: Refs

    def _validate(self) -> None:
        _choice(self.status, "verified disputed unresolved")
        unique(self.checks, "name")


@dataclass(frozen=True, kw_only=True)
class AblationArm(CurriculumRecord):
    arm_id: str
    candidate_id: str
    control_arm_id: str | None
    components_json: str
    data_case_ids: IDs
    starting_system: SystemConfig
    optimizer_digest: str
    reward_digest: str
    seed_policy: str
    seeds: tuple[int | None, ...]
    planned_budget_json: str
    actual_budget_json: str | None
    knowledge_report_json: str | None

    def _validate(self) -> None:
        _choice(self.seed_policy, "shared independent unknown")
        components = read_curriculum_object(self.components_json)
        if set(components) != set(COMPONENTS):
            raise ValueError("component roster requires exact fields")
        for value in components.values():
            if value is not None:
                if type(value) is not str:
                    raise ValueError("component requires a digest or null")
                _digest(value)
        for budget in (self.planned_budget_json, self.actual_budget_json):
            if budget is None:
                continue
            data = read_curriculum_object(budget)
            if set(data) != {"train", "evaluation"}:
                raise ValueError("budget requires train and evaluation")
            for row in data.values():
                if type(row) is not dict or set(row) != {"compute", "unit", "examples", "tokens"}:
                    raise ValueError("budget requires exact fields")
                if type(row["unit"]) is not str or not row["unit"].strip():
                    raise ValueError("budget requires a unit")
                for name in ("compute", "examples", "tokens"):
                    allowed = (int, float) if name == "compute" else (int,)
                    if type(row[name]) not in allowed or row[name] < 0:
                        raise ValueError("invalid budget amount")


@dataclass(frozen=True, kw_only=True)
class CurationExposure(CurriculumRecord):
    ordinal: int
    case_ids: IDs
    candidate_ids: IDs
    round_ids: IDs
    source_json: str
    evidence_refs: Refs


COMPONENTS = (
    "synthetic",
    "causal_pairs",
    "laundering",
    "sensitive_debate",
    "pluralistic",
    "knowledge",
)
_RECORDS = (
    CurationPolicy,
    EpisodeSource,
    BankEntry,
    BankSnapshot,
    EvidenceCheck,
    ReplayEvidence,
    LessonReview,
    UtilitySlot,
    UtilityObservation,
    UtilityStudy,
    MilestoneReview,
    BoundaryCandidate,
    GenerationReview,
    AblationArm,
    CurationExposure,
)


def curriculum_digest(record: object) -> str:
    """Bind one exact supported record; a digest does not authenticate its claims."""
    if type(record) not in _RECORDS:
        raise ValueError("expected an exact curriculum record")
    return content_digest(record.to_dict())  # type: ignore[attr-defined]


def unique(values: tuple[Any, ...], field: str, *, key: Any = None) -> None:
    """Reject ambiguous identities before joining records."""
    identities = [key(value) if key else getattr(value, field) for value in values]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate {field}")


def validate_episode_roster(episodes: tuple[EpisodeSource, ...]) -> None:
    """Resolve the complete ancestry without recursive graph traversal."""
    unique(episodes, "episode_id")
    parents = {episode.episode_id: set(episode.parent_episode_ids) for episode in episodes}
    if any(parent not in parents for group in parents.values() for parent in group):
        raise ValueError("unknown episode ancestor")
    remaining = dict(parents)
    while remaining:
        ready = {name for name, group in remaining.items() if not group}
        if not ready:
            raise ValueError("episode ancestry cycle")
        remaining = {name: group - ready for name, group in remaining.items() if name not in ready}
