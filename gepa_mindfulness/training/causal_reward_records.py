"""Bounded, exact input records for explicitly authorized causal reward composition."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import FactorChange, canonical_json, content_digest
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)

from .contracts import RewardRequest
from .trajectory import Trajectory

JsonObject = dict[str, object]
_MAX_BYTES = 4 * 1024 * 1024


def validate_input_envelope(payload: object) -> None:
    """Reject non-JSON, cyclic, deep or oversized input without recursive traversal.

    Args:
        payload: Complete JSON input; root depth is one and maximum depth is 32.

    Raises:
        ValueError: Input exceeds 4 MiB, contains nonfinite values or is not JSON.
    """
    active: set[int] = set()
    stack: list[tuple[Any, int, bool]] = [(payload, 1, False)]
    size = 0
    while stack:
        value, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(value))
            continue
        if depth > 32:
            raise ValueError("input nesting exceeds 32")
        if type(value) in (dict, list):
            if id(value) in active:
                raise ValueError("cyclic input")
            active.add(id(value))
            stack.append((value, depth, True))
            size += 2 + max(0, len(value) - 1)
            if type(value) is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        raise ValueError("JSON keys must be strings")
                    size += _encoded_size(key) + 1
                    stack.append((item, depth + 1, False))
            else:
                stack.extend((item, depth + 1, False) for item in value)
        elif value is None or type(value) in (str, int, float, bool):
            if type(value) is float and not math.isfinite(value):
                raise ValueError("nonfinite input")
            size += _encoded_size(value)
        else:
            raise ValueError("input must contain only JSON values")
        if size > _MAX_BYTES:
            raise ValueError("input exceeds 4 MiB")


def _encoded_size(value: object) -> int:
    if isinstance(value, str) and len(value) > _MAX_BYTES:
        raise ValueError("input exceeds 4 MiB")
    try:
        return len(canonical_json(value).encode("utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid JSON scalar") from error


def _object(data: object, keys: str) -> dict[str, Any]:
    if type(data) is not dict or set(data) != set(keys.split()):
        raise ValueError("unexpected or missing fields")
    return data


def _wire(data: object, schema: str, keys: str) -> dict[str, Any]:
    validate_input_envelope(data)
    result = _object(data, "schema_version " + keys)
    if result["schema_version"] != schema:
        raise ValueError("unsupported schema_version")
    return result


def _text(value: object, name: str, limit: int = 256) -> None:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} requires nonblank text of at most {limit} characters")


def _digest(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected lowercase SHA-256 digest")


def _number(value: object, *, nonnegative: bool = False) -> None:
    if type(value) not in (int, float):
        raise ValueError("expected finite number, not boolean")
    value = cast(float, value)
    try:
        valid = math.isfinite(value) and (not nonnegative or value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("invalid finite number")


def _tuple(value: object, maximum: int, *, nonempty: bool = True) -> tuple[Any, ...]:
    if type(value) is not tuple or len(value) > maximum or (nonempty and not value):
        raise ValueError("invalid tuple size or type")
    return value


def _array(value: object) -> list[Any]:
    if type(value) is not list:
        raise ValueError("wire sequence must be an array")
    return value


def _refs(
    refs: tuple[EvidenceReference, ...], maximum: int = 256, *, nonempty: bool = True
) -> None:
    _tuple(refs, maximum, nonempty=nonempty)
    for ref in refs:
        if type(ref) is not EvidenceReference or not ref.is_observable:
            raise ValueError("expected observable evidence reference")
        _text(ref.reference_id, "reference_id")
    if len({ref.reference_id for ref in refs}) != len(refs):
        raise ValueError("duplicate evidence reference ID")


def _read_refs(value: object) -> tuple[EvidenceReference, ...]:
    return tuple(EvidenceReference.from_dict(item) for item in _array(value))


def _json_object(value: str) -> dict[str, Any]:
    if type(value) is not str or len(value) > _MAX_BYTES:
        raise ValueError("expected bounded JSON object string")
    # Check encoded nesting before the JSON parser can recurse through hostile text.
    depth, quoted, escaped = 0, False, False
    for char in value:
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
                raise ValueError("input nesting exceeds 32")
        elif char in "]}":
            depth -= 1

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        result = json.loads(value, object_pairs_hook=unique_pairs)
    except (ValueError, RecursionError) as error:
        raise ValueError("invalid JSON object") from error
    validate_input_envelope(result)
    if type(result) is not dict:
        raise ValueError("expected JSON object")
    return result


def _request_wire(request: RewardRequest) -> JsonObject:
    if type(request) is not RewardRequest or type(request.trajectory) is not Trajectory:
        raise ValueError("expected RewardRequest with Trajectory")
    _refs(request.observable_references, nonempty=False)
    data = {
        "trajectory": request.trajectory.to_dict(),
        "observable_references": [r.to_dict() for r in request.observable_references],
    }
    validate_input_envelope(data)
    return json.loads(canonical_json(data))


def _read_request(data: object) -> RewardRequest:
    validate_input_envelope(data)
    values = _object(data, "trajectory observable_references")
    if type(values["trajectory"]) is not dict:
        raise ValueError("trajectory must be an object")
    request = RewardRequest(
        Trajectory.from_dict(values["trajectory"]), _read_refs(values["observable_references"])
    )
    if canonical_json(_request_wire(request)) != canonical_json(data):
        raise ValueError("noncanonical trajectory or request")
    return request


def request_digest(request: RewardRequest) -> str:
    """Bind every canonical trajectory field and its allowed observable references."""
    return content_digest(_request_wire(request))


def _cell_wire(cell: V5EvaluationCell) -> JsonObject:
    return {
        "case_id": cell.case_id,
        "case_version": cell.case_version,
        "stripe_id": cell.stripe_id,
        "subtype": cell.subtype,
        "repeat_id": cell.repeat_id,
        "seed": cell.seed,
        "model_version": cell.model_version,
        "harness_version": cell.harness_version,
    }


def _read_cell(value: object) -> V5EvaluationCell:
    return V5EvaluationCell(
        **_object(
            value,
            "case_id case_version stripe_id subtype repeat_id seed model_version harness_version",
        )
    )


def _evaluator_wire(value: TrustedEvaluatorContract) -> JsonObject:
    if type(value) is not TrustedEvaluatorContract:
        raise ValueError("expected evaluator contract")
    data: JsonObject = {
        "evaluator_id": value.evaluator_id,
        "evaluator_version": value.evaluator_version,
        "contract_id": value.contract_id,
    }
    for name, item in data.items():
        _text(item, name)
    return data


def _read_evaluator(value: object) -> TrustedEvaluatorContract:
    return TrustedEvaluatorContract(**_object(value, "evaluator_id evaluator_version contract_id"))


def _provenance_wire(value: RewardProvenance) -> JsonObject:
    if type(value) is not RewardProvenance or value.public_rationale_comparison is not None:
        raise ValueError("expected causal reward provenance")
    return {
        "component_name": value.component_name,
        "verification_method": value.verification_method,
        "route": value.route.value,
        "evidence_refs": [r.to_dict() for r in value.evidence_refs],
        "evaluator": None if value.evaluator is None else _evaluator_wire(value.evaluator),
        "public_rationale_comparison": None,
    }


def _read_provenance(value: object) -> RewardProvenance:
    data = _object(
        value,
        "component_name verification_method route evidence_refs evaluator "
        "public_rationale_comparison",
    )
    if data["public_rationale_comparison"] is not None:
        raise ValueError("causal provenance cannot include a rationale comparison")
    return RewardProvenance(
        component_name=data["component_name"],
        verification_method=data["verification_method"],
        route=VerificationRoute(data["route"]),
        evidence_refs=_read_refs(data["evidence_refs"]),
        evaluator=None if data["evaluator"] is None else _read_evaluator(data["evaluator"]),
    )


@dataclass(frozen=True, kw_only=True)
class CausalRewardPolicy:
    """Explicit experiment and provider configuration; disabled by default."""

    policy_id: str
    policy_version: str
    provider_id: str
    provider_version: str
    provider_config_digest: str
    enabled: bool = False
    weight: float = 0.0
    max_adjustment: float = 0.0
    experiment_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        for name in ("policy_id", "policy_version", "provider_id", "provider_version"):
            _text(getattr(self, name), name)
        _digest(self.provider_config_digest)
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        _number(self.weight, nonnegative=True)
        _number(self.max_adjustment, nonnegative=True)
        _refs(self.experiment_refs, 16, nonempty=False)
        if any(
            r.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD for r in self.experiment_refs
        ):
            raise ValueError("experiment authorization requires external records")
        if self.weight > 0 and (self.max_adjustment <= 0 or not self.experiment_refs):
            raise ValueError("positive weight requires positive cap and experiment references")
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> JsonObject:
        """Return a detached exact policy wire object."""
        return {
            "schema_version": "causal-reward-policy-v1",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "provider_config_digest": self.provider_config_digest,
            "enabled": self.enabled,
            "weight": self.weight,
            "max_adjustment": self.max_adjustment,
            "experiment_refs": [r.to_dict() for r in self.experiment_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> CausalRewardPolicy:
        """Validate and restore the complete versioned policy."""
        d = _wire(
            data,
            "causal-reward-policy-v1",
            "policy_id policy_version provider_id "
            "provider_version provider_config_digest enabled weight max_adjustment "
            "experiment_refs",
        )
        return cls(
            policy_id=d["policy_id"],
            policy_version=d["policy_version"],
            provider_id=d["provider_id"],
            provider_version=d["provider_version"],
            provider_config_digest=d["provider_config_digest"],
            enabled=d["enabled"],
            weight=d["weight"],
            max_adjustment=d["max_adjustment"],
            experiment_refs=_read_refs(d["experiment_refs"]),
        )


@dataclass(frozen=True, init=False)
class CausalRewardContext:
    """Two canonical arm snapshots plus complete retained source provenance."""

    _before_json: str
    _target_json: str
    before_cell: V5EvaluationCell
    target_cell: V5EvaluationCell
    before_case_mapping: tuple[str, str]
    target_case_mapping: tuple[str, str]
    before_family_id: str
    target_family_id: str
    before_lineage: tuple[str, ...]
    target_lineage: tuple[str, ...]
    before_eligibility: str | None
    target_eligibility: str | None
    before_factors: tuple[tuple[str, str], ...]
    target_factors: tuple[tuple[str, str], ...]
    change: FactorChange
    seed_policy: str
    sources: tuple[tuple[EvidenceReference, str], ...]
    target_event_refs: tuple[EvidenceReference, ...]

    def __init__(
        self,
        *,
        before: RewardRequest,
        target: RewardRequest,
        before_cell: V5EvaluationCell,
        target_cell: V5EvaluationCell,
        before_case_mapping: tuple[str, str],
        target_case_mapping: tuple[str, str],
        before_family_id: str,
        target_family_id: str,
        before_lineage: tuple[str, ...],
        target_lineage: tuple[str, ...],
        before_eligibility: str | None,
        target_eligibility: str | None,
        before_factors: tuple[tuple[str, str], ...],
        target_factors: tuple[tuple[str, str], ...],
        change: FactorChange,
        seed_policy: str,
        sources: tuple[tuple[EvidenceReference, str], ...],
        target_event_refs: tuple[EvidenceReference, ...],
    ) -> None:
        values = locals().copy()
        for name in ("self", "before", "target"):
            del values[name]
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_before_json", canonical_json(_request_wire(before)))
        object.__setattr__(self, "_target_json", canonical_json(_request_wire(target)))
        self._validate()

    @property
    def before(self) -> RewardRequest:
        """Return a new detached comparison request on every access."""
        return _read_request(json.loads(self._before_json))

    @property
    def target(self) -> RewardRequest:
        """Return a new detached target request on every access."""
        return _read_request(json.loads(self._target_json))

    def _validate(self) -> None:
        before, target = self.before.trajectory, self.target.trajectory
        for arm in ("before", "target"):
            request = getattr(self, arm)
            cell = getattr(self, arm + "_cell")
            if type(cell) is not V5EvaluationCell:
                raise ValueError("expected V5EvaluationCell")
            _read_cell(_cell_wire(cell))
            for value in (cell.model_version, cell.harness_version):
                _text(value, "cell identity")
            mapping = _tuple(getattr(self, arm + "_case_mapping"), 2)
            case = next(c for c in load_case_manifest().cases if c.id == cell.case_id)
            if mapping != (request.trajectory.case_id, case.key):
                raise ValueError("case mapping must match literal trajectory ID and manifest key")
            for value in mapping:
                _text(value, "case mapping")
            if (request.trajectory.model_identifier, request.trajectory.seed) != (
                cell.model_version,
                cell.seed,
            ):
                raise ValueError("trajectory model and seed must match its cell")
            for name in (
                "trajectory_id",
                "model_identifier",
                "policy_version",
                "backend_name",
                "backend_version",
            ):
                _text(getattr(request.trajectory, name), name)
            _text(getattr(self, arm + "_family_id"), "family_id")
            lineage = _tuple(getattr(self, arm + "_lineage"), 256)
            for item in lineage:
                _text(item, "lineage")
            if len(set(lineage)) != len(lineage):
                raise ValueError("duplicate lineage ID")
            label = getattr(self, arm + "_eligibility")
            if label is not None and type(label) is not str:
                raise ValueError("eligibility must be string or None")
            factors = _tuple(getattr(self, arm + "_factors"), 64)
            for pair in factors:
                if len(_tuple(pair, 2)) != 2:
                    raise ValueError("factor entry must have two fields")
                _text(pair[0], "factor")
                # FactorChange validates JSON values, but permits only changed values.
                if type(pair[1]) is not str:
                    raise ValueError("factor value must be canonical JSON text")
                validate_input_envelope(pair[1])
                try:
                    parsed = json.loads(pair[1])
                    validate_input_envelope(parsed)
                    if canonical_json(parsed) != pair[1]:
                        raise ValueError("noncanonical factor JSON")
                except (ValueError, RecursionError) as error:
                    raise ValueError("factor value must be canonical JSON") from error
            if len(dict(factors)) != len(factors):
                raise ValueError("duplicate factor key")
        if before.trajectory_id == target.trajectory_id:
            raise ValueError("arm trajectory IDs must differ")
        for name in ("model_identifier", "policy_version", "backend_name", "backend_version"):
            if getattr(before, name) != getattr(target, name):
                raise ValueError("arm model, policy and backend must match")
        if (self.before_cell.repeat_id, self.before_cell.harness_version) != (
            self.target_cell.repeat_id,
            self.target_cell.harness_version,
        ):
            raise ValueError("repeat and harness must match")
        if self.seed_policy not in ("shared", "per_arm") or type(self.seed_policy) is not str:
            raise ValueError("seed_policy must be shared or per_arm")
        if self.seed_policy == "shared" and before.seed != target.seed:
            raise ValueError("shared seeds must match")
        if type(self.change) is not FactorChange:
            raise ValueError("expected FactorChange")
        _text(self.change.factor, "factor")
        left, right = dict(self.before_factors), dict(self.target_factors)
        if set(left) != set(right) or {k for k in left if left[k] != right[k]} != {
            self.change.factor
        }:
            raise ValueError("exactly the declared factor must change")
        if (left[self.change.factor], right[self.change.factor]) != (
            self.change.before,
            self.change.after,
        ):
            raise ValueError("factor change values mismatch")
        _tuple(self.sources, 256)
        for entry in self.sources:
            if len(_tuple(entry, 2)) != 2:
                raise ValueError("source entry requires reference and JSON object")
            _json_object(entry[1])
        refs = tuple(entry[0] for entry in self.sources)
        _refs(refs)
        allowed = set(self.before.observable_references) | set(self.target.observable_references)
        if set(refs) != allowed:
            raise ValueError("source index must exactly cover both requests' allowed references")
        _refs(self.target_event_refs)
        if not set(self.target_event_refs).issubset(self.target.observable_references):
            raise ValueError("target event evidence must belong to target request")
        if not any(
            r.source_kind
            in (EvidenceSourceKind.OBSERVABLE_OUTPUT, EvidenceSourceKind.OBSERVABLE_ACTION)
            for r in self.target_event_refs
        ):
            raise ValueError("target event requires action or output evidence")
        object.__setattr__(
            self,
            "sources",
            tuple((ref, canonical_json(_json_object(source))) for ref, source in self.sources),
        )
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> JsonObject:
        """Return complete detached arms, coordinates, source objects and intervention."""
        return {
            "schema_version": "causal-reward-context-v1",
            "before": json.loads(self._before_json),
            "target": json.loads(self._target_json),
            "before_cell": _cell_wire(self.before_cell),
            "target_cell": _cell_wire(self.target_cell),
            "before_case_mapping": list(self.before_case_mapping),
            "target_case_mapping": list(self.target_case_mapping),
            "before_family_id": self.before_family_id,
            "target_family_id": self.target_family_id,
            "before_lineage": list(self.before_lineage),
            "target_lineage": list(self.target_lineage),
            "before_eligibility": self.before_eligibility,
            "target_eligibility": self.target_eligibility,
            "before_factors": [list(pair) for pair in self.before_factors],
            "target_factors": [list(pair) for pair in self.target_factors],
            "change": self.change.to_dict(),
            "seed_policy": self.seed_policy,
            "sources": [[ref.to_dict(), json.loads(source)] for ref, source in self.sources],
            "target_event_refs": [r.to_dict() for r in self.target_event_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> CausalRewardContext:
        """Restore exact nested schemas without legacy trajectory normalization."""
        d = _wire(
            data,
            "causal-reward-context-v1",
            "before target before_cell target_cell "
            "before_case_mapping target_case_mapping before_family_id target_family_id "
            "before_lineage target_lineage before_eligibility target_eligibility "
            "before_factors target_factors change seed_policy sources target_event_refs",
        )
        sources = []
        for entry in _array(d["sources"]):
            if len(_array(entry)) != 2 or type(entry[1]) is not dict:
                raise ValueError("source entry requires reference and object")
            sources.append((EvidenceReference.from_dict(entry[0]), canonical_json(entry[1])))
        return cls(
            before=_read_request(d["before"]),
            target=_read_request(d["target"]),
            before_cell=_read_cell(d["before_cell"]),
            target_cell=_read_cell(d["target_cell"]),
            before_case_mapping=tuple(_array(d["before_case_mapping"])),
            target_case_mapping=tuple(_array(d["target_case_mapping"])),
            before_family_id=d["before_family_id"],
            target_family_id=d["target_family_id"],
            before_lineage=tuple(_array(d["before_lineage"])),
            target_lineage=tuple(_array(d["target_lineage"])),
            before_eligibility=d["before_eligibility"],
            target_eligibility=d["target_eligibility"],
            before_factors=tuple(tuple(_array(p)) for p in _array(d["before_factors"])),
            target_factors=tuple(tuple(_array(p)) for p in _array(d["target_factors"])),
            change=FactorChange.from_dict(d["change"]),
            seed_policy=d["seed_policy"],
            sources=tuple(sources),
            target_event_refs=_read_refs(d["target_event_refs"]),
        )


@dataclass(frozen=True, kw_only=True)
class CausalRewardVerdict:
    """Verifier claims bound to complete context and policy, never self-authenticating."""

    context_digest: str
    policy_digest: str
    evaluator: TrustedEvaluatorContract
    status: str
    human_required: bool
    relevance: str
    single_factor_verified: bool | None
    before_correct: bool | None
    after_correct: bool | None
    action_changed: bool | None
    change_justified: bool | None
    required_update: bool | None
    update_satisfied: bool | None
    severe_violation: bool | None
    reason: str
    evidence_refs: tuple[EvidenceReference, ...]
    provenance: RewardProvenance

    def __post_init__(self) -> None:
        _digest(self.context_digest)
        _digest(self.policy_digest)
        _evaluator_wire(self.evaluator)
        if type(self.status) is not str or self.status not in (
            "verified",
            "unresolved",
            "disputed",
        ):
            raise ValueError("invalid verdict status")
        if type(self.relevance) is not str or self.relevance not in (
            "irrelevant",
            "relevant",
            "unknown",
        ):
            raise ValueError("invalid relevance")
        if type(self.human_required) is not bool:
            raise ValueError("human_required must be boolean")
        for name in (
            "single_factor_verified",
            "before_correct",
            "after_correct",
            "action_changed",
            "change_justified",
            "required_update",
            "update_satisfied",
            "severe_violation",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError("verdict facts require exact boolean or None")
        if (
            self.relevance == "irrelevant"
            and self.required_update is True
            or self.action_changed is False
            and self.change_justified is True
            or self.required_update is False
            and self.update_satisfied is False
        ):
            raise ValueError("contradictory verdict facts")
        _text(self.reason, "reason", 4096)
        _refs(self.evidence_refs)
        provenance = _read_provenance(_provenance_wire(self.provenance))
        if provenance.component_name != "verified_causal_penalty":
            raise ValueError("component must be verified_causal_penalty")
        _text(provenance.verification_method, "verification_method", 4096)
        _refs(provenance.evidence_refs, nonempty=False)
        if provenance.evaluator is not None and provenance.evaluator != self.evaluator:
            raise ValueError("provenance evaluator must match verdict evaluator")
        if not set(provenance.evidence_refs).issubset(self.evidence_refs):
            raise ValueError("provenance evidence must be included in verdict evidence")
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> JsonObject:
        """Return complete claims and the exclusive provenance route."""
        return {
            "schema_version": "causal-reward-verdict-v1",
            "context_digest": self.context_digest,
            "policy_digest": self.policy_digest,
            "evaluator": _evaluator_wire(self.evaluator),
            "status": self.status,
            "human_required": self.human_required,
            "relevance": self.relevance,
            "single_factor_verified": self.single_factor_verified,
            "before_correct": self.before_correct,
            "after_correct": self.after_correct,
            "action_changed": self.action_changed,
            "change_justified": self.change_justified,
            "required_update": self.required_update,
            "update_satisfied": self.update_satisfied,
            "severe_violation": self.severe_violation,
            "reason": self.reason,
            "evidence_refs": [r.to_dict() for r in self.evidence_refs],
            "provenance": _provenance_wire(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: object) -> CausalRewardVerdict:
        """Restore exact claims without conferring authentication."""
        d = _wire(
            data,
            "causal-reward-verdict-v1",
            "context_digest policy_digest evaluator status "
            "human_required relevance single_factor_verified before_correct after_correct "
            "action_changed change_justified required_update update_satisfied severe_violation "
            "reason evidence_refs provenance",
        )
        return cls(
            context_digest=d["context_digest"],
            policy_digest=d["policy_digest"],
            evaluator=_read_evaluator(d["evaluator"]),
            status=d["status"],
            human_required=d["human_required"],
            relevance=d["relevance"],
            single_factor_verified=d["single_factor_verified"],
            before_correct=d["before_correct"],
            after_correct=d["after_correct"],
            action_changed=d["action_changed"],
            change_justified=d["change_justified"],
            required_update=d["required_update"],
            update_satisfied=d["update_satisfied"],
            severe_violation=d["severe_violation"],
            reason=d["reason"],
            evidence_refs=_read_refs(d["evidence_refs"]),
            provenance=_read_provenance(d["provenance"]),
        )


@dataclass(frozen=True, kw_only=True)
class CausalRewardAccounting:
    """Exhaustive host claims about overlap for the distinct causal failure."""

    context_digest: str
    verdict_digest: str
    policy_digest: str
    baseline_total: float
    provider_id: str
    provider_version: str
    provider_config_digest: str
    layers: tuple[tuple[str, str], ...]
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for value in (
            self.context_digest,
            self.verdict_digest,
            self.policy_digest,
            self.provider_config_digest,
        ):
            _digest(value)
        _number(self.baseline_total)
        _text(self.provider_id, "provider_id")
        _text(self.provider_version, "provider_version")
        _tuple(self.layers, 32, nonempty=False)
        for entry in self.layers:
            if len(_tuple(entry, 2)) != 2:
                raise ValueError("layer requires name and overlap disposition")
            _text(entry[0], "layer")
            if type(entry[1]) is not str or entry[1] not in (
                "not_counted",
                "already_counted",
                "unknown",
            ):
                raise ValueError("invalid layer overlap disposition")
        if len(dict(self.layers)) != len(self.layers):
            raise ValueError("duplicate accounting layer")
        _refs(self.evidence_refs)
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> JsonObject:
        """Return exact bindings and per-layer causal overlap claims."""
        return {
            "schema_version": "causal-reward-accounting-v1",
            "context_digest": self.context_digest,
            "verdict_digest": self.verdict_digest,
            "policy_digest": self.policy_digest,
            "baseline_total": self.baseline_total,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "provider_config_digest": self.provider_config_digest,
            "layers": [list(pair) for pair in self.layers],
            "evidence_refs": [r.to_dict() for r in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> CausalRewardAccounting:
        """Restore bounded overlap claims; the host still authenticates their semantics."""
        d = _wire(
            data,
            "causal-reward-accounting-v1",
            "context_digest verdict_digest policy_digest "
            "baseline_total provider_id provider_version provider_config_digest "
            "layers evidence_refs",
        )
        return cls(
            context_digest=d["context_digest"],
            verdict_digest=d["verdict_digest"],
            policy_digest=d["policy_digest"],
            baseline_total=d["baseline_total"],
            provider_id=d["provider_id"],
            provider_version=d["provider_version"],
            provider_config_digest=d["provider_config_digest"],
            layers=tuple(tuple(_array(p)) for p in _array(d["layers"])),
            evidence_refs=_read_refs(d["evidence_refs"]),
        )


def _bindings(
    context: CausalRewardContext,
    policy: CausalRewardPolicy,
    verdict: CausalRewardVerdict | None,
    accounting: CausalRewardAccounting | None,
    baseline_total: float | None = None,
) -> None:
    context_hash, policy_hash = reward_record_digest(context), reward_record_digest(policy)
    allowed = {ref for ref, _ in context.sources}
    if verdict is not None:
        if (verdict.context_digest, verdict.policy_digest) != (context_hash, policy_hash):
            raise ValueError("verdict binding mismatch")
        if not set(verdict.evidence_refs).issubset(allowed):
            raise ValueError("verdict evidence outside context sources")
        if not set(context.target_event_refs).issubset(verdict.evidence_refs):
            raise ValueError("verdict must retain target event evidence")
    if accounting is not None:
        if (accounting.context_digest, accounting.policy_digest) != (context_hash, policy_hash):
            raise ValueError("accounting binding mismatch")
        if verdict is not None and accounting.verdict_digest != reward_record_digest(verdict):
            raise ValueError("accounting verdict binding mismatch")
        if (
            accounting.provider_id,
            accounting.provider_version,
            accounting.provider_config_digest,
        ) != (policy.provider_id, policy.provider_version, policy.provider_config_digest):
            raise ValueError("provider configuration binding mismatch")
        if baseline_total is not None and accounting.baseline_total != baseline_total:
            raise ValueError("accounting baseline mismatch")
        if not set(accounting.evidence_refs).issubset(allowed):
            raise ValueError("accounting evidence outside context sources")


@dataclass(frozen=True, kw_only=True)
class CausalRewardUseRequest:
    """Complete actual scoring transaction submitted for fresh host authorization."""

    context: CausalRewardContext
    verdict: CausalRewardVerdict
    accounting: CausalRewardAccounting
    policy: CausalRewardPolicy
    baseline_total: float
    integrity_snapshot: str | None
    event_key: str

    def __post_init__(self) -> None:
        for record, kind in (
            (self.context, CausalRewardContext),
            (self.verdict, CausalRewardVerdict),
            (self.accounting, CausalRewardAccounting),
            (self.policy, CausalRewardPolicy),
        ):
            if type(record) is not kind:
                raise ValueError("use request requires exact typed records")
        _number(self.baseline_total)
        _digest(self.event_key)
        if self.integrity_snapshot is not None:
            object.__setattr__(
                self, "integrity_snapshot", canonical_json(_json_object(self.integrity_snapshot))
            )
        _bindings(self.context, self.policy, self.verdict, self.accounting, self.baseline_total)
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> JsonObject:
        """Return detached full records and the actual complete integrity breakdown."""
        return {
            "schema_version": "causal-reward-use-request-v1",
            "context": self.context.to_dict(),
            "verdict": self.verdict.to_dict(),
            "accounting": self.accounting.to_dict(),
            "policy": self.policy.to_dict(),
            "baseline_total": self.baseline_total,
            "integrity_snapshot": (
                None if self.integrity_snapshot is None else json.loads(self.integrity_snapshot)
            ),
            "event_key": self.event_key,
        }

    @classmethod
    def from_dict(cls, data: object) -> CausalRewardUseRequest:
        """Restore audit input, never optimizer permission."""
        d = _wire(
            data,
            "causal-reward-use-request-v1",
            "context verdict accounting policy " "baseline_total integrity_snapshot event_key",
        )
        if d["integrity_snapshot"] is not None and type(d["integrity_snapshot"]) is not dict:
            raise ValueError("integrity_snapshot must be an object or None")
        return cls(
            context=CausalRewardContext.from_dict(d["context"]),
            verdict=CausalRewardVerdict.from_dict(d["verdict"]),
            accounting=CausalRewardAccounting.from_dict(d["accounting"]),
            policy=CausalRewardPolicy.from_dict(d["policy"]),
            baseline_total=d["baseline_total"],
            integrity_snapshot=(
                None if d["integrity_snapshot"] is None else canonical_json(d["integrity_snapshot"])
            ),
            event_key=d["event_key"],
        )


def reward_record_digest(record: object) -> str:
    """Bind only an exact new input record's complete wire representation.

    Raises:
        ValueError: The input is not one of the five causal reward wire record types.
    """
    if type(record) not in (
        CausalRewardPolicy,
        CausalRewardContext,
        CausalRewardVerdict,
        CausalRewardAccounting,
        CausalRewardUseRequest,
    ):
        raise ValueError("expected exact causal reward record")
    typed_record = cast(
        CausalRewardPolicy
        | CausalRewardContext
        | CausalRewardVerdict
        | CausalRewardAccounting
        | CausalRewardUseRequest,
        record,
    )
    payload = typed_record.to_dict()
    validate_input_envelope(payload)
    return content_digest(payload)
