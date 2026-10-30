"""Opt-in structured decision proposals and separate recorded native-depth diagnostics."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from math import isclose
from typing import Protocol

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import ImprovementRecord, _encode, record_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.training.causal_reward_records import (
    _json_object,
    validate_input_envelope,
)

from .epistemic_routing import (
    RoutingAssessment,
    RoutingBackend,
    RoutingFeatures,
    RoutingPolicy,
    RoutingRequest,
    evaluate_routing_backend,
    prepare_routing,
)
from .schemas import RecommendedAction as Action

ACTIONS = tuple(Action)


def _probability(value: object) -> None:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("probability must be a finite number in [0,1], not boolean")


def _refs(values: tuple[EvidenceReference, ...]) -> None:
    if not 1 <= len(values) <= 32 or len({v.reference_id for v in values}) != len(values):
        raise ValueError("nonempty unique bounded public evidence required")
    if any(len(v.reference_id) > 256 for v in values):
        raise ValueError("evidence identity exceeds limit")


@dataclass(frozen=True, kw_only=True)
class _DecisionRecord(ImprovementRecord):
    training_eligibility: str = "DEVELOPMENT"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.training_eligibility != "DEVELOPMENT"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("decision records cannot confer optimizer or runtime authority")
        validate_input_envelope(_encode(self))

    def _schema(self, expected: str) -> None:
        if self.schema_version != expected:
            raise ValueError("unsupported decision schema")


@dataclass(frozen=True)
class PublicDecisionContext(_DecisionRecord):
    state_digest: str
    task: str
    artifact: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "public-decision-context-v1"

    def _validate(self) -> None:
        self._schema("public-decision-context-v1")
        if len(self.task) > 8192 or len(self.artifact) > 65536:
            raise ValueError("public context text exceeds limit")
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class DecisionPacket(_DecisionRecord):
    context: PublicDecisionContext
    features_json: str
    schema_version: str = "bounded-decision-packet-v1"

    def _validate(self) -> None:
        self._schema("bounded-decision-packet-v1")
        if len(self.features_json) > 65536:
            raise ValueError("public routing features exceed diagnostic packet limit")
        features = _json_object(self.features_json)
        if canonical_json(features) != self.features_json or set(features) != {
            f.name for f in fields(RoutingFeatures)
        }:
            raise ValueError("packet requires exact canonical routing features")
        if self.context.state_digest != features["input_digest"]:
            raise ValueError("public state binding mismatch")


@dataclass(frozen=True)
class DecisionReadout(_DecisionRecord):
    """Probabilities follow ACTIONS order; counts/calibration identity are host captures."""

    packet_digest: str
    model_version: str
    depth: int
    probabilities: tuple[float, ...]
    input_tokens: int
    output_tokens: int
    calibration_digest: str | None = None
    schema_version: str = "bounded-decision-readout-v1"

    def _validate(self) -> None:
        self._schema("bounded-decision-readout-v1")
        if len(self.model_version) > 128 or not 1 <= self.depth <= 16:
            raise ValueError("invalid readout model version or depth")
        if min(self.input_tokens, self.output_tokens) < 0:
            raise ValueError("token counts must be nonnegative exact integers")
        if len(self.probabilities) != len(ACTIONS):
            raise ValueError("readout requires one probability per existing action")
        for value in self.probabilities:
            _probability(value)
        if not isclose(sum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one within binary64 absolute 1e-12")

    @property
    def action(self) -> Action:
        """Use stable existing action order to break equal-probability ties."""
        return ACTIONS[max(range(len(ACTIONS)), key=self.probabilities.__getitem__)]

    @property
    def confidence(self) -> float:
        return float(max(self.probabilities))


@dataclass(frozen=True)
class DecisionBackend:
    name: str
    version: str
    model_version: str
    kind: str
    predict: Callable[[DecisionPacket], DecisionReadout]

    def __post_init__(self) -> None:
        if any(
            type(s) is not str or not s.strip() or len(s) > 128
            for s in (self.name, self.version, self.model_version)
        ):
            raise ValueError("backend identities require bounded nonblank text")
        if self.kind not in ("jev", "generative") or not callable(self.predict):
            raise ValueError("unsupported decision backend or callback")


@dataclass(frozen=True)
class DecisionAssessment:
    packet: DecisionPacket
    readout: DecisionReadout | None
    routing: RoutingAssessment
    backend_kind: str
    error: str | None
    confidence_escalation: bool

    @property
    def raw_action(self) -> Action | None:
        return self.readout.action if self.readout is not None and self.error is None else None

    @property
    def confidence(self) -> float | None:
        return self.readout.confidence if self.raw_action is not None else None

    def to_dict(self) -> dict:
        result = dict(
            schema_version="bounded-decision-assessment-v1",
            training_eligibility="DEVELOPMENT",
            optimizer_input=False,
            confers_authority=False,
            packet=self.packet.to_dict(),
            readout=self.readout.to_dict() if self.readout else None,
            routing=self.routing.to_dict(),
            backend_kind=self.backend_kind,
            error=self.error,
            raw_action=self.raw_action.value if self.raw_action else None,
            confidence=self.confidence,
            confidence_escalation=self.confidence_escalation,
            calibration_qualified=False,
        )
        validate_input_envelope(result)
        return result


def prepare_decision_packet(
    features: RoutingFeatures, public: PublicDecisionContext
) -> DecisionPacket:
    """Bind a host-public artifact to independently prepared routing features."""
    if type(features) is not RoutingFeatures or type(public) is not PublicDecisionContext:
        raise ValueError("expected exact prepared features and public context")
    public = PublicDecisionContext.from_dict(public.to_dict())
    return DecisionPacket(public, canonical_json(features.to_dict()))


def evaluate_decision_backend(
    features: RoutingFeatures,
    public: PublicDecisionContext,
    backend: DecisionBackend,
    *,
    confidence_threshold: float,
) -> DecisionAssessment:
    """Delegate all admissibility/override decisions to the established routing guard."""
    _probability(confidence_threshold)
    if type(backend) is not DecisionBackend:
        raise ValueError("expected exact decision backend")
    backend = replace(backend)
    features = replace(features)
    packet = prepare_decision_packet(features, public)
    original = packet.to_dict()
    validate_input_envelope(
        dict(
            packet=original,
            routing_features=features.to_dict(),
            metadata_capacity="x" * 8192,
        )
    )
    digest = record_digest(packet)
    captured = {"readout": None, "error": None, "confidence_escalation": False}

    def propose(_: RoutingFeatures) -> Action:
        detached = DecisionPacket.from_dict(original)
        try:
            value = backend.predict(detached)
        except Exception:
            captured["error"] = "callback_error"
            raise ValueError("decision callback failed") from None
        try:
            unchanged = detached.to_dict() == original
        except Exception:
            unchanged = False
        if not unchanged:
            captured["error"] = "mutated_packet"
            raise ValueError("decision packet mutated")
        try:
            if type(value) is not DecisionReadout:
                raise ValueError("wrong readout type")
            value = DecisionReadout.from_dict(value.to_dict())
        except Exception:
            captured["error"] = "invalid_readout"
            raise ValueError("invalid decision readout") from None
        captured["readout"] = value
        if (value.packet_digest, value.model_version, value.depth) != (
            digest,
            backend.model_version,
            1,
        ):
            captured["error"] = "readout_binding_mismatch"
            raise ValueError("decision readout binding mismatch")
        if value.confidence < confidence_threshold:
            captured["confidence_escalation"] = True
            return Action.ESCALATE
        return value.action

    routing = evaluate_routing_backend(
        features, RoutingBackend(backend.name, backend.version, propose)
    )
    return DecisionAssessment(
        packet,
        captured["readout"],
        routing,
        backend.kind,
        captured["error"],
        captured["confidence_escalation"],
    )


def route_decision(
    request: RoutingRequest,
    *,
    public: PublicDecisionContext,
    backend: DecisionBackend,
    confidence_threshold: float,
    policy: RoutingPolicy = RoutingPolicy(),
) -> DecisionAssessment | None:
    """Propose a route only when enabled; host timeouts, verification and grants remain required."""
    policy = replace(policy)
    if not policy.enabled:
        return None
    return evaluate_decision_backend(
        prepare_routing(request, policy), public, backend, confidence_threshold=confidence_threshold
    )


@dataclass(frozen=True)
class LoopedBackendSpec(_DecisionRecord):
    model_version: str
    checkpoint_digest: str
    architecture: str
    native_loop_pretrained: bool
    resumes_hidden_state: bool
    shared_readout: bool
    max_depth: int
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "native-loop-backend-spec-v1"

    def _validate(self) -> None:
        self._schema("native-loop-backend-spec-v1")
        if (
            not 1 <= self.max_depth <= 16
            or max(len(self.model_version), len(self.architecture)) > 128
        ):
            raise ValueError("invalid native model identity or depth")
        _refs(self.evidence_refs)


class NativeLoopBackend(Protocol):
    """External native session interface; no compatible model implementation is provided."""

    spec: LoopedBackendSpec

    def start(self, packet: DecisionPacket) -> object:
        """Encode the request once in an operator-owned native loop runtime."""
        ...

    def step(self, state: object) -> tuple[object, DecisionReadout]:
        """Resume hidden state by one native step, sharing checkpoint and readout head."""
        ...


def analyze_depth_trace(
    packet: DecisionPacket,
    spec: LoopedBackendSpec,
    readouts: tuple[DecisionReadout, ...],
    *,
    agreement_scores: tuple[float, ...] | None,
    agreement_threshold: float,
    confidence_threshold: float,
    expected_actions: tuple[Action, ...] | None = None,
    enabled: bool = False,
) -> dict:
    """Compare stopping rules on an already computed trace; establish no real compute savings."""
    if enabled is not True:
        raise ValueError("depth diagnostics require enabled=True")
    _probability(agreement_threshold)
    _probability(confidence_threshold)
    if type(packet) is not DecisionPacket or type(spec) is not LoopedBackendSpec:
        raise ValueError("expected exact packet and native backend declaration")
    packet, spec = DecisionPacket.from_dict(packet.to_dict()), LoopedBackendSpec.from_dict(
        spec.to_dict()
    )
    if not (spec.native_loop_pretrained and spec.resumes_hidden_state and spec.shared_readout):
        raise ValueError("compatible native loop and shared readout declaration required")
    if type(readouts) is not tuple or len(readouts) != spec.max_depth:
        raise ValueError("complete native depth trace required")
    if any(type(r) is not DecisionReadout for r in readouts):
        raise ValueError("expected exact depth readouts")
    readouts = tuple(DecisionReadout.from_dict(r.to_dict()) for r in readouts)
    digest = record_digest(packet)
    if any(
        (r.packet_digest, r.model_version, r.depth) != (digest, spec.model_version, i)
        for i, r in enumerate(readouts, 1)
    ):
        raise ValueError("depth trace input, model or contiguous depth binding mismatch")
    if agreement_scores is not None:
        if type(agreement_scores) is not tuple or len(agreement_scores) != len(readouts) - 1:
            raise ValueError("one agreement estimate per nonterminal depth required")
        for value in agreement_scores:
            _probability(value)
    if expected_actions is not None and (
        type(expected_actions) is not tuple
        or not expected_actions
        or any(type(a) is not Action for a in expected_actions)
        or len(set(expected_actions)) != len(expected_actions)
    ):
        raise ValueError("expected actions must be independently supplied distinct labels")
    deepest = readouts[-1]
    confidence_depth = next(
        (r.depth for r in readouts if r.confidence >= confidence_threshold), deepest.depth
    )
    stopper_depth = next(
        (i for i, p in enumerate(agreement_scores or (), 1) if p >= agreement_threshold),
        deepest.depth,
    )
    selected = readouts[stopper_depth - 1]
    report = dict(
        schema_version="native-depth-diagnostic-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        packet=packet.to_dict(),
        spec=spec.to_dict(),
        readouts=[r.to_dict() for r in readouts],
        agreement_scores=list(agreement_scores) if agreement_scores is not None else None,
        agreement_threshold=agreement_threshold,
        confidence_threshold=confidence_threshold,
        expected_actions=[a.value for a in expected_actions] if expected_actions else None,
        executed_depth=deepest.depth,
        confidence_selected_depth=confidence_depth,
        stopper_selected_depth=stopper_depth,
        stopper_agrees_with_deepest=selected.action == deepest.action,
        agreement_targets=[r.action == deepest.action for r in readouts[:-1]],
        stopper_label_agreement=selected.action in expected_actions if expected_actions else None,
        actual_compute_savings_established=False,
        native_backend_independently_verified=False,
    )
    validate_input_envelope(report)
    report["result_digest"] = content_digest(report)
    return report
