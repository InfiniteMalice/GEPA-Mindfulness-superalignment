"""Opt-in decision learning from verified, action-bound epistemic histories."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from inspect import Parameter, signature
from typing import Any, cast

from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from ..core.epistemic_process import (
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    VerifiedProcessComponent,
)
from ..core.reward_provenance import RewardProvenance, TrustedEvaluatorContract, VerificationRoute
from ..factuality_observability.schemas import RecommendedAction
from ..verification.epistemic_reconciliation import EpistemicReconciliation
from ..verification.state import parse_rfc3339_datetime
from .contrastive import _digest, _text
from .eligibility import TrainingEligibility, require_training_eligible

ACTIONS = tuple(RecommendedAction)
ALLOWED_COMPONENTS = frozenset(
    EpistemicProcessComponent(name)
    for name in (
        "calibration",
        "consequence_prediction",
        "belief_update",
        "contradiction_handling",
        "missing_evidence_detection",
        "justified_abstention",
        "recovery",
    )
)


class Behavior(str, Enum):
    """Experimental strata; these do not add canonical cases or generate reward."""

    MISMATCH = "model_mismatch"
    INDEPENDENT = "independent_support"
    CORRELATED = "correlated_support"
    CONTRADICTION = "contradiction"
    MISSING = "missing_evidence"
    UNRESOLVED = "unresolved_evidence"
    HIGH_STAKES = "high_stakes"
    SCOPED_SUCCESS = "scoped_success"


@dataclass(frozen=True, slots=True)
class TrajectoryExample:
    """Host-authored input; retained provenance may be mutable until preparation."""

    example_id: str
    source_group: str
    behavior: Behavior
    context: str
    events: tuple[EventEnvelope, ...]
    source_record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """Policy-visible context and numeric history, without labels or record identities."""

    context: str
    history_json: str


@dataclass(frozen=True, slots=True)
class PreparedTrajectory:
    """Evaluator-only snapshot; pass only its input field to a policy."""

    example_id: str
    source_group: str
    behavior: Behavior
    eligibility: TrainingEligibility
    input: DecisionInput
    source_json: str
    source_digest: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DecisionVerifier:
    """Host-trusted evaluator; scores every ACTIONS entry using external behavior evidence.

    The host authenticates evidence and enforces the contract's scoring semantics. Merely
    naming a contract does not establish trust. Never supply an untrusted model as verifier.
    """

    contract: TrustedEvaluatorContract
    assess: Callable[[PreparedTrajectory], tuple[EpistemicProcessAssessment, ...]]

    def __post_init__(self) -> None:
        """Require an explicit, complete evaluator contract and callback."""
        if type(self.contract) is not TrustedEvaluatorContract or not callable(self.assess):
            raise ValueError("verifier requires a trusted evaluator contract and callback")
        TrustedEvaluatorContract.__post_init__(self.contract)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _retained_eligibility(source: Mapping[str, Any]) -> TrainingEligibility:
    restrictions = tuple(TrainingEligibility)
    labels = []
    pending: list[Any] = [source]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            if "training_eligibility" in value:
                try:
                    labels.append(TrainingEligibility(value["training_eligibility"]))
                except (ValueError, TypeError) as error:
                    raise ValueError("invalid retained training_eligibility") from error
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return max(labels, key=restrictions.index)


def _uncertainty(state: Any) -> dict[str, Any]:
    return {"estimator_version": state.estimator_version} | {
        name: getattr(state, name)
        for name in ("world_uncertainty", "model_uncertainty", "monitor_uncertainty")
    }


def _history(events: tuple[EventEnvelope, ...]) -> list[dict[str, Any]]:
    if not events or events[-1].event_type != StructuredEventType.EPISTEMIC_RECONCILIATION:
        raise ValueError("history must end in epistemic reconciliation")
    validate_action_bound_sequence(events)
    times = [parse_rfc3339_datetime(event.timestamp, "timestamp") for event in events]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("history must be chronological")
    by_id = {event.event_id: event for event in events}
    history = []
    context = None
    previous = None
    observations: set[str] = set()
    for event in events:
        if event.event_type != StructuredEventType.EPISTEMIC_RECONCILIATION:
            continue
        record = EpistemicReconciliation.from_dict(event.payload)
        update = record.update
        current = update.prior_state.context
        if context is not None and current != context:
            raise ValueError("history requires one evaluation context")
        context = current
        if record.observation_event_id in observations:
            raise ValueError("history cannot count an observation twice")
        observations.add(record.observation_event_id)
        if previous is not None and (
            update.prior_state.action_id != previous.action_id
            or update.prior_state.prediction_commit_id != previous.prediction_commit_id
        ):
            raise ValueError("successive updates must retain prior action and prediction links")
        previous = update.posterior_state
        if any(not ref.is_observable for ref in update.evidence_refs):
            raise ValueError("decision histories require observable evidence")
        for verifier_id in record.verification_event_ids:
            payload = by_id[verifier_id].payload
            if "verified" in payload:
                supported = payload["verified"] is True
            else:
                result = payload.get("result", {})
                supported = (
                    payload.get("verification_level") == "relational_evidence"
                    and isinstance(result, Mapping)
                    and result.get("claimed_outcome_supported") is True
                    and result.get("provenance_intact") is True
                )
            if not supported:
                raise ValueError("history requires successful outcome verification")
        observation = by_id[record.observation_event_id]
        action = by_id[observation.parent_event_ids[0]]
        prediction = by_id[record.prediction_event_id]
        measurements = {item.measurement_id: item for item in update.measurements}
        # Only bound numerical fields cross this projection, not arbitrary payload text.
        history.append(
            dict(
                prediction=[
                    binding.innovation.predicted_measurement for binding in record.bindings
                ],
                prediction_confidence=prediction.payload["confidence"],
                action=action.payload["action_class"],
                observation=[binding.innovation.actual_measurement for binding in record.bindings],
                dimensions=[
                    measurements[b.measurement_id].target_dimension for b in record.bindings
                ],
                representations=[
                    measurements[b.measurement_id].representation_id for b in record.bindings
                ],
                verified=True,
                prior=_uncertainty(update.prior_state),
                residuals=[binding.innovation.residual for binding in record.bindings],
                posterior=_uncertainty(update.posterior_state),
            )
        )
    if context is None:
        raise ValueError("history requires a reconciliation context")
    for event in events:
        context.validate_event(event)
    return history


def prepare_trajectories(
    examples: Iterable[TrajectoryExample], *, for_training: bool
) -> tuple[PreparedTrajectory, ...]:
    """Snapshot a whole catalog and validate causal histories before callbacks.

    Args:
        examples: Authored trajectories retaining complete admission provenance.
        for_training: Require explicit TRAIN, otherwise require an explicit non-TRAIN split.

    Returns:
        Immutable evaluator records and their narrow policy inputs.

    Raises:
        ValueError: Identity, split, provenance, evidence, chronology or bindings are invalid.
    """
    if type(for_training) is not bool:
        raise ValueError("for_training must be boolean")
    prepared = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for example in examples:
        if type(example) is not TrajectoryExample or type(example.behavior) is not Behavior:
            raise ValueError("expected a typed trajectory and behavior")
        example_id = _text(example.example_id, "example_id")
        group = _text(example.source_group, "source_group")
        public_context = _text(example.context, "context")
        if not isinstance(example.events, (list, tuple)) or any(
            type(event) is not EventEnvelope for event in example.events
        ):
            raise ValueError("events must contain EventEnvelope records")
        source = freeze_json_mapping(
            dict(
                example_id=example_id,
                source_group=group,
                behavior=example.behavior.value,
                context=public_context,
                events=[event.to_dict() for event in example.events],
                source_record=example.source_record,
            ),
            field_name="decision trajectory",
        )
        snapshot = cast(dict[str, Any], thaw_json_mapping(source))
        metadata = snapshot["source_record"]
        if not isinstance(metadata, Mapping):
            raise ValueError("source_record must retain admission provenance")
        try:
            eligibility = TrainingEligibility(metadata.get("training_eligibility"))
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory requires explicit training_eligibility") from error
        if for_training:
            require_training_eligible(source)
        elif eligibility is TrainingEligibility.TRAIN:
            raise ValueError("evaluation requires non-TRAIN trajectories")
        else:
            eligibility = _retained_eligibility(source)
        events = tuple(EventEnvelope(**event) for event in snapshot["events"])
        history_json = _json(_history(events))
        policy_input = DecisionInput(public_context, history_json)
        fingerprint = _digest([" ".join(public_context.split()), history_json])
        if example_id in ids or fingerprint in fingerprints:
            raise ValueError("duplicate trajectory identity or policy input")
        ids.add(example_id)
        fingerprints.add(fingerprint)
        prepared.append(
            PreparedTrajectory(
                example_id,
                group,
                example.behavior,
                eligibility,
                policy_input,
                _json(snapshot),
                _digest(snapshot),
                fingerprint,
            )
        )
    if not prepared:
        raise ValueError("trajectory catalog must not be empty")
    return tuple(prepared)


def _verified_tables(
    examples: tuple[PreparedTrajectory, ...], verifier: DecisionVerifier
) -> tuple[tuple[tuple[float, ...], ...], str, dict[str, Any]]:
    if type(verifier) is not DecisionVerifier:
        raise ValueError("a trusted DecisionVerifier is required")
    DecisionVerifier.__post_init__(verifier)
    contract = TrustedEvaluatorContract(**asdict(verifier.contract))
    assess = verifier.assess
    tables = []
    evidence = []
    for example in examples:
        assessments = assess(replace(example, input=replace(example.input)))
        if type(assessments) is not tuple or len(assessments) != len(ACTIONS):
            raise ValueError("verifier must assess every canonical action in ACTIONS order")
        components = None
        for assessment in assessments:
            if type(assessment) is not EpistemicProcessAssessment:
                raise ValueError("verifier must return EpistemicProcessAssessment records")
            EpistemicProcessAssessment.__post_init__(assessment)
            for item in assessment.verified_components:
                if (
                    type(item) is not VerifiedProcessComponent
                    or type(item.provenance) is not RewardProvenance
                    or type(item.provenance.evaluator) is not TrustedEvaluatorContract
                ):
                    raise ValueError("decision scores require canonical component provenance")
            names = frozenset(item.component for item in assessment.verified_components)
            if not names or not names <= ALLOWED_COMPONENTS:
                raise ValueError("decision scores require allowed verified components")
            if components is not None and names != components:
                raise ValueError("all actions require the same verified component set")
            components = names
            for item in assessment.verified_components:
                VerifiedProcessComponent.__post_init__(item)
                RewardProvenance.__post_init__(item.provenance)
                TrustedEvaluatorContract.__post_init__(
                    cast(TrustedEvaluatorContract, item.provenance.evaluator)
                )
                if (
                    item.provenance.route is not VerificationRoute.TRUSTED_EVALUATOR
                    or item.provenance.evaluator != contract
                ):
                    raise ValueError("score provenance must match the trusted evaluator contract")
        scores = tuple(EpistemicProcessAssessment.optimizer_score(item) for item in assessments)
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError("verified decision scores must be finite unit-interval values")
        tables.append(scores)
        # Grounding diagnostics are excluded from both reward and assessment identity.
        evidence.append(
            [
                example.source_digest,
                [[asdict(c) for c in a.verified_components] for a in assessments],
            ]
        )
    return tuple(tables), _digest(evidence), asdict(contract)


def _configuration(enabled: bool, seed: int) -> None:
    if enabled is not True:
        raise ValueError("dynamic uncertainty experiments require enabled=True")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def train_decisions(
    examples: Iterable[TrajectoryExample],
    score: Callable[[DecisionInput], Any],
    optimizer: Any,
    *,
    verifier: DecisionVerifier,
    epochs: int = 1,
    seed: int = 0,
    enabled: bool = False,
) -> dict[str, Any]:
    """Maximize expected externally verified decision score with a caller-owned policy.

    No language, uncertainty value, residual or private state directly generates reward.
    The host supplies a trusted behavioral evaluator and owns model mode/checkpoints.
    A runtime callback/optimizer failure stops the loop; earlier updates are not rolled back.

    Args:
        examples: Explicitly admitted TRAIN records covering all eight behavior strata.
        score: Return eight differentiable logits in the fixed ACTIONS order.
        optimizer: Torch optimizer supporting step() without required arguments.
        verifier: Authenticated host evaluator of externally observable decisions.
        epochs: Visits per record, an integer from 1 to 1000.
        seed: Local shuffle seed, from zero through 2**32 - 1.
        enabled: Literal True explicitly opts in.

    Returns:
        Source/assessment digests, contract identity, losses and exposure counts.

    Raises:
        ValueError: Admission, verification, configuration or optimization is invalid.
        ImportError: Optional Torch is unavailable.
    """
    _configuration(enabled, seed)
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be an integer in [1, 1000]")
    prepared = prepare_trajectories(examples, for_training=True)
    if {item.behavior for item in prepared} != set(Behavior):
        raise ValueError("training requires all eight behavior strata")
    import torch

    if not callable(score) or not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("training requires a scorer and torch optimizer")
    try:
        step_parameters = signature(optimizer.step).parameters.values()
    except (TypeError, ValueError) as error:
        raise ValueError("optimizer.step must have an inspectable signature") from error
    if any(
        p.default is Parameter.empty
        and p.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        for p in step_parameters
    ):
        raise ValueError("optimizer.step must be callable without required arguments")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if not parameters or any(not torch.isfinite(p).all().item() for p in parameters):
        raise ValueError("optimizer parameters must be finite")
    tables, assessment_digest, evaluator = _verified_tables(prepared, verifier)
    if any(max(table) == min(table) for table in tables):
        raise ValueError("training requires informative verified decision scores")
    counts = dict.fromkeys((behavior.value for behavior in Behavior), 0)
    losses = []
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(prepared)))
        rng.shuffle(order)
        for index in order:
            example = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            logits = score(replace(example.input))
            if (
                not isinstance(logits, torch.Tensor)
                or logits.shape != (len(ACTIONS),)
                or not logits.is_floating_point()
                or not logits.requires_grad
                or not torch.isfinite(logits).all().item()
            ):
                raise ValueError("scorer must return eight finite differentiable floating logits")
            # Preserve verified preferences when the policy emits low-precision logits.
            objective_logits = logits if logits.dtype == torch.float64 else logits.float()
            target = objective_logits.new_tensor(tables[index])
            if target.max().item() == target.min().item():
                raise ValueError("verified scores lose informativeness in objective precision")
            loss = -(objective_logits.softmax(dim=0) * target).sum()
            if not torch.isfinite(loss).item():
                raise ValueError("decision loss must be finite")
            loss.backward()
            gradients = []
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    gradient = gradient.coalesce().values()
                elif gradient.layout != torch.strided:
                    raise ValueError("optimizer requires dense or sparse COO gradients")
                gradients.append(gradient)
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                optimizer.zero_grad(set_to_none=True)
                raise ValueError("optimizer must receive finite gradients")
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in parameters):
                raise ValueError("optimizer produced nonfinite parameters; restore checkpoint")
            counts[example.behavior.value] += 1
            losses.append(float(loss.detach().item()))
    return dict(
        schema_version="dynamic-uncertainty-training-v1",
        seed=seed,
        epochs=epochs,
        dataset_digest=_digest([(p.example_id, p.source_digest) for p in prepared]),
        assessment_digest=assessment_digest,
        evaluator=evaluator,
        source_groups=sorted({p.source_group for p in prepared}),
        updates_by_behavior=counts,
        losses=losses,
        confers_authority=False,
    )
