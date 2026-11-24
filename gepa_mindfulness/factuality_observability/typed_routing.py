"""Selective routing of host-typed procedural decisions on public frozen states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import cast

from evaluation.improvement_records import record_digest

from .decision_routing import _DecisionRecord, _probability
from .epistemic_routing import RoutingBackend, RoutingFeatures, evaluate_routing_backend
from .schemas import RecommendedAction as Action

TASK_TYPES = frozenset({"deterministic", "enumerable", "generative", "unknown"})
TARGETS = frozenset(
    {
        "clarification",
        "epistemic_uncertainty",
        "semantic_inconsistency",
        "jailbreak_suspicion",
        "authorization_status",
        "diagnostic_intervention",
        "escalation",
        "reasoning_depth",
    }
)


@dataclass(frozen=True)
class RoutingTask(_DecisionRecord):
    """Only the enumerable class has a fast prediction vocabulary."""

    case_id: str
    state_digest: str
    target: str
    kind: str
    choices: tuple[Action, ...]
    high_consequence: bool
    schema_version: str = "mdac-routing-task-v1"

    def _validate(self) -> None:
        self._schema("mdac-routing-task-v1")
        if self.kind not in TASK_TYPES or self.target not in TARGETS:
            raise ValueError("unsupported typed task or routing target")
        if (self.kind == "enumerable") != bool(self.choices):
            raise ValueError("only enumerable tasks take a nonempty set of choices")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("duplicate enumerable choice")


@dataclass(frozen=True)
class FastPrediction(_DecisionRecord):
    task_digest: str
    model_version: str
    choice: Action | None
    confidence: float | None
    schema_version: str = "mdac-fast-prediction-v1"

    def _validate(self) -> None:
        self._schema("mdac-fast-prediction-v1")
        if (self.choice is None) != (self.confidence is None):
            raise ValueError("unknown predictions have null choice and confidence")
        if self.confidence is not None:
            _probability(self.confidence)


@dataclass(frozen=True)
class CalibrationLabel(_DecisionRecord):
    case_id: str
    model_version: str
    target: str
    choice: Action | None
    confidence: float | None
    expected: Action
    label_source: str
    evaluator_id: str
    evidence_ref: str
    schema_version: str = "mdac-fast-label-v1"

    def _validate(self) -> None:
        self._schema("mdac-fast-label-v1")
        if self.target not in TARGETS:
            raise ValueError("unsupported routing target")
        if self.label_source not in {"verified_execution", "independent_label"}:
            raise ValueError("independent label or verified execution evidence required")
        if (self.choice is None) != (self.confidence is None):
            raise ValueError("unknown choices require null confidence")
        if self.confidence is not None:
            _probability(self.confidence)


@dataclass(frozen=True)
class RoutingCalibration(_DecisionRecord):
    """Recompute the operating point from labeled rows, grouping ties atomically."""

    labels: tuple[CalibrationLabel, ...]
    target_accuracy: float
    minimum_examples: int
    schema_version: str = "mdac-routing-calibration-v1"

    def _validate(self) -> None:
        self._schema("mdac-routing-calibration-v1")
        _probability(self.target_accuracy)
        if not 1 <= self.minimum_examples <= 10000 or not 1 <= len(self.labels) <= 10000:
            raise ValueError("calibration requires bounded labels and a positive minimum sample")
        if len(set(self.sample_ids)) != len(self.labels):
            raise ValueError("duplicate calibration case")
        if len({(row.model_version, row.target) for row in self.labels}) != 1:
            raise ValueError("calibration requires one checkpoint and task target")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.case_id for row in self.labels)

    def operating_point(self) -> tuple[float | None, int, int]:
        scored = sorted(
            (
                (row.confidence, row.choice == row.expected)
                for row in self.labels
                if row.confidence is not None
            ),
            reverse=True,
        )
        count = correct = 0
        best: tuple[float | None, int, int] = (None, 0, 0)
        for confidence, group in groupby(scored, key=lambda x: x[0]):
            for _, ok in group:
                count += 1
                correct += ok
            if count >= self.minimum_examples and correct / count >= self.target_accuracy:
                best = (confidence, count, correct)
        return best

    @property
    def threshold(self) -> float | None:
        return self.operating_point()[0]

    @property
    def accepted(self) -> int:
        return self.operating_point()[1]

    @property
    def correct(self) -> int:
        return self.operating_point()[2]


def calibrate_threshold(
    labels: tuple[CalibrationLabel, ...], *, target_accuracy: float = 0.95, minimum_examples: int
) -> RoutingCalibration:
    """Find maximum empirical coverage meeting accuracy and independent sample requirements."""
    return RoutingCalibration(labels, target_accuracy, minimum_examples)


@dataclass(frozen=True)
class TypedRoutingInput(_DecisionRecord):
    task: RoutingTask
    prediction: FastPrediction | None = None
    calibration: RoutingCalibration | None = None
    schema_version: str = "mdac-typed-routing-input-v1"

    @classmethod
    def from_dict(cls, value: object) -> TypedRoutingInput:
        return cast(TypedRoutingInput, super().from_dict(value))

    def _validate(self) -> None:
        self._schema("mdac-typed-routing-input-v1")
        if self.task.kind != "enumerable" and (self.prediction or self.calibration):
            raise ValueError("only enumerable decisions accept fast predictions or calibration")


def evaluate_typed_task(features: RoutingFeatures, value: TypedRoutingInput) -> dict:
    """Return a guarded proposal for an already captured state. No action is dispatched."""
    if type(value) is not TypedRoutingInput or type(features) is not RoutingFeatures:
        raise ValueError("exact typed routing input and prepared features required")
    value = TypedRoutingInput.from_dict(value.to_dict())
    task, prediction, calibration = value.task, value.prediction, value.calibration
    if task.state_digest != features.input_digest:
        raise ValueError("typed task state mismatch")
    if prediction and (
        prediction.task_digest != record_digest(task)
        or (prediction.choice is not None and prediction.choice not in task.choices)
    ):
        raise ValueError("fast prediction task binding or choice mismatch")
    if calibration:
        first = calibration.labels[0]
        if (
            task.case_id in calibration.sample_ids
            or task.target != first.target
            or (prediction and prediction.model_version != first.model_version)
        ):
            raise ValueError("calibration target, checkpoint, or held-out case mismatch")
    threshold = None if calibration is None else calibration.threshold
    if task.high_consequence:
        proposed, reason = Action.ROUTE_EXTERNAL, "independent_verification"
    elif task.kind == "deterministic":
        proposed, reason = features.required_action, "deterministic_guard"
    elif task.kind == "generative":
        proposed, reason = Action.DECOMPOSE_AND_VERIFY, "stronger_reasoning"
    elif task.kind == "unknown" or prediction is None or prediction.choice is None:
        proposed, reason = Action.ESCALATE, "unknown"
    elif threshold is None or prediction.confidence is None or prediction.confidence < threshold:
        proposed, reason = Action.ESCALATE, "uncalibrated_or_low_confidence"
    else:
        proposed, reason = prediction.choice, "enumerated_fast_choice"
    # Escalation must survive a guard whose sole other admissible route could be retrieval.
    if proposed not in features.allowed_actions and reason in {
        "independent_verification",
        "stronger_reasoning",
    }:
        proposed = Action.ESCALATE
    route = evaluate_routing_backend(
        features, RoutingBackend("typed-procedure", "v1", lambda _: proposed)
    )
    return dict(
        schema_version="mdac-typed-route-v1",
        task_digest=record_digest(task),
        reason=reason,
        raw_choice=(
            None if prediction is None or prediction.choice is None else prediction.choice.value
        ),
        confidence=None if prediction is None else prediction.confidence,
        threshold=threshold,
        routing=route.to_dict(),
        confers_authority=False,
    )
