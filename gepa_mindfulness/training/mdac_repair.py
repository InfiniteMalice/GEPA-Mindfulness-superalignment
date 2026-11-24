"""Opt-in, host-evaluated failure-anchor repair and retentive supervision."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any

from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping

from .eligibility import require_training_eligible
from .seeds import validate_seed
from .trajectory import RolloutRequest

CHECKS = ("truthfulness", "authorization", "constitution", "usefulness", "uncertainty")


class AnchorScale(str, Enum):
    RESPONSE = "immediate_response"
    DECISION = "intermediate_reasoning_or_action_decision"
    INTERACTION = "multi_turn_interaction"
    TRAJECTORY = "extended_agent_trajectory"
    ROUND = "training_round_regression"


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {name}")


@dataclass(frozen=True)
class FailureAnchor:
    """Captured public state and location in a trajectory or training round."""

    scale: AnchorScale
    trajectory_or_round: str
    coordinate: str
    policy_version: str
    prompt: str
    snapshot: Mapping[str, object]
    original_decision: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.scale, AnchorScale):
            raise ValueError("unknown anchor scale")
        for name in (
            "trajectory_or_round",
            "coordinate",
            "policy_version",
            "prompt",
            "original_decision",
        ):
            _text(getattr(self, name), name)
        for name in ("snapshot", "metadata"):
            object.__setattr__(
                self, name, freeze_json_mapping(getattr(self, name), field_name=name)
            )

    @property
    def location(self) -> tuple[str, str, str]:
        return (self.scale.value, self.trajectory_or_round, self.coordinate)


@dataclass(frozen=True)
class Alternative:
    """An observable decision, replay policy and retained proposal provenance."""

    decision: str
    policy_version: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _text(self.decision, "decision")
        _text(self.policy_version, "policy_version")
        object.__setattr__(
            self, "metadata", freeze_json_mapping(self.metadata, field_name="metadata")
        )


@dataclass(frozen=True)
class ReplayCall:
    """The host restores the detached snapshot and runs a sandbox continuation."""

    anchor: FailureAnchor
    alternative: Alternative
    evaluator: str
    seed: int
    candidate: bool
    retention: bool

    @property
    def decision(self) -> str:
        if self.candidate and not self.retention:
            return self.alternative.decision
        return self.anchor.original_decision

    @property
    def policy_version(self) -> str:
        if self.candidate:
            return self.alternative.policy_version
        return self.anchor.policy_version

    def restore(self) -> dict[str, object]:
        return thaw_json_mapping(self.anchor.snapshot)

    @property
    def digest(self) -> str:
        return content_digest(
            dict(
                version="mdac-continuation-v1",
                location=list(self.anchor.location),
                source=thaw_json_mapping(self.anchor.metadata),
                state=self.restore(),
                prompt=self.anchor.prompt,
                original=self.anchor.original_decision,
                incumbent=self.anchor.policy_version,
                proposal=self.alternative.decision,
                candidate_policy=self.alternative.policy_version,
                candidate_source=thaw_json_mapping(self.alternative.metadata),
                candidate=self.candidate,
                retention=self.retention,
                seed=self.seed,
                evaluator=self.evaluator,
            )
        )


@dataclass(frozen=True)
class Continuation:
    """Host verdict at the whole continuation horizon with locally verified progress."""

    request_digest: str
    restored_state_digest: str
    checks: Mapping[str, bool | None]
    evidence: EvidenceReference
    verified_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.checks) != set(CHECKS) or any(
            type(value) is not bool and value is not None for value in self.checks.values()
        ):
            raise ValueError("every continuation criterion requires a boolean or unknown")
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        if type(self.evidence) is not EvidenceReference or (
            self.evidence.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD
        ):
            raise ValueError("sandbox continuation requires external evidence")
        if type(self.verified_steps) is not tuple:
            raise ValueError("verified_steps must be an immutable sequence of public step IDs")
        for step in self.verified_steps:
            _text(step, "verified step ID")

    @property
    def passed(self) -> bool:
        return all(value is True for value in self.checks.values())


@dataclass(frozen=True)
class RepairExample:
    """Canonical actor request plus the separately verified supervision target."""

    request: RolloutRequest
    target: str


@dataclass(frozen=True)
class RepairResult:
    reason: str
    trials: tuple[Continuation, ...]
    training_example: RepairExample | None

    @property
    def verified(self) -> bool:
        return self.reason == "verified"


def replay_repair(
    anchor: FailureAnchor,
    alternative: Alternative,
    retention: tuple[FailureAnchor, ...],
    *,
    sandbox: Callable[[ReplayCall], Continuation],
    evaluator: str,
    seeds: tuple[int, ...],
    enabled: bool = False,
    export_training: bool = False,
) -> RepairResult | None:
    """Compare incumbent and candidate in separately restored environments for each seed.

    The trusted host owns sandbox execution, independent evaluation and source custody.
    On retention cases both policies start with the retained decision and generate onward.
    """

    if enabled is not True:
        return None
    _text(evaluator, "evaluator")
    if type(seeds) is not tuple or len(set(seeds)) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct confirmation seeds required")
    for seed in seeds:
        if validate_seed(seed) is None:
            raise ValueError("explicit replay seed required")
    if alternative.decision == anchor.original_decision:
        raise ValueError("repair requires a distinct decision")
    locations = {anchor.location, *(item.location for item in retention)}
    if not retention or len(locations) != 1 + len(retention):
        raise ValueError("distinct retention anchors required")
    if any(item.policy_version != anchor.policy_version for item in retention):
        raise ValueError("retention anchors must bind the same incumbent")
    trials: list[Continuation] = []
    evidence_ids: set[str] = set()
    failures: list[str] = []
    for item in (anchor, *retention):
        retaining = item is not anchor
        for seed in seeds:
            pair = []
            for candidate in (False, True):
                call = ReplayCall(item, alternative, evaluator, seed, candidate, retaining)
                trial = sandbox(call)
                if type(trial) is not Continuation or (
                    trial.request_digest != call.digest
                    or trial.restored_state_digest != content_digest(call.restore())
                    or trial.evidence.reference_id in evidence_ids
                ):
                    raise ValueError("invalid restored state, replay binding or repeated evidence")
                evidence_ids.add(trial.evidence.reference_id)
                trials.append(trial)
                pair.append(trial)
            before, after = pair
            if retaining:
                if not before.passed or not after.passed:
                    failures.append("retention failed")
            elif False not in before.checks.values():
                failures.append("failure not reproduced")
            elif not after.passed:
                failures.append("candidate continuation failed")
    example = None
    if not failures and export_training is True:
        sources = (anchor.metadata, alternative.metadata, *(item.metadata for item in retention))
        for metadata in sources:
            if metadata.get("training_eligibility") != "TRAIN":
                raise ValueError("training_eligibility must be explicit TRAIN for all sources")
            require_training_eligible(metadata)
        metadata = freeze_json_mapping(
            dict(
                training_eligibility="TRAIN",
                source_record=thaw_json_mapping(anchor.metadata),
                alternative_source=thaw_json_mapping(alternative.metadata),
                retention_sources=[thaw_json_mapping(x.metadata) for x in retention],
                anchor=list(anchor.location),
                evaluator=evaluator,
                replay_evidence=[t.evidence.to_dict() for t in trials],
            ),
            field_name="verified repair provenance",
        )
        example = RepairExample(
            RolloutRequest(anchor.prompt, policy_version=anchor.policy_version, metadata=metadata),
            alternative.decision,
        )
    return RepairResult(failures[0] if failures else "verified", tuple(trials), example)


def retentive_loss(
    repair: Any,
    retention: Any,
    generalization: Any,
    *,
    retention_weight: float,
    generalization_weight: float,
) -> Any:
    """Combine trainer-computed scalar losses while preserving native autograd graphs."""

    for weight in (retention_weight, generalization_weight):
        if type(weight) not in (int, float) or not isfinite(weight) or weight < 0:
            raise ValueError("loss weights must be finite and nonnegative")
    return repair + retention_weight * retention + generalization_weight * generalization
