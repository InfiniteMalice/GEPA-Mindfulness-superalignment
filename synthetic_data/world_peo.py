"""Export longitudinal offline simulations through the existing action-bound PEO stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_reconciliation import (
    EpistemicReconciliation,
    OutcomeMeasurementBinding,
    make_epistemic_reconciliation_event,
)
from gepa_mindfulness.verification.epistemic_state import (
    EpistemicContext,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    UncertaintyUpdateRecord,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
    make_verification_result_event,
)
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .worlds import SyntheticWorld, _action, _enabled, _records, _text, render_world, simulate


@dataclass(frozen=True)
class EpisodeStep:
    """Caller-supplied prospective prediction, not an oracle-generated target."""

    action_id: str
    predicted_success: float
    confidence: float

    def __post_init__(self) -> None:
        _text(self.action_id)
        # Reuse the existing finite, non-boolean confidence contract for both unit values.
        PredictionCommit("check", {}, self.predicted_success, ())
        PredictionCommit("check", {}, self.confidence, ())


def build_episode(
    world: SyntheticWorld,
    steps: tuple[EpisodeStep, ...],
    *,
    context: EpistemicContext,
    episode_id: str,
    start_timestamp: str,
    enabled: bool = False,
) -> dict[str, Any]:
    """Commit predictions before each offline attempt and return evaluator-only JSON data.

    The complete export contains latent truth and labels: never use it as an actor prompt.
    Each step's actor_prompt is the separate public projection available before that attempt.
    Verification certifies simulator contract consistency only, not real-world correctness.
    """
    _enabled(enabled)
    _text(episode_id)
    if type(world) is not SyntheticWorld or type(context) is not EpistemicContext:
        raise ValueError("world and context must be validated typed records")
    _records(steps, EpisodeStep)
    if not steps:
        raise ValueError("an episode requires at least one step")
    start = parse_rfc3339_datetime(start_timestamp, "start_timestamp")
    events: list[EventEnvelope] = []
    snapshots = [world.to_dict()]
    step_records = []
    evidence_records: dict[str, dict[str, Any]] = {}
    provenance = world.provenance
    previous_action: str | None = None
    previous_prediction: str | None = None
    for index, step in enumerate(steps):
        action = _action(world, step.action_id)
        prefix = f"{episode_id}:{index}"
        prediction_id, action_id, observation_id = (
            f"{prefix}:prediction",
            f"{prefix}:action",
            f"{prefix}:observation",
        )
        state_ref = EvidenceReference(f"{prefix}:state", EvidenceSourceKind.EXTERNAL_RECORD)
        result_ref = EvidenceReference(f"{prefix}:result", EvidenceSourceKind.OBSERVABLE_OUTPUT)

        def metadata(suffix: str, offset: int, parents: tuple[str, ...] = ()) -> dict[str, Any]:
            return dict(
                event_id=f"{prefix}:{suffix}",
                timestamp=(start + timedelta(seconds=index * 6 + offset)).isoformat(),
                run_id=context.run_id,
                repeat_id=context.repeat_id,
                model_version=context.system.model_version,
                harness_version=context.system.harness_version,
                parent_event_ids=parents,
            )

        def estimate(
            snapshot: SyntheticWorld,
            suffix: str,
            evidence: EvidenceReference,
            linked_action: str | None,
            linked_prediction: str | None,
        ) -> EpistemicStateEstimate:
            hidden = sum(action.actor_id not in fact.visible_to for fact in snapshot.facts)
            return EpistemicStateEstimate(
                estimate_id=f"{prefix}:{suffix}",
                context=context,
                estimator_version="hidden-fact-fraction-v1",
                world_uncertainty=hidden / len(snapshot.facts),
                model_uncertainty=None,
                monitor_uncertainty=None,
                action_id=linked_action,
                prediction_commit_id=linked_prediction,
                evidence_refs=(evidence,),
                provenance=provenance,
            )

        prior = estimate(world, "prior", state_ref, previous_action, previous_prediction)
        actor_prompt = render_world(world, actor_id=action.actor_id, enabled=True)
        parents = (events[-1].event_id,) if events else ()
        events.append(
            make_prediction_commit_event(
                PredictionCommit(
                    prediction_id,
                    {"success": step.predicted_success},
                    step.confidence,
                    (state_ref.reference_id,),
                ),
                **metadata("p", 0, parents),
            )
        )
        # This is the offline simulator operation. The target action can be denied.
        operation = ActionRecord(
            action_id, f"simulate:{action.action_id}", True, "offline_simulation", prediction_id
        )
        events.append(
            make_action_event(
                operation, StructuredEventType.ACTION_PROPOSED, **metadata("a", 1, (f"{prefix}:p",))
            )
        )
        events.append(
            make_action_event(
                operation, StructuredEventType.ACTION_EXECUTED, **metadata("x", 2, (f"{prefix}:a",))
            )
        )
        transition = simulate(world, action.action_id, enabled=True)
        actual = float(transition.success)
        evidence_records[state_ref.reference_id] = dict(
            world_digest=world.digest,
            actor_id=action.actor_id,
        )
        evidence_records[result_ref.reference_id] = dict(
            before_digest=world.digest,
            after_digest=transition.after.digest,
            action_id=action.action_id,
            success=actual,
        )
        events.append(
            make_outcome_observation_event(
                OutcomeObservation(
                    observation_id, action_id, {"success": actual}, (result_ref.reference_id,)
                ),
                **metadata("o", 3, (f"{prefix}:x",)),
            )
        )
        consistent = simulate(world, action.action_id, enabled=True).after == transition.after
        events.append(
            make_verification_result_event(
                VerificationResult(
                    f"{prefix}:synthetic-world-replay",
                    "boolean-world-v1",
                    observation_id,
                    consistent,
                    (result_ref.reference_id,),
                ),
                **metadata("v", 4, (f"{prefix}:o",)),
            )
        )
        measurement = EpistemicMeasurement(
            measurement_id=f"{prefix}:measurement",
            context=context,
            source=ConfidenceSource.TOOL_RESULT,
            target_dimension="simulation_success",
            representation_id="boolean-success-v1",
            value=actual,
            uncertainty=None,
            evidence_refs=(result_ref,),
            provenance=provenance,
        )
        innovation = InnovationRecord(
            innovation_id=f"{prefix}:innovation",
            context=context,
            prediction_commit_id=prediction_id,
            observation_id=observation_id,
            predicted_measurement=step.predicted_success,
            actual_measurement=actual,
            evidence_refs=(result_ref,),
            provenance=provenance,
        )
        update = UncertaintyUpdateRecord(
            update_id=f"{prefix}:update",
            prior_state=prior,
            measurements=(measurement,),
            posterior_state=estimate(
                transition.after, "posterior", result_ref, action_id, prediction_id
            ),
            update_method="visibility-count-and-recorded-comparison",
            evidence_refs=(state_ref, result_ref),
            provenance=provenance,
        )
        reconciliation = EpistemicReconciliation(
            update=update,
            prediction_event_id=f"{prefix}:p",
            observation_event_id=f"{prefix}:o",
            verification_event_ids=(f"{prefix}:v",),
            bindings=(
                OutcomeMeasurementBinding(
                    measurement_id=measurement.measurement_id,
                    innovation=innovation,
                    prediction_path=("success",),
                    observation_path=("success",),
                ),
            ),
        )
        events.append(
            make_epistemic_reconciliation_event(
                reconciliation,
                event_id=f"{prefix}:r",
                timestamp=metadata("r", 5)["timestamp"],
            )
        )
        step_records.append(
            dict(
                **asdict(step),
                actor_prompt=actor_prompt,
                expected_judgment=transition.reason,
                success=transition.success,
                before_digest=world.digest,
                after_digest=transition.after.digest,
            )
        )
        world = transition.after
        snapshots.append(world.to_dict())
        previous_action, previous_prediction = action_id, prediction_id
    validate_action_bound_sequence(events)
    return dict(
        schema_version="synthetic-world-peo-v1",
        episode_id=episode_id,
        training_eligibility=world.training_eligibility.value,
        verification_scope="simulator_consistency_only",
        evidence_records=evidence_records,
        worlds=snapshots,
        steps=step_records,
        events=[event.to_dict() for event in events],
    )
