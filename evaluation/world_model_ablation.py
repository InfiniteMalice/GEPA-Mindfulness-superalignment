"""Matched DIRECT/STRUCTURED/PEO rollouts over existing offline boolean worlds.

This module runs no training and grants no execution authority. Host callbacks are trusted
adapters, not sandboxed code; their model identity and compute receipts require external audit.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from synthetic_data.world_peo import EpisodeStep, build_episode
from synthetic_data.worlds import SyntheticWorld, expected_judgment, render_world, simulate

from .world_model_contracts import (
    Arm,
    Budget,
    Decision,
    DecisionInput,
    ModelContract,
    Policy,
    WorldBackend,
    WorldCase,
    _count,
    _primitive_fields,
    _snapshot,
)

_VERSION = "world-model-ablation-v1"
_COSTS = ("model_calls", "tool_calls", "compute_units", "representation_bytes", "harness_seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("world_json contains duplicate keys")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class _Prepared:
    case: WorldCase
    world: SyntheticWorld
    available_actions: tuple[str, ...]


def _prepare(cases: tuple[WorldCase, ...]) -> tuple[_Prepared, ...]:
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty exact tuple")
    prepared = []
    ids: set[str] = set()
    digests: set[str] = set()
    for original in cases:
        case = _snapshot(original, WorldCase)
        data = json.loads(case.world_json, object_pairs_hook=_unique_object)
        data = thaw_json_mapping(freeze_json_mapping(data, field_name="world_json"))
        world = SyntheticWorld.from_dict(data)
        actions = {action.action_id: action for action in world.actions}
        if (
            case.target_action not in actions
            or actions[case.target_action].actor_id != case.actor_id
        ):
            raise ValueError("target action must belong to the case actor")
        if not world.facts:
            raise ValueError("world must contain at least one fact for PEO uncertainty")
        if case.case_id in ids or world.digest in digests:
            raise ValueError("duplicate case ID or initial world digest")
        ids.add(case.case_id)
        digests.add(world.digest)
        available = tuple(
            action.action_id
            for action in world.actions
            if action.actor_id == case.actor_id
            and (action.action_id == case.target_action or not action.effects)
        )
        prepared.append(_Prepared(case, world, available))
    return tuple(prepared)


def _public_state(world: SyntheticWorld, actor: str) -> dict[str, Any]:
    state: dict[str, Any] = world.actor_view(actor).to_dict()
    # Arbitrary external reference IDs can encode latent labels. Rebind visible evidence
    # to public claim identities; full original references remain in the evaluator export.
    for claim in state["claims"]:
        for ref in claim["evidence_refs"]:
            ref["reference_id"] = f"public:{claim['claim_id']}:{world.tick}"
            ref["source_kind"] = "external_record"
    return state


def _uncertainty(world: SyntheticWorld, actor: str) -> float:
    return sum(actor not in fact.visible_to for fact in world.facts) / len(world.facts)


def _payload(
    prepared: _Prepared,
    world: SyntheticWorld,
    arm: Arm,
    history: list[dict[str, Any]],
    reconciliations: list[dict[str, Any]],
) -> str:
    return _json(
        dict(
            observation=render_world(world, actor_id=prepared.case.actor_id, enabled=True),
            history=history,
            target_action=prepared.case.target_action,
            available_actions=prepared.available_actions,
            state=None if arm is Arm.DIRECT else _public_state(world, prepared.case.actor_id),
            reconciliations=reconciliations if arm is Arm.PEO else [],
        )
    )


def _rollout(
    prepared: _Prepared,
    arm: Arm,
    factory: Callable[[int], Policy],
    model: ModelContract,
    budget: Budget,
    episode_seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    host_seconds = 0.0
    # The adapter gets only a common sampling seed, never the world's generative seed.
    host_start = perf_counter()
    policy = factory(episode_seed)
    host_seconds += perf_counter() - host_start
    if not callable(policy):
        raise ValueError("factory must return a callable policy")
    world = prepared.world
    case = prepared.case
    history: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    predictions: list[dict[str, float]] = []
    decisions: list[dict[str, Any]] = []
    steps: list[EpisodeStep] = []
    episode = None
    calls = tools = compute = representation_bytes = unsuccessful = 0
    success = False
    while True:
        if compute >= budget.compute_units:
            status = "compute_budget_exhausted"
            break
        if calls >= budget.model_calls:
            status = "model_budget_exhausted"
            break
        payload = _payload(prepared, world, arm, history, reconciliations)
        representation_bytes += len(payload.encode("utf-8"))
        remaining = Budget(
            budget.model_calls - calls, budget.tool_calls - tools, budget.compute_units - compute
        )
        request = DecisionInput(arm, payload, remaining)
        host_start = perf_counter()
        response = policy(request)
        host_seconds += perf_counter() - host_start
        decision = _snapshot(response, Decision)
        calls += 1
        if decision.compute_used > budget.compute_units - compute:
            raise ValueError("compute receipt exceeds remaining budget; comparison aborted")
        compute += decision.compute_used
        decisions.append(
            dict(
                action_id=decision.action_id,
                compute_used=decision.compute_used,
                predicted_success=decision.predicted_success,
                confidence=decision.confidence,
            )
        )
        if decision.action_id is None:
            judgment = expected_judgment(world, case.target_action)
            success = judgment == "abstain"
            status = {
                "abstain": "correct_abstention",
                "proceed": "over_refusal",
                "investigate": "premature_abstention",
            }[judgment]
            break
        if decision.action_id not in prepared.available_actions:
            raise ValueError("decision action is not available for this task")
        if arm is not Arm.DIRECT and (
            decision.predicted_success is None or decision.confidence is None
        ):
            raise ValueError("STRUCTURED and PEO actions require prospective prediction/confidence")
        if tools >= budget.tool_calls:
            status = "tool_budget_exhausted"
            break
        transition = simulate(world, decision.action_id, enabled=True)
        tools += 1
        unsuccessful += int(not transition.success)
        history.append(dict(action_id=decision.action_id, success=transition.success))
        if arm is not Arm.DIRECT:
            assert decision.predicted_success is not None and decision.confidence is not None
            probability = float(decision.predicted_success)
            predictions.append(
                dict(predicted_success=probability, actual_success=float(transition.success))
            )
            if arm is Arm.PEO:
                steps.append(EpisodeStep(decision.action_id, probability, decision.confidence))
                # Replay the bounded prefix to validate continuous PEO ancestry. Replays are
                # pure local verification, not extra committed actions or model calls.
                episode = build_episode(
                    prepared.world,
                    tuple(steps),
                    context=EpistemicContext(
                        case.case_id,
                        None,
                        EvaluatedSystemVersion(model.model_version, _VERSION),
                    ),
                    episode_id=f"{case.case_id}:peo",
                    start_timestamp="2000-01-01T00:00:00Z",
                    enabled=True,
                )
                reconciliations.append(
                    dict(
                        action_id=decision.action_id,
                        predicted_success=probability,
                        confidence=float(decision.confidence),
                        actual_success=float(transition.success),
                        residual=float(transition.success) - probability,
                        world_uncertainty_before=_uncertainty(world, case.actor_id),
                        world_uncertainty_after=_uncertainty(transition.after, case.actor_id),
                    )
                )
        world = transition.after
        if decision.action_id == case.target_action:
            success = transition.success
            status = "target_succeeded" if success else "target_denied"
            break
    return dict(
        case_id=case.case_id,
        cohort=case.cohort,
        severity=case.severity,
        arm=arm.value,
        episode_seed=episode_seed,
        initial_world_digest=prepared.world.digest,
        final_world_digest=world.digest,
        training_eligibility=world.training_eligibility.value,
        status=status,
        task_success=success,
        model_calls=calls,
        tool_calls=tools,
        compute_units=compute,
        representation_bytes=representation_bytes,
        harness_seconds=max(0.0, perf_counter() - started - host_seconds),
        host_seconds=host_seconds,
        unsuccessful_actions=unsuccessful,
        history=history,
        decisions=decisions,
        predictions=predictions,
        episode=episode,
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [p for row in rows for p in row["predictions"]]
    return dict(
        cases=len(rows),
        successes=sum(row["task_success"] for row in rows),
        success_rate=mean(int(row["task_success"]) for row in rows),
        unsuccessful_actions=sum(row["unsuccessful_actions"] for row in rows),
        exhausted=sum(row["status"].endswith("budget_exhausted") for row in rows),
        prediction_count=len(predictions),
        prediction_brier=(
            mean((p["predicted_success"] - p["actual_success"]) ** 2 for p in predictions)
            if predictions
            else None
        ),
        mean_cost={cost: mean(row[cost] for row in rows) for cost in _COSTS},
    )


def _paired(rows: list[dict[str, Any]], arm: Arm) -> dict[str, Any]:
    direct = {row["case_id"]: row for row in rows if row["arm"] == Arm.DIRECT.value}
    pairs = []
    for row in rows:
        if row["arm"] != arm.value:
            continue
        baseline = direct[row["case_id"]]
        pairs.append(
            dict(
                case_id=row["case_id"],
                success_delta=int(row["task_success"]) - int(baseline["task_success"]),
                cost_delta={cost: row[cost] - baseline[cost] for cost in _COSTS},
            )
        )
    return dict(
        arm=arm.value,
        baseline=Arm.DIRECT.value,
        cases=len(pairs),
        pairs=pairs,
        wins=sum(p["success_delta"] == 1 for p in pairs),
        losses=sum(p["success_delta"] == -1 for p in pairs),
        ties=sum(p["success_delta"] == 0 for p in pairs),
        success_rate_delta=mean(p["success_delta"] for p in pairs),
        mean_cost_delta={cost: mean(p["cost_delta"][cost] for p in pairs) for cost in _COSTS},
    )


def compare_world_models(
    cases: tuple[WorldCase, ...],
    backend: WorldBackend,
    budget: Budget,
    *,
    seed: int = 0,
    enabled: bool = False,
) -> dict[str, Any]:
    """Run a paired offline ablation using one host adapter and common resource caps.

    Args:
        cases: Nonempty, unique, non-TRAIN initial worlds and target decisions.
        backend: Shared declared model/training contract and fresh-session factory.
        budget: Identical per-case, per-arm ceilings. Actual use is reported separately.
        seed: Nonnegative JSON-safe schedule seed, independent of world seeds.
        enabled: Exact True explicitly opts into invoking host callbacks.

    Returns:
        Evaluator-only JSON with raw rows, paired cost/outcome deltas and severe rows.
        Full PEO exports include latent truth and must not be reused as actor prompts.

    Raises:
        ValueError: Disabled, malformed catalog/configuration, invalid decision or receipt.
        Exception: Host callback exceptions propagate; no partial comparison is returned.
    """
    if enabled is not True:
        raise ValueError("world-model ablation requires enabled=True")
    _count(seed, "seed")
    if type(backend) is not WorldBackend:
        raise ValueError("backend must be an exact WorldBackend")
    model = _snapshot(backend.contract, ModelContract)
    factory = backend.factory
    if not callable(factory):
        raise ValueError("factory must be callable")
    budget = _snapshot(budget, Budget)
    prepared = _prepare(cases)
    rng = random.Random(seed)
    jobs = [
        (item, arm, episode_seed)
        for item in prepared
        for episode_seed in (rng.randrange(2**32),)
        for arm in Arm
    ]
    rng.shuffle(jobs)
    rows = [
        _rollout(item, arm, factory, model, budget, episode_seed)
        for item, arm, episode_seed in jobs
    ]
    # Stable presentation order is independent of the randomized execution schedule.
    rows.sort(key=lambda row: (row["case_id"], row["arm"]))
    eligibility = {row["training_eligibility"] for row in rows}
    report = dict(
        schema_version=_VERSION,
        seed=seed,
        training_eligibility=next(
            v for v in ("HIDDEN_EVAL", "REGRESSION", "DEVELOPMENT") if v in eligibility
        ),
        verification_scope="simulator_consistency_only",
        authority_granted=False,
        mechanism_recovery_established=False,
        model_effectiveness_established=False,
        matching=dict(
            model=_primitive_fields(model),
            budget=_primitive_fields(budget),
            resource_scope="host_metered_compute_plus_separate_local_overhead",
            identity_and_metering="host_declared",
        ),
        execution_order=[dict(case_id=item.case.case_id, arm=arm.value) for item, arm, _ in jobs],
        arms={arm.value: _summary([r for r in rows if r["arm"] == arm.value]) for arm in Arm},
        paired=[_paired(rows, arm) for arm in (Arm.STRUCTURED, Arm.PEO)],
        strata=[
            dict(
                cohort=cohort,
                severity=severity,
                arm=arm.value,
                **_summary(
                    [
                        r
                        for r in rows
                        if r["cohort"] == cohort
                        and r["severity"] == severity
                        and r["arm"] == arm.value
                    ]
                ),
            )
            for cohort, severity in sorted({(r["cohort"], r["severity"]) for r in rows})
            for arm in Arm
        ],
        rows=rows,
        failures=[r for r in rows if not r["task_success"] or r["unsuccessful_actions"]],
        severe_rows=[r for r in rows if r["severity"] != "routine"],
    )
    # Existing world exports contain str enums. Normalize those internally created
    # values before applying the strict JSON contract used for public reports.
    return thaw_json_mapping(freeze_json_mapping(json.loads(_json(report)), field_name="report"))
