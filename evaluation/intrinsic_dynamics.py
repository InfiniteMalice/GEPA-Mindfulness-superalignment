"""Opt-in frozen linear action probes and separate latent-rollout ranking diagnostics."""

# Standard library
from __future__ import annotations

import random
from collections import defaultdict
from statistics import correlation

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.debate_records import _record
from gepa_mindfulness.verification.diagnostic_records import strings
from synthetic_data.worlds import SyntheticWorld, simulate

from .causal_records import canonical_json, content_digest
from .improvement_records import SystemConfig
from .intrinsic_dynamics_records import (
    MAX_ROWS,
    DynamicsProbeConfig,
    ProbeTransition,
    RolloutCost,
    count,
)


def _records(rows: object, cls: type) -> tuple:
    """Snapshot a bounded complete roster before numerical work or any simulation."""
    if type(rows) is not tuple or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("1..512 exact record tuple required")
    return tuple(_record(row, cls) for row in rows)


def _report(system: SystemConfig, **values) -> dict:
    """Retain host identity and explicit non-authority in bounded JSON-safe reports."""
    if type(system) is not SystemConfig:
        raise ValueError("exact SystemConfig required")
    result = dict(
        schema_version="intrinsic-dynamics-report-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_eligible=False,
        authority_granted=False,
        mechanism_recovery_established=False,
        source_reference="REF-INTRINSIC-DYNAMICS",
        source_binding="host_declared_not_authenticated",
        system=SystemConfig.from_dict(system.to_dict()).to_dict(),
        **values,
    )
    if len(canonical_json(result).encode("utf-8")) > 4 * 1048576:
        raise ValueError("diagnostic output capacity exceeded")
    return result


def capture_world_transition(
    initial_world: SyntheticWorld,
    action_ids: tuple[str, ...],
    index: int,
    *,
    trajectory_id: str,
    before: tuple[float, ...],
    after: tuple[float, ...],
    predicted_after: tuple[float, ...] | None = None,
    enabled: bool = False,
) -> ProbeTransition | None:
    """Bind supplied latents to an existing offline simulation's action and state lineage.

    This evaluator-only bridge does not encode states, prove the supplied representations
    came from them, grant real execution authority or admit synthetic worlds for training.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit boolean enabled required")
    if not enabled:
        return None
    if type(initial_world) is not SyntheticWorld:
        raise ValueError("exact initial SyntheticWorld required")
    world = SyntheticWorld.from_dict(initial_world.to_dict())
    if world.tick != 0 or world.training_eligibility.value != "DEVELOPMENT":
        raise ValueError("initial DEVELOPMENT world required; no held-out relabeling")
    if type(action_ids) is not tuple or not 1 <= len(action_ids) <= MAX_ROWS:
        raise ValueError("bounded action prefix required")
    count(index, "index", len(action_ids) - 1)
    vocabulary = tuple(sorted(action.action_id for action in world.actions))
    if any(type(action) is not str or action not in vocabulary for action in action_ids):
        raise ValueError("unknown action in prefix")
    for action in action_ids[:index]:
        world = simulate(world, action, enabled=True).after
    transition = simulate(world, action_ids[index], enabled=True)
    digest = content_digest(
        dict(
            root=initial_world.digest,
            prefix=action_ids[: index + 1],
            before=world.digest,
            after=transition.after.digest,
        )
    )
    return ProbeTransition(
        trajectory_id,
        index,
        initial_world.digest,
        digest,
        action_ids[index],
        vocabulary,
        before,
        after,
        (EvidenceReference(f"synthetic-transition:{digest}", EvidenceSourceKind.EXTERNAL_RECORD),),
        predicted_after,
    )


def _score(truth, predicted, test_rows) -> dict:
    """Report coordinate-averaged errors and variance-weighted R2 without clipping failures."""
    import numpy as np

    squared = (truth - predicted) ** 2
    sse = np.sum(squared, axis=0)
    sst = np.sum((truth - np.mean(truth, axis=0)) ** 2, axis=0)
    total = float(np.sum(sst))
    r2 = None if total == 0 else 1 - float(np.sum(sse)) / total
    choices = np.argmax(predicted, axis=1)
    ties = np.sum(predicted == np.max(predicted, axis=1, keepdims=True), axis=1) > 1
    return dict(
        mse=float(np.mean(squared)),
        r2=r2,
        r2_status="constant_targets" if r2 is None else "available",
        per_coordinate_r2=[None if v == 0 else 1 - float(e / v) for e, v in zip(sse, sst)],
        accuracy=float(np.mean(choices == np.argmax(truth, axis=1))),
        tied_predictions=int(np.sum(ties)),
        tie_break="first_declared_action",
        predictions=[
            dict(
                trajectory_id=r.trajectory_id,
                step_index=r.step_index,
                action_id=r.action_id,
                predicted_action=r.action_vocabulary[int(c)],
                scores=[float(v) for v in p],
            )
            for r, c, p in zip(test_rows, choices, predicted)
        ],
    )


def run_action_probes(
    rows: tuple[ProbeTransition, ...],
    *,
    system: SystemConfig,
    fit_trajectories: tuple[str, ...],
    test_trajectories: tuple[str, ...],
    config: DynamicsProbeConfig = DynamicsProbeConfig(),
) -> dict | None:
    """Fit fresh bounded linear probes on frozen latents using a complete trajectory split.

    Four feature controls share every fit/test row. Only the pair control additionally uses
    seeded shuffled FIT labels; test targets never enter fitting. This is a discrete linear
    adaptation, not the source's MLP probe or shared-encoder training reproduction.
    """
    config = _record(config, DynamicsProbeConfig)
    if not config.enabled:
        return None
    if type(system) is not SystemConfig:
        raise ValueError("exact SystemConfig required")
    rows = _records(rows, ProbeTransition)
    fit_ids = strings(fit_trajectories, "fit_trajectories", required=True)
    test_ids = strings(test_trajectories, "test_trajectories", required=True)
    if set(fit_ids) & set(test_ids) or set(fit_ids + test_ids) != {r.trajectory_id for r in rows}:
        raise ValueError("disjoint complete trajectory rosters required")
    if len({(r.trajectory_id, r.step_index) for r in rows}) != len(rows) or len(
        {r.transition_digest for r in rows}
    ) != len(rows):
        raise ValueError("duplicate step or transition")
    roots = defaultdict(set)
    for row in rows:
        roots[row.trajectory_id].add(row.root_world_digest)
    if any(len(values) != 1 for values in roots.values()) or (
        {r.root_world_digest for r in rows if r.trajectory_id in fit_ids}
        & {r.root_world_digest for r in rows if r.trajectory_id in test_ids}
    ):
        raise ValueError("inconsistent or overlapping root source trajectories")
    vocabulary = rows[0].action_vocabulary
    if any(r.action_vocabulary != vocabulary or len(r.before) != len(rows[0].before) for r in rows):
        raise ValueError("common action vocabulary and latent dimensions required")
    rows = tuple(sorted(rows, key=lambda r: (r.trajectory_id, r.step_index)))
    fit = tuple(r for r in rows if r.trajectory_id in fit_ids)
    test = tuple(r for r in rows if r.trajectory_id in test_ids)
    permutation = list(range(len(fit)))
    random.Random(config.seed).shuffle(permutation)
    import numpy as np

    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            fit_y = np.eye(len(vocabulary))[[vocabulary.index(r.action_id) for r in fit]]
            test_y = np.eye(len(vocabulary))[[vocabulary.index(r.action_id) for r in test]]
            probes = {}
            for arm in ("before", "after", "delta", "pair", "pair_shuffled"):

                def features(items):
                    before = np.array([r.before for r in items], dtype=np.float64)
                    after = np.array([r.after for r in items], dtype=np.float64)
                    values = dict(
                        before=before,
                        after=after,
                        delta=after - before,
                        pair=np.concatenate((before, after), axis=1),
                    )
                    values = values["pair" if arm == "pair_shuffled" else arm]
                    return np.column_stack((np.ones(len(items)), values))

                design = features(fit)
                targets = fit_y[permutation] if arm == "pair_shuffled" else fit_y
                weights, _, rank, singular = np.linalg.lstsq(design, targets, rcond=1e-12)
                predicted = features(test) @ weights
                if not np.isfinite(predicted).all():
                    raise ValueError("nonfinite action predictions")
                probes[arm] = dict(
                    **_score(test_y, predicted, test),
                    rank=int(rank),
                    design_columns=int(design.shape[1]),
                    rank_deficient=bool(rank < design.shape[1]),
                    retained_condition=(
                        None if rank == 0 else float(singular[0] / singular[rank - 1])
                    ),
                )
            observed = tuple(r for r in test if r.predicted_after is not None)
            forward = dict(
                metric="one_step_latent_mse",
                observed=len(observed),
                planned=len(test),
                missing_step_ids=[
                    [r.trajectory_id, r.step_index] for r in test if r.predicted_after is None
                ],
                mse=(
                    None
                    if not observed
                    else float(
                        np.mean(
                            [
                                (np.array(r.predicted_after) - np.array(r.after)) ** 2
                                for r in observed
                            ]
                        )
                    )
                ),
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("unsupported numerical action probe") from error
    return _report(
        system,
        kind="frozen_linear_action_probes",
        config=config.to_dict(),
        numpy_version=np.__version__,
        dtype="float64",
        rcond=1e-12,
        action_vocabulary=list(vocabulary),
        fit_trajectories=list(fit_ids),
        test_trajectories=list(test_ids),
        fit_rows=len(fit),
        test_rows=len(test),
        shuffled_fit_indices=permutation,
        probes=probes,
        forward=forward,
        source_rows=[r.to_dict() for r in rows],
        dataset_digest=content_digest([r.to_dict() for r in rows]),
    )


def _ranks(values: list[float]) -> list[float]:
    """Average equal-value ranks rather than selecting a favorable tie order."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def compare_rollout_costs(
    rows: tuple[RolloutCost, ...], *, system: SystemConfig, enabled: bool = False
) -> dict | None:
    """Compare supplied predicted/realized costs separately for each decision and horizon.

    The host must register candidates before collecting costs, bind them to executed plans
    and keep encoder/cost definitions fixed. This function authenticates none of those facts
    and never executes or selects a candidate. Missing entries remain unscored.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit boolean enabled required")
    if not enabled:
        return None
    rows = _records(rows, RolloutCost)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.trajectory_id, row.decision_id, row.horizon)].append(row)
    groups = []
    for (trajectory, decision, horizon), items in sorted(grouped.items()):
        if len({r.candidate_id for r in items}) != len(items) or len(
            {r.plan_digest for r in items}
        ) != len(items):
            raise ValueError("duplicate candidate or plan within decision/horizon")
        predicted = [r.predicted_cost for r in items]
        actual = [r.realized_cost for r in items]
        score = None
        if None in predicted or None in actual:
            status = "missing_costs"
        elif len(items) < 2:
            status = "too_few_candidates"
        elif len(set(predicted)) == 1 or len(set(actual)) == 1:
            status = "constant_costs"
        else:
            score = correlation(_ranks(predicted), _ranks(actual))
            status = "available"
        groups.append(
            dict(
                trajectory_id=trajectory,
                decision_id=decision,
                horizon=horizon,
                candidates=len(items),
                spearman=score,
                status=status,
            )
        )
    return _report(
        system, kind="latent_rollout_cost_ranking", groups=groups, rows=[r.to_dict() for r in rows]
    )
