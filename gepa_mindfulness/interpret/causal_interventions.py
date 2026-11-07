"""Experimental actual patching, restoration, path and controlled mediation experiments.

All numeric runs remain unresolved diagnostics until independent host verification.
No parameter surgery, optimizer signal, intent inference or external execution occurs.
"""

# Standard library
from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import _record
from gepa_mindfulness.verification.runtime_governance import TrustedClock
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
)
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence

from .activation_simulation import replace_selected_activations
from .native_causal_records import (
    CausalCommit,
    CausalPlan,
    EffectPrediction,
    InterventionMode,
    InterventionSite,
    LogitContrast,
    run_measurement_digest,
)
from .native_neurons import FrozenNativeModel
from .native_records import NativeInput

EXECUTOR_ID = "gepa-native-causal-executor"
EXECUTOR_VERSION = "1"


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("causal experiments require enabled=True")


def _now(clock: TrustedClock) -> str:
    value = clock.now()
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("causal experiment clock requires an aware datetime")
    return value.astimezone(timezone.utc).isoformat()


def freeze_causal_plan(
    plan: CausalPlan,
    *,
    prediction: PredictionCommit,
    action: ActionRecord,
    clock: TrustedClock,
    enabled: bool = False,
) -> CausalCommit:
    """Bind the existing immutable prediction/action contracts before actual execution."""
    _enabled(enabled)
    snapshot = _record(plan, CausalPlan)
    if type(prediction) is not PredictionCommit or type(action) is not ActionRecord:
        raise ValueError("exact prediction/action records required")
    return CausalCommit(
        snapshot,
        canonical_json(prediction.to_dict()),
        canonical_json(action.to_dict()),
        _now(clock),
    )


def _selection(tensor: Any, site: InterventionSite) -> Any:
    return tensor[:, site.token_positions, :][:, :, site.columns].detach().clone()


def _run(
    backend: FrozenNativeModel,
    context: NativeInput,
    sites: dict[str, InterventionSite],
    patches: dict[str, Any],
    metric: LogitContrast,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Multi-hook trusted executor: capture actual order, originals and applied values."""
    import torch

    backend._assert_frozen()
    observed: dict[str, dict[str, Any]] = {}
    tensors: dict[str, Any] = {}
    order: list[str] = []
    handles = []

    def transform(name: str, value: Any) -> Any:
        site = sites[name]
        if name in observed:
            raise ValueError("intervention hook fired more than once")
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != (1, len(context.token_ids), site.hook.width)
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("actual causal hook tensor is incompatible or nonfinite")
        order.append(name)
        tensors[name] = value.detach().clone()
        changed = (
            value
            if name not in patches
            else replace_selected_activations(
                value,
                patches[name],
                token_positions=site.token_positions,
                component_indices=site.columns,
            )
        )
        before, after = _selection(value, site), _selection(changed, site)
        observed[name] = {
            "site": site.to_dict(),
            "before": before[0].cpu().tolist(),
            "after": after[0].detach().cpu().tolist(),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "delta_l2": (after.double() - before.double()).norm(dim=-1)[0].cpu().tolist(),
        }
        return changed

    def make_before(name: str):
        def before(module: Any, inputs: Any) -> Any:
            return (transform(name, inputs[0]),) + inputs[1:]

        return before

    def make_after(name: str):
        def after(module: Any, inputs: Any, output: Any) -> Any:
            if sites[name].hook.location == "output[0]":
                if type(output) is not tuple or not output:
                    raise ValueError("causal hook requires tuple output")
                return (transform(name, output[0]),) + output[1:]
            return transform(name, output)

        return after

    try:
        for name, site in sites.items():
            module = backend._model.get_submodule(site.hook.module_path)
            handle = (
                module.register_forward_pre_hook(make_before(name))
                if site.hook.location == "input"
                else module.register_forward_hook(make_after(name))
            )
            handles.append(handle)
        with torch.no_grad():
            ids = torch.tensor([context.token_ids], dtype=torch.long, device=backend._device)
            mask = torch.tensor([context.attention_mask], dtype=torch.long, device=backend._device)
            logits = backend._model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            if tuple(logits.shape) != (
                1,
                len(context.token_ids),
                backend._model.config.vocab_size,
            ) or not bool(torch.isfinite(logits).all()):
                raise ValueError("invalid causal model output")
            raw = logits[0].detach().cpu().double().tolist()
    finally:
        for handle in reversed(handles):
            handle.remove()
    backend._assert_frozen()
    if set(observed) != set(sites):
        raise ValueError("causal hook did not fire")
    if "mediator" in sites and order.index("source") >= order.index("mediator"):
        raise ValueError("actual mediator hook order must follow source")
    score = raw[metric.position][metric.target_token] - raw[metric.position][metric.reference_token]
    return {
        "input": context.to_dict(),
        "logits": raw,
        "score": score,
        "sites": observed,
        "hook_order": order,
    }, tensors


def _effect_outcome(plan: CausalPlan, primary: float, control: float | None) -> str:
    prediction = plan.prediction
    signed = prediction.direction * primary
    if not prediction.minimum_effect <= signed <= prediction.maximum_effect:
        return "FALSIFIED"
    if (
        control is None
        or prediction.direction * (primary - control) < prediction.minimum_control_gap
    ):
        return "NOT_SPECIFIC"
    return "MET"


def _make_events(
    commit: CausalCommit, report: dict[str, Any], timestamp: str
) -> list[dict[str, Any]]:
    plan = commit.plan
    prefix = content_digest({"commit": commit.digest, "action": commit.action_record.action_id})
    metadata = {
        "run_id": plan.experiment_id,
        "model_version": plan.model_digest,
        "harness_version": EXECUTOR_ID + ":" + EXECUTOR_VERSION,
        "seed": plan.seed,
    }
    prediction = make_prediction_commit_event(
        commit.prediction_record,
        event_id=prefix + ":prediction",
        timestamp=commit.created_at,
        **metadata,
    )
    proposed = make_action_event(
        commit.action_record,
        "action_proposed",
        event_id=prefix + ":proposed",
        timestamp=commit.created_at,
        parent_event_ids=(prediction.event_id,),
        **metadata,
    )
    executed = make_action_event(
        commit.action_record,
        "action_executed",
        event_id=prefix + ":executed",
        timestamp=timestamp,
        parent_event_ids=(proposed.event_id,),
        **metadata,
    )
    observation = OutcomeObservation(
        prefix + ":observation",
        commit.action_record.action_id,
        {"measurement_digest": report["measurement_digest"]},
        tuple(r.reference_id for r in plan.evidence_refs),
    )
    outcome = make_outcome_observation_event(
        observation,
        event_id=prefix + ":outcome",
        timestamp=timestamp,
        parent_event_ids=(executed.event_id,),
        **metadata,
    )
    events = (prediction, proposed, executed, outcome)
    validate_action_bound_sequence(events)
    return [json.loads(canonical_json(e.to_dict())) for e in events]


def execute_causal_plan(
    backend: FrozenNativeModel,
    commit: CausalCommit,
    *,
    clock: TrustedClock,
    enabled: bool = False,
) -> dict[str, Any]:
    """Execute real interventions; raw measurements never self-admit causal evidence."""
    _enabled(enabled)
    frozen = _record(commit, CausalCommit)
    plan = frozen.plan
    if type(backend) is not FrozenNativeModel or backend.identity.digest != plan.model_digest:
        raise ValueError("causal plan requires its exact frozen model")
    sites = {"source": plan.source}
    if plan.mediator is not None:
        sites["mediator"] = plan.mediator
    for site in sites.values():
        if site.hook != backend.hook(layer=site.hook.layer, kind=site.hook.kind):
            raise ValueError("causal hook differs from verified architecture")
    if (
        max(plan.metric.target_token, plan.metric.reference_token)
        >= backend._model.config.vocab_size
    ):
        raise ValueError("metric token outside vocabulary")
    if any(
        t >= backend._model.config.vocab_size
        for i in (plan.donor, plan.recipient)
        for t in i.token_ids
    ):
        raise ValueError("causal input token outside vocabulary")
    capacity = getattr(backend._model.config, "max_position_embeddings", 128)
    if len(plan.donor.token_ids) > min(128, capacity):
        raise ValueError("causal input exceeds model context capacity")
    if (
        20 * len(plan.donor.token_ids) * backend._model.config.vocab_size
        > plan.max_distribution_values
    ):
        raise ValueError("causal distribution evidence budget exceeded before execution")
    timestamp = _now(clock)
    if parse_rfc3339_datetime(timestamp, "executed_at") < parse_rfc3339_datetime(
        frozen.created_at, "created_at"
    ):
        raise ValueError("execution clock precedes prediction commitment")
    conditions: dict[str, dict[str, Any]] = {}
    donor, d = _run(backend, plan.donor, sites, {}, plan.metric)
    recipient, r = _run(backend, plan.recipient, sites, {}, plan.metric)
    donor["effect"], recipient["effect"] = 0.0, 0.0
    conditions.update(donor=donor, recipient=recipient)

    def run(
        name: str,
        context: NativeInput,
        local_sites: dict[str, InterventionSite],
        patches: dict[str, Any],
        reference: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record, tensors = _run(backend, context, local_sites, patches, plan.metric)
        record["effect"] = record["score"] - reference
        record["reference_score"] = reference
        conditions[name] = record
        return record, tensors

    import torch

    source_d, source_r = _selection(d["source"], plan.source), _selection(r["source"], plan.source)
    zero = torch.zeros_like(source_d)
    identity_context = (
        plan.donor
        if plan.mode in (InterventionMode.ABLATION, InterventionMode.RESTORATION)
        else plan.recipient
    )
    identity_values = source_d if identity_context is plan.donor else source_r
    identity_ref = donor["score"] if identity_context is plan.donor else recipient["score"]
    identity, _ = run(
        "identity", identity_context, sites, {"source": identity_values}, identity_ref
    )
    if identity["effect"] != 0.0:
        raise ValueError("identity intervention did not preserve the exact matched output")
    ablated, _ = run("ablation", plan.donor, sites, {"source": zero}, donor["score"])

    def pipeline(
        prefix: str, local_sites: dict[str, InterventionSite], values: Any
    ) -> tuple[float, dict[str, Any] | None]:
        mode = plan.mode
        if mode is InterventionMode.ABLATION:
            measured, _ = run(prefix, plan.donor, local_sites, {"source": values}, donor["score"])
            return donor["score"] - measured["score"], None
        if mode is InterventionMode.RESTORATION:
            assert plan.mediator is not None
            loss, _ = run(
                prefix + "_source_ablated",
                plan.donor,
                local_sites,
                {"source": values},
                donor["score"],
            )
            restored, _ = run(
                prefix,
                plan.donor,
                local_sites,
                {"source": values, "mediator": _selection(d["mediator"], plan.mediator)},
                loss["score"],
            )
            return restored["effect"], None
        patched, intermediate = run(
            prefix if mode is not InterventionMode.PATH_PATCH else prefix + "_source_patch",
            plan.recipient,
            local_sites,
            {"source": values},
            recipient["score"],
        )
        if mode is InterventionMode.PATH_PATCH:
            assert plan.mediator is not None
            via_mediator, _ = run(
                prefix,
                plan.recipient,
                local_sites,
                {"mediator": _selection(intermediate["mediator"], plan.mediator)},
                recipient["score"],
            )
            return via_mediator["effect"], None
        if mode is InterventionMode.MEDIATION:
            assert plan.mediator is not None
            restored, _ = run(
                "mediator_restored" if prefix == "intervention" else prefix + "_mediator_restored",
                plan.recipient,
                local_sites,
                {"source": values, "mediator": _selection(r["mediator"], plan.mediator)},
                recipient["score"],
            )
            total, direct = patched["effect"], restored["effect"]
            details = {
                "total_effect": total,
                "direct_effect": direct,
                "mediated_effect": total - direct,
                "causal_assumptions": [
                    "Controlled mediator restoration, not a complete natural mediation model.",
                    "Other paths, interactions and off-manifold states remain possible.",
                ],
            }
            return total - direct, details
        return patched["effect"], None

    value = source_d
    if plan.mode in (InterventionMode.ABLATION, InterventionMode.RESTORATION):
        value = zero
    elif plan.mode is InterventionMode.ACTIVATION_REPLACE:
        value = torch.tensor(
            [plan.replacement_values], dtype=source_r.dtype, device=source_r.device
        )
    primary, mediation = pipeline("intervention", sites, value)
    if plan.mode is InterventionMode.COMPONENT_SWAP:
        run("reverse_swap", plan.donor, sites, {"source": source_r}, donor["score"])
    rng = random.Random(plan.seed)
    width = (
        plan.source.hook.heads
        if plan.source.hook.kind.value == "attention_head"
        else plan.source.hook.width
    )
    eligible = [i for i in range(width) if i not in plan.source.component_indices]
    random_effect = None
    if len(eligible) >= len(plan.source.component_indices):
        random_site = replace(
            plan.source,
            component_indices=tuple(rng.sample(eligible, len(plan.source.component_indices))),
        )
        random_sites = {**sites, "source": random_site}
        alternative_d = _selection(d["source"], random_site)
        alternative_value = alternative_d
        if plan.mode in (InterventionMode.ABLATION, InterventionMode.RESTORATION):
            alternative_value = torch.zeros_like(alternative_d)
        elif plan.mode is InterventionMode.ACTIVATION_REPLACE:
            alternative_value = value
        random_effect, _ = pipeline("random_component", random_sites, alternative_value)
    baseline_source = (
        source_d
        if plan.mode in (InterventionMode.ABLATION, InterventionMode.RESTORATION)
        else source_r
    )
    magnitude = (value.double() - baseline_source.double()).norm(dim=-1, keepdim=True)
    direction = torch.tensor(
        [[[rng.gauss(0.0, 1.0) for _ in plan.source.columns] for _ in plan.source.token_positions]],
        dtype=torch.float64,
        device=value.device,
    )
    fallback = torch.zeros_like(direction)
    fallback[:, :, 0] = 1.0
    direction = torch.where(
        (direction.norm(dim=-1, keepdim=True) == 0).expand_as(direction), fallback, direction
    )
    direction /= direction.norm(dim=-1, keepdim=True)
    random_values = (baseline_source.double() + direction * magnitude).to(value.dtype)
    direction_effect, _ = pipeline("magnitude_matched_direction", sites, random_values)
    report = {
        "schema_version": "native-causal-run-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "evidence_status": "UNRESOLVED",
        "measurement_authentication": "NOT_PERFORMED",
        "executor_id": EXECUTOR_ID,
        "executor_version": EXECUTOR_VERSION,
        "plan_digest": plan.digest,
        "commit_digest": frozen.digest,
        "commit": frozen.to_dict(),
        "model": backend.identity.to_dict(),
        "conditions": conditions,
        "primary_effect": primary,
        "control_effect": random_effect,
        "direction_control_effect": direction_effect,
        "mediation": mediation,
        "prediction_result": _effect_outcome(plan, primary, random_effect),
        "executed_at": timestamp,
        "evidence_refs": [e.to_dict() for e in plan.evidence_refs],
        "control_roles": {
            "random_component": "specificity at matched component budget",
            "magnitude_matched_direction": "perturbation sensitivity; singleton degeneracy applies",
        },
        "limitations": [
            "These test a functional effect; ethical understanding and intent remain unverified.",
            "Failed single-unit necessity tests do not establish irrelevance under redundancy.",
            "Path patching can aggregate several routes to a mediator; no unique edge is presumed.",
        ],
    }
    report["measurement_digest"] = run_measurement_digest(report)
    report["events"] = _make_events(frozen, report, timestamp)
    validate_input_envelope(report)
    return report


__all__ = [
    "CausalPlan",
    "CausalCommit",
    "InterventionSite",
    "InterventionMode",
    "LogitContrast",
    "EffectPrediction",
    "freeze_causal_plan",
    "execute_causal_plan",
]
