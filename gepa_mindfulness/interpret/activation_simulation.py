"""Experimental activation substitution with real downstream KL and matched controls.

Only copied frozen models and closed predictors are used. Results measure these finite
interventions, not semantic explanation, ethical understanding or training eligibility.
"""

# Standard library
from __future__ import annotations

import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

# Local
from gepa_mindfulness.verification.cva_records import _record, array
from gepa_mindfulness.verification.runtime_governance import TrustedClock

from .activation_simulation_records import ComponentControl, LogitChoiceProbe, SimulationControls
from .explanation_tests import (
    ExplanationCommit,
    TokenFeature,
    _batch_for_split,
    fit_ridge_predictor,
    require_causal_prefix_isolation,
)
from .native_neurons import FrozenNativeModel
from .native_records import HookKind, HookSpec, NativeBatch, NativeInput, index, indices

CONDITIONS = (
    "explanation",
    "identity",
    "ablation",
    "mean",
    "random_predictor",
    "shuffled_labels",
    "magnitude_matched_random_direction",
    "random_neurons",
    "matched_complexity",
    "cross_context",
)


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("activation simulation requires enabled=True")


def downstream_kl(original: Any, substituted: Any) -> Any:
    """Compute KL(P_original || P_substituted) at each [batch,position], in float64."""
    import torch

    if (
        type(original) is not torch.Tensor
        or type(substituted) is not torch.Tensor
        or original.ndim != 3
        or original.shape != substituted.shape
        or original.dtype != substituted.dtype
        or original.device != substituted.device
        or not original.is_floating_point()
        or original.device.type == "meta"
        or min(original.shape) < 1
    ):
        raise ValueError("KL requires matching finite [batch,position,vocab] floating logits")
    if not bool(torch.isfinite(original).all() and torch.isfinite(substituted).all()):
        raise ValueError("nonfinite KL logits")
    log_p = original.double().log_softmax(-1)
    log_q = substituted.double().log_softmax(-1)
    result = (log_p.exp() * (log_p - log_q)).sum(-1)
    if not bool(torch.isfinite(result).all()) or bool((result < -1e-12).any()):
        raise ValueError("numeric KL breakdown")
    # Correct only representational negative roundoff; the underlying value is nonnegative.
    return result.clamp_min(0.0)


def replace_selected_activations(
    original: Any,
    replacement: Any,
    *,
    token_positions: tuple[int, ...],
    component_indices: tuple[int, ...],
) -> Any:
    """Copy a [B,T,D] tensor and replace an exactly shaped [B,P,C] selection."""
    import torch

    positions = indices(token_positions, 128)
    components = indices(component_indices, 64)
    if (
        type(original) is not torch.Tensor
        or type(replacement) is not torch.Tensor
        or original.ndim != 3
        or not original.is_floating_point()
        or original.device.type == "meta"
        or original.device != replacement.device
        or original.dtype != replacement.dtype
    ):
        raise ValueError("substitution requires exact floating tensor dtype/device compatibility")
    if (
        tuple(replacement.shape) != (original.shape[0], len(positions), len(components))
        or max(positions) >= original.shape[1]
        or max(components) >= original.shape[2]
    ):
        raise ValueError("substitution tensor shape/position/component mismatch")
    if not bool(torch.isfinite(original).all() and torch.isfinite(replacement).all()):
        raise ValueError("substitution contains nonfinite values")
    result = original.clone()
    for p, position in enumerate(positions):
        for c, component in enumerate(components):
            result[:, position, component] = replacement[:, p, c]
    return result


def _commits(commits: tuple[ExplanationCommit, ...]) -> tuple[ExplanationCommit, ...]:
    values = tuple(_record(c, ExplanationCommit) for c in array(commits, 16, required=True))
    first = values[0]
    if len({c.component_index for c in values}) != len(values):
        raise ValueError("duplicate component commitments")
    for c in values:
        if (
            c.model_digest != first.model_digest
            or c.hook_digest != first.hook_digest
            or c.splits != first.splits
            or c.evaluation_positions != first.evaluation_positions
        ):
            raise ValueError("group commitments require matched model/hook/splits/positions")
        for feature in c.hypothesis.predictor.features:
            if feature.offset > 0 or (
                feature.other_offset is not None and feature.other_offset > 0
            ):
                raise ValueError("future-token features are not permitted in causal simulation")
    return values


def prepare_simulation_controls(
    commits: tuple[ExplanationCommit, ...],
    generation_batches: tuple[NativeBatch, ...],
    *,
    seed: int,
    logit_positions: tuple[int, ...] | None = None,
    probes: tuple[LogitChoiceProbe, ...] = (),
    clock: TrustedClock | None = None,
    enabled: bool = False,
) -> SimulationControls:
    """Freeze generation-only controls and output positions before any locked measurement.

    Temporal custody/authenticity still require host verification. Original source batches
    are retained, and simulation independently replays them on the actual frozen model.
    """
    _enabled(enabled)
    frozen = _commits(commits)
    index(seed, maximum=2**32)
    batches = tuple(_record(b, NativeBatch) for b in array(generation_batches, 16, required=True))
    if len(frozen) != len(batches):
        raise ValueError("one generation batch required per commitment")
    rng = random.Random(seed)
    controls = []
    for commit, source in zip(frozen, batches):
        batch = _batch_for_split(source, commit.splits.generation, "generation")
        if (
            batch.digest != commit.generation_batch_digest
            or batch.model.digest != commit.model_digest
            or batch.hook.digest != commit.hook_digest
            or batch.component_indices != (commit.component_index,)
            or batch.token_positions != commit.evaluation_positions
        ):
            raise ValueError("generation batch differs from frozen prediction source")
        target = [m.activation[0] for m in batch.measurements]
        mean = math.fsum(target) / len(target)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in target) / len(target))
        vocabulary = sorted({t for i in batch.inputs for t in i.token_ids})
        features = []
        for i, feature in enumerate(commit.hypothesis.predictor.features):
            if feature.kind == "token_is":
                new = TokenFeature(
                    f"random-context-{i}", feature.kind, feature.offset, rng.choice(vocabulary)
                )
            elif feature.kind == "position_is":
                new = TokenFeature(
                    f"random-position-{i}", feature.kind, 0, rng.randrange(batch.tensor_shape[1])
                )
            else:
                a, b = rng.sample((-2, -1, 0), 2)
                new = TokenFeature(f"random-relation-{i}", feature.kind, a, None, b)
            features.append(new)
        predictor = fit_ridge_predictor(
            commit.splits, batch, features=tuple(features), ridge=1e-6, enabled=True
        )
        controls.append(
            ComponentControl(
                commit.digest,
                batch.digest,
                commit.component_index,
                float(mean),
                float(std),
                predictor,
            )
        )
    first = frozen[0]
    positions = (
        tuple(p for p in first.evaluation_positions if p + 1 < batches[0].tensor_shape[1])
        if logit_positions is None
        else logit_positions
    )
    indices(positions, 128)
    if max(positions) >= batches[0].tensor_shape[1]:
        raise ValueError("declared logit position outside generation geometry")
    require_causal_prefix_isolation(
        first.splits, tuple(sorted(set(first.evaluation_positions + positions)))
    )
    width = batches[0].hook.width
    available = [i for i in range(width) if i not in {c.component_index for c in frozen}]
    random_components = (
        tuple(rng.sample(available, len(frozen))) if len(available) >= len(frozen) else ()
    )
    shift = rng.randrange(1, len(frozen)) if len(frozen) > 1 else 0
    permutation = tuple((i + shift) % len(frozen) for i in range(len(frozen)))
    probe_values = tuple(_record(p, LogitChoiceProbe) for p in array(probes, 32))
    if any(p.example_id not in {r.example_id for r in first.splits.locked} for p in probe_values):
        raise ValueError("probe names a foreign locked example")
    now = datetime.now(timezone.utc) if clock is None else clock.now()
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("control clock requires an aware datetime")
    return SimulationControls(
        first.model_digest,
        first.hook_digest,
        first.splits.digest,
        tuple(controls),
        batches,
        first.evaluation_positions,
        positions,
        permutation,
        random_components,
        probe_values,
        seed,
        now.astimezone(timezone.utc).isoformat(),
    )


def _forward(
    backend: FrozenNativeModel,
    hook: HookSpec,
    ids: Any,
    mask: Any,
    positions: tuple[int, ...],
    components: tuple[int, ...],
    replacement: Any = None,
) -> tuple[Any, Any]:
    """Trusted executor; no callbacks or code are accepted from model/scenario content."""
    import torch

    backend._assert_frozen()
    captured = []

    def transform(value: Any) -> Any:
        expected = (ids.shape[0], ids.shape[1], hook.width)
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != expected
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("actual intervention hook tensor is incompatible or nonfinite")
        captured.append(value.detach().clone())
        return (
            value
            if replacement is None
            else replace_selected_activations(
                value, replacement, token_positions=positions, component_indices=components
            )
        )

    def before(module: Any, inputs: Any) -> Any:
        return (transform(inputs[0]),) + inputs[1:]

    def after(module: Any, inputs: Any, output: Any) -> Any:
        if hook.location == "output[0]":
            if type(output) is not tuple or not output:
                raise ValueError("intervention hook requires tuple output")
            return (transform(output[0]),) + output[1:]
        return transform(output)

    module = backend._model.get_submodule(hook.module_path)
    handle = (
        module.register_forward_pre_hook(before)
        if hook.location == "input"
        else module.register_forward_hook(after)
    )
    try:
        with torch.no_grad():
            logits = backend._model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            if (
                logits.shape[:2] != ids.shape
                or logits.ndim != 3
                or not bool(torch.isfinite(logits).all())
            ):
                raise ValueError("invalid model output distribution")
            logits = logits.detach().clone()
    finally:
        handle.remove()
    backend._assert_frozen()
    if len(captured) != 1:
        raise ValueError("intervention hook did not fire exactly once")
    return logits, captured[0]


def _predict_values(
    predictors: tuple[Any, ...],
    contexts: tuple[NativeInput, ...],
    positions: tuple[int, ...],
    *,
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    # Predictor only receives public token contexts, never actual activation tensors.
    return torch.tensor(
        [
            [[p.predict(context, position) for p in predictors] for position in positions]
            for context in contexts
        ],
        dtype=dtype,
        device=device,
    )


def _calibration(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    measured = [r for r in rows if r[f"{prefix}_nll"] is not None]
    bins: list[dict[str, Any]] = []
    for i in range(10):
        values = [r for r in measured if min(9, int(r[f"{prefix}_confidence"] * 10)) == i]
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": len(values),
                "confidence": (
                    sum(r[f"{prefix}_confidence"] for r in values) / len(values) if values else None
                ),
                "accuracy": (
                    sum(r[f"{prefix}_correct"] for r in values) / len(values) if values else None
                ),
            }
        )
    return {
        "count": len(measured),
        "bins": bins,
        "ece": (
            sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins if b["count"])
            / len(measured)
            if measured
            else None
        ),
    }


def _distribution_rows(
    original: Any,
    substituted: Any,
    contexts: tuple[NativeInput, ...],
    positions: tuple[int, ...],
) -> list[dict[str, Any]]:
    kl = downstream_kl(original, substituted)
    log_p, log_q = original.double().log_softmax(-1), substituted.double().log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()
    rows = []
    for b, context in enumerate(contexts):
        for position in positions:
            row: dict[str, Any] = {
                "example_id": context.example_id,
                "position": position,
                "kl_original_to_substituted": float(kl[b, position].item()),
                "substituted_logits": substituted[b, position].cpu().tolist(),
            }
            gold = (
                context.token_ids[position + 1]
                if position + 1 < len(context.token_ids) and context.attention_mask[position + 1]
                else None
            )
            row["gold_next_token"] = gold
            for name, log_probs, probs in (("original", log_p, p), ("substituted", log_q, q)):
                vector = probs[b, position]
                prediction = int(vector.argmax().item())
                row[f"{name}_prediction"] = prediction
                row[f"{name}_confidence"] = float(vector[prediction].item())
                row[f"{name}_correct"] = None if gold is None else prediction == gold
                row[f"{name}_nll"] = (
                    None if gold is None else float(-log_probs[b, position, gold].item())
                )
                row[f"{name}_brier"] = (
                    None
                    if gold is None
                    else float((vector.square().sum() - 2 * vector[gold] + 1).item())
                )
            row["prediction_changed"] = row["original_prediction"] != row["substituted_prediction"]
            rows.append(row)
    return rows


def _probe_results(
    probes: tuple[LogitChoiceProbe, ...],
    original: Any,
    substituted: Any,
    contexts: tuple[NativeInput, ...],
) -> list[dict[str, Any]]:
    results = []
    indexed = {c.example_id: i for i, c in enumerate(contexts)}
    for probe in probes:
        b = indexed[probe.example_id]
        scores = [c.token_id for c in probe.choices]
        old = int(original[b, probe.position, scores].argmax().item())
        new = int(substituted[b, probe.position, scores].argmax().item())
        results.append(
            {
                "probe_id": probe.probe_id,
                "original_choice": probe.choices[old].label,
                "substituted_choice": probe.choices[new].label,
                "expected_label": probe.expected_label,
                "changed": old != new,
                "original_correct": (
                    None
                    if probe.expected_label is None
                    else probe.choices[old].label == probe.expected_label
                ),
                "substituted_correct": (
                    None
                    if probe.expected_label is None
                    else probe.choices[new].label == probe.expected_label
                ),
            }
        )
    return results


def simulate_activations(
    backend: FrozenNativeModel,
    hook: HookSpec,
    commits: tuple[ExplanationCommit, ...],
    locked_inputs: tuple[NativeInput, ...],
    *,
    controls: SimulationControls,
    logit_positions: tuple[int, ...],
    max_distribution_values: int = 1_000_000,
    enabled: bool = False,
) -> dict[str, Any]:
    """Substitute frozen explanation predictions and real controls on a copied model.

    Retain evaluated logits so KL can be recomputed. Large evidence requests fail before
    execution; no unrun distribution is replaced by a fabricated score. No parameter changes.
    """
    _enabled(enabled)
    import torch

    indices(logit_positions, 128)
    if type(backend) is not FrozenNativeModel:
        raise ValueError("exact frozen native backend required")
    selected = _record(hook, HookSpec)
    if selected != backend.hook(layer=selected.layer, kind=selected.kind):
        raise ValueError("simulation hook differs from verified architecture location")
    if selected.kind is HookKind.ATTENTION_HEAD:
        raise ValueError("vector-head explanation predictors are not supported")
    frozen = _commits(commits)
    control = _record(controls, SimulationControls)
    first = frozen[0]
    contexts = tuple(_record(c, NativeInput) for c in array(locked_inputs, 8, required=True))
    refs = {r.example_id: r for r in first.splits.locked}
    if len({c.example_id for c in contexts}) != len(contexts) or {
        c.example_id for c in contexts
    } != set(refs):
        raise ValueError("simulation requires exactly the frozen locked examples")
    from .explanation_tests import ExampleRef

    if any(
        ExampleRef.from_input(c, distribution=refs[c.example_id].distribution) != refs[c.example_id]
        for c in contexts
    ):
        raise ValueError("simulation locked input differs from the committed context")
    if first.model_digest != backend.identity.digest or first.hook_digest != selected.digest:
        raise ValueError("simulation model/hook differs from frozen predictor")
    if logit_positions != control.logit_positions:
        raise ValueError("logit positions differ from frozen simulation controls")
    positions = first.evaluation_positions
    components = tuple(c.component_index for c in frozen)
    if any(c.component_index >= selected.width for c in frozen):
        raise ValueError("simulation component outside hook")
    if len({len(c.token_ids) for c in contexts}) != 1 or any(
        max(positions + logit_positions) >= len(c.token_ids) for c in contexts
    ):
        raise ValueError("locked geometry incompatible with frozen positions")
    if any(not c.attention_mask[p] for c in contexts for p in positions + logit_positions):
        raise ValueError("simulation requires measured coverage at every frozen position")
    require_causal_prefix_isolation(
        first.splits, tuple(sorted(set(first.evaluation_positions + logit_positions)))
    )
    # Rebuild declarative statistics/baselines from their original generation evidence.
    rebuilt = prepare_simulation_controls(
        frozen,
        control.generation_batches,
        seed=control.seed,
        logit_positions=control.logit_positions,
        probes=control.probes,
        enabled=True,
    )
    if replace(rebuilt, created_at=control.created_at) != control:
        raise ValueError("simulation controls differ from their generation-only construction")
    vocabulary = backend._model.config.vocab_size
    if any(choice.token_id >= vocabulary for probe in control.probes for choice in probe.choices):
        raise ValueError("probe token outside model vocabulary")
    if type(max_distribution_values) is not int or not 1 <= max_distribution_values <= 2_000_000:
        raise ValueError("distribution evidence budget must be in 1..2000000")
    if (len(CONDITIONS) + 1) * len(contexts) * len(
        logit_positions
    ) * vocabulary > max_distribution_values:
        raise ValueError("distribution evidence budget exceeded before execution")
    if any(t >= vocabulary for c in contexts for t in c.token_ids):
        raise ValueError("locked token outside model vocabulary")
    backend._assert_frozen()
    # Replay source evidence instead of trusting self-labelled generation activation records.
    for source in control.generation_batches:
        replay = backend.capture(
            selected,
            torch.tensor(
                [c.token_ids for c in source.inputs], dtype=torch.long, device=backend._device
            ),
            attention_mask=torch.tensor(
                [c.attention_mask for c in source.inputs], dtype=torch.long, device=backend._device
            ),
            component_indices=source.component_indices,
            example_ids=tuple(c.example_id for c in source.inputs),
            token_positions=source.token_positions,
            enabled=True,
        )
        if replay != source:
            raise ValueError("generation source does not match independent frozen-model replay")
    ids = torch.tensor([c.token_ids for c in contexts], dtype=torch.long, device=backend._device)
    mask = torch.tensor(
        [c.attention_mask for c in contexts], dtype=torch.long, device=backend._device
    )
    if ids.shape[1] > min(128, getattr(backend._model.config, "max_position_embeddings", 128)):
        raise ValueError("locked input exceeds supported position capacity")
    original, actual = _forward(backend, selected, ids, mask, positions, components)
    base_values = actual[:, positions, :][:, :, components]
    predictions = tuple(c.hypothesis.predictor for c in frozen)
    predicted = _predict_values(
        predictions, contexts, positions, dtype=actual.dtype, device=actual.device
    )
    rng = random.Random(control.seed)
    means = torch.tensor(
        [c.mean for c in control.components], dtype=actual.dtype, device=actual.device
    )
    random_values = torch.tensor(
        [
            [[rng.gauss(c.mean, c.std) for c in control.components] for _ in positions]
            for _ in contexts
        ],
        dtype=actual.dtype,
        device=actual.device,
    )
    direction = torch.tensor(
        [[[rng.gauss(0.0, 1.0) for _ in components] for _ in positions] for _ in contexts],
        dtype=torch.float64,
        device=actual.device,
    )
    norm = direction.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, :, 0] = 1.0
    direction = torch.where((norm == 0).expand_as(direction), fallback, direction)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    magnitude = (predicted.double() - base_values.double()).norm(dim=-1, keepdim=True)
    matched_random = (base_values.double() + direction * magnitude).to(actual.dtype)
    values = {
        "explanation": predicted,
        "identity": base_values.clone(),
        "ablation": torch.zeros_like(base_values),
        "mean": means.expand_as(base_values).clone(),
        "random_predictor": random_values,
        "shuffled_labels": (
            _predict_values(
                tuple(predictions[i] for i in control.label_permutation),
                contexts,
                positions,
                dtype=actual.dtype,
                device=actual.device,
            )
            if len(frozen) > 1
            else None
        ),
        "magnitude_matched_random_direction": matched_random,
        "random_neurons": predicted if control.random_component_indices else None,
        "matched_complexity": _predict_values(
            tuple(c.matched_predictor for c in control.components),
            contexts,
            positions,
            dtype=actual.dtype,
            device=actual.device,
        ),
        "cross_context": base_values.roll(1, 0) if len(contexts) > 1 else None,
    }
    results: dict[str, Any] = {}
    for name in CONDITIONS:
        value = values[name]
        targets = control.random_component_indices if name == "random_neurons" else components
        if value is None:
            results[name] = {
                "status": "UNAVAILABLE",
                "reason": "insufficient matched control units/examples",
                "summary": None,
                "per_position": [],
            }
            continue
        changed, matched_actual = _forward(backend, selected, ids, mask, positions, targets, value)
        if not torch.equal(actual, matched_actual):
            raise ValueError("matched baseline activation changed between intervention runs")
        rows = _distribution_rows(original, changed, contexts, logit_positions)
        raw_target = actual[:, positions, :][:, :, targets]
        errors = (value.double() - raw_target.double()).abs()
        activation_rows = [
            {
                "example_id": context.example_id,
                "position": position,
                "component_index": component,
                "actual": float(raw_target[b, p, c].item()),
                "requested": (
                    float(predictions[c].predict(context, position))
                    if name == "explanation"
                    else None
                ),
                "applied": float(value[b, p, c].item()),
            }
            for b, context in enumerate(contexts)
            for p, position in enumerate(positions)
            for c, component in enumerate(targets)
        ]
        nll_rows = [r for r in rows if r["original_nll"] is not None]
        results[name] = {
            "status": "MEASURED",
            "component_indices": list(targets),
            "summary": {
                "mean_kl": sum(r["kl_original_to_substituted"] for r in rows) / len(rows),
                "mean_delta_nll": (
                    sum(r["substituted_nll"] - r["original_nll"] for r in nll_rows) / len(nll_rows)
                    if nll_rows
                    else None
                ),
                "nll_count": len(nll_rows),
                "mean_activation_mae": float(errors.mean().item()),
                "top_token_change_rate": sum(r["prediction_changed"] for r in rows) / len(rows),
            },
            "per_position": rows,
            "activation_predictions": activation_rows,
            "calibration_original": _calibration(rows, "original"),
            "calibration_substituted": _calibration(rows, "substituted"),
            "behavior_probes": _probe_results(control.probes, original, changed, contexts),
            "groups": {
                group: {
                    "count": len(group_rows),
                    "mean_kl": sum(r["kl_original_to_substituted"] for r in group_rows)
                    / len(group_rows),
                }
                for group in sorted({r.distribution for r in first.splits.locked})
                if (group_rows := [r for r in rows if refs[r["example_id"]].distribution == group])
            },
        }
    return {
        "schema_version": "activation-simulation-report-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "evidence_status": "PREDICTIVE_HYPOTHESIS",
        "measurement_authentication": "NOT_PERFORMED",
        "source_model_state_digest": backend.identity.state_digest,
        "model": backend.identity.to_dict(),
        "hook": selected.to_dict(),
        "controls": control.to_dict(),
        "commits": [c.to_dict() for c in frozen],
        "inputs": [c.to_dict() for c in contexts],
        "original_logits": [
            {
                "example_id": context.example_id,
                "position": p,
                "logits": original[b, p].cpu().tolist(),
            }
            for b, context in enumerate(contexts)
            for p in logit_positions
        ],
        "conditions": results,
        "activation_coverage": [
            {
                "component_index": component,
                "count": base_values[:, :, c].numel(),
                "positive_count": int((base_values[:, :, c] > 0).sum().item()),
                "negative_count": int((base_values[:, :, c] < 0).sum().item()),
                "zero_count": int((base_values[:, :, c] == 0).sum().item()),
                "weak_threshold": frozen[c].activation_threshold,
                "near_zero_threshold": frozen[c].activation_threshold * 0.01,
                "weak_count": int(
                    (
                        (base_values[:, :, c].abs() < frozen[c].activation_threshold)
                        & (base_values[:, :, c] != 0)
                    )
                    .sum()
                    .item()
                ),
                "near_zero_count": int(
                    (base_values[:, :, c].abs() <= frozen[c].activation_threshold * 0.01)
                    .sum()
                    .item()
                ),
                "applied_activation_mae": float(
                    (predicted[:, :, c].double() - base_values[:, :, c].double())
                    .abs()
                    .mean()
                    .item()
                ),
            }
            for c, component in enumerate(components)
        ],
        "forward_counts": {
            "generation_replay": len(frozen),
            "baseline": 1,
            "intervention": sum(r["status"] == "MEASURED" for r in results.values()),
        },
        "limitations": [
            "Low KL or constant substitution alone does not establish an explained neuron.",
            "One frozen same-hook group is tested; cross-layer paths require separate tests.",
            "Single-component random directions are degenerate one-dimensional controls.",
            "Logit-choice semantics and custody/authenticity require independent host validation.",
        ],
    }
