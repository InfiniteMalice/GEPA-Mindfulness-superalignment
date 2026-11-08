"""Experimental actual native/dictionary comparisons; predictions never see locked targets."""

# Standard library
from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Any

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import _record, array
from gepa_mindfulness.verification.runtime_governance import TrustedClock

from .activation_simulation import _calibration, _distribution_rows, _forward
from .comparative_records import (
    ArmFit,
    ComparisonCommit,
    ComparisonProtocol,
    FeatureDictionary,
)
from .explanation_tests import ActivationPredictor, ExampleRef
from .native_neurons import FrozenNativeModel
from .native_records import HookSpec, NativeBatch, NativeInput

CONDITIONS = (
    "identity",
    "reconstruction",
    "explanation",
    "ablation",
    "mean",
    "random_predictor",
    "matched_complexity",
    "random_features",
    "shuffled_labels",
)


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("interpretation comparison requires enabled=True")


def native_dictionary(
    backend: FrozenNativeModel, hook: HookSpec, *, enabled: bool = False
) -> FeatureDictionary:
    """An explicit identity coordinate map, not a learned feature decomposition."""
    _enabled(enabled)
    if type(backend) is not FrozenNativeModel:
        raise ValueError("comparison requires the supported frozen backend")
    selected = _record(hook, HookSpec)
    if selected != backend.hook(layer=selected.layer, kind=selected.kind) or selected.width > 64:
        raise ValueError("native dictionary requires its verified bounded hook")
    backend._assert_frozen()
    eye = tuple(tuple(float(i == j) for j in range(selected.width)) for i in range(selected.width))
    zero = (0.0,) * selected.width
    return FeatureDictionary(
        "native:" + selected.digest,
        "1",
        backend.identity.digest,
        selected.digest,
        eye,
        zero,
        eye,
        zero,
        "linear",
        "native",
        (
            EvidenceReference(
                "native-coordinates:" + selected.digest, EvidenceSourceKind.EXTERNAL_RECORD
            ),
        ),
    )


def encode_decode(dictionary: FeatureDictionary, activations: Any) -> tuple[Any, Any]:
    """Closed affine/ReLU encoder and affine decoder in float64; no executable model payload."""
    import torch

    d = _record(dictionary, FeatureDictionary)
    if (
        type(activations) is not torch.Tensor
        or activations.ndim != 3
        or activations.shape[-1] != d.width
        or activations.layout is not torch.strided
        or not activations.is_floating_point()
        or activations.device.type == "meta"
        or not bool(torch.isfinite(activations).all())
    ):
        raise ValueError("dictionary requires finite dense [batch,position,width] activations")

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=activations.device)

    z = activations.double() @ tensor(d.encoder) + tensor(d.encoder_bias)
    if d.nonlinearity == "relu":
        z = z.clamp_min(0.0)
    reconstruction = z @ tensor(d.decoder) + tensor(d.decoder_bias)
    if not bool(torch.isfinite(z).all() and torch.isfinite(reconstruction).all()):
        raise ValueError("dictionary computation is nonfinite")
    return z, reconstruction


def _inputs(
    values: tuple[NativeInput, ...], refs: tuple[ExampleRef, ...], name: str
) -> tuple[NativeInput, ...]:
    contexts = tuple(_record(v, NativeInput) for v in array(values, 8, required=True))
    indexed = {v.example_id: v for v in contexts}
    if len(indexed) != len(contexts) or set(indexed) != {r.example_id for r in refs}:
        raise ValueError(name + " split roster mismatch")
    for ref in refs:
        if ExampleRef.from_input(indexed[ref.example_id], distribution=ref.distribution) != ref:
            raise ValueError(name + " context differs from frozen split")
    # Canonical manifest order prevents favorable reordering of seeded controls.
    return tuple(indexed[r.example_id] for r in refs)


def _backend(
    backend: FrozenNativeModel, p: ComparisonProtocol, inputs: tuple[NativeInput, ...]
) -> None:
    if type(backend) is not FrozenNativeModel or backend.identity.digest != p.model_digest:
        raise ValueError("comparison model differs from frozen protocol")
    if backend._device.type != "cpu":
        raise ValueError("this deterministic comparison backend currently supports CPU only")
    if p.hook != backend.hook(layer=p.hook.layer, kind=p.hook.kind):
        raise ValueError("comparison hook differs from verified architecture")
    backend._assert_frozen()
    lengths = {len(i.token_ids) for i in inputs}
    if len(lengths) != 1:
        raise ValueError("each matched input batch must have identical sequence length")
    length = len(inputs[0].token_ids)
    capacity = getattr(backend._model.config, "max_position_embeddings", 128)
    if length > min(128, capacity):
        raise ValueError("comparison context exceeds architecture capacity")
    positions = p.activation_positions + p.logit_positions
    if max(positions) >= length or any(not i.attention_mask[t] for i in inputs for t in positions):
        raise ValueError("comparison position is out of range or masked")
    vocabulary = backend._model.config.vocab_size
    if max(p.metric.target_token, p.metric.reference_token) >= vocabulary or any(
        t >= vocabulary for i in inputs for t in i.token_ids
    ):
        raise ValueError("comparison token outside model vocabulary")
    if p.budget.max_forwards_per_arm < len(CONDITIONS):
        raise ValueError("forward budget cannot support the matched condition roster")
    planned = (
        len(inputs)
        * length
        * vocabulary
        * (1 + len(CONDITIONS) * sum(a.dictionary is not None for a in p.arms))
    )
    if planned > p.budget.max_distribution_values:
        raise ValueError("comparison distribution evidence budget exceeded before execution")
    _report_preflight(p, inputs, planned)


def _report_preflight(p, inputs, planned, *, source_bytes=0):
    # Conservative serialized envelope bound, not just the tensor allocation budget.
    active = tuple(a for a in p.arms if a.dictionary is not None)
    batch, positions, width = len(inputs), len(p.activation_positions), p.hook.width
    selected = sum(len(a.selected_features) for a in active)
    features = sum(a.dictionary.feature_count for a in active if a.dictionary is not None)
    native_values = (
        batch
        * positions
        * (width * (1 + len(active) * (len(CONDITIONS) + 1)) + features + 6 * selected)
    )
    rows = batch * (
        len(p.logit_positions) * (1 + len(CONDITIONS) * len(active))
        + positions * selected
        + len(active)
    )
    id_bytes = max(len(canonical_json(i.example_id).encode("utf-8")) for i in inputs)
    metadata = rows * (2048 + id_bytes)
    summaries = (1 + len(active) * (len(CONDITIONS) + batch + 2)) * 16384
    fixed = source_bytes + 3 * len(canonical_json(p.to_dict()).encode("utf-8"))
    # Finite binary64 JSON numbers need at most 32 bytes including separators.
    upper_bytes = fixed + metadata + summaries + 32 * (planned + native_values)
    if upper_bytes > 4 * 1024 * 1024:
        raise ValueError("comparison serialized report byte budget exceeded before execution")


def _capture(backend, p, inputs):
    import torch

    return backend.capture(
        p.hook,
        torch.tensor([i.token_ids for i in inputs], dtype=torch.long),
        attention_mask=torch.tensor([i.attention_mask for i in inputs], dtype=torch.long),
        component_indices=tuple(range(p.hook.width)),
        example_ids=tuple(i.example_id for i in inputs),
        token_positions=p.activation_positions,
        enabled=True,
    )


def _batch_tensor(batch: NativeBatch):
    import torch

    observed = {
        (m.example_id, m.token_position, m.component_index): m.activation[0]
        for m in batch.measurements
    }
    return torch.tensor(
        [
            [
                [observed[i.example_id, p, c] for c in batch.component_indices]
                for p in batch.token_positions
            ]
            for i in batch.inputs
        ],
        dtype=torch.float64,
    )


def _fit(features, contexts, positions, targets, ridge):
    import torch

    rows = [
        [f.value(context, position) for f in features] + [1.0]
        for context in contexts
        for position in positions
    ]
    x = torch.tensor(rows, dtype=torch.float64)
    y = targets.reshape(-1, targets.shape[-1])
    solution = torch.linalg.solve(
        x.T @ x + ridge * torch.eye(x.shape[1], dtype=torch.float64), x.T @ y
    )
    if not bool(torch.isfinite(solution).all()):
        raise ValueError("comparison ridge fit numerical breakdown")
    return tuple(
        ActivationPredictor(
            features,
            tuple(float(v) for v in solution[:-1, k]),
            float(solution[-1, k]),
            "ridge",
            tuple(i.example_id for i in contexts),
        )
        for k in range(y.shape[1])
    )


def _make_fits(
    p: ComparisonProtocol, generation: NativeBatch, vocabulary: int
) -> tuple[ArmFit, ...]:
    raw = _batch_tensor(generation)
    fits = []
    for arm in p.arms:
        if arm.dictionary is None:
            continue
        z, _ = encode_decode(arm.dictionary, raw)
        targets = z[:, :, arm.selected_features]
        rng = random.Random(p.seed)
        matched = []
        for f in arm.public_features:
            if f.kind == "token_is":
                feature = replace(f, token_id=rng.randrange(vocabulary))
            elif f.kind == "position_is":
                feature = replace(f, token_id=rng.randrange(len(generation.inputs[0].token_ids)))
            else:
                feature = replace(f, offset=-rng.randrange(3), other_offset=-rng.randrange(3))
            matched.append(feature)
        other = [v for v in range(arm.dictionary.feature_count) if v not in arm.selected_features]
        random_features = (
            tuple(rng.sample(other, len(arm.selected_features)))
            if len(other) >= len(arm.selected_features)
            else ()
        )
        size = len(arm.selected_features)
        permutation = tuple((i + 1) % size for i in range(size))
        flat = targets.reshape(-1, size)
        fits.append(
            ArmFit(
                arm.arm_id,
                _fit(
                    arm.public_features, generation.inputs, p.activation_positions, targets, p.ridge
                ),
                _fit(tuple(matched), generation.inputs, p.activation_positions, targets, p.ridge),
                tuple(float(v) for v in flat.mean(0)),
                tuple(float(v) for v in flat.std(0, unbiased=False)),
                random_features,
                permutation,
            )
        )
    return tuple(fits)


def prepare_comparison(
    backend: FrozenNativeModel,
    protocol: ComparisonProtocol,
    generation_inputs: tuple[NativeInput, ...],
    *,
    clock: TrustedClock,
    enabled: bool = False,
) -> ComparisonCommit:
    """Capture/freeze generation-only fits and controls before locked evaluation."""
    _enabled(enabled)
    p = _record(protocol, ComparisonProtocol)
    contexts = _inputs(generation_inputs, p.manifest.generation, "generation")
    _backend(backend, p, contexts)
    batch = _capture(backend, p, contexts)
    fits = _make_fits(p, batch, backend._model.config.vocab_size)
    now = clock.now()
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("comparison clock requires aware datetime")
    return ComparisonCommit(p, batch, fits, now.astimezone(timezone.utc).isoformat())


def _predict(predictors, contexts, positions):
    import torch

    return torch.tensor(
        [
            [[v.predict(context, position) for v in predictors] for position in positions]
            for context in contexts
        ],
        dtype=torch.float64,
    )


def _summary(rows):
    nll = [r["substituted_nll"] - r["original_nll"] for r in rows if r["original_nll"] is not None]
    return {
        "positions": len(rows),
        "mean_kl": sum(r["kl_original_to_substituted"] for r in rows) / len(rows),
        "nll_count": len(nll),
        "mean_nll_change": sum(nll) / len(nll) if nll else None,
        "top_token_changes": sum(r["prediction_changed"] for r in rows),
        "original_calibration": _calibration(rows, "original"),
        "substituted_calibration": _calibration(rows, "substituted"),
    }


def _evaluate_arm(backend, p, arm, fit, contexts, ids, mask, baseline, actual):
    import torch

    started = perf_counter_ns()
    positions = p.activation_positions
    native = actual[:, positions, :].double()
    z, reconstruction = encode_decode(arm.dictionary, native)
    decoder = torch.tensor(arm.dictionary.decoder, dtype=torch.float64)
    predicted = _predict(fit.predictors, contexts, positions)
    matched = _predict(fit.matched_predictors, contexts, positions)
    selected = arm.selected_features
    target = z[:, :, selected]
    mean = torch.tensor(fit.means, dtype=torch.float64).expand_as(target)
    rng = random.Random(p.seed)
    random_values = torch.tensor(
        [
            [[rng.gauss(m, s) for m, s in zip(fit.means, fit.stds)] for _ in positions]
            for _ in contexts
        ],
        dtype=torch.float64,
    )
    replacements = {
        "identity": native,
        "reconstruction": reconstruction,
        "explanation": native + (predicted - target) @ decoder[list(selected)],
        "ablation": native - target @ decoder[list(selected)],
        "mean": native + (mean - target) @ decoder[list(selected)],
        "random_predictor": native + (random_values - target) @ decoder[list(selected)],
        "matched_complexity": native + (matched - target) @ decoder[list(selected)],
    }
    unavailable = {}
    if fit.random_indices:
        replacements["random_features"] = (
            native - z[:, :, fit.random_indices] @ decoder[list(fit.random_indices)]
        )
    else:
        unavailable["random_features"] = "No disjoint feature group fits the declared budget."
    if len(selected) > 1:
        replacements["shuffled_labels"] = (
            native + (predicted[:, :, fit.label_permutation] - target) @ decoder[list(selected)]
        )
    else:
        unavailable["shuffled_labels"] = "Group-label control requires at least two features."
    conditions = {}
    metric_logits = {}
    for name in CONDITIONS:
        if name not in replacements:
            conditions[name] = {"status": "UNAVAILABLE", "reason": unavailable[name]}
            continue
        replacement = replacements[name].to(dtype=actual.dtype, device=actual.device)
        output, observed = _forward(
            backend, p.hook, ids, mask, positions, tuple(range(p.hook.width)), replacement
        )
        if not torch.equal(actual, observed):
            raise ValueError("comparison baseline activation changed between matched conditions")
        if name == "identity" and not torch.equal(output, baseline):
            raise ValueError("comparison identity failed exact distribution preservation")
        rows = _distribution_rows(baseline, output, contexts, p.logit_positions)
        conditions[name] = {
            "status": "MEASURED",
            "rows": rows,
            "summary": _summary(rows),
            "applied_native_values": replacement.cpu().tolist(),
            "mean_native_change_l2": float((replacement.double() - native).norm(dim=-1).mean()),
        }
        metric_logits[name] = (
            output[:, p.metric.position, p.metric.target_token].double()
            - output[:, p.metric.position, p.metric.reference_token].double()
        )
    original_metric = (
        baseline[:, p.metric.position, p.metric.target_token].double()
        - baseline[:, p.metric.position, p.metric.reference_token].double()
    )
    causal = []
    prediction = arm.causal_prediction
    for b, context in enumerate(contexts):
        effect = float(metric_logits["ablation"][b] - original_metric[b])
        control = (
            None
            if "random_features" not in metric_logits
            else float(metric_logits["random_features"][b] - original_metric[b])
        )
        signed = prediction.direction * effect
        outcome = "MET"
        if not prediction.minimum_effect <= signed <= prediction.maximum_effect:
            outcome = "FALSIFIED"
        elif (
            control is None
            or prediction.direction * (effect - control) < prediction.minimum_control_gap
        ):
            outcome = "NOT_SPECIFIC"
        causal.append(
            {
                "example_id": context.example_id,
                "ablation_effect": effect,
                "random_feature_effect": control,
                "prediction_result": outcome,
            }
        )
    error = (predicted - target).abs()
    error_rows = [
        {
            "example_id": c.example_id,
            "position": pos,
            "feature_index": k,
            "actual": float(target[b, j, n]),
            "predicted": float(predicted[b, j, n]),
            "absolute_error": float(error[b, j, n]),
        }
        for b, c in enumerate(contexts)
        for j, pos in enumerate(positions)
        for n, k in enumerate(selected)
    ]
    all_rows = conditions["explanation"]["rows"]
    distributions = {r.example_id: r.distribution for r in p.manifest.locked}
    groups = {}
    for distribution in sorted(set(distributions.values())):
        group_rows = [r for r in all_rows if distributions[r["example_id"]] == distribution]
        group_error = [
            r["absolute_error"]
            for r in error_rows
            if distributions[r["example_id"]] == distribution
        ]
        groups[distribution] = {
            "summary": _summary(group_rows),
            "feature_mae": sum(group_error) / len(group_error),
        }
    mse = float((native - reconstruction).square().mean())
    feature_mae = float(error.mean())
    normalized_errors = [
        float(error[:, :, i].mean()) / scale for i, scale in enumerate(fit.stds) if scale > 0.0
    ]
    normalized_mae = sum(normalized_errors) / len(normalized_errors) if normalized_errors else None
    return {
        "arm": arm.to_dict(),
        "status": "MEASURED",
        "conditions": conditions,
        "reconstruction": {
            "mse": mse,
            "mae": float((native - reconstruction).abs().mean()),
            "residual_retained_in_feature_substitutions": True,
            "decoded_native_values": reconstruction.tolist(),
        },
        "prediction": {
            "feature_mae": feature_mae,
            "generation_std_normalized_mae": normalized_mae,
            "normalized_feature_count": len(normalized_errors),
            "zero_generation_scale_features": [
                selected[i] for i, scale in enumerate(fit.stds) if scale == 0.0
            ],
            "rows": error_rows,
            "criterion": "MET" if feature_mae <= arm.max_feature_mae else "FALSIFIED",
            "actual_feature_values": z.tolist(),
        },
        "causal": {
            "evidence_status": "UNRESOLVED",
            "prediction": prediction.to_dict(),
            "rows": causal,
            "meaning": "Declared feature-group ablation contrast only.",
        },
        "coverage": {
            "negative_native_values": int((native < 0).sum()),
            "weak_native_values": int((native.abs() <= p.weak_activation_threshold).sum()),
            "near_zero_native_values": int(
                (native.abs() <= p.weak_activation_threshold * 0.01).sum()
            ),
            "negative_selected_features": int((target < 0).sum()),
            "negative_feature_pole_meaningful": arm.dictionary.nonlinearity == "linear",
            "threshold": p.weak_activation_threshold,
        },
        "by_distribution": groups,
        "complexity": {
            "native_width": arm.dictionary.width,
            "dictionary_features": arm.dictionary.feature_count,
            "dictionary_parameters": sum(map(len, arm.dictionary.encoder + arm.dictionary.decoder))
            + len(arm.dictionary.encoder_bias)
            + len(arm.dictionary.decoder_bias),
            "predictor_parameters": len(selected) * (len(arm.public_features) + 1),
            "selected_decoder_nonzeros": int((decoder[list(selected)] != 0).sum()),
            "mean_selected_nonzero_features": float((target.abs() > 1e-12).double().sum(-1).mean()),
        },
        "cost": {
            "forward_calls": len(replacements),
            "wall_time_ns": perf_counter_ns() - started,
            "timing_scope": "Locked dictionary/prediction/intervention evaluation, CPU.",
        },
        "unexplained_behavior": "Semantic behavior, other components and routes remain unverified.",
    }


def evaluate_comparison(
    backend: FrozenNativeModel,
    commit: ComparisonCommit,
    locked_inputs: tuple[NativeInput, ...],
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Replay generation then perform actual matched locked interventions; no superiority claim."""
    _enabled(enabled)
    import torch

    frozen = _record(commit, ComparisonCommit)
    p = frozen.protocol
    contexts = _inputs(locked_inputs, p.manifest.locked, "locked")
    _backend(backend, p, contexts)
    planned = (
        len(contexts)
        * len(contexts[0].token_ids)
        * backend._model.config.vocab_size
        * (1 + len(CONDITIONS) * sum(a.dictionary is not None for a in p.arms))
    )
    _report_preflight(
        p, contexts, planned, source_bytes=len(canonical_json(frozen.to_dict()).encode("utf-8"))
    )
    start = perf_counter_ns()
    replay = _capture(backend, p, frozen.generation.inputs)
    if replay != frozen.generation:
        raise ValueError("generation source differs from actual frozen-model replay")
    if _make_fits(p, replay, backend._model.config.vocab_size) != frozen.fits:
        raise ValueError("generation fits or derived controls differ from deterministic replay")
    shared_ns = perf_counter_ns() - start
    ids = torch.tensor([c.token_ids for c in contexts], dtype=torch.long)
    mask = torch.tensor([c.attention_mask for c in contexts], dtype=torch.long)
    baseline, actual = _forward(
        backend, p.hook, ids, mask, p.activation_positions, tuple(range(p.hook.width))
    )
    indexed = {f.arm_id: f for f in frozen.fits}
    arms = []
    for arm in p.arms:
        if arm.dictionary is None:
            arms.append(
                {"arm": arm.to_dict(), "status": "UNAVAILABLE", "reason": arm.unavailable_reason}
            )
        else:
            arms.append(
                _evaluate_arm(
                    backend, p, arm, indexed[arm.arm_id], contexts, ids, mask, baseline, actual
                )
            )
    backend._assert_frozen()
    result = {
        "schema_version": "interpretation-comparison-report-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "evidence_status": "PREDICTIVE_HYPOTHESIS",
        "measurement_authentication": "NOT_PERFORMED",
        "commit": frozen.to_dict(),
        "locked_inputs": [c.to_dict() for c in contexts],
        "original_logits": [
            {"example_id": c.example_id, "position": pos, "logits": baseline[b, pos].cpu().tolist()}
            for b, c in enumerate(contexts)
            for pos in p.logit_positions
        ],
        "original_native_values": actual[:, p.activation_positions, :].double().tolist(),
        "arms": arms,
        "shared_cost": {
            "generation_replay_forward_calls": 1,
            "locked_baseline_forward_calls": 1,
            "generation_replay_and_refit_ns": shared_ns,
        },
        "method_superiority": "NOT_ESTABLISHED",
        "alignment_improvement": "NOT_ESTABLISHED",
        "limitations": [
            "Fixed dictionary origin/training claims require external provenance.",
            "Context references do not certify explanatory use or causal validity.",
            "Matched feature count does not imply matched direction magnitude or semantics.",
            "Residual retention can hide computation; inspect reconstruction separately.",
            "Timings exclude original discovery/training and are not time-space theorems.",
            "Inputs, seeds and distributions are not automatically independent samples.",
        ],
    }
    validate_input_envelope(result)
    result["result_digest"] = content_digest(result)
    return result
