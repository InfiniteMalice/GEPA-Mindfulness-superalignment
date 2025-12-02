"""White-box linear probe utilities for deception detection."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from ..train.grn import GRNSettings, build_grn
from ..utils.imports import optional_import

logger = logging.getLogger(__name__)

np = optional_import("numpy")
torch = optional_import("torch")

np_module: Any | None = cast(Any, np) if np is not None else None
torch_module: Any | None = cast(Any, torch) if torch is not None else None


@dataclass
class ProbeWeights:
    """Container describing linear probe parameters."""

    weights: List[float]
    bias: float = 0.0
    metadata: Dict[str, Any] | None = None

    @property
    def dimension(self) -> int:
        return len(self.weights)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(vec: Sequence[float], weights: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(vec, weights))


def _ensure_float_list(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values]


def _normalise_tokens(
    tokens: Sequence[Sequence[float]],
    grn_settings: GRNSettings,
    module: Any | None = None,
) -> List[List[float]]:
    """Apply GRN normalisation to token vectors when available."""
    if not tokens:
        return []
    if not grn_settings.enabled:
        return [list(token) for token in tokens]
    if torch_module is None:
        logger.warning("GRN requested for probes but torch is unavailable; skipping")
        return [list(token) for token in tokens]
    if module is None:
        logger.debug("GRN module unavailable; returning tokens unchanged")
        return [list(token) for token in tokens]
    with torch_module.no_grad():
        # Shape: [num_tokens, hidden_dim] expected by GlobalResponseNorm
        param = next(module.parameters(), None)
        buffer = next(module.buffers(), None)
        target_dtype = (
            param.dtype
            if param is not None
            else buffer.dtype if buffer is not None else torch_module.float32
        )
        target_device = (
            param.device if param is not None else buffer.device if buffer is not None else None
        )
        tensor = torch_module.tensor(tokens, dtype=target_dtype, device=target_device)
        normalised = module(tensor)
    return [_ensure_float_list(row.tolist()) for row in normalised.detach().cpu().unbind(dim=0)]


def extract_hidden_states(
    model: Any,
    inputs: Mapping[str, Any],
    layers: Sequence[int] | None = None,
    pool: str = "mean",
) -> Optional[Dict[str, Any]]:
    """Extract hidden activations from a model, returning JSON-friendly tensors.

    The function attempts several strategies in a best-effort manner, falling back to
    synthetic placeholders when activations are unavailable. When a closed API is
    detected, ``None`` is returned so downstream callers can gracefully degrade.
    """

    if inputs is None:
        logger.warning("No inputs provided for hidden state extraction")
        return None

    if isinstance(inputs, Mapping) and "activations" in inputs:
        cached = inputs["activations"]
        if isinstance(cached, Mapping):
            return {
                "layers": {
                    str(layer): {
                        "tokens": [
                            _ensure_float_list(token) for token in cached_layer.get("tokens", [])
                        ],
                        "token_to_step": list(cached_layer.get("token_to_step", [])),
                    }
                    for layer, cached_layer in cached.items()
                    if isinstance(cached_layer, Mapping)
                },
                "pool": pool,
            }

    if model is None:
        logger.info("Model handle unavailable; returning None for hidden states")
        return None

    layer_list = list(layers or [])

    if torch_module is not None:
        try:
            model_device = getattr(model, "device", "cpu")
            call_kwargs: Dict[str, Any] = {}
            if hasattr(model, "eval"):
                model.eval()
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:  # pragma: no cover - optional capability
                    logger.debug("Unable to move model to CPU", exc_info=True)
            if hasattr(model, "__call__"):
                call_kwargs["output_hidden_states"] = True
                with torch_module.no_grad():
                    outputs = model(**inputs, **call_kwargs)
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    logger.warning("Model response missing hidden states; returning None")
                    if hasattr(model, "to"):
                        try:
                            model.to(model_device)
                        except Exception:  # pragma: no cover
                            logger.debug("Unable to restore model device", exc_info=True)
                    return None
                processed: Dict[str, Any] = {}
                selected_layers: Iterable[Tuple[int, Any]]
                if layer_list:
                    selected_layers = [
                        (idx, hidden_states[idx]) for idx in layer_list if idx < len(hidden_states)
                    ]
                else:
                    selected_layers = enumerate(hidden_states)
                for idx, tensor in selected_layers:
                    if tensor is None:
                        continue
                    cpu_tensor = tensor.detach().to("cpu")
                    arr = (
                        cpu_tensor.numpy().tolist()
                        if hasattr(cpu_tensor, "numpy")
                        else cpu_tensor.tolist()
                    )
                    flat_tokens: List[List[float]] = []
                    if isinstance(arr, list) and arr and isinstance(arr[0], list):
                        # Hugging Face models often return [batch, tokens, hidden]
                        batch_acts = arr[0] if isinstance(arr[0], list) else arr
                        for token in batch_acts:
                            if isinstance(token, list):
                                flat_tokens.append(_ensure_float_list(token))
                    processed[str(idx)] = {
                        "tokens": flat_tokens,
                        "token_to_step": list(range(len(flat_tokens))),
                    }
                if hasattr(model, "to"):
                    try:
                        model.to(model_device)
                    except Exception:  # pragma: no cover
                        logger.debug("Unable to restore model device", exc_info=True)
                return {"layers": processed, "pool": pool}
        except Exception as err:  # pragma: no cover - exercised in live setups
            logger.warning("Torch-based extraction failed: %s", err)
            return None

    custom_extractor = getattr(model, "get_hidden_states", None)
    if callable(custom_extractor):
        try:
            extracted = custom_extractor(inputs=inputs, layers=layer_list, pool=pool)
        except Exception as err:  # pragma: no cover - best effort hook
            logger.warning("Custom extractor errored: %s", err)
            return None
        if isinstance(extracted, Mapping):
            return {
                "layers": {
                    str(key): {
                        "tokens": [_ensure_float_list(vec) for vec in value.get("tokens", [])],
                        "token_to_step": list(value.get("token_to_step", [])),
                    }
                    for key, value in extracted.get("layers", {}).items()
                    if isinstance(value, Mapping)
                },
                "pool": extracted.get("pool", pool),
            }

    logger.info("Unable to obtain hidden activations; returning None")
    return None


def _load_json_weights(path: Path) -> Optional[ProbeWeights]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as err:
        logger.error("Unable to read probe weights at %s: %s", path, err)
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Probe weights at %s not JSON encoded", path)
        return None
    weights = payload.get("weights")
    if not isinstance(weights, Iterable):
        logger.error("Malformed probe weights in %s", path)
        return None
    bias = float(payload.get("bias", 0.0))
    metadata = None
    if isinstance(payload.get("metadata"), Mapping):
        metadata = dict(payload.get("metadata") or {})
    return ProbeWeights(
        weights=_ensure_float_list(weights),
        bias=bias,
        metadata=metadata or {},
    )


def load_probe(weights_path: str | Path) -> Optional[ProbeWeights]:
    """Load a linear probe from disk in JSON, NumPy, or Torch format."""

    path = Path(weights_path)
    if not path.exists():
        logger.warning("Probe weights missing at %s", path)
        return None

    loader_attempts = [_load_json_weights]

    if np_module is not None:

        def _load_numpy(path: Path) -> Optional[ProbeWeights]:
            try:
                blob = np_module.load(path, allow_pickle=True)
            except Exception:  # pragma: no cover - optional dependency
                logger.debug("Unable to load numpy weights from %s", path, exc_info=True)
                return None
            arr = blob.item() if hasattr(blob, "item") else blob
            if isinstance(arr, Mapping) and "weights" in arr:
                bias_val = float(arr.get("bias", 0.0))
                metadata_val = {}
                if isinstance(arr.get("metadata"), Mapping):
                    metadata_val = dict(arr.get("metadata", {}))
                return ProbeWeights(
                    weights=_ensure_float_list(arr["weights"]),
                    bias=bias_val,
                    metadata=metadata_val,
                )
            if np_module is not None and hasattr(arr, "tolist"):
                weights_list = arr.tolist()
                return ProbeWeights(weights=_ensure_float_list(weights_list))
            return None

        loader_attempts.append(_load_numpy)

    if torch_module is not None:

        def _load_torch(path: Path) -> Optional[ProbeWeights]:
            try:
                blob = torch_module.load(path, map_location="cpu")
            except Exception:  # pragma: no cover - optional dependency
                logger.debug("Unable to load torch weights from %s", path, exc_info=True)
                return None
            if isinstance(blob, Mapping) and "weights" in blob:
                metadata_val = {}
                if isinstance(blob.get("metadata"), Mapping):
                    metadata_val = dict(blob.get("metadata", {}))
                return ProbeWeights(
                    weights=_ensure_float_list(blob["weights"]),
                    bias=float(blob.get("bias", 0.0)),
                    metadata=metadata_val,
                )
            if hasattr(blob, "numpy"):
                weights_list = blob.numpy().tolist()
                return ProbeWeights(weights=_ensure_float_list(weights_list))
            return None

        loader_attempts.append(_load_torch)

    for loader in loader_attempts:
        probe = loader(path)
        if probe is not None:
            logger.info("Loaded probe weights via %s from %s", loader.__name__, path)
            return probe

    logger.error("Failed to load probe weights at %s", path)
    return None


def _pool_tokens(tokens: Sequence[Sequence[float]], mode: str) -> Optional[List[float]]:
    if not tokens:
        return None
    if mode == "cls":
        return list(tokens[0])
    if mode == "last":
        return list(tokens[-1])
    if mode != "mean":
        logger.warning("Unknown pooling mode %s; defaulting to mean", mode)
    length = len(tokens)
    dim = len(tokens[0])
    acc = [0.0] * dim
    for token in tokens:
        for idx, value in enumerate(token):
            acc[idx] += float(value)
    return [value / length for value in acc]


def _assign_steps(
    scores: List[Tuple[int, float]],
    token_to_step: Sequence[int] | None,
) -> List[Dict[str, Any]]:
    if not scores:
        return []
    mapping = token_to_step or []
    grouped: MutableMapping[int, List[float]] = {}
    for token_idx, score in scores:
        step_idx = mapping[token_idx] if token_idx < len(mapping) else token_idx
        grouped.setdefault(int(step_idx), []).append(float(score))
    aggregated: List[Dict[str, Any]] = []
    for step_idx in sorted(grouped):
        step_scores = grouped[step_idx]
        aggregated.append(
            {
                "step": step_idx,
                "score": sum(step_scores) / max(len(step_scores), 1),
            }
        )
    return aggregated


def _coerce_labels(labels: Optional[Sequence[Any]]) -> Optional[List[int]]:
    if labels is None:
        return None
    coerced: List[int] = []
    for value in labels:
        try:
            coerced.append(int(value))
        except (TypeError, ValueError):
            coerced.append(0)
    positives = sum(1 for value in coerced if value == 1)
    negatives = sum(1 for value in coerced if value == 0)
    if positives == 0 or negatives == 0:
        return None
    return coerced


def _sorted_scores(
    scores: Sequence[float],
    labels: Sequence[int],
) -> List[Tuple[float, int]]:
    paired = list(zip(scores, labels))
    paired.sort(key=lambda item: (-item[0], item[1]))
    return paired


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if not scores:
        return None
    order = _sorted_scores(scores, labels)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    tp = 0.0
    fp = 0.0
    prev_score = None
    prev_tp = 0.0
    prev_fp = 0.0
    area = 0.0
    for score, label in order:
        if prev_score is not None and score != prev_score:
            area += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
            prev_score = score
        elif prev_score is None:
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    area += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return area / (pos * neg)


def auprc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if not scores:
        return None
    order = _sorted_scores(scores, labels)
    tp = 0.0
    fp = 0.0
    prev_score = None
    area = 0.0
    last_recall = 0.0
    pos = sum(labels)
    if pos == 0:
        return None
    for score, label in order:
        if prev_score is None:
            prev_score = score
        if score != prev_score:
            recall = tp / pos
            precision = tp / max(tp + fp, 1e-9)
            area += precision * (recall - last_recall)
            last_recall = recall
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    recall = tp / pos
    precision = tp / max(tp + fp, 1e-9)
    area += precision * (recall - last_recall)
    return area


def fpr_at_tpr(
    scores: Sequence[float],
    labels: Sequence[int],
    target_tpr: float = 0.8,
) -> Optional[float]:
    if not scores:
        return None
    order = _sorted_scores(scores, labels)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    tp = 0
    fp = 0
    for score, label in order:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        if tpr >= target_tpr:
            return fp / neg
    return fp / neg if neg else None


def threshold_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    fixed: float = 0.01,
) -> Optional[float]:
    if not scores:
        return None
    order = _sorted_scores(scores, labels)
    pos = sum(labels)
    neg = len(labels) - pos
    if neg == 0:
        return None
    fp = 0
    for score, label in order:
        if label == 1:
            continue
        fp += 1
        fpr = fp / neg
        if fpr >= fixed:
            return score
    return order[-1][0] if order else None


def infer_probe(
    activations: Optional[Dict[str, Any]],
    probe: Optional[ProbeWeights],
    pooling: str = "mean",
    threshold_config: Optional[Mapping[str, Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    grn_config: GRNSettings | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run probe inference on activations and compute metrics."""

    if probe is None:
        return {
            "status": "unavailable",
            "reason": "probe-missing",
            "scores": {"per_token": [], "per_step": []},
        }

    if activations is None or not activations.get("layers"):
        return {
            "status": "unavailable",
            "reason": "activations-missing",
            "scores": {"per_token": [], "per_step": []},
            "probe": {"dimension": probe.dimension},
        }

    labels_vec = _coerce_labels(labels)
    grn_settings = (
        grn_config if isinstance(grn_config, GRNSettings) else GRNSettings.from_mapping(grn_config)
    )
    grn_module = build_grn(grn_settings)
    layers = activations.get("layers", {})
    token_scores: List[Tuple[int, float]] = []
    per_token_payload: List[Dict[str, Any]] = []
    token_map: List[int] = []

    for layer_name, payload in layers.items():
        if not isinstance(payload, Mapping):
            continue
        tokens = _normalise_tokens(payload.get("tokens", []), grn_settings, grn_module)
        token_to_step = payload.get("token_to_step")
        for token_idx, token in enumerate(tokens):
            if len(token) != probe.dimension:
                continue
            logit = _dot(token, probe.weights) + probe.bias
            score = _sigmoid(logit)
            token_scores.append((token_idx, score))
            per_token_payload.append(
                {
                    "layer": layer_name,
                    "index": token_idx,
                    "score": score,
                }
            )
        pooled = _pool_tokens(tokens, pooling)
        if pooled is None:
            continue
        logit = _dot(pooled, probe.weights) + probe.bias
        pooled_score = _sigmoid(logit)
        per_token_payload.append(
            {
                "layer": layer_name,
                "index": "pooled",
                "score": pooled_score,
            }
        )
        if token_scores and not token_map:
            token_map = list(token_to_step or [])

    per_step_payload = _assign_steps(token_scores, token_map)

    score_values = [item[1] for item in token_scores] or [
        entry["score"] for entry in per_step_payload
    ]
    metric_scores = [entry["score"] for entry in per_step_payload]
    metric_labels = labels_vec if labels_vec is not None else None

    computed_metrics: Dict[str, Optional[float]] = {}
    threshold_value: Optional[float] = None
    decision_threshold_source = None
    if metric_labels is not None and metric_scores:
        computed_metrics["auroc"] = auroc(metric_scores, metric_labels)
        computed_metrics["auprc"] = auprc(metric_scores, metric_labels)
        computed_metrics["fpr_at_tpr80"] = fpr_at_tpr(metric_scores, metric_labels, target_tpr=0.8)
        if threshold_config and threshold_config.get("type") == "fixed_fpr":
            threshold_value = threshold_at_fpr(
                metric_scores,
                metric_labels,
                fixed=float(threshold_config.get("fpr", 0.01)),
            )
            decision_threshold_source = "fixed_fpr"
    if threshold_value is None and probe.metadata:
        threshold_candidate = probe.metadata.get("threshold")
        if threshold_candidate is not None:
            try:
                threshold_value = float(threshold_candidate)
                decision_threshold_source = "metadata"
            except (TypeError, ValueError):
                logger.debug("Invalid threshold metadata: %s", threshold_candidate)
    if threshold_value is None and score_values:
        threshold_value = sum(score_values) / len(score_values)
        decision_threshold_source = "mean-score"

    flagged_steps: List[int] = []
    if threshold_value is not None:
        for entry in per_step_payload:
            entry["decision"] = bool(entry["score"] >= threshold_value)
            if entry["decision"]:
                flagged_steps.append(int(entry["step"]))
        for entry in per_token_payload:
            if isinstance(entry.get("index"), int):
                entry["decision"] = bool(entry["score"] >= threshold_value)
    else:
        for entry in per_step_payload:
            entry["decision"] = False
        for entry in per_token_payload:
            entry["decision"] = False

    return {
        "status": "ok",
        "probe": {
            "dimension": probe.dimension,
            "metadata": probe.metadata or {},
        },
        "scores": {
            "per_token": per_token_payload,
            "per_step": per_step_payload,
            "threshold": threshold_value,
            "threshold_source": decision_threshold_source,
            "max_score": max((entry["score"] for entry in per_step_payload), default=None),
        },
        "metrics": {key: value for key, value in computed_metrics.items() if value is not None},
        "summary": {
            "flagged_steps": flagged_steps,
            "total_steps": len(per_step_payload),
            "pooling": pooling,
        },
    }


__all__ = [
    "auprc",
    "auroc",
    "extract_hidden_states",
    "fpr_at_tpr",
    "infer_probe",
    "load_probe",
    "ProbeWeights",
    "threshold_at_fpr",
]
