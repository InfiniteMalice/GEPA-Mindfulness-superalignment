"""Experimental, falsifiable context predictors with locked input/split commitments.

Closed predictors see only public tokens/positions, never target activations. These tools
compute diagnostics from supplied measurements. They do not authenticate measurements,
prove temporal custody, admit evidence or establish causal understanding.
"""

# Standard library
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_enum,
    _read_records,
    _record,
    array,
    digest,
    enum,
    text,
    texts,
)
from gepa_mindfulness.verification.runtime_governance import TrustedClock
from gepa_mindfulness.verification.state import parse_rfc3339_datetime

from .native_records import NativeBatch, NativeInput, finite, index, indices


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("explanation experiments require enabled=True")


class ExplanationKind(str, Enum):
    ARBITRARY_MIXTURE = "arbitrary_mixture"
    SHARED_FEATURE = "shared_feature"
    RELATED_CONTEXT = "related_context"
    HIERARCHICAL_GENERALIZATION = "hierarchical_generalization"
    CONDITIONAL_CONJUNCTION = "conditional_conjunction"
    MULTIPLE_MECHANISMS = "multiple_mechanisms"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExampleRef(CVARecord):
    example_id: str
    input_digest: str
    context_digest: str
    distribution: str

    schema_version = "native-example-reference-v1"

    def __post_init__(self) -> None:
        text(self.example_id, "example_id")
        digest(self.input_digest)
        digest(self.context_digest)
        text(self.distribution, "distribution")

    @classmethod
    def from_input(cls, context: NativeInput, *, distribution: str) -> ExampleRef:
        value = _record(context, NativeInput)
        fingerprint = content_digest(
            {"tokens": list(value.token_ids), "mask": list(value.attention_mask)}
        )
        return cls(value.example_id, value.digest, fingerprint, distribution)


@dataclass(frozen=True)
class SplitManifest(CVARecord):
    generation: tuple[ExampleRef, ...]
    development: tuple[ExampleRef, ...]
    locked: tuple[ExampleRef, ...]

    schema_version = "native-split-manifest-v1"
    restorers = {
        name: lambda v: _read_records(v, ExampleRef)
        for name in ("generation", "development", "locked")
    }

    def __post_init__(self) -> None:
        ids: set[str] = set()
        contexts: set[str] = set()
        for group in (self.generation, self.development, self.locked):
            for raw in array(group, 256, required=True):
                ref = _record(raw, ExampleRef)
                if ref.example_id in ids:
                    raise ValueError("example ID overlaps within or across splits")
                if ref.context_digest in contexts:
                    raise ValueError("exact token context overlaps within or across splits")
                ids.add(ref.example_id)
                contexts.add(ref.context_digest)


@dataclass(frozen=True)
class TokenFeature(CVARecord):
    name: str
    kind: str
    offset: int
    token_id: int | None = None
    other_offset: int | None = None

    schema_version = "native-token-feature-v1"

    def __post_init__(self) -> None:
        text(self.name, "feature name")
        if self.kind not in ("token_is", "tokens_equal", "position_is"):
            raise ValueError("unsupported closed feature kind")
        if type(self.offset) is not int or not -32 <= self.offset <= 32:
            raise ValueError("feature offset must be in -32..32")
        if self.kind == "tokens_equal":
            if (
                self.token_id is not None
                or type(self.other_offset) is not int
                or not -32 <= self.other_offset <= 32
            ):
                raise ValueError("token relation requires a second offset and no token ID")
        else:
            index(self.token_id)
            if self.other_offset is not None or (self.kind == "position_is" and self.offset != 0):
                raise ValueError("invalid equality/position feature parameters")

    def value(self, context: NativeInput, position: int) -> float:
        if self.kind == "position_is":
            return float(position == self.token_id)
        first = position + self.offset
        if not 0 <= first < len(context.token_ids) or not context.attention_mask[first]:
            return 0.0
        if self.kind == "token_is":
            return float(context.token_ids[first] == self.token_id)
        assert self.other_offset is not None
        second = position + self.other_offset
        return float(
            0 <= second < len(context.token_ids)
            and context.attention_mask[second] == 1
            and context.token_ids[first] == context.token_ids[second]
        )


@dataclass(frozen=True)
class ActivationPredictor(CVARecord):
    features: tuple[TokenFeature, ...]
    weights: tuple[float, ...]
    bias: float
    kind: str
    fit_example_ids: tuple[str, ...]

    schema_version = "native-context-linear-predictor-v1"
    restorers = {
        "features": lambda v: _read_records(v, TokenFeature),
        "weights": _read_array,
        "fit_example_ids": _read_array,
    }

    def __post_init__(self) -> None:
        features = tuple(_record(f, TokenFeature) for f in array(self.features, 16, required=True))
        if len({f.name for f in features}) != len(features):
            raise ValueError("duplicate feature names")
        array(self.weights, 16, required=True)
        if len(features) != len(self.weights):
            raise ValueError("feature/weight length mismatch")
        for value in self.weights + (self.bias,):
            finite(value)
            if abs(value) > 1_000_000:
                raise ValueError("predictor coefficient exceeds declared bound")
        if self.kind not in ("rule", "ridge"):
            raise ValueError("unsupported predictor")
        texts(self.fit_example_ids, required=self.kind == "ridge", maximum=256)

    def predict(self, context: NativeInput, position: int) -> float:
        public = _record(context, NativeInput)
        index(position, maximum=len(public.token_ids))
        self.__post_init__()
        return self.bias + sum(
            w * f.value(public, position) for w, f in zip(self.weights, self.features)
        )


@dataclass(frozen=True)
class ExplanationHypothesis(CVARecord):
    hypothesis_id: str
    description: str
    mechanism_kind: ExplanationKind
    competitors: tuple[str, ...]
    falsification_predictions: tuple[str, ...]
    predictor: ActivationPredictor
    generation_example_ids: tuple[str, ...]

    schema_version = "native-explanation-hypothesis-v1"
    restorers = {
        "mechanism_kind": lambda v: _read_enum(v, ExplanationKind),
        "competitors": _read_array,
        "falsification_predictions": _read_array,
        "predictor": ActivationPredictor.from_dict,
        "generation_example_ids": _read_array,
    }

    def __post_init__(self) -> None:
        text(self.hypothesis_id, "hypothesis_id")
        text(self.description, "description", 4096)
        enum(self.mechanism_kind, ExplanationKind)
        texts(self.competitors, required=True)
        texts(self.falsification_predictions, required=True)
        _record(self.predictor, ActivationPredictor)
        texts(self.generation_example_ids, required=True, maximum=256)


@dataclass(frozen=True)
class ExplanationCommit(CVARecord):
    hypothesis: ExplanationHypothesis
    splits: SplitManifest
    generation_batch_digest: str
    model_digest: str
    hook_digest: str
    component_index: int
    evaluation_positions: tuple[int, ...]
    created_at: str
    max_mae: float
    activation_threshold: float

    schema_version = "native-explanation-commit-v1"
    restorers = {
        "hypothesis": ExplanationHypothesis.from_dict,
        "splits": SplitManifest.from_dict,
        "evaluation_positions": _read_array,
    }

    def __post_init__(self) -> None:
        hypothesis = _record(self.hypothesis, ExplanationHypothesis)
        splits = _record(self.splits, SplitManifest)
        for value in (self.generation_batch_digest, self.model_digest, self.hook_digest):
            digest(value)
        index(self.component_index)
        for position in indices(self.evaluation_positions, 128):
            index(position, maximum=128)
        parse_rfc3339_datetime(self.created_at, "created_at")
        finite(self.max_mae)
        finite(self.activation_threshold)
        if self.max_mae < 0 or self.activation_threshold <= 0:
            raise ValueError("invalid preregistered metric thresholds")
        ids = {r.example_id for r in splits.generation}
        if (
            set(hypothesis.generation_example_ids) != ids
            or not set(hypothesis.predictor.fit_example_ids) <= ids
        ):
            raise ValueError("hypothesis/predictor fitting must use declared generation data only")


def _batch_for_split(batch: NativeBatch, refs: tuple[ExampleRef, ...], name: str) -> NativeBatch:
    value = _record(batch, NativeBatch)
    actual = {i.example_id: i for i in value.inputs}
    if set(actual) != {r.example_id for r in refs}:
        raise ValueError(f"batch must contain exactly the {name} split")
    for ref in refs:
        if ExampleRef.from_input(actual[ref.example_id], distribution=ref.distribution) != ref:
            raise ValueError(f"batch differs from committed {name} context")
    if len(value.component_indices) != 1 or any(len(m.activation) != 1 for m in value.measurements):
        raise ValueError("explanation prediction currently requires one scalar component")
    if {m.example_id for m in value.measurements} != set(actual):
        raise ValueError(f"{name} split lacks per-example measurement coverage")
    return value


def fit_ridge_predictor(
    manifest: SplitManifest,
    generation: NativeBatch,
    *,
    features: tuple[TokenFeature, ...],
    ridge: float,
    enabled: bool = False,
) -> ActivationPredictor:
    """Bounded deterministic ridge fit, using generation observations only; no dependency solver."""
    _enabled(enabled)
    splits = _record(manifest, SplitManifest)
    batch = _batch_for_split(generation, splits.generation, "generation")
    features = tuple(_record(f, TokenFeature) for f in array(features, 16, required=True))
    finite(ridge)
    if not 1e-6 <= ridge <= 1e6:
        raise ValueError("ridge must lie in [1e-6,1e6]")
    contexts = {i.example_id: i for i in batch.inputs}
    rows = [
        [f.value(contexts[m.example_id], m.token_position) for f in features] + [1.0]
        for m in batch.measurements
    ]
    target = [m.activation[0] for m in batch.measurements]
    n = len(features) + 1
    normal = [
        [sum(row[i] * row[j] for row in rows) + (ridge if i == j else 0.0) for j in range(n)]
        + [sum(row[i] * y for row, y in zip(rows, target))]
        for i in range(n)
    ]
    # Partial-pivot elimination, bounded to 17 unknowns; reject numerical breakdown.
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(normal[row][column]))
        normal[column], normal[pivot] = normal[pivot], normal[column]
        divisor = normal[column][column]
        if abs(divisor) < 1e-12:
            raise ValueError("ridge fit numerical breakdown")
        normal[column] = [v / divisor for v in normal[column]]
        for row in range(n):
            if row == column:
                continue
            factor = normal[row][column]
            normal[row] = [a - factor * b for a, b in zip(normal[row], normal[column])]
    solution = tuple(row[-1] for row in normal)
    return ActivationPredictor(
        features, solution[:-1], solution[-1], "ridge", tuple(i.example_id for i in batch.inputs)
    )


def freeze_explanation(
    hypothesis: ExplanationHypothesis,
    manifest: SplitManifest,
    generation: NativeBatch,
    *,
    clock: TrustedClock,
    max_mae: float,
    activation_threshold: float,
    enabled: bool = False,
) -> ExplanationCommit:
    """Digest a prediction protocol; host must enforce temporal custody and locked-data secrecy."""
    _enabled(enabled)
    h = _record(hypothesis, ExplanationHypothesis)
    splits = _record(manifest, SplitManifest)
    batch = _batch_for_split(generation, splits.generation, "generation")
    now = clock.now()
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("host clock must return an aware datetime")
    return ExplanationCommit(
        h,
        splits,
        batch.digest,
        batch.model.digest,
        batch.hook.digest,
        batch.component_indices[0],
        batch.token_positions,
        now.astimezone(timezone.utc).isoformat(),
        max_mae,
        activation_threshold,
    )


def generate_context_hypotheses(
    generation: NativeBatch,
    manifest: SplitManifest,
    *,
    max_hypotheses: int = 4,
    enabled: bool = False,
) -> tuple[ExplanationHypothesis, ...]:
    """Generate bounded lexical/offset hypotheses and an explicit unknown competitor.

    All generation observations contribute, including weak/zero/negative activations.
    Token associations are not noun detectors or established semantic mechanisms.
    Richer relational/generalization hypotheses may be supplied through the same typed API.
    """
    _enabled(enabled)
    splits = _record(manifest, SplitManifest)
    batch = _batch_for_split(generation, splits.generation, "generation")
    if type(max_hypotheses) is not int or not 2 <= max_hypotheses <= 8:
        raise ValueError("max_hypotheses must be in 2..8")
    contexts = {i.example_id: i for i in batch.inputs}
    ids = tuple(contexts)
    candidates = []
    for offset in (-1, 0, 1):
        counts: Counter[int] = Counter()
        for m in batch.measurements:
            context = contexts[m.example_id]
            position = m.token_position + offset
            if 0 <= position < len(context.token_ids) and context.attention_mask[position]:
                counts[context.token_ids[position]] += 1
        if not counts:
            continue
        token = min(counts, key=lambda t: (-counts[t], t))
        feature = TokenFeature(f"token_{token}_offset_{offset}", "token_is", offset, token)
        predictor = fit_ridge_predictor(
            splits, batch, features=(feature,), ridge=1e-6, enabled=True
        )
        candidates.append(
            ExplanationHypothesis(
                f"context-{offset}-{token}",
                f"Token {token} at relative offset {offset} is a candidate context association. "
                "Its semantic or causal meaning is unknown.",
                ExplanationKind.RELATED_CONTEXT,
                ("arbitrary unrelated mixture", "multiple mechanisms"),
                (
                    f"Changing the token at offset {offset} should change predicted activation by "
                    f"{predictor.weights[0]} when other context features are fixed.",
                ),
                predictor,
                ids,
            )
        )
        if len(candidates) == max_hypotheses - 1:
            break
    mean = sum(m.activation[0] for m in batch.measurements) / len(batch.measurements)
    candidates.append(
        ExplanationHypothesis(
            "unknown-mechanism",
            "No shared semantic mechanism established; constant baseline hypothesis.",
            ExplanationKind.UNKNOWN,
            ("shared contextual feature", "multiple distinct mechanisms"),
            ("The constant predictor fails if held-out changes exceed the frozen tolerance.",),
            ActivationPredictor(
                (TokenFeature("unused", "position_is", 0, 128),), (0.0,), float(mean), "rule", ids
            ),
            ids,
        )
    )
    return tuple(candidates)


def generate_near_misses(
    hypothesis: ExplanationHypothesis,
    generation: NativeBatch,
    manifest: SplitManifest,
    *,
    replacement_tokens: tuple[int, ...],
    enabled: bool = False,
) -> dict[str, Any]:
    """Propose up to 32 new context perturbations; no activation/behavior is asserted.

    Only tokens already in the generation vocabulary are used. Host researchers must
    reserve new examples, freeze predictions, and measure them on the declared model.
    """
    _enabled(enabled)
    h = _record(hypothesis, ExplanationHypothesis)
    splits = _record(manifest, SplitManifest)
    batch = _batch_for_split(generation, splits.generation, "generation")
    if set(h.generation_example_ids) != {r.example_id for r in splits.generation}:
        raise ValueError("hypothesis uses foreign generation examples")
    contexts = {i.example_id: i for i in batch.inputs}
    vocabulary = {t for i in batch.inputs for t in i.token_ids}
    array(replacement_tokens, 16, required=True)
    for token in replacement_tokens:
        index(token)
    if (
        len(set(replacement_tokens)) != len(replacement_tokens)
        or not set(replacement_tokens) <= vocabulary
    ):
        raise ValueError("replacement tokens require unique generation-vocabulary members")
    seen = {r.context_digest for r in splits.generation + splits.development + splits.locked}
    candidates = []
    for measurement in batch.measurements:
        context = contexts[measurement.example_id]
        position = measurement.token_position
        predicted = h.predictor.predict(context, position)
        offsets = {f.offset for f in h.predictor.features if f.kind != "position_is"}
        offsets.update(f.other_offset for f in h.predictor.features if f.other_offset is not None)
        for offset in sorted(offsets):
            changed_position = position + offset
            if (
                not 0 <= changed_position < len(context.token_ids)
                or not context.attention_mask[changed_position]
            ):
                continue
            for token in replacement_tokens:
                tokens = list(context.token_ids)
                tokens[changed_position] = token
                fingerprint = content_digest(
                    {"tokens": tokens, "mask": list(context.attention_mask)}
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                candidate = NativeInput(
                    "near-miss:" + fingerprint, tuple(tokens), context.attention_mask
                )
                after = h.predictor.predict(candidate, position)
                candidates.append(
                    {
                        "input": candidate.to_dict(),
                        "source_input_digest": context.digest,
                        "token_position": position,
                        "changed_position": changed_position,
                        "predicted_activation": after,
                        "predicted_change": after - predicted,
                        "measurement_status": "NOT_MEASURED",
                    }
                )
                if len(candidates) == 32:
                    break
            if len(candidates) == 32:
                break
        if len(candidates) == 32:
            break
    return {
        "schema_version": "native-near-miss-proposals-v1",
        "maturity": "experimental",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "hypothesis_digest": h.digest,
        "split_digest": splits.digest,
        "generation_batch_digest": batch.digest,
        "candidates": candidates,
        "measurement_status": "NOT_MEASURED",
    }


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mae": None,
            "mse": None,
            "false_positives": None,
            "false_negatives": None,
            "sign_errors": None,
        }
    errors = [r["predicted"] - r["actual"] for r in rows]
    squares = [v * v for v in errors]
    if any(not math.isfinite(v) for v in errors + squares):
        raise ValueError("numeric metric overflow; choose a justified evaluation scale")
    return {
        "count": len(rows),
        "mae": sum(abs(v) / len(errors) for v in errors),
        "mse": sum(v / len(errors) for v in squares),
        "false_positives": sum(r["predicted"] >= threshold > r["actual"] for r in rows),
        "false_negatives": sum(r["actual"] >= threshold > r["predicted"] for r in rows),
        "positive_opportunities": sum(r["actual"] >= threshold for r in rows),
        "negative_opportunities": sum(r["actual"] < threshold for r in rows),
        "sign_errors": sum(
            r["predicted"] == 0 or (r["actual"] > 0) != (r["predicted"] > 0)
            for r in rows
            if r["actual"] != 0
        ),
        "negative_pole_count": sum(r["actual"] < 0 for r in rows),
        "weak_activation_count": sum(0 < abs(r["actual"]) < threshold for r in rows),
    }


def evaluate_explanation(
    commit: ExplanationCommit,
    locked: NativeBatch,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Compute actual held-out prediction errors; no causal status or authenticity is granted."""
    _enabled(enabled)
    frozen = _record(commit, ExplanationCommit)
    batch = _batch_for_split(locked, frozen.splits.locked, "locked")
    if batch.token_positions != frozen.evaluation_positions:
        raise ValueError("locked token positions differ from the frozen evaluation coverage")
    if (
        batch.model.digest != frozen.model_digest
        or batch.hook.digest != frozen.hook_digest
        or batch.component_indices != (frozen.component_index,)
    ):
        raise ValueError("locked model/hook/component differs from frozen prediction protocol")
    contexts = {i.example_id: i for i in batch.inputs}
    distributions = {r.example_id: r.distribution for r in frozen.splits.locked}
    rows = []
    for measurement in batch.measurements:
        # Only the closed public context crosses the prediction boundary.
        predicted = frozen.hypothesis.predictor.predict(
            contexts[measurement.example_id], measurement.token_position
        )
        rows.append(
            {
                "example_id": measurement.example_id,
                "token_position": measurement.token_position,
                "input_digest": measurement.input_digest,
                "measurement_digest": measurement.digest,
                "distribution": distributions[measurement.example_id],
                "predicted": predicted,
                "actual": measurement.activation[0],
            }
        )
    metrics = _metrics(rows, frozen.activation_threshold)
    groups = {
        name: _metrics([r for r in rows if r["distribution"] == name], frozen.activation_threshold)
        for name in sorted(set(distributions.values()))
    }
    id_metrics = _metrics(
        [r for r in rows if r["distribution"] != "ood"], frozen.activation_threshold
    )
    ood = groups.get("ood", {}).get("mae")
    return {
        "schema_version": "native-explanation-evaluation-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "evidence_status": "PREDICTIVE_HYPOTHESIS",
        "measurement_authentication": "NOT_PERFORMED",
        "commit_digest": frozen.digest,
        "commit": frozen.to_dict(),
        "source_batch_digest": batch.digest,
        "source_measurements": batch.to_dict(),
        "metrics": metrics,
        "groups": groups,
        "rows": rows,
        "prediction_result": (
            "PREREGISTERED_CRITERION_MET"
            if metrics["mae"] <= frozen.max_mae
            else "PREDICTIONS_FALSIFIED"
        ),
        "ood_mae_degradation": (
            ood - id_metrics["mae"] if ood is not None and id_metrics["mae"] is not None else None
        ),
        "downstream_simulation": "NOT_RUN",
        "causal_interventions": "NOT_RUN",
        "limitations": [
            "A frozen record does not prove temporal custody or measurement authenticity.",
            "The host must isolate locked data and independently register any accepted evidence.",
            "Prediction accuracy alone does not establish a clean neuron or causal explanation.",
        ],
    }
