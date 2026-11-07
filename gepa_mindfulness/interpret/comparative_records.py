"""Experimental fixed-dictionary comparison contracts; no record grants scientific authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass

# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_records,
    _record,
    array,
    digest,
    references,
    text,
)
from gepa_mindfulness.verification.diagnostic_records import restore_refs
from gepa_mindfulness.verification.state import parse_rfc3339_datetime

from .explanation_tests import ActivationPredictor, ExampleRef, SplitManifest, TokenFeature
from .native_causal_records import EffectPrediction, LogitContrast
from .native_records import HookKind, HookSpec, NativeBatch, finite, index, indices

METHODS = ("native", "native_context", "sparse_autoencoder", "other_decomposition")


def _floats(values: tuple[float, ...], maximum: int = 64) -> None:
    for value in array(values, maximum, required=True):
        finite(value)
        if abs(value) > 1e6:
            raise ValueError("dictionary coefficient exceeds declared bound")


def _matrix(values: tuple[tuple[float, ...], ...]) -> None:
    for row in array(values, 64, required=True):
        _floats(row)
    if len({len(row) for row in values}) != 1:
        raise ValueError("matrix rows have different widths")


def _read_matrix(value: object) -> tuple:
    return tuple(_read_array(row) for row in _read_array(value))


@dataclass(frozen=True)
class FeatureDictionary(CVARecord):
    dictionary_id: str
    version: str
    model_digest: str
    hook_digest: str
    encoder: tuple[tuple[float, ...], ...]
    encoder_bias: tuple[float, ...]
    decoder: tuple[tuple[float, ...], ...]
    decoder_bias: tuple[float, ...]
    nonlinearity: str
    origin: str
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "interpretation-feature-dictionary-v1"
    restorers = {
        "encoder": _read_matrix,
        "decoder": _read_matrix,
        "encoder_bias": _read_array,
        "decoder_bias": _read_array,
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        text(self.dictionary_id, "dictionary_id")
        text(self.version, "dictionary version")
        digest(self.model_digest)
        digest(self.hook_digest)
        _matrix(self.encoder)
        _matrix(self.decoder)
        _floats(self.encoder_bias)
        _floats(self.decoder_bias)
        if (
            len(self.encoder[0]) != self.feature_count
            or len(self.decoder) != self.feature_count
            or len(self.decoder[0]) != self.width
            or len(self.decoder_bias) != self.width
        ):
            raise ValueError("dictionary encoder/decoder geometry mismatch")
        if self.nonlinearity not in ("linear", "relu"):
            raise ValueError("unsupported dictionary nonlinearity")
        if self.origin not in ("native", "constructed", "external_learned", "external_unspecified"):
            raise ValueError("unsupported dictionary origin claim")
        references(self.evidence_refs)

    @property
    def width(self) -> int:
        return len(self.encoder)

    @property
    def feature_count(self) -> int:
        return len(self.encoder_bias)

    @property
    def is_native(self) -> bool:
        eye = tuple(tuple(float(i == j) for j in range(self.width)) for i in range(self.width))
        return (
            self.nonlinearity == "linear"
            and self.feature_count == self.width
            and self.encoder == self.decoder == eye
            and not any(self.encoder_bias + self.decoder_bias)
        )


@dataclass(frozen=True)
class ComparisonArm(CVARecord):
    arm_id: str
    method: str
    dictionary: FeatureDictionary | None
    selected_features: tuple[int, ...]
    public_features: tuple[TokenFeature, ...]
    causal_prediction: EffectPrediction
    hypothesis: str
    max_feature_mae: float
    context_refs: tuple[EvidenceReference, ...] = ()
    unavailable_reason: str | None = None

    schema_version = "interpretation-comparison-arm-v1"
    restorers = {
        "dictionary": lambda v: None if v is None else FeatureDictionary.from_dict(v),
        "selected_features": _read_array,
        "public_features": lambda v: _read_records(v, TokenFeature),
        "causal_prediction": EffectPrediction.from_dict,
        "context_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        text(self.arm_id, "arm_id")
        if type(self.method) is not str or self.method not in METHODS:
            raise ValueError("unsupported comparison method")
        selected = indices(self.selected_features, 8)
        features = tuple(
            _record(f, TokenFeature) for f in array(self.public_features, 16, required=True)
        )
        if len({f.name for f in features}) != len(features):
            raise ValueError("duplicate public feature names")
        if any(
            f.offset > 0 or (f.other_offset is not None and f.other_offset > 0) for f in features
        ):
            raise ValueError(
                "comparison predictors require causal-prefix features, never future tokens"
            )
        _record(self.causal_prediction, EffectPrediction)
        text(self.hypothesis, "hypothesis", 4096)
        finite(self.max_feature_mae)
        if self.max_feature_mae < 0:
            raise ValueError("prediction tolerance must be nonnegative")
        references(self.context_refs, required=False)
        if self.method == "native_context" and not self.context_refs:
            raise ValueError("contextual method requires explicit circuit-context references")
        if self.dictionary is None:
            text(self.unavailable_reason, "unavailable_reason", 2048)
        else:
            dictionary = _record(self.dictionary, FeatureDictionary)
            if self.unavailable_reason is not None:
                raise ValueError("available dictionary cannot be labelled unavailable")
            if max(selected) >= dictionary.feature_count:
                raise ValueError("feature selection exceeds dictionary")
            if self.method in ("native", "native_context") and not dictionary.is_native:
                raise ValueError("native analysis requires actual identity coordinates")
            if self.method == "sparse_autoencoder" and dictionary.nonlinearity != "relu":
                raise ValueError("this SAE adapter supports fixed ReLU dictionaries only")


@dataclass(frozen=True)
class ComparisonBudget(CVARecord):
    selected_features: int
    public_feature_count: int
    max_forwards_per_arm: int
    max_distribution_values: int

    schema_version = "interpretation-comparison-budget-v1"

    def __post_init__(self) -> None:
        for value in (
            self.selected_features,
            self.public_feature_count,
            self.max_forwards_per_arm,
            self.max_distribution_values,
        ):
            if type(value) is not int:
                raise ValueError("comparison budgets require exact integers")
        if (
            not 1 <= self.selected_features <= 8
            or not 1 <= self.public_feature_count <= 16
            or not 1 <= self.max_forwards_per_arm <= 32
            or not 1 <= self.max_distribution_values <= 2000000
        ):
            raise ValueError("comparison budget exceeds supported bounds")


@dataclass(frozen=True)
class ComparisonProtocol(CVARecord):
    protocol_id: str
    model_digest: str
    hook: HookSpec
    manifest: SplitManifest
    arms: tuple[ComparisonArm, ...]
    budget: ComparisonBudget
    activation_positions: tuple[int, ...]
    logit_positions: tuple[int, ...]
    metric: LogitContrast
    seed: int
    ridge: float
    weak_activation_threshold: float = 0.01

    schema_version = "interpretation-comparison-protocol-v1"
    restorers = {
        "hook": HookSpec.from_dict,
        "manifest": SplitManifest.from_dict,
        "arms": lambda v: _read_records(v, ComparisonArm),
        "budget": ComparisonBudget.from_dict,
        "activation_positions": _read_array,
        "logit_positions": _read_array,
        "metric": LogitContrast.from_dict,
    }

    def __post_init__(self) -> None:
        text(self.protocol_id, "protocol_id")
        digest(self.model_digest)
        hook = _record(self.hook, HookSpec)
        if hook.kind is HookKind.ATTENTION_HEAD or hook.width > 64:
            raise ValueError("comparison supports at most 64 scalar coordinates, not head vectors")
        manifest = _record(self.manifest, SplitManifest)
        if any(
            len(group) > 8 for group in (manifest.generation, manifest.development, manifest.locked)
        ):
            raise ValueError("comparison supports at most eight inputs per split")
        budget = _record(self.budget, ComparisonBudget)
        arms = tuple(_record(a, ComparisonArm) for a in array(self.arms, 8, required=True))
        if len({a.arm_id for a in arms}) != len(arms) or set(a.method for a in arms) != set(
            METHODS
        ):
            raise ValueError(
                "all four methods, with unique arm identities, must remain in the roster"
            )
        if not any(a.dictionary is not None for a in arms):
            raise ValueError("comparison requires at least one available method")
        for arm in arms:
            if (
                len(arm.selected_features) != budget.selected_features
                or len(arm.public_features) != budget.public_feature_count
            ):
                raise ValueError("methods do not match declared selection/predictor budget")
            if arm.dictionary is not None and (
                arm.dictionary.model_digest != self.model_digest
                or arm.dictionary.hook_digest != hook.digest
                or arm.dictionary.width != hook.width
            ):
                raise ValueError("dictionary model/hook differs from matched comparison")
        for position in indices(self.activation_positions, 16) + indices(self.logit_positions, 16):
            index(position, maximum=128)
        metric = _record(self.metric, LogitContrast)
        if metric.position not in self.logit_positions:
            raise ValueError("causal contrast must be among frozen output positions")
        index(self.seed, maximum=2**32)
        finite(self.ridge)
        finite(self.weak_activation_threshold)
        if not 1e-6 <= self.ridge <= 1e6 or self.weak_activation_threshold <= 0:
            raise ValueError("invalid ridge or declared activation threshold")


@dataclass(frozen=True)
class ArmFit(CVARecord):
    arm_id: str
    predictors: tuple[ActivationPredictor, ...]
    matched_predictors: tuple[ActivationPredictor, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    random_indices: tuple[int, ...]
    label_permutation: tuple[int, ...]

    schema_version = "interpretation-arm-fit-v1"
    restorers = {
        "predictors": lambda v: _read_records(v, ActivationPredictor),
        "matched_predictors": lambda v: _read_records(v, ActivationPredictor),
        "means": _read_array,
        "stds": _read_array,
        "random_indices": _read_array,
        "label_permutation": _read_array,
    }

    def __post_init__(self) -> None:
        text(self.arm_id, "arm_id")
        for predictors in (self.predictors, self.matched_predictors):
            for predictor in array(predictors, 8, required=True):
                _record(predictor, ActivationPredictor)
        count = len(self.predictors)
        _floats(self.means, 8)
        _floats(self.stds, 8)
        if (
            len(self.matched_predictors) != count
            or len(self.means) != count
            or len(self.stds) != count
        ):
            raise ValueError("fit/control shape mismatch")
        if any(s < 0 for s in self.stds):
            raise ValueError("negative descriptive standard deviation")
        array(self.random_indices, 8)
        if self.random_indices:
            indices(self.random_indices, 8)
            if len(self.random_indices) != count:
                raise ValueError("random-feature budget mismatch")
        if sorted(indices(self.label_permutation, 8)) != list(range(count)):
            raise ValueError("invalid group-label permutation")


@dataclass(frozen=True)
class ComparisonCommit(CVARecord):
    protocol: ComparisonProtocol
    generation: NativeBatch
    fits: tuple[ArmFit, ...]
    created_at: str

    schema_version = "interpretation-comparison-commit-v1"
    restorers = {
        "protocol": ComparisonProtocol.from_dict,
        "generation": NativeBatch.from_dict,
        "fits": lambda v: _read_records(v, ArmFit),
    }

    def __post_init__(self) -> None:
        p = _record(self.protocol, ComparisonProtocol)
        generation = _record(self.generation, NativeBatch)
        if (
            generation.model.digest != p.model_digest
            or generation.hook != p.hook
            or generation.component_indices != tuple(range(p.hook.width))
            or generation.token_positions != p.activation_positions
        ):
            raise ValueError("generation model/hook/positions differ from comparison")
        actual = {i.example_id: i for i in generation.inputs}
        if set(actual) != {r.example_id for r in p.manifest.generation}:
            raise ValueError("generation split differs from commitment")
        for ref in p.manifest.generation:
            if ExampleRef.from_input(actual[ref.example_id], distribution=ref.distribution) != ref:
                raise ValueError("generation context differs from split")
        fits = tuple(_record(f, ArmFit) for f in array(self.fits, 8, required=True))
        available = [a for a in p.arms if a.dictionary is not None]
        if [f.arm_id for f in fits] != [a.arm_id for a in available]:
            raise ValueError("fit roster differs from available methods")
        for fit, arm in zip(fits, available):
            if len(fit.predictors) != len(arm.selected_features):
                raise ValueError("fit feature count differs from protocol")
            for predictor in fit.predictors + fit.matched_predictors:
                if (
                    predictor.kind != "ridge"
                    or predictor.fit_example_ids != tuple(actual)
                    or len(predictor.features) != p.budget.public_feature_count
                ):
                    raise ValueError("fit does not use the declared generation/predictor budget")
            if any(v.features != arm.public_features for v in fit.predictors):
                raise ValueError("fit features differ from frozen explanation")
        parse_rfc3339_datetime(self.created_at, "created_at")
