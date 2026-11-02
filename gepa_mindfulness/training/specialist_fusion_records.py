"""Versioned numerical research proposals retain existing evidence and source restrictions."""

# Standard library
from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass

# Third-party
# Local
from evaluation.causal_records import canonical_json
from evaluation.improvement_records import SystemConfig
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.artifact_records import record_tuple
from gepa_mindfulness.verification.debate_records import _digest, _record
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    _text,
    choice,
    public_refs,
    restore_records,
    restore_refs,
)

from .causal_reward_records import _json_value, validate_input_envelope
from .eligibility import require_training_eligible

MAX_DIMENSION = 64
MAX_TOKENS = 4096
MAX_SPECIALISTS = 8


def number(value: object) -> float:
    """Require finite real JSON scalars without bool/string or enormous-integer coercion."""
    if type(value) not in (float, int):
        raise ValueError("expected finite nonboolean number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("number outside binary64 range") from error
    if not math.isfinite(result):
        raise ValueError("expected finite number")
    return result


def matrix(value: object) -> tuple[tuple[float, ...], ...]:
    """Detach a bounded rectangular matrix, with no numerical dependency or array hooks."""
    if type(value) not in (list, tuple) or not 1 <= len(value) <= MAX_DIMENSION:
        raise ValueError("matrix must have 1..64 rows")
    result = []
    width = None
    for row in value:
        if type(row) not in (list, tuple) or not 1 <= len(row) <= MAX_DIMENSION:
            raise ValueError("matrix must have 1..64 columns")
        if width is not None and len(row) != width:
            raise ValueError("ragged matrix")
        width = len(row)
        result.append(tuple(number(item) for item in row))
    return tuple(result)


def require_float32(values: tuple[tuple[float, ...], ...]) -> None:
    """Reject a false FP32 declaration without importing the optional numeric backend."""
    try:
        if any(struct.unpack("f", struct.pack("f", v))[0] != v for row in values for v in row):
            raise ValueError("matrix values are not float32-rounded")
    except (OverflowError, struct.error) as error:
        raise ValueError("matrix exceeds float32 range") from error


def bounded_text(value: object, name: str) -> None:
    """Keep record coordinates bounded without implying authenticity."""
    _text(value, name)
    if len(value.encode("utf-8")) > 512:
        raise ValueError(f"oversized {name}")


@dataclass(frozen=True)
class TFOPDConfig:
    """Disabled research settings; fixed CPU bounds are repository controls, not paper settings."""

    enabled: bool = False
    alpha: float = 0.9
    accumulation_dtype: str = "float32"
    storage_dtype: str = "float32"
    condition_limit: float = 1e12

    def __post_init__(self) -> None:
        """Reject implicit activation, unknown precision and numerical fallback requests."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not 0 <= number(self.alpha) <= 1 or number(self.condition_limit) < 1:
            raise ValueError("invalid shrinkage or condition limit")
        for dtype in (self.accumulation_dtype, self.storage_dtype):
            choice(dtype, "dtype", ("float32", "float64"))

    def to_dict(self) -> dict:
        """Retain all numerical choices in an exact v1 configuration."""
        self.__post_init__()
        return dict(schema_version="tfopd-config-v1", **asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> TFOPDConfig:
        """Unknown fields cannot introduce fallback or promotion permissions."""
        expected = {"schema_version", *cls.__dataclass_fields__}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid TF-OPD configuration fields")
        if value["schema_version"] != "tfopd-config-v1":
            raise ValueError("unsupported TF-OPD configuration")
        return cls(**{key: item for key, item in value.items() if key != "schema_version"})


@dataclass(frozen=True)
class FusionSource(DiagnosticRecord):
    """Declared host provenance does not authenticate checkpoints, rollouts or correctness."""

    base_checkpoint_digest: str
    specialist_checkpoint_digest: str
    architecture_id: str
    layer_id: str
    system: SystemConfig
    rollout_digest: str
    source_metadata_json: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version = "tfopd-source-v1"
    restorers = {"system": SystemConfig.from_dict, "evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        """Reject restricted source lineage through the unchanged shared training guard."""
        for value in (
            self.base_checkpoint_digest,
            self.specialist_checkpoint_digest,
            self.rollout_digest,
        ):
            _digest(value)
        for name in ("architecture_id", "layer_id"):
            bounded_text(getattr(self, name), name)
        if type(self.system) is not SystemConfig:
            raise ValueError("exact existing SystemConfig required")
        object.__setattr__(self, "system", SystemConfig.from_dict(self.system.to_dict()))
        if (
            type(self.source_metadata_json) is not str
            or len(self.source_metadata_json.encode("utf-8")) > 16384
        ):
            raise ValueError("source metadata exceeds capacity")
        metadata = _json_value(self.source_metadata_json)
        if type(metadata) is not dict or metadata.get("training_eligibility") != "TRAIN":
            raise ValueError("host-admitted explicit TRAIN source metadata required")
        require_training_eligible(metadata)
        for name in ("runtime_version", "selection_policy_version", "evaluator_version"):
            bounded_text(metadata.get(name), name)
        if type(metadata.get("sampling_config")) is not dict:
            raise ValueError("source sampling configuration required")
        object.__setattr__(self, "source_metadata_json", canonical_json(metadata))
        refs = public_refs(self.evidence_refs, required=True)
        if len(refs) > 32:
            raise ValueError("too many source references")
        for ref in refs:
            bounded_text(ref.reference_id, "reference_id")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class LayerStatistics(DiagnosticRecord):
    """A detached input moment and source binding, not proof of on-policy collection."""

    source: FusionSource
    token_count: int
    gram: tuple[tuple[float, ...], ...]
    input_digest: str
    accumulation_dtype: str
    schema_version = "tfopd-layer-statistics-v1"
    restorers = {"source": FusionSource.from_dict}

    def __post_init__(self) -> None:
        """Keep actual token counts, bounded moments and independently visible provenance."""
        object.__setattr__(self, "source", _record(self.source, FusionSource))
        if type(self.token_count) is not int or not 1 <= self.token_count <= MAX_TOKENS:
            raise ValueError("statistics require 1..4096 actual tokens")
        gram = matrix(self.gram)
        if len(gram) != len(gram[0]):
            raise ValueError("input Gram must be square")
        object.__setattr__(self, "gram", gram)
        _digest(self.input_digest)
        choice(self.accumulation_dtype, "accumulation_dtype", ("float32", "float64"))
        if self.accumulation_dtype == "float32":
            require_float32(gram)


@dataclass(frozen=True)
class FusionCandidate(DiagnosticRecord):
    """Immutable candidate numbers; never a model update, accepted baseline or certificate."""

    sources: tuple[FusionSource, ...]
    statistics_digests: tuple[str, ...]
    base_weight_digest: str
    specialist_weight_digests: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    domain_weights: tuple[float, ...]
    config_json: str
    diagnostics_json: str
    optimizer_eligible: bool = False
    authority_granted: bool = False
    schema_version = "tfopd-fusion-candidate-v1"
    restorers = {
        "sources": lambda value: restore_records(value, FusionSource),
        "statistics_digests": tuple,
        "specialist_weight_digests": tuple,
        "domain_weights": tuple,
    }

    def __post_init__(self) -> None:
        """Revalidate the complete numerical proposal and refuse authority extensions."""
        if self.optimizer_eligible is not False or self.authority_granted is not False:
            raise ValueError("fusion candidates grant no optimizer or promotion authority")
        if type(self.sources) is not tuple or not 1 <= len(self.sources) <= MAX_SPECIALISTS:
            raise ValueError("candidate requires 1..8 sources")
        object.__setattr__(self, "sources", record_tuple(self.sources, FusionSource))
        for values in (self.statistics_digests, self.specialist_weight_digests):
            if type(values) is not tuple or len(values) != len(self.sources):
                raise ValueError("candidate source/digest count mismatch")
            for value in values:
                _digest(value)
        _digest(self.base_weight_digest)
        if type(self.domain_weights) is not tuple or len(self.domain_weights) != len(self.sources):
            raise ValueError("candidate domain weight count mismatch")
        if any(number(value) <= 0 for value in self.domain_weights):
            raise ValueError("domain weights must be positive")
        config = TFOPDConfig.from_dict(_json_value(self.config_json))
        if config.enabled is not True:
            raise ValueError("candidate requires an enabled numerical configuration")
        weights = matrix(self.weights)
        if config.storage_dtype == "float32":
            require_float32(weights)
        object.__setattr__(self, "weights", weights)
        diagnostics = _json_value(self.diagnostics_json)
        if type(diagnostics) is not dict:
            raise ValueError("candidate diagnostics must be an object")
        validate_input_envelope(diagnostics)
        object.__setattr__(self, "config_json", canonical_json(config.to_dict()))
        object.__setattr__(self, "diagnostics_json", canonical_json(diagnostics))
