"""Default-off TF-OPD input moments and linear-layer candidate arithmetic, without execution."""

# Standard library
from __future__ import annotations

import math
import zlib

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.verification.artifact_records import artifact_digest
from gepa_mindfulness.verification.debate_records import _record

from .specialist_fusion_records import (
    MAX_DIMENSION,
    MAX_SPECIALISTS,
    MAX_TOKENS,
    FusionCandidate,
    FusionSource,
    LayerStatistics,
    TFOPDConfig,
    number,
)


def _enabled(config: TFOPDConfig) -> bool:
    """Validate only inert configuration before inspecting source records or numerical inputs."""
    if type(config) is not TFOPDConfig:
        raise ValueError("exact TFOPDConfig required")
    config.__post_init__()
    return config.enabled


def _array(value: object, maximum_rows: int):
    """Copy bounded dense real arrays; object hooks, bools and coercible strings are rejected."""
    import numpy as np

    if type(value) is np.ndarray:
        if value.ndim != 2 or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dense float32/float64 matrices required")
        rows, columns = value.shape
    elif type(value) in (tuple, list):
        rows = len(value)
        if not 1 <= rows <= maximum_rows:
            raise ValueError("matrix row capacity exceeded")
        columns = None
        for row in value:
            if type(row) not in (tuple, list) or not 1 <= len(row) <= MAX_DIMENSION:
                raise ValueError("invalid matrix row")
            if columns is not None and len(row) != columns:
                raise ValueError("ragged matrix")
            columns = len(row)
            for element in row:
                number(element)
        value = np.asarray(value, dtype=np.float64)
    else:
        raise ValueError("plain rows or exact NumPy ndarray required")
    if not 1 <= rows <= maximum_rows or not 1 <= columns <= MAX_DIMENSION:
        raise ValueError("matrix dimension capacity exceeded")
    if not np.isfinite(value).all():
        raise ValueError("nonfinite matrix")
    return np.array(value, copy=True)


def _norm(value) -> float:
    """Use scaled hypot to retain tiny updates without squared-norm underflow."""
    result = math.hypot(*(float(v) for v in value.flat))
    if not math.isfinite(result):
        raise ValueError("diagnostic norm exceeds binary64 range")
    return result


def _log_energy(delta, gram) -> float | None:
    """Scale both factors before the quadratic form so tiny energies retain their ratio."""
    import numpy as np

    delta_scale = float(np.max(np.abs(delta)))
    gram_scale = float(np.max(np.abs(gram)))
    if delta_scale == 0 or gram_scale == 0:
        return None
    normalized = delta / delta_scale
    energy = float(np.sum((normalized @ (gram / gram_scale)) * normalized))
    if energy < 0 or not math.isfinite(energy):
        raise ValueError("invalid rounded teacher energy")
    if energy == 0:
        return None
    return 2 * math.log(delta_scale) + math.log(gram_scale) + math.log(energy)


def _represented_energy(log_energy: float | None) -> tuple[float | None, str]:
    """Distinguish an exact computed zero from a positive value outside binary64 range."""
    if log_energy is None:
        return 0.0, "zero"
    try:
        value = math.exp(log_energy)
    except OverflowError:
        return None, "not_representable"
    return (value, "available") if value > 0 else (None, "not_representable")


def _energy_share(numerator_log: float | None, denominator_log: float | None):
    """Return a ratio only when reference information and numeric range support one."""
    if denominator_log is None:
        return None, "undefined_zero_reference_energy"
    if numerator_log is None:
        return 0.0, "zero"
    return _represented_energy(numerator_log - denominator_log)


def screen_rollout_text(text: str, *, finished: bool, enabled: bool = False) -> dict | None:
    """Diagnose only the paper-inspired text screen; this is not behavioral verification.

    UTF-8 compressed bytes are compared with UTF-8 tail bytes, an explicit adaptation of the
    paper's character-tail description. A flag cannot supply reward, certify correctness or
    classify necessary repeated actions as unproductive. The host retains and reviews cohorts.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(text) is not str or len(text) > 262144 or type(finished) is not bool:
        raise ValueError("bounded text and explicit finished status required")
    tail = text[-20000:].encode("utf-8")
    compressed = len(zlib.compress(tail))
    flag = bool(tail) and compressed * 20 < len(tail)
    return dict(
        schema_version="tfopd-text-screen-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_eligible=False,
        authority_granted=False,
        source_reference="REF-TF-OPD",
        text_digest=content_digest(text),
        tail_characters=min(len(text), 20000),
        tail_bytes=len(tail),
        compressed_bytes=compressed,
        compression_flag=flag,
        finished=finished,
        keep_by_text_screen=finished and bool(text.strip()) and not flag,
        codec="zlib-wrapped-deflate",
        compression_level=-1,
        codec_version=zlib.ZLIB_VERSION,
        codec_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
    )


def collect_input_statistics(
    source: FusionSource, inputs: object, *, config: TFOPDConfig = TFOPDConfig()
) -> LayerStatistics | None:
    """Compute actual token-normalized moments from host-supplied admitted activation rows.

    This function does not generate rollouts, install hooks or authenticate their provenance.
    FP32 accumulation follows the paper; FP64 is an explicit numerical ablation.
    """
    if not _enabled(config):
        return None
    source = _record(source, FusionSource)
    import numpy as np

    values = _array(inputs, MAX_TOKENS)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            accumulation = values.astype(config.accumulation_dtype)
            gram = (accumulation.T @ accumulation) / np.asarray(
                len(values), dtype=config.accumulation_dtype
            )
        if not np.isfinite(gram).all():
            raise ValueError("nonfinite input moments")
        if np.any((np.diag(gram) == 0) & np.any(values != 0, axis=0)):
            raise ValueError("nonzero activation energy underflowed in accumulation")
    except FloatingPointError as error:
        raise ValueError("input statistics overflowed") from error
    return LayerStatistics(
        source,
        len(values),
        tuple(tuple(float(v) for v in row) for row in gram),
        content_digest(values.tolist()),
        config.accumulation_dtype,
    )


def fuse_linear_layer(
    base: object,
    specialists: tuple[object, ...],
    statistics: tuple[LayerStatistics, ...],
    *,
    domain_weights: tuple[float, ...] | None = None,
    config: TFOPDConfig = TFOPDConfig(),
) -> FusionCandidate | None:
    """Solve the source's FP64 delta equation and return inert, explicitly rounded numbers.

    The host owns checkpoint/data authentication and all protected evaluation and admission.
    No live tensors, checkpoints, optimizer state or stores are modified. Singular systems
    fail visibly; there is no implicit regularizer, pseudo-inverse or baseline substitution.
    """
    if not _enabled(config):
        return None
    if (
        type(specialists) is not tuple
        or type(statistics) is not tuple
        or not 1 <= len(statistics) <= MAX_SPECIALISTS
        or len(specialists) != len(statistics)
    ):
        raise ValueError("1..8 matching specialist/statistics pairs required")
    records = tuple(_record(value, LayerStatistics) for value in statistics)
    sources = tuple(record.source for record in records)
    first = sources[0]
    if len({s.specialist_checkpoint_digest for s in sources}) != len(sources) or any(
        (s.base_checkpoint_digest, s.architecture_id, s.layer_id)
        != (first.base_checkpoint_digest, first.architecture_id, first.layer_id)
        for s in sources
    ):
        raise ValueError("incompatible declared base/architecture/layer or duplicate specialist")
    if len(canonical_json([r.to_dict() for r in records]).encode("utf-8")) > 1048576:
        raise ValueError("input record capacity exceeded")
    if domain_weights is None:
        domain_weights = (1.0,) * len(records)
    if type(domain_weights) is not tuple or len(domain_weights) != len(records):
        raise ValueError("matching domain weights required")
    weights = tuple(number(v) for v in domain_weights)
    if any(v <= 0 for v in weights):
        raise ValueError("domain weights must be positive")
    import numpy as np

    original = _array(base, MAX_DIMENSION)
    teachers = tuple(_array(value, MAX_DIMENSION) for value in specialists)
    if any(value.shape != original.shape for value in teachers):
        raise ValueError("specialist matrix shape mismatch")
    grams = tuple(_array(record.gram, MAX_DIMENSION).astype(np.float64) for record in records)
    width = original.shape[1]
    for gram in grams:
        if gram.shape != (width, width) or not np.array_equal(gram, gram.T):
            raise ValueError("symmetric shape-matched Gram required")
        if np.linalg.eigvalsh(gram)[0] < 0:
            raise ValueError("non-PSD Gram is unsupported; no implicit repair")
    reference = original.astype(np.float64)
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            deltas = tuple(value.astype(np.float64) - reference for value in teachers)
            adjusted = tuple(
                weight * (config.alpha * gram + (1 - config.alpha) * np.diag(np.diag(gram)))
                for weight, gram in zip(weights, grams)
            )
            lhs = np.sum(adjusted, axis=0, dtype=np.float64)
            condition = float(np.linalg.cond(lhs))
            if not math.isfinite(condition) or condition > config.condition_limit:
                raise ValueError("singular or ill-conditioned Gram sum")
            rhs = np.sum([delta @ gram for delta, gram in zip(deltas, adjusted)], axis=0)
            update = np.linalg.solve(lhs.T, rhs.T).T
            solved = reference + update
            stored = solved.astype(config.storage_dtype)
            if not np.isfinite(stored).all():
                raise ValueError("candidate storage overflow")
            teacher_rows = []
            for delta, gram, teacher in zip(deltas, adjusted, teachers):
                residual = update - delta
                numerator_log = _log_energy(residual, gram)
                denominator_log = _log_energy(delta, gram)
                denominator, energy_status = _represented_energy(denominator_log)
                share, share_status = _energy_share(numerator_log, denominator_log)
                stored_log = _log_energy(
                    stored.astype(np.float64) - teacher.astype(np.float64), gram
                )
                stored_share, stored_status = _energy_share(stored_log, denominator_log)
                base_norm = _norm(reference)
                delta_norm = _norm(delta)
                teacher_rows.append(
                    dict(
                        unreproduced_share=share,
                        unreproduced_share_status=share_status,
                        stored_unreproduced_share=stored_share,
                        stored_unreproduced_share_status=stored_status,
                        reference_energy=denominator,
                        reference_energy_status=energy_status,
                        reference_log_energy=denominator_log,
                        residual_log_energy=numerator_log,
                        update_norm=delta_norm,
                        base_relative_update=None if base_norm == 0 else delta_norm / base_norm,
                        large_delta_warning=None if base_norm == 0 else delta_norm > base_norm / 2,
                    )
                )
            diagnostics = dict(
                source_reference="REF-TF-OPD",
                numpy_version=np.__version__,
                source_binding="declared_not_authenticated",
                solve_dtype="float64",
                unreproduced_share_basis="fp64_delta_solve_before_storage",
                stored_share_basis="rounded_candidate_minus_supplied_teacher",
                base_input_dtype=str(original.dtype),
                specialist_input_dtypes=[str(value.dtype) for value in teachers],
                statistics_accumulation_dtypes=[record.accumulation_dtype for record in records],
                storage_dtype=config.storage_dtype,
                condition_number=condition,
                solve_residual_norm=_norm(update @ lhs - rhs),
                solve_update_norm=_norm(update),
                stored_update_norm=_norm(stored.astype(np.float64) - reference),
                quantization_error_norm=_norm(stored.astype(np.float64) - solved),
                teachers=teacher_rows,
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("unsupported numerical fusion system") from error
    result = FusionCandidate(
        sources,
        tuple(artifact_digest(record) for record in records),
        content_digest(reference.tolist()),
        tuple(content_digest(value.tolist()) for value in teachers),
        tuple(tuple(float(v) for v in row) for row in stored),
        weights,
        canonical_json(config.to_dict()),
        canonical_json(diagnostics),
    )
    if len(canonical_json(result.to_dict()).encode("utf-8")) > 4 * 1048576:
        raise ValueError("candidate output capacity exceeded")
    return result
