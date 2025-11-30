"""Training utilities for Mindful Trace GEPA."""

from typing import NoReturn

from ..utils.imports import optional_import

_dist = optional_import("mindful_trace_gepa.train.dist")

if _dist is not None:
    get_accelerator = _dist.get_accelerator
    wrap_model_optimizer = _dist.wrap_model_optimizer
    save_sharded = _dist.save_sharded
else:  # pragma: no cover - fallback when torch/accelerate are missing

    def _missing(*_args: object, **_kwargs: object) -> NoReturn:
        """Raise ImportError when training distribution helpers are unavailable."""

        raise ImportError(
            "mindful_trace_gepa.train.dist is unavailable; check optional dependencies or import"
            " failures"
        )

    get_accelerator = _missing
    wrap_model_optimizer = _missing
    save_sharded = _missing

__all__ = ["get_accelerator", "wrap_model_optimizer", "save_sharded"]
