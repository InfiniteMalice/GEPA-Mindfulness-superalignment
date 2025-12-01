"""Training utilities for Mindful Trace GEPA."""

from typing import TYPE_CHECKING, NoReturn

from ..utils.imports import optional_import

_dist = optional_import("mindful_trace_gepa.train.dist")

if _dist is not None:
    get_accelerator = _dist.get_accelerator
    wrap_model_optimizer = _dist.wrap_model_optimizer
    save_sharded = _dist.save_sharded
else:  # pragma: no cover - fallback when torch/accelerate are missing

    def _missing(*_args: object, **_kwargs: object) -> NoReturn:
        """Raise ImportError when torch or accelerate dependencies are unavailable."""

        raise ImportError(
            "mindful_trace_gepa.train.dist requires torch and accelerate; install with"
            " 'pip install torch accelerate' or check for import failures"
        )

    get_accelerator = _missing
    wrap_model_optimizer = _missing
    save_sharded = _missing

if TYPE_CHECKING:
    from .dist import get_accelerator, save_sharded, wrap_model_optimizer

__all__ = ["get_accelerator", "save_sharded", "wrap_model_optimizer"]
