"""Compatibility helpers for configuring TRL PPO components."""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from trl import PPOConfig as TRLPPOConfig
from trl import PPOTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _available_ppo_config_fields() -> set[str]:
    """Collect constructor fields exposed by the TRL PPOConfig."""

    available: set[str] = set()

    dataclass_fields = getattr(TRLPPOConfig, "__dataclass_fields__", None)
    if dataclass_fields:
        available.update(dataclass_fields.keys())

    try:
        signature = inspect.signature(TRLPPOConfig)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        available.update(signature.parameters.keys())

    return available


def make_trl_ppo_config(training_config: "TrainingConfig") -> TRLPPOConfig:
    """Construct a TRL PPOConfig compatible with the installed TRL version."""

    ppo_settings = training_config.ppo
    ppo_config_kwargs: dict[str, Any] = {
        "learning_rate": ppo_settings.learning_rate,
        "mini_batch_size": ppo_settings.mini_batch_size,
        "batch_size": ppo_settings.batch_size,
    }

    epoch_value = ppo_settings.ppo_epochs
    available_fields = _available_ppo_config_fields()
    if "ppo_epochs" in available_fields:
        ppo_config_kwargs["ppo_epochs"] = epoch_value
    elif "num_epochs" in available_fields:
        ppo_config_kwargs["num_epochs"] = epoch_value
    elif "num_train_epochs" in available_fields:
        ppo_config_kwargs["num_train_epochs"] = epoch_value
    else:
        LOGGER.warning(
            "Unable to determine PPO epoch parameter; using TRL defaults.",
        )

    return TRLPPOConfig(**ppo_config_kwargs)


@lru_cache(maxsize=1)
def _ppo_trainer_config_keywords() -> list[str]:
    """Determine PPOTrainer keyword names that accept the config object."""

    try:
        signature = inspect.signature(PPOTrainer)
    except (TypeError, ValueError):
        return ["config", "ppo_config"]

    keywords: list[str] = []
    for candidate in ("config", "ppo_config"):
        parameter = signature.parameters.get(candidate)
        if parameter and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            keywords.append(candidate)

    return keywords or ["config", "ppo_config"]


def create_ppo_trainer(
    *,
    ppo_config: TRLPPOConfig,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    ref_model: "PreTrainedModel | None" = None,
) -> PPOTrainer:
    """Create a PPOTrainer compatible with the installed TRL signature."""

    base_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
    }
    if ref_model is not None:
        base_kwargs["ref_model"] = ref_model

    candidate_errors: list[str] = []
    for keyword in _ppo_trainer_config_keywords():
        keyword_kwargs = dict(base_kwargs)
        keyword_kwargs[keyword] = ppo_config
        try:
            return PPOTrainer(**keyword_kwargs)
        except TypeError as exc:  # pragma: no cover - depends on TRL version
            candidate_errors.append(f"{keyword}: {exc}")

    LOGGER.warning(
        "Unable to determine PPOTrainer config keyword; passing positionally.",
    )

    try:
        return PPOTrainer(
            ppo_config,
            base_kwargs["model"],
            base_kwargs.get("ref_model"),
            base_kwargs["tokenizer"],
        )
    except TypeError as exc:  # pragma: no cover - defensive guard
        error_detail = "; ".join(candidate_errors + [f"positional: {exc}"])
        raise TypeError(
            "Unable to construct PPOTrainer with available configuration "
            f"options: {error_detail}"
        ) from exc


__all__ = [
    "create_ppo_trainer",
    "make_trl_ppo_config",
]
