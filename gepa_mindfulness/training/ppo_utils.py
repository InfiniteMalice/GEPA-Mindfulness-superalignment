"""Helpers for constructing TRL PPO components across library versions."""

from __future__ import annotations

import inspect
import logging
from typing import Optional

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from trl import PPOConfig as TRLPPOConfig
from trl import PPOTrainer

from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _available_fields(cls: type) -> set[str]:
    """Collect available constructor fields for TRL dataclasses across versions."""

    collected: set[str] = set()

    dataclass_fields = getattr(cls, "__dataclass_fields__", None)
    if dataclass_fields:
        collected.update(dataclass_fields.keys())

    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        collected.update(signature.parameters.keys())

    return collected


def make_trl_ppo_config(training_config: TrainingConfig) -> TRLPPOConfig:
    """Create a TRL PPOConfig compatible with multiple TRL releases."""

    ppo_config_kwargs = {
        "learning_rate": training_config.ppo.learning_rate,
        "mini_batch_size": training_config.ppo.mini_batch_size,
        "batch_size": training_config.ppo.batch_size,
    }

    available_fields = _available_fields(TRLPPOConfig)

    epoch_value = training_config.ppo.ppo_epochs
    if "ppo_epochs" in available_fields:
        ppo_config_kwargs["ppo_epochs"] = epoch_value
    elif "num_epochs" in available_fields:
        ppo_config_kwargs["num_epochs"] = epoch_value
    elif "num_train_epochs" in available_fields:
        ppo_config_kwargs["num_train_epochs"] = epoch_value
    else:
        LOGGER.warning(
            "Unable to determine PPO epoch parameter; using TRL defaults."
        )

    return TRLPPOConfig(**ppo_config_kwargs)


def _candidate_keywords() -> list[str]:
    """Determine keyword names accepted by PPOTrainer for the config argument."""

    try:
        signature = inspect.signature(PPOTrainer)
    except (TypeError, ValueError):
        return ["config", "ppo_config"]

    keywords: list[str] = []
    for candidate in ("config", "ppo_config"):
        parameter = signature.parameters.get(candidate)
        if parameter and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            keywords.append(candidate)

    return keywords


def create_ppo_trainer(
    ppo_config: TRLPPOConfig,
    *,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    ref_model: Optional[AutoModelForCausalLM] = None,
) -> PPOTrainer:
    """Instantiate PPOTrainer while handling signature drift between TRL versions."""

    candidate_errors: list[str] = []

    for keyword in _candidate_keywords():
        try:
            return PPOTrainer(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                **{keyword: ppo_config},
            )
        except TypeError as exc:  # pragma: no cover - depends on TRL version
            candidate_errors.append(f"{keyword}: {exc}")

    LOGGER.warning(
        "Unable to determine PPOTrainer config keyword; passing positionally."
    )

    try:
        return PPOTrainer(ppo_config, model, ref_model, tokenizer)
    except TypeError as exc:  # pragma: no cover - defensive guard
        error_detail = "; ".join(candidate_errors + [f"positional: {exc}"])
        raise TypeError(
            "Unable to construct PPOTrainer with available configuration options: "
            f"{error_detail}"
        ) from exc
