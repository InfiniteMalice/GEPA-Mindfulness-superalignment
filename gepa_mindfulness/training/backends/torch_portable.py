"""Factory for the portable PyTorch policy backend."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from torch import nn

from gepa_mindfulness.training.runtime_config import RLRunConfig

from .base import TokenizerLike
from .torch_policy import TorchPolicyBackend, TrainingMode


def create_portable_backend(
    config: RLRunConfig,
    *,
    policy_model: nn.Module | None = None,
    tokenizer: TokenizerLike | None = None,
    reference_model: nn.Module | None = None,
    training_mode: TrainingMode = "full",
    lora_config: Mapping[str, object] | None = None,
) -> TorchPolicyBackend:
    """Create a configured backend from injected or Hugging Face model assets."""
    if not isinstance(config, RLRunConfig):
        raise TypeError("config must be an RLRunConfig")
    if (policy_model is None) != (tokenizer is None):
        raise ValueError("policy_model and tokenizer must be supplied together")
    if policy_model is None:
        policy_model, tokenizer = _load_transformers_assets(config.policy.model_name)
    if tokenizer is None:  # pragma: no cover - narrowed by the checks above
        raise AssertionError("tokenizer must be available")
    if training_mode == "lora":
        reference_model = deepcopy(policy_model) if reference_model is None else reference_model
        policy_model = _apply_lora(policy_model, lora_config)
        reference_model = _apply_lora(reference_model, lora_config)
        adapter_identifier = "peft-lora"
    elif training_mode == "full":
        if lora_config is not None:
            raise ValueError("lora_config requires training_mode='lora'")
        adapter_identifier = None
    else:
        raise ValueError("training_mode must be 'full' or 'lora'")
    return TorchPolicyBackend(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        device=config.runtime.device,
        learning_rate=config.algorithm.learning_rate,
        max_new_tokens=config.policy.max_new_tokens,
        max_grad_norm=config.algorithm.max_grad_norm,
        training_mode=training_mode,
        model_identifier=(
            config.hybrid.model_id
            if config.runtime.backend == "mojo-vulkan-llamacpp"
            else config.policy.model_name
        ),
        adapter_identifier=adapter_identifier,
    )


def _load_transformers_assets(model_name: str) -> tuple[nn.Module, TokenizerLike]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:  # pragma: no cover - train extra is installed in CI
        raise RuntimeError(
            "Portable model loading requires the optional 'transformers' dependency."
        ) from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Model {model_name!r} is not available locally; downloads are disabled. "
            "Provide a local model directory or populate the Hugging Face cache first."
        ) from error
    return model, tokenizer


def _apply_lora(
    policy_model: nn.Module,
    config: Mapping[str, object] | None,
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "PEFT LoRA mode requires the optional 'peft' dependency; install peft first."
        ) from error
    values: dict[str, object] = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    if config is not None:
        values.update(config)
    return get_peft_model(policy_model, LoraConfig(**values))


__all__ = ["create_portable_backend"]
