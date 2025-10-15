import inspect
import logging
from typing import Any

from trl import PPOTrainer

from ..core.rewards import RewardSignal, RewardWeights
from ..core.tracing import CircuitTracerLogger
from .configs import TrainingConfig
from .ppo_utils import TRLPPOConfig

LOGGER = logging.getLogger(__name__)


def _get_available_ppo_fields() -> set[str]:
    """
    Introspect TRLPPOConfig to identify valid fields.
    Returns a set of field names that can be passed to TRLPPOConfig.
    """
    available: set[str] = set()

    dataclass_fields = getattr(TRLPPOConfig, "__dataclass_fields__", None)
    if dataclass_fields:
        available.update(dataclass_fields.keys())
        return available

    try:
        signature = inspect.signature(TRLPPOConfig)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        available.update(
            param
            for param in signature.parameters.keys()
            if param not in ("self", "args", "kwargs")
        )

    return available


class PPOPipeline:
    """
    Orchestrates PPO training with GEPA rewards and Circuit Tracer logging.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        reward_weights: RewardWeights,
        circuit_logger: CircuitTracerLogger | None = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_weights = reward_weights
        self.circuit_logger = circuit_logger

        # Build PPO trainer
        self._init_ppo_trainer()

    def _init_ppo_trainer(self) -> None:
        """
        Initialize PPO trainer with defensive compatibility handling.
        """
        # Get base kwargs for trainer construction
        base_kwargs = {
            "model": self.model,
            "ref_model": self.ref_model,
            "tokenizer": self.tokenizer,
        }

        # Convert TrainingConfig to PPO config kwargs
        ppo_config_dict = self.config.model_dump()

        # Get available TRLPPOConfig fields
        available_fields: set[str] = set()

        dataclass_fields = getattr(TRLPPOConfig, "__dataclass_fields__", None)
        if dataclass_fields:
            available_fields.update(dataclass_fields.keys())
        else:
            try:
                signature = inspect.signature(TRLPPOConfig)
            except (TypeError, ValueError):
                signature = None

            if signature is not None:
                available_fields.update(
                    param
                    for param in signature.parameters.keys()
                    if param not in ("self", "args", "kwargs")
                )

        # Filter config dict to only include valid fields
        ppo_config_kwargs = {
            k: v for k, v in ppo_config_dict.items() if k in available_fields
        }

        # Log filtered fields for debugging
        filtered_out = set(ppo_config_dict.keys()) - set(ppo_config_kwargs.keys())
        if filtered_out:
            LOGGER.debug(f"Filtered out unsupported PPO config fields: {filtered_out}")

        # Create PPO config
        ppo_config = TRLPPOConfig(**ppo_config_kwargs)

        # Try creating trainer with different signatures
        candidate_errors = []

        # Attempt 1: Try with **base_kwargs unpacked
        try:
            self.ppo_trainer = PPOTrainer(config=ppo_config, **base_kwargs)
            return
        except TypeError as exc:
            candidate_errors.append(f"config+kwargs: {exc}")

        # Attempt 2: Try with positional arguments
        try:
            self.ppo_trainer = PPOTrainer(
                ppo_config,
                base_kwargs["model"],
                base_kwargs["ref_model"],
                base_kwargs["tokenizer"],
            )
            return
        except TypeError as exc:
            candidate_errors.append(f"positional: {exc}")

        # If all attempts failed, raise with detailed error
        error_detail = "; ".join(candidate_errors)
        raise TypeError(
            "Unable to construct PPOTrainer with available configuration options: "
            f"{error_detail}"
        )

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Execute one PPO training step with GEPA rewards.
        """
        # Generate responses
        query_tensors = batch["input_ids"]
        response_tensors = self.ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **self.config.generation_kwargs,
        )

        # Compute GEPA rewards
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            query_text = self.tokenizer.decode(query, skip_special_tokens=True)
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)

            # Get GEPA score (placeholder - implement actual scoring)
            reward_signal = self._compute_gepa_reward(query_text, response_text)
            rewards.append(reward_signal.total_reward)

        # PPO update
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

        return stats

    def _compute_gepa_reward(self, query: str, response: str) -> RewardSignal:
        """
        Compute GEPA reward signal for a query-response pair.
        Placeholder - actual implementation depends on scoring logic.
        """
        # TODO: Implement actual GEPA scoring
        return RewardSignal(
            task_reward=0.0,
            gepa_reward=0.0,
            honesty_reward=0.0,
            hallucination_penalty=0.0,
            total_reward=0.0,
        )
