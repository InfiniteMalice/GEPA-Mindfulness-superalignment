"""End-to-end orchestration for the GEPA mindfulness alignment pipeline."""

from __future__ import annotations

import inspect
import logging
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig as TRLPPOConfig
from trl import PPOTrainer

from ..core.abstention import enforce_abstention, honesty_reward_from_trace
from ..core.adversarial import sample_adversarial_batch
from ..core.contemplative_principles import (
    ContemplativePrinciple,
    GEPAPrinciples,
    GEPAPrincipleScore,
)
from ..core.imperatives import AlignmentImperative, ImperativeEvaluator, ImperativeSignal
from ..core.rewards import RewardSignal, RewardWeights
from ..core.tracing import CircuitTracerLogger
from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    prompt: str
    response: str
    reward: float
    trace_summary: dict
    contradiction_report: dict


class TrainingOrchestrator:
    """High-level driver coordinating GEPA, tracing, and PPO training."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tracing = CircuitTracerLogger()
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.policy_model)
        self.policy_model = AutoModelForCausalLM.from_pretrained(config.model.policy_model).to(
            self.device
        )
        self.reward_weights = RewardWeights.from_mapping(config.reward_weights.dict())
        self.ppo_trainer: Optional[PPOTrainer] = None

    def build_ppo_trainer(self) -> None:
        if self.ppo_trainer is not None:
            return
        ppo_config_kwargs = {
            "learning_rate": self.config.ppo.learning_rate,
            "mini_batch_size": self.config.ppo.mini_batch_size,
            "batch_size": self.config.ppo.batch_size,
        }

        epoch_value = self.config.ppo.ppo_epochs
        available_fields: set[str] = set()

        dataclass_fields = getattr(TRLPPOConfig, "__dataclass_fields__", None)
        if dataclass_fields:
            available_fields.update(dataclass_fields.keys())

        try:
            signature = inspect.signature(TRLPPOConfig)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            available_fields.update(signature.parameters.keys())

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

        ppo_config = TRLPPOConfig(**ppo_config_kwargs)

        trainer_kwargs = {
            "model": self.policy_model,
            "ref_model": None,
            "tokenizer": self.tokenizer,
        }

        try:
            trainer_signature = inspect.signature(PPOTrainer)
        except (TypeError, ValueError):
            trainer_signature = None

        config_keyword = None
        if trainer_signature is not None:
            parameters = trainer_signature.parameters
            for candidate in ("config", "ppo_config"):
                if candidate in parameters:
                    config_keyword = candidate
                    break

        if config_keyword is not None:
            trainer_kwargs[config_keyword] = ppo_config
            self.ppo_trainer = PPOTrainer(**trainer_kwargs)
        else:
            LOGGER.warning(
                "Unable to determine PPOTrainer config parameter; passing positionally."
            )
            self.ppo_trainer = PPOTrainer(ppo_config, **trainer_kwargs)

    def _sample_prompt(self, dataset: Iterable[str]) -> str:
        if isinstance(dataset, list):
            return random.choice(dataset)
        materialized = list(dataset)
        if not materialized:
            raise ValueError("Dataset must contain at least one prompt")
        return random.choice(materialized)

    def _gepa_scores(self, trace_payload: dict) -> GEPAPrinciples:
        scores = {
            ContemplativePrinciple.MINDFULNESS: GEPAPrincipleScore(
                value=0.8 if "mindfulness" in trace_payload else 0.6,
                rationale=trace_payload.get("framing", ""),
            ),
            ContemplativePrinciple.EMPATHY: GEPAPrincipleScore(
                value=0.7 if "empathy" in trace_payload else 0.5,
                rationale=trace_payload.get("evidence", ""),
            ),
            ContemplativePrinciple.PERSPECTIVE: GEPAPrincipleScore(
                value=0.75,
                rationale=trace_payload.get("tensions", ""),
            ),
            ContemplativePrinciple.AGENCY: GEPAPrincipleScore(
                value=0.65,
                rationale=trace_payload.get("decision", ""),
            ),
        }
        return GEPAPrinciples(scores=scores)

    def _imperative_scores(self, trace_payload: dict) -> ImperativeEvaluator:
        evaluator = ImperativeEvaluator.from_iterable(
            [
                (
                    AlignmentImperative.REDUCE_SUFFERING,
                    ImperativeSignal(
                        support=0.8, opposition=0.1, rationale=trace_payload.get("framing", "")
                    ),
                ),
                (
                    AlignmentImperative.INCREASE_PROSPERITY,
                    ImperativeSignal(
                        support=0.6, opposition=0.2, rationale=trace_payload.get("decision", "")
                    ),
                ),
                (
                    AlignmentImperative.INCREASE_KNOWLEDGE,
                    ImperativeSignal(
                        support=0.7, opposition=0.15, rationale=trace_payload.get("reflection", "")
                    ),
                ),
            ]
        )
        return evaluator

    def run_step(self, prompt: str) -> RolloutResult:
        self.build_ppo_trainer()
        assert self.ppo_trainer is not None
        with self.tracing.trace(prompt=prompt) as trace:
            self.tracing.log_event("framing", f"Evaluating prompt: {prompt}")
            self.tracing.log_event(
                "evidence", "No external evidence available, relying on model priors."
            )
            self.tracing.log_event("tensions", "Balance honesty with usefulness.")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.policy_model.generate(
                **inputs, max_length=inputs["input_ids"].shape[-1] + 64
            )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.tracing.log_event("decision", decoded)
            self.tracing.log_event("reflection", "Assessed alignment outcomes for this rollout.")

        trace_payload = trace.summary()
        confidence = min(0.99, 0.5 + random.random() * 0.5)
        decision = enforce_abstention(
            decoded, confidence, threshold=self.config.confidence_threshold
        )

        gepa = self._gepa_scores(trace_payload)
        imperatives = self._imperative_scores(trace_payload)
        honesty_reward = honesty_reward_from_trace(
            confidence=confidence,
            mindfulness=gepa.scores[ContemplativePrinciple.MINDFULNESS].value,
            emptiness=gepa.scores[ContemplativePrinciple.PERSPECTIVE].value,
        )
        reward_signal = RewardSignal(
            task_success=1.0 if decision.metadata["abstained"] == 0.0 else 0.0,
            gepa_score=gepa.aggregate(),
            honesty_reward=honesty_reward,
            hallucination_score=0.2 if "hallucination" in prompt.lower() else 0.0,
            imperatives_truth=imperatives.aggregate(),
        )
        combined_reward = reward_signal.combined(self.reward_weights)
        self.ppo_trainer.step([prompt], [decision.output], torch.tensor([combined_reward]))

        return RolloutResult(
            prompt=prompt,
            response=decision.output,
            reward=combined_reward,
            trace_summary=trace_payload,
            contradiction_report=imperatives.contradiction_report(),
        )

    def run(self, dataset: Iterable[str]) -> List[RolloutResult]:
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        dataset_list = [prompt for prompt in dataset if prompt]
        if not dataset_list:
            raise ValueError("Dataset must contain at least one non-empty prompt")
        results: List[RolloutResult] = []
        for step in range(self.config.max_steps):
            prompt = self._sample_prompt(dataset_list)
            result = self.run_step(prompt)
            results.append(result)
            LOGGER.info("Step %s reward: %s", step, result.reward)
        return results

    def run_adversarial_eval(self) -> List[RolloutResult]:
        scenarios = sample_adversarial_batch(self.config.adversarial_batch)
        return [self.run_step(scenario.prompt) for scenario in scenarios]
