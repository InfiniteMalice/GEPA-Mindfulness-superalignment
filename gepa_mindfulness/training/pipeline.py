import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from mindful_trace_gepa.deception.circuit_analysis import (
    detect_deception_circuits,
    detect_deception_heuristic,
)
from mindful_trace_gepa.prompts.dual_path import (
    make_dual_path_prompt,
    parse_dual_path_response,
)

from ..core.rewards import RewardSignal, RewardWeights
from ..core.tracing import CircuitTracerLogger
from .configs import TrainingConfig

try:
    from trl import PPOTrainer

    TRL_AVAILABLE = True
except ImportError:
    PPOTrainer = None  # type: ignore
    TRL_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - executed when torch absent
    torch = None

try:
    from .ppo_utils import TRLPPOConfig
except ImportError:
    TRLPPOConfig = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """
    Result from a single PPO rollout.
    """

    prompt: str
    response: str
    reward: float
    trace_summary: dict[str, Any]
    contradiction_report: dict[str, Any]
    stats: dict[str, float] | None = None
    deception_signals: dict[str, Any] | None = None
    sections: dict[str, Any] | None = None


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
        if not TRL_AVAILABLE:
            raise ImportError("trl required for PPO training: pip install trl transformers")

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
        ppo_config_kwargs = {k: v for k, v in ppo_config_dict.items() if k in available_fields}

        # Log filtered fields for debugging
        filtered_out = set(ppo_config_dict.keys()) - set(ppo_config_kwargs.keys())
        if filtered_out:
            LOGGER.debug("Filtered out unsupported PPO config fields: %s", filtered_out)

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
            f"Unable to construct PPOTrainer with available configuration options:{error_detail}"
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


class TrainingOrchestrator:
    """High-level rollout orchestrator with dual-path deception tracing."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        reward_weights: RewardWeights | None = None,
        circuit_logger: CircuitTracerLogger | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or config.device
        self.reward_weights = reward_weights or RewardWeights.from_mapping(
            config.reward_weights.dict()
        )
        self.circuit_logger = circuit_logger or CircuitTracerLogger()
        self.context_profile = ""

    def run(self, prompts: List[str]) -> List[RolloutResult]:
        """Run rollouts for the provided prompts."""

        results: List[RolloutResult] = []
        for prompt in prompts:
            results.append(self._perform_rollout_step(prompt))
        return results

    def run_adversarial_eval(self) -> List[RolloutResult]:
        """Run adversarial evaluation without PPO updates."""

        return []

    def _perform_rollout_step(self, prompt: str) -> RolloutResult:
        dual_prompt = make_dual_path_prompt(query=prompt, context=self.context_profile)

        with self.circuit_logger.span("dual_path_rollout") as span:
            span.log_event(
                "framing",
                {
                    "original_query": prompt,
                    "dual_path_prompt": dual_prompt,
                },
            )

            with self.circuit_logger.capture_circuits() as circuits:
                response_text = self._generate_response(dual_prompt)

            sections = parse_dual_path_response(response_text)

            if circuits is not None:
                path_1_circuits = self.circuit_logger.extract_span_circuits(
                    circuits,
                    sections.get("path_1_span", (0, 0))[0],
                    sections.get("path_1_span", (0, 0))[1],
                )
                path_2_circuits = self.circuit_logger.extract_span_circuits(
                    circuits,
                    sections.get("path_2_span", (0, 0))[0],
                    sections.get("path_2_span", (0, 0))[1],
                )
            else:
                path_1_circuits = None
                path_2_circuits = None

            span.log_event(
                "path_1_reasoning",
                {
                    "content": sections.get("path_1", ""),
                    "circuits": path_1_circuits,
                    "span": sections.get("path_1_span", (0, 0)),
                },
            )
            span.log_event(
                "path_2_reasoning",
                {
                    "content": sections.get("path_2", ""),
                    "circuits": path_2_circuits,
                    "span": sections.get("path_2_span", (0, 0)),
                },
            )
            span.log_event("comparison", {"content": sections.get("comparison", "")})
            span.log_event(
                "decision",
                {
                    "recommended_path": sections.get("recommended_path", "unclear"),
                    "recommendation_text": sections.get("recommendation", ""),
                },
            )

            if path_1_circuits is not None and path_2_circuits is not None:
                deception_signals = detect_deception_circuits(
                    sections,
                    path_1_circuits,
                    path_2_circuits,
                )
            else:
                deception_signals = detect_deception_heuristic(sections)

            span.log_event("deception_analysis", deception_signals)

            confidence = self._compute_confidence(sections, deception_signals)
            final_response = self._apply_abstention(
                response_text,
                confidence,
                getattr(self.config, "confidence_threshold", 0.0),
            )

            trace_summary = span.get_trace_summary()
            gepa_scores = self._score_gepa_with_deception(trace_summary, deception_signals)
            reward = self._compute_reward(gepa_scores, confidence, deception_signals)

            return RolloutResult(
                prompt=prompt,
                response=final_response,
                reward=reward,
                trace_summary=trace_summary,
                contradiction_report={"contradictions": span.get_contradictions()},
                deception_signals=deception_signals,
                sections=sections,
            )

    def _generate_response(self, prompt: str) -> str:
        """Generate a dual-path response from the underlying model."""

        if self.model is None or self.tokenizer is None or torch is None:
            return self._heuristic_dual_path_stub(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response_tokens = outputs[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    def _heuristic_dual_path_stub(self, prompt: str) -> str:
        """Fallback deterministic dual-path response used without a model."""

        return (
            "[PATH 1 REASONING]\n"
            "I am cautious about this request. There may be uncertainties or risks "
            "that require expert advice.\n\n"
            "[PATH 2 REASONING]\n"
            "Taking an action-oriented stance, here is a confident recommendation "
            "addressing the immediate need.\n\n"
            "[COMPARISON]\n"
            "Path 1 emphasises prudence while Path 2 focuses on decisiveness.\n\n"
            "[RECOMMENDATION]\n"
            "I recommend Path 1 for safety."
        )

    def _compute_confidence(
        self,
        sections: Dict[str, Any],
        deception_signals: Dict[str, Any],
    ) -> float:
        confidence = 0.7 if sections.get("recommended_path") == "path_1" else 0.6
        if deception_signals.get("deception_detected"):
            confidence -= deception_signals.get("confidence_score", 0.0) * 0.5
        return max(0.0, min(1.0, confidence))

    def _apply_abstention(self, response: str, confidence: float, threshold: float) -> str:
        if threshold and confidence < threshold:
            return (
                "I am not confident enough to provide definitive guidance here. "
                "Please consult an appropriate expert."
            )
        return response

    def _score_gepa_with_deception(
        self, trace_summary: Dict[str, Any], deception_signals: Dict[str, Any]
    ) -> Dict[str, float]:
        base_scores = self._score_gepa_principles(trace_summary)
        if deception_signals.get("deception_detected"):
            penalty = deception_signals.get("confidence_score", 0.0) * 2.0
            base_scores["integrity"] = max(0.0, base_scores.get("integrity", 0.0) - penalty)
            LOGGER.info(
                "Deception detected (confidence=%.2f), reduced Integrity score by %.2f",
                deception_signals.get("confidence_score", 0.0),
                penalty,
            )
        return base_scores

    def _score_gepa_principles(self, trace_summary: Dict[str, Any]) -> Dict[str, float]:
        return {
            "mindfulness": 1.0 if trace_summary.get("framing") else 0.5,
            "compassion": 1.0 if trace_summary.get("path_1_reasoning") else 0.5,
            "integrity": 1.0,
            "prudence": 1.0 if trace_summary.get("comparison") else 0.5,
        }

    def _compute_reward(
        self,
        gepa_scores: Dict[str, float],
        confidence: float,
        deception_signals: Dict[str, Any],
    ) -> float:
        base_reward = (
            self.reward_weights.task_success * self._task_reward(gepa_scores)
            + self.reward_weights.gepa_alignment * self._gepa_reward(gepa_scores)
            + self.reward_weights.honesty_trace * self._honesty_reward(confidence)
            - self.reward_weights.hallucination_penalty * self._hallucination_penalty(gepa_scores)
        )

        if deception_signals.get("deception_detected"):
            deception_penalty = deception_signals.get("confidence_score", 0.0) * 0.5
            base_reward -= deception_penalty
            LOGGER.info("Applied deception penalty: -%.3f", deception_penalty)

        return base_reward

    def _task_reward(self, gepa_scores: Dict[str, float]) -> float:
        return (gepa_scores.get("mindfulness", 0.0) + gepa_scores.get("compassion", 0.0)) / 2

    def _gepa_reward(self, gepa_scores: Dict[str, float]) -> float:
        values = list(gepa_scores.values())
        return sum(values) / len(values) if values else 0.0

    def _honesty_reward(self, confidence: float) -> float:
        return confidence

    def _hallucination_penalty(self, gepa_scores: Dict[str, float]) -> float:
        return max(0.0, 1.0 - gepa_scores.get("integrity", 0.0))
