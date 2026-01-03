"""DAPO hybrid trainer integrating GEPA mindfulness scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from gepa_mindfulness.core.rewards import GEPARewardCalculator, HallucinationConfig, RewardWeights

PromptMetadata = Mapping[str, object]
GEPAComponentFn = Callable[[str, str], Mapping[str, float]]

LOGGER = logging.getLogger(__name__)

try:
    _GEPA_POLICY_SPEC = importlib.util.find_spec("gepa_dapo_grn.policy_interfaces")
except ModuleNotFoundError:
    _GEPA_POLICY_SPEC = None

if _GEPA_POLICY_SPEC is not None:
    from gepa_dapo_grn.policy_interfaces import Policy
else:

    class Policy:  # type: ignore[no-redef]
        """Fallback Policy base when gepa_dapo_grn is unavailable."""

        pass


if TYPE_CHECKING:
    from gepa_dapo_grn import (
        CurriculumTracker,
        DAPOTrainer,
        GEPAFeedback,
        GRNConfig,
        SafetyController,
    )


@dataclass
class GEPAFeedbackConfig:
    """Mapping configuration for GEPA reward and tag dimensions."""

    reward_dim_map: dict[str, str] = field(default_factory=dict)
    tag_dim_map: dict[str, str] = field(default_factory=dict)
    default_confidence: float = 0.6
    trace_summary_chars: int = 200


@dataclass
class DAPOHybridConfig:
    """Configuration for running the GEPA-guided DAPO hybrid trainer."""

    model_name: str
    output_dir: Path
    batch_size: int
    max_steps: int
    eval_interval: int
    learning_rate: float
    max_new_tokens: int
    temperature: float
    reward_mixer_weights: dict[str, float]
    gepa_feedback: GEPAFeedbackConfig
    use_grn: bool
    grn_location: str


@dataclass
class PromptExample:
    """Single prompt example with optional reference and metadata."""

    prompt: str
    reference: str | list[str] | None
    meta: PromptMetadata


class JSONLLogger:
    """Append structured training records for downstream GEPA analysis."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, object]) -> None:
        """Append a JSON record to the log file."""

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _require_gepa_dapo_grn() -> Mapping[str, Any]:
    try:
        from gepa_dapo_grn import (  # type: ignore[import-not-found]
            CurriculumTracker,
            DAPOConfig,
            DAPOTrainer,
            GEPAFeedback,
            GRNConfig,
            RewardMixerConfig,
            SafetyController,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("gepa_dapo_grn is required for DAPO hybrid training.") from exc
    return {
        "CurriculumTracker": CurriculumTracker,
        "DAPOConfig": DAPOConfig,
        "DAPOTrainer": DAPOTrainer,
        "GEPAFeedback": GEPAFeedback,
        "GRNConfig": GRNConfig,
        "RewardMixerConfig": RewardMixerConfig,
        "SafetyController": SafetyController,
    }


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    last_period = snippet.rfind(".")
    if last_period > 20:
        return snippet[: last_period + 1]
    return snippet


def _infer_prompt_len(
    *,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    full_ids: Sequence[int],
) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    eos_id = tokenizer.eos_token_id
    if prompt_ids and eos_id is not None and prompt_ids[-1] == eos_id:
        trimmed = prompt_ids[:-1]
        if full_ids[: len(trimmed)] == trimmed:
            return len(trimmed)
    bos_id = tokenizer.bos_token_id
    if prompt_ids and bos_id is not None and prompt_ids[0] == bos_id:
        trimmed = prompt_ids[1:]
        if full_ids[: len(trimmed)] == trimmed:
            return len(trimmed)
    prompt_ids_no_special = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids_no_special)] == prompt_ids_no_special:
        return len(prompt_ids_no_special)
    LOGGER.warning("Prompt tokenization mismatch; using fallback prompt length.")
    return len(prompt_ids_no_special)


class HfPolicyAdapter(Policy):
    """Wrap a HuggingFace causal LM in the DAPO policy interface."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Generate responses for prompts using sampling."""

        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def log_probs(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> list[list[float]]:
        """Compute token log probabilities for each completion."""

        log_prob_batches: list[list[float]] = []
        for prompt, completion in zip(prompts, completions):
            full_text = prompt + completion
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=self.tokenizer.is_fast,
                return_special_tokens_mask=self.tokenizer.is_fast,
            ).to(self.device)
            if self.tokenizer.is_fast:
                offsets = encoded.pop("offset_mapping")[0]
                special_mask = encoded.pop("special_tokens_mask")[0]
                prompt_len = sum(
                    1
                    for (start, end), special in zip(offsets, special_mask)
                    if end <= len(prompt) and not special
                )
            else:
                prompt_len = _infer_prompt_len(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    full_ids=encoded.input_ids[0].tolist(),
                )
            with torch.no_grad():
                logits = self.model(**encoded).logits[0]
            response_ids = encoded.input_ids[0, prompt_len:]
            if response_ids.numel() == 0:
                log_prob_batches.append([])
                continue
            positions = torch.arange(response_ids.shape[0], device=self.device) + prompt_len - 1
            positions = positions.clamp(min=0, max=logits.shape[0] - 1)
            token_logits = logits[positions]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            indices = torch.arange(response_ids.shape[0], device=self.device)
            gathered = log_probs[indices, response_ids]
            log_prob_batches.append([float(value) for value in gathered])
        return log_prob_batches


def default_reward_calculator() -> GEPARewardCalculator:
    """Return a default GEPA reward calculator with standard weights."""

    weights = RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
    hallucination = HallucinationConfig(
        confidence_threshold=0.75,
        confident_wrong_penalty=-2.0,
        uncertain_wrong_penalty=-0.5,
        appropriate_abstention_reward=0.5,
        lazy_abstention_penalty=-0.2,
    )
    return GEPARewardCalculator(weights=weights, hallucination=hallucination)


def default_gepa_components(prompt: str, completion: str) -> Mapping[str, float]:
    """Score GEPA components using exposed mindfulness scoring utilities."""

    from gepa_mindfulness.core.thought_alignment import compute_epistemic_score, compute_match_score

    trace, answer = _split_completion(completion)
    match_score = compute_match_score(trace=trace, answer=answer, context=prompt)
    epistemic = compute_epistemic_score(trace)
    aggregate = (match_score + epistemic) / 2.0
    return {
        "mindfulness": aggregate,
        "empathy": match_score,
        "perspective": epistemic,
        "agency": aggregate,
    }


def _split_completion(completion: str) -> tuple[str, str]:
    """Split a completion into reasoning trace and final answer."""

    markers = ["Final:", "Answer:"]
    for marker in markers:
        if marker in completion:
            prefix, suffix = completion.rsplit(marker, maxsplit=1)
            trace = prefix.strip()
            answer = suffix.strip()
            if trace and answer:
                return trace, answer
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return "", lines[0]
    return "\n".join(lines[:-1]), lines[-1]


def _build_gepa_feedback(
    feedback_cls: type[Any],
    reward_dimensions: Mapping[str, float],
    tag_dimensions: Mapping[str, float],
    meta: Mapping[str, object],
    abstained: bool,
) -> Any:
    params = signature(feedback_cls).parameters
    payload: dict[str, object] = {}
    field_map: dict[str, object] = {
        "reward_dimensions": reward_dimensions,
        "reward_dims": reward_dimensions,
        "rewards": reward_dimensions,
        "reward_vector": reward_dimensions,
        "reward_components": reward_dimensions,
        "tag_dimensions": tag_dimensions,
        "tag_dims": tag_dimensions,
        "tags": tag_dimensions,
        "tag_vector": tag_dimensions,
        "tag_components": tag_dimensions,
        "meta": meta,
        "metadata": meta,
        "info": meta,
        "abstained": abstained,
        "abstention": abstained,
        "did_abstain": abstained,
    }
    for name in params:
        if name in field_map:
            payload[name] = field_map[name]
    required = [
        name
        for name, param in params.items()
        if param.default is param.empty and name not in payload
    ]
    if required:
        raise TypeError(
            "GEPAFeedback missing required params: "
            f"{required}. Available keys: {sorted(payload.keys())}"
        )
    elif not payload:
        LOGGER.warning("GEPAFeedback signature not recognized; payload is empty.")
    return feedback_cls(**payload)


def score_with_gepa(
    prompt: str,
    completion: str,
    *,
    reward_calculator: GEPARewardCalculator,
    component_scorer: GEPAComponentFn,
    feedback_config: GEPAFeedbackConfig,
    meta: PromptMetadata,
    reference_answers: Sequence[str] | str | None = None,
) -> Any:
    """Compute GEPA feedback for a single prompt/response pair.

    Args:
        prompt: Prompt string given to the policy.
        completion: Generated response from the policy.
        reward_calculator: GEPA reward calculator used to aggregate GEPA signals.
        component_scorer: Callable that produces GEPA component scores.
        feedback_config: Mapping from GEPA outputs into reward/tag vectors.
        meta: Metadata for downstream logging and analytics.
        reference_answers: Optional reference answers for correctness scoring.

    Returns:
        GEPAFeedback populated with reward and tag dimensions for DAPO training.
    """

    gepa_scores = component_scorer(prompt, completion)
    trace_summary = {"reflection": _truncate_text(completion, feedback_config.trace_summary_chars)}
    breakdown = reward_calculator.compute_reward(
        response=completion,
        reference_answers=reference_answers,
        gepa_scores=gepa_scores,
        imperatives=None,
        confidence=feedback_config.default_confidence,
        trace_summary=trace_summary,
    )
    reward_dimensions = {
        reward_key: float(gepa_scores.get(source_key, 0.0))
        for reward_key, source_key in feedback_config.reward_dim_map.items()
    }
    tag_dimensions = {
        tag_key: float(gepa_scores.get(source_key, 0.0))
        for tag_key, source_key in feedback_config.tag_dim_map.items()
    }
    reward_dimensions["gepa_total"] = breakdown.total
    meta_payload = dict(meta)
    meta_payload["gepa_components"] = dict(gepa_scores)
    meta_payload["reward_breakdown"] = {
        "task_success": breakdown.task_success,
        "gepa_alignment": breakdown.gepa_alignment,
        "honesty": breakdown.honesty,
        "hallucination": breakdown.hallucination,
        "total": breakdown.total,
    }
    classes = _require_gepa_dapo_grn()
    return _build_gepa_feedback(
        classes["GEPAFeedback"],
        reward_dimensions=reward_dimensions,
        tag_dimensions=tag_dimensions,
        meta=meta_payload,
        abstained=breakdown.abstention_quality is not None,
    )


def _batch_iter(data: Sequence[PromptExample], batch_size: int) -> Iterable[list[PromptExample]]:
    for index in range(0, len(data), batch_size):
        yield list(data[index : index + batch_size])


def _create_config(cls: type[Any], payload: Mapping[str, object]) -> Any:
    params = signature(cls).parameters
    kwargs = {key: value for key, value in payload.items() if key in params}
    return cls(**kwargs)


def _create_dapo_trainer(
    *,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    dapo_config: Any,
    grn_config: Any,
    reward_mixer: Any,
    curriculum_tracker: Any,
    safety_controller: Any,
) -> Any:
    classes = _require_gepa_dapo_grn()
    params = signature(classes["DAPOTrainer"]).parameters
    kwargs: dict[str, object] = {}
    mapping = {
        "policy": policy,
        "optimizer": optimizer,
        "dapo_config": dapo_config,
        "grn_config": grn_config,
        "reward_mixer_config": reward_mixer,
        "reward_mixer": reward_mixer,
        "curriculum_tracker": curriculum_tracker,
        "safety_controller": safety_controller,
    }
    for name, value in mapping.items():
        if name in params:
            kwargs[name] = value
    return classes["DAPOTrainer"](**kwargs)


def _resolve_weights(
    curriculum_tracker: CurriculumTracker,
    prompts: Sequence[str],
    feedbacks: Sequence[GEPAFeedback],
) -> list[float]:
    if hasattr(curriculum_tracker, "get_sample_weights"):
        return list(curriculum_tracker.get_sample_weights(prompts, feedbacks))
    if hasattr(curriculum_tracker, "weights_for_prompts"):
        return list(curriculum_tracker.weights_for_prompts(prompts))
    return [1.0 for _ in prompts]


def _update_safety(
    safety_controller: SafetyController,
    feedbacks: Sequence[GEPAFeedback],
    trainer: DAPOTrainer,
) -> Mapping[str, float]:
    if hasattr(safety_controller, "update"):
        safety_controller.update(feedbacks)
    if hasattr(safety_controller, "apply_to_trainer"):
        safety_controller.apply_to_trainer(trainer)
    if hasattr(safety_controller, "metrics"):
        return dict(safety_controller.metrics)
    return {}


def _apply_grn(
    policy: Policy,
    grn_config: GRNConfig,
    feedbacks: Sequence[GEPAFeedback],
    trainer: DAPOTrainer,
) -> None:
    if hasattr(policy, "apply_grn"):
        policy.apply_grn(grn_config, feedbacks)
    elif hasattr(trainer, "apply_grn"):
        trainer.apply_grn(grn_config, feedbacks)


def train_loop(
    *,
    config: DAPOHybridConfig,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    dataset: Sequence[PromptExample],
    eval_dataset: Sequence[PromptExample],
    reward_calculator: GEPARewardCalculator,
    component_scorer: GEPAComponentFn,
    curriculum_tracker: CurriculumTracker,
    safety_controller: SafetyController,
    logger: JSONLLogger,
) -> None:
    """Run the DAPO hybrid training loop with GEPA feedback."""

    classes = _require_gepa_dapo_grn()
    dapo_config = _create_config(
        classes["DAPOConfig"],
        {
            "clip_ratio": 0.2,
            "kl_target": 0.01,
            "batch_size": config.batch_size,
        },
    )
    grn_config = _create_config(
        classes["GRNConfig"],
        {
            "enabled": config.use_grn,
            "location": config.grn_location,
        },
    )
    reward_mixer = _create_config(
        classes["RewardMixerConfig"],
        {"weights": config.reward_mixer_weights},
    )
    trainer = _create_dapo_trainer(
        policy=policy,
        optimizer=optimizer,
        dapo_config=dapo_config,
        grn_config=grn_config,
        reward_mixer=reward_mixer,
        curriculum_tracker=curriculum_tracker,
        safety_controller=safety_controller,
    )

    step = 0
    for batch in _batch_iter(dataset, config.batch_size):
        if step >= config.max_steps:
            break
        prompts = [item.prompt for item in batch]
        references = [item.reference for item in batch]
        completions = policy.generate(
            prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        feedbacks = [
            score_with_gepa(
                prompt,
                completion,
                reward_calculator=reward_calculator,
                component_scorer=component_scorer,
                feedback_config=config.gepa_feedback,
                meta=item.meta,
                reference_answers=reference,
            )
            for prompt, completion, reference, item in zip(prompts, completions, references, batch)
        ]
        sample_weights = _resolve_weights(curriculum_tracker, prompts, feedbacks)
        safety_metrics = _update_safety(safety_controller, feedbacks, trainer)
        if config.use_grn:
            _apply_grn(policy, grn_config, feedbacks, trainer)
        train_payload = {
            "prompts": prompts,
            "completions": completions,
            "feedbacks": feedbacks,
            "sample_weights": sample_weights,
        }
        metrics = trainer.train_step(train_payload)
        logger.log(
            {
                "step": step,
                "record_type": "batch",
                "metrics": metrics,
                "safety": safety_metrics,
            }
        )
        for prompt, completion, feedback, weight in zip(
            prompts, completions, feedbacks, sample_weights
        ):
            feedback_meta = getattr(feedback, "meta", None)
            reward_total = None
            if isinstance(feedback_meta, Mapping):
                breakdown = feedback_meta.get("reward_breakdown")
                if isinstance(breakdown, Mapping):
                    reward_total = breakdown.get("total")
            record = {
                "step": step,
                "record_type": "sample",
                "prompt": prompt,
                "completion": completion,
                "sample_weight": weight,
                "reward_total": reward_total,
                "reward_dimensions": getattr(feedback, "reward_dimensions", None),
                "tag_dimensions": getattr(feedback, "tag_dimensions", None),
                "meta": feedback_meta,
            }
            logger.log(record)
        if (step + 1) % config.eval_interval == 0:
            _run_eval(
                config=config,
                policy=policy,
                dataset=eval_dataset,
                reward_calculator=reward_calculator,
                component_scorer=component_scorer,
                logger=logger,
            )
        step += 1


def _run_eval(
    *,
    config: DAPOHybridConfig,
    policy: Policy,
    dataset: Sequence[PromptExample],
    reward_calculator: GEPARewardCalculator,
    component_scorer: GEPAComponentFn,
    logger: JSONLLogger,
) -> None:
    """Evaluate the policy on a held-out dataset and log GEPA feedback."""

    for batch in _batch_iter(dataset, config.batch_size):
        prompts = [item.prompt for item in batch]
        references = [item.reference for item in batch]
        completions = policy.generate(
            prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        feedbacks = [
            score_with_gepa(
                prompt,
                completion,
                reward_calculator=reward_calculator,
                component_scorer=component_scorer,
                feedback_config=config.gepa_feedback,
                meta={"split": "eval", **item.meta},
                reference_answers=reference,
            )
            for prompt, completion, reference, item in zip(prompts, completions, references, batch)
        ]
        for prompt, completion, feedback in zip(prompts, completions, feedbacks):
            feedback_meta = getattr(feedback, "meta", None)
            reward_total = None
            if isinstance(feedback_meta, Mapping):
                breakdown = feedback_meta.get("reward_breakdown")
                if isinstance(breakdown, Mapping):
                    reward_total = breakdown.get("total")
            record = {
                "step": "eval",
                "record_type": "sample",
                "prompt": prompt,
                "completion": completion,
                "reward_total": reward_total,
                "reward_dimensions": getattr(feedback, "reward_dimensions", None),
                "tag_dimensions": getattr(feedback, "tag_dimensions", None),
                "meta": feedback_meta,
            }
            logger.log(record)


def _load_jsonl_dataset(path: Path) -> list[PromptExample]:
    """Load a JSONL dataset containing prompts and optional references."""

    examples: list[PromptExample] = []
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt = str(payload.get("prompt", ""))
        reference = payload.get("reference")
        meta = payload.get("meta", {})
        examples.append(PromptExample(prompt=prompt, reference=reference, meta=meta))
    return examples


def _build_default_feedback_config() -> GEPAFeedbackConfig:
    return GEPAFeedbackConfig(
        reward_dim_map={
            "mindfulness": "mindfulness",
            "empathy": "empathy",
            "perspective": "perspective",
            "agency": "agency",
        },
        tag_dim_map={
            "mindfulness": "mindfulness",
            "epistemic": "perspective",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEPA-guided DAPO hybrid trainer")
    parser.add_argument("--model", default="gpt2", help="Base policy model name")
    parser.add_argument("--dataset", default="", help="Path to JSONL dataset")
    parser.add_argument("--eval-dataset", default="", help="Path to eval JSONL dataset")
    parser.add_argument("--output-dir", default="runs/dapo_hybrid", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="RL batch size")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--eval-interval", type=int, default=5, help="Eval frequency")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens per rollout")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temp")
    parser.add_argument("--use-grn", action="store_true", help="Enable GRN modulation")
    parser.add_argument("--grn-location", default="attention", help="GRN insertion site")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for policy execution",
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default="{}",
        help="JSON mapping of reward mixer weights",
    )
    return parser.parse_args()


def _parse_reward_weights(payload: str) -> dict[str, float]:
    if not payload:
        return {}
    try:
        data = json.loads(payload)
        return {str(key): float(value) for key, value in data.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid reward weights JSON: {exc}") from exc


def _default_examples() -> list[PromptExample]:
    return [
        PromptExample(
            prompt="Explain why mindful breathing helps with focus.",
            reference=None,
            meta={"task_id": "demo_1"},
        ),
        PromptExample(
            prompt="What is a safe way to pause when unsure?",
            reference=None,
            meta={"task_id": "demo_2"},
        ),
    ]


def run() -> None:
    """Entry point for running the DAPO hybrid trainer."""

    args = _parse_args()
    reward_mixer_weights = _parse_reward_weights(args.reward_weights)
    output_dir = Path(args.output_dir)
    dataset = _load_jsonl_dataset(Path(args.dataset)) or _default_examples()
    eval_dataset = _load_jsonl_dataset(Path(args.eval_dataset)) or dataset[:1]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    policy = HfPolicyAdapter(model, tokenizer, device=args.device)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.learning_rate)

    config = DAPOHybridConfig(
        model_name=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reward_mixer_weights=reward_mixer_weights,
        gepa_feedback=_build_default_feedback_config(),
        use_grn=bool(args.use_grn),
        grn_location=args.grn_location,
    )

    reward_calculator = default_reward_calculator()
    classes = _require_gepa_dapo_grn()
    curriculum_tracker = classes["CurriculumTracker"]()
    safety_controller = classes["SafetyController"]()
    logger = JSONLLogger(output_dir / "metrics.jsonl")

    train_loop(
        config=config,
        policy=policy,
        optimizer=optimizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        reward_calculator=reward_calculator,
        component_scorer=default_gepa_components,
        curriculum_tracker=curriculum_tracker,
        safety_controller=safety_controller,
        logger=logger,
    )


if __name__ == "__main__":
    run()
