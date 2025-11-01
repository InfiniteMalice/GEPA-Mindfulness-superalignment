"""Tests for the Phase 0 baseline evaluator."""

from __future__ import annotations

import json

from gepa_mindfulness.core.rewards import GEPARewardCalculator, HallucinationConfig, RewardWeights
from gepa_mindfulness.evaluation.baseline_evaluator import (
    BaselineEvaluator,
    EvaluationExample,
    load_evaluation_dataset,
)
from gepa_mindfulness.evaluation.context_classifier import ContextClassifier


def _reward_calculator() -> GEPARewardCalculator:
    weights = RewardWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    hallucination = HallucinationConfig(
        confidence_threshold=0.75,
        confident_wrong_penalty=-1.0,
        uncertain_wrong_penalty=-0.5,
        appropriate_abstention_reward=0.5,
        lazy_abstention_penalty=-0.5,
    )
    return GEPARewardCalculator(
        weights=weights,
        hallucination=hallucination,
        abstention_threshold=0.5,
    )


def test_evaluate_dataset_with_attribution_graph(tmp_path, tiny_model, dummy_tokenizer) -> None:
    reward_calculator = _reward_calculator()
    classifier = ContextClassifier()
    evaluator = BaselineEvaluator(
        model=tiny_model,
        tokenizer=dummy_tokenizer,
        reward_calculator=reward_calculator,
        context_classifier=classifier,
        extract_ags=True,
        ag_method="gradient_x_activation",
        output_dir=tmp_path,
    )
    example = EvaluationExample(
        prompt="hello world ",
        response="hello",
        ground_truth="hello",
        context_type="growth",
        metadata={},
    )
    results = evaluator.evaluate_dataset(dataset=[example], dataset_name="test")
    assert len(results) == 1
    result = results[0]
    assert result.attribution_graph is not None
    assert result.gepa_aggregate is not None
    assert (tmp_path / "test_results.jsonl").exists()


def test_load_evaluation_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    payload = {"prompt": "hello world", "response": "hello", "ground_truth": "hello"}
    dataset_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    classifier = ContextClassifier()
    examples = load_evaluation_dataset(dataset_path=dataset_path, context_classifier=classifier)
    assert len(examples) == 1
    assert examples[0].context_type in {"neutral", "growth", "emotional", "tension"}
