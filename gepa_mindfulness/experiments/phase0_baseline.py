"""Phase 0 baseline experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from gepa_mindfulness.core.rewards import (
    GEPARewardCalculator,
    HallucinationConfig,
    RewardWeights,
)
from gepa_mindfulness.evaluation.baseline_evaluator import (
    BaselineEvaluator,
    load_evaluation_dataset,
)
from gepa_mindfulness.evaluation.context_classifier import ContextClassifier
from gepa_mindfulness.experiments.correlation_analysis import CorrelationAnalyzer


def run_phase0_experiment(*, config_path: Path) -> None:
    """Run Phase 0 evaluation and correlation analysis."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    reward_cfg = config.get("rewards", {})
    weights = RewardWeights.from_mapping(reward_cfg.get("weights", {}))
    hallucination_payload = reward_cfg.get("hallucination", {})
    hallucination_cfg = HallucinationConfig(
        confidence_threshold=float(hallucination_payload.get("confidence_threshold", 0.75)),
        confident_wrong_penalty=float(hallucination_payload.get("confident_wrong_penalty", -1.0)),
        uncertain_wrong_penalty=float(hallucination_payload.get("uncertain_wrong_penalty", -0.5)),
        appropriate_abstention_reward=float(
            hallucination_payload.get("appropriate_abstention_reward", 0.5)
        ),
        lazy_abstention_penalty=float(hallucination_payload.get("lazy_abstention_penalty", -0.5)),
    )
    reward_calculator = GEPARewardCalculator(
        weights=weights,
        hallucination=hallucination_cfg,
        abstention_threshold=reward_cfg.get("abstention_threshold"),
    )

    context_classifier = ContextClassifier()
    output_dir = Path(
        config.get("output", {}).get("results_dir", "experiments/phase0_baselines/results")
    )
    evaluator = BaselineEvaluator(
        model=model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        context_classifier=context_classifier,
        extract_ags=config.get("model", {}).get("extract_ags", True),
        ag_method=config.get("model", {}).get("ag_method", "gradient_x_activation"),
        output_dir=output_dir,
    )

    datasets = config.get("datasets", [])
    for dataset_cfg in datasets:
        dataset_path = Path(dataset_cfg["path"])
        dataset_name = dataset_cfg.get("name", dataset_path.stem)
        examples = load_evaluation_dataset(
            dataset_path=dataset_path,
            context_classifier=context_classifier,
        )
        evaluator.evaluate_dataset(dataset=examples, dataset_name=dataset_name)

    analysis_dir = Path(
        config.get("output", {}).get("analysis_dir", "experiments/phase0_baselines/analysis")
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analyzer = CorrelationAnalyzer(results_dir=output_dir)
    heatmap_path = analysis_dir / "gepa_ag_correlation.png"
    report_path = analysis_dir / "gepa_ag_correlation.md"
    analyzer.visualize_correlations(output_path=heatmap_path)
    analyzer.generate_report(output_path=report_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 baseline experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase0_baseline.yaml"),
        help="Path to experiment configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_phase0_experiment(config_path=args.config)


if __name__ == "__main__":
    main()
