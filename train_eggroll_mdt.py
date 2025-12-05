"""Command-line entrypoint for EGGROLL + MDT training."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Mapping

from mindful_trace_gepa.train.eggroll_mdt_trainer import EGGROLLConfig, EGGROLLMDTTrainer
from mindful_trace_gepa.utils.imports import optional_import

LOGGER = logging.getLogger(__name__)

torch = optional_import("torch")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EGGROLL + MDT training")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--use-grn-confidence", action="store_true")
    parser.add_argument("--use-grn-probes", action="store_true")
    parser.add_argument("--use-grn-fitness", action="store_true")
    return parser


def dummy_eval_fn(model: Any, candidate_id: int) -> Mapping[str, Any]:
    # Placeholder evaluation to keep entrypoint self-contained. Real workloads should supply
    # a richer evaluation pipeline.
    return {
        "task_reward": 0.0,
        "ethics_score": 0.0,
        "deception_penalty": 0.0,
        "confidence_metric": 0.0,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if torch is None:
        raise ImportError("torch is required to run the EGGROLL + MDT entrypoint")
    config = EGGROLLConfig(
        generations=args.generations,
        population_size=args.population,
        rank=args.rank,
        geometry_weight=args.geometry_weight,
        use_grn_for_confidence=args.use_grn_confidence,
        use_grn_for_probes=args.use_grn_probes,
        use_grn_for_fitness=args.use_grn_fitness,
    )
    LOGGER.info("Initialising EGGROLL + MDT trainer with config: %s", config)
    model = torch.nn.Linear(2, 1)
    trainer = EGGROLLMDTTrainer(model=model, eval_fn=dummy_eval_fn, config=config)
    result = trainer.run()
    LOGGER.info("Training complete. Generations logged: %d", len(result["logs"]))


if __name__ == "__main__":
    main()
