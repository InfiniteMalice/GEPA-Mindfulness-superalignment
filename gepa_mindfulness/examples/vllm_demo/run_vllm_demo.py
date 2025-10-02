"""vLLM targeted demonstration using preconfigured engine endpoint."""
from __future__ import annotations

from pathlib import Path

from gepa_mindfulness.training.configs import load_training_config
from gepa_mindfulness.training.pipeline import TrainingOrchestrator

SAMPLE_PROMPTS = [
    "Audit the model's reasoning for covert misalignment.",
    "Explain how to remain honest when uncertain about facts.",
]


def main() -> None:
    config = load_training_config(Path(__file__).resolve().parents[2] / "configs" / "vllm.yaml")
    orchestrator = TrainingOrchestrator(config=config)
    results = orchestrator.run(SAMPLE_PROMPTS)
    for result in results:
        print({
            "prompt": result.prompt,
            "response": result.response,
            "reward": result.reward,
            "trace": result.trace_summary,
            "contradictions": result.contradiction_report,
        })


if __name__ == "__main__":
    main()
