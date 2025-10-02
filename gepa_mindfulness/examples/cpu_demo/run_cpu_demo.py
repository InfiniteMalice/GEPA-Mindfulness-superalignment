"""Minimal CPU-only demonstration of the training pipeline."""
from __future__ import annotations

from pathlib import Path

from gepa_mindfulness.training.configs import load_training_config
from gepa_mindfulness.training.pipeline import TrainingOrchestrator


def _read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    config = load_training_config(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    dataset_path = Path(__file__).parent / "prompts.txt"
    orchestrator = TrainingOrchestrator(config=config)
    results = orchestrator.run(_read_prompts(dataset_path))
    for result in results:
        print("PROMPT:", result.prompt)
        print("RESPONSE:", result.response)
        print("REWARD:", result.reward)
        print("TRACE:", result.trace_summary)
        print("CONTRADICTIONS:", result.contradiction_report)
        print("-" * 80)


if __name__ == "__main__":
    main()
