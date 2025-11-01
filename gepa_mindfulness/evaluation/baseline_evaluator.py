"""Baseline evaluator orchestrating Phase 0 measurements."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - executed when torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from tqdm import tqdm

try:  # pragma: no cover - optional dependency
    from transformers import PreTrainedModel, PreTrainedTokenizer

    TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - executed when transformers missing
    PreTrainedModel = Any  # type: ignore[assignment]
    PreTrainedTokenizer = Any  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

from gepa_mindfulness.core.rewards import GEPARewardCalculator
from gepa_mindfulness.evaluation.context_classifier import ContextClassifier
from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionGraph,
    extract_attribution_graph,
)
from gepa_mindfulness.interpret.graph_metrics import compute_all_metrics


@dataclass
class EvaluationExample:
    """Single evaluation example with optional model response."""

    prompt: str
    response: str
    ground_truth: str | List[str] | None
    context_type: str
    metadata: Dict[str, object]


@dataclass
class EvaluationResult:
    """Aggregate metrics for a single evaluation example."""

    example_id: str
    context_type: str
    gepa_components: Dict[str, float]
    gepa_aggregate: float
    ag_metrics: Dict[str, float]
    hallucination_detected: bool
    abstention_detected: bool
    confidence: float
    attribution_graph: Optional[AttributionGraph]


class BaselineEvaluator:
    """Comprehensive evaluation pipeline for Phase 0 baselines."""

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_calculator: GEPARewardCalculator,
        context_classifier: ContextClassifier,
        extract_ags: bool = True,
        ag_method: str = "gradient_x_activation",
        output_dir: Path = Path("experiments/phase0_baselines/results"),
    ) -> None:
        """Initialise the evaluator with model, tokenizer, and helpers."""

        _require_dependency("torch", TORCH_AVAILABLE)
        _require_dependency("transformers", TRANSFORMERS_AVAILABLE)

        self.model = model
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.context_classifier = context_classifier
        self.extract_ags = extract_ags
        self.ag_method = ag_method
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(
        self,
        dataset: List[EvaluationExample],
        *,
        dataset_name: str,
    ) -> List[EvaluationResult]:
        """Evaluate a dataset and persist incremental results."""

        results: List[EvaluationResult] = []
        for index, example in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            example_id = f"{dataset_name}_{index}"
            result = self._evaluate_example(example=example, example_id=example_id)
            results.append(result)
            if (index + 1) % 10 == 0:
                self._save_results(results=results, dataset_name=dataset_name)
        self._save_results(results=results, dataset_name=dataset_name)
        return results

    def _evaluate_example(
        self,
        *,
        example: EvaluationExample,
        example_id: str,
    ) -> EvaluationResult:
        """Evaluate a single example and return metrics."""

        if not example.response:
            example.response = self._generate_response(prompt=example.prompt)

        confidence = self._compute_confidence(prompt=example.prompt, response=example.response)
        trace = self._get_trace(prompt=example.prompt, response=example.response)
        gepa_components = self._compute_gepa_components(trace=trace)
        trace_summary = self._build_trace_summary(trace=trace)
        reward = self.reward_calculator.compute_reward(
            response=example.response,
            reference_answers=example.ground_truth,
            gepa_scores=gepa_components,
            imperatives=None,
            confidence=confidence,
            trace_summary=trace_summary,
        )

        ag: Optional[AttributionGraph] = None
        ag_metrics: Dict[str, float] = {}
        if self.extract_ags:
            ag = extract_attribution_graph(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=example.prompt,
                response=example.response,
                method=self.ag_method,
                layers=[-3, -2, -1],
                threshold=0.01,
            )
            ag_metrics = compute_all_metrics(ag)

        hallucination = self._detect_hallucination(
            response=example.response,
            confidence=confidence,
            ground_truth=example.ground_truth,
        )
        abstention = self._detect_abstention(response=example.response)

        return EvaluationResult(
            example_id=example_id,
            context_type=example.context_type,
            gepa_components=gepa_components,
            gepa_aggregate=reward.total,
            ag_metrics=ag_metrics,
            hallucination_detected=hallucination,
            abstention_detected=abstention,
            confidence=confidence,
            attribution_graph=ag,
        )

    def _generate_response(self, *, prompt: str) -> str:
        """Generate a model response for ``prompt`` using sampling."""

        _require_dependency("torch", TORCH_AVAILABLE)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _compute_confidence(self, *, prompt: str, response: str) -> float:
        """Estimate average token probability for the ``response``."""

        _require_dependency("torch", TORCH_AVAILABLE)

        full_text = prompt + response
        encoded = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        if encoded.input_ids.shape[1] <= prompt_len:
            return 0.0

        with torch.no_grad():
            logits = self.model(**encoded).logits[0]

        response_ids = encoded.input_ids[0, prompt_len:]
        if response_ids.numel() == 0:
            return 0.0

        positions = torch.arange(response_ids.shape[0], device=self.model.device) + prompt_len - 1
        positions = positions.clamp(min=0, max=logits.shape[0] - 1)
        token_logits = logits[positions]
        probs = torch.softmax(token_logits, dim=-1)
        indices = torch.arange(response_ids.shape[0], device=self.model.device)
        gathered = probs[indices, response_ids]
        return float(gathered.mean().item())

    def _get_trace(self, *, prompt: str, response: str) -> Dict[str, float | List[str]]:
        """Return a mock trace used for compatibility with reward signals."""

        _ = (prompt, response)
        return {
            "mindfulness": 0.5,
            "empathy": 0.5,
            "perspective": 0.5,
            "agency": 0.5,
            "reduce_suffering": 0.5,
            "increase_prosperity": 0.5,
            "increase_knowledge": 0.5,
            "tensions": [],
        }

    @staticmethod
    def _compute_gepa_components(*, trace: Dict[str, float | List[str]]) -> Dict[str, float]:
        """Extract GEPA components from the trace payload."""

        keys = [
            "mindfulness",
            "empathy",
            "perspective",
            "agency",
            "reduce_suffering",
            "increase_prosperity",
            "increase_knowledge",
        ]
        components: Dict[str, float] = {}
        for key in keys:
            value = trace.get(key, 0.0)
            components[key] = float(value) if isinstance(value, (int, float)) else 0.0
        return components

    @staticmethod
    def _build_trace_summary(*, trace: Dict[str, float | List[str]]) -> Dict[str, str]:
        """Convert the trace payload into a summary for reward calculation."""

        tensions = trace.get("tensions", [])
        return {
            "evidence": "evidence" if trace else "",
            "tensions": "present" if tensions else "",
            "reflection": "reflection" if trace else "",
        }

    @staticmethod
    def _detect_hallucination(
        *,
        response: str,
        confidence: float,
        ground_truth: str | List[str] | None,
    ) -> bool:
        """Return ``True`` if the response contradicts provided ground truth."""

        if ground_truth is None:
            return False

        if isinstance(ground_truth, str):
            references = [ground_truth]
        else:
            references = [str(value) for value in ground_truth]
        response_norm = response.strip().lower()
        is_correct = any(response_norm == ref.strip().lower() for ref in references)
        return not is_correct and confidence > 0.75

    @staticmethod
    def _detect_abstention(*, response: str) -> bool:
        """Detect abstention-like responses based on simple heuristics."""

        abstention_markers = [
            "i don't know",
            "i cannot",
            "i can't",
            "i'm not sure",
            "uncertain",
        ]
        lower = response.lower()
        return any(marker in lower for marker in abstention_markers)

    def _save_results(
        self,
        *,
        results: List[EvaluationResult],
        dataset_name: str,
    ) -> None:
        """Persist results to disk along with attribution graph summaries."""

        results_file = self.output_dir / f"{dataset_name}_results.jsonl"
        with results_file.open("w", encoding="utf-8") as handle:
            for result in results:
                payload = asdict(result)
                payload.pop("attribution_graph", None)
                handle.write(json.dumps(payload) + "\n")

        if not self.extract_ags:
            return

        ag_dir = self.output_dir / "attribution_graphs" / dataset_name
        ag_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            graph = result.attribution_graph
            if graph is None:
                continue
            graph_file = ag_dir / f"{result.example_id}.json"
            self._save_attribution_graph(graph=graph, filepath=graph_file)

    @staticmethod
    def _save_attribution_graph(*, graph: AttributionGraph, filepath: Path) -> None:
        """Persist a lightweight summary of an attribution graph."""

        summary = {
            "prompt": graph.prompt,
            "response": graph.response,
            "method": graph.method,
            "metadata": graph.metadata,
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
        }
        with filepath.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


def load_evaluation_dataset(
    *,
    dataset_path: Path,
    context_classifier: ContextClassifier,
) -> List[EvaluationExample]:
    """Load a JSONL dataset and annotate it with context labels."""

    examples: List[EvaluationExample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload["prompt"]
            response = payload.get("response", "")
            ground_truth = payload.get("ground_truth")
            trace = payload.get("trace")
            classification = context_classifier.classify(prompt, trace)
            examples.append(
                EvaluationExample(
                    prompt=prompt,
                    response=response,
                    ground_truth=ground_truth,
                    context_type=classification["type"],
                    metadata={
                        "source": str(dataset_path),
                        "index": index,
                        "context_confidence": classification["confidence"],
                    },
                )
            )
    return examples


def _require_dependency(name: str, available: bool) -> None:
    """Raise an informative error when a required dependency is missing."""

    if not available:
        raise ModuleNotFoundError(
            f"{name} is required for baseline evaluation. "
            "Install the optional dependency to use this feature."
        )
