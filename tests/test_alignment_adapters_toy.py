import importlib
from pathlib import Path

import pytest

from evaluation.suites.common import DatasetUnavailableError, load_jsonl_cases, score_response

FIXTURES = Path(__file__).parent / "fixtures" / "alignment_battery"


@pytest.mark.parametrize(
    ("module_name", "fixture_name", "suite"),
    [
        ("evaluation.suites.factuality.simpleqa_adapter", "simpleqa_toy.jsonl", "simpleqa"),
        ("evaluation.suites.factuality.truthfulqa_adapter", "truthfulqa_toy.jsonl", "truthfulqa"),
        (
            "evaluation.suites.sycophancy.anthropic_sycophancy_adapter",
            "sycophancy_toy.jsonl",
            "anthropic_sycophancy",
        ),
        (
            "evaluation.suites.robustness.prompt_injection_hierarchy",
            "prompt_injection_toy.jsonl",
            "prompt_injection_hierarchy",
        ),
        ("evaluation.suites.privacy.pii_leakage_probe", "privacy_toy.jsonl", "pii_leakage"),
        (
            "evaluation.suites.deception.apollo_style_scheming",
            "deception_toy.jsonl",
            "apollo_scheming",
        ),
        ("evaluation.suites.bias_toxicity.bbq_adapter", "bias_toxicity_toy.jsonl", "bbq"),
    ],
)
def test_toy_adapters_load_local_fixtures(module_name, fixture_name, suite) -> None:
    module = importlib.import_module(module_name)

    cases = module.load_examples(str(FIXTURES / fixture_name), limit=1)

    assert len(cases) == 1
    assert cases[0].suite == suite
    assert cases[0].prompt


def test_load_jsonl_cases_respects_zero_limit() -> None:
    cases = load_jsonl_cases(
        FIXTURES / "simpleqa_toy.jsonl",
        suite="simpleqa",
        category="factuality",
        limit=0,
    )

    assert cases == []


def test_load_jsonl_cases_rejects_directories(tmp_path) -> None:
    with pytest.raises(DatasetUnavailableError, match="not a file"):
        load_jsonl_cases(tmp_path, suite="simpleqa", category="factuality")


def test_score_response_uses_model_answer_over_fixture_expected_outcome() -> None:
    cases = load_jsonl_cases(
        FIXTURES / "simpleqa_toy.jsonl",
        suite="simpleqa",
        category="factuality",
        limit=1,
    )

    result = score_response(cases[0], "Definitely not Paris")

    assert cases[0].metadata["expected_outcome"] == "correct"
    assert result.outcome == "incorrect"
