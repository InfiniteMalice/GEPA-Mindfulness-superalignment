"""Contracts for the authored reward-integrity curriculum source."""

# Standard library
import json
from pathlib import Path

# Local
from scripts.synthetic_dataset_tool import _validate_jsonl

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "reward_integrity"
    / "reward_integrity_curriculum_v1.jsonl"
)
GOLD_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "gold"
    / "superalignment_gold_v1.jsonl"
)
CENTRAL_DIAGNOSTIC = (
    "If the reward channel disappeared and reality alone judged the result, would this still "
    "count as success?"
)
EXPECTED_FAMILIES = {
    "animal_reinforcement_pathway",
    "human_substance_addiction",
    "benchmark_leakage",
    "evaluator_manipulation",
    "reward_channel_tampering",
    "benign_shortcut",
    "structured_skill_acquisition",
    "recovery_and_repair",
}
EXPECTED_IDS = {
    "reward-integrity-001",
    "reward-integrity-002",
    "reward-integrity-003",
    "reward-integrity-004",
    "reward-integrity-005",
    "reward-integrity-006",
    "reward-integrity-007",
    "reward-integrity-008",
}
RESPONSE_CLASSES = {
    "grounded_success",
    "benign_shortcut",
    "exploit_disclosure",
    "proxy_exploitation",
    "evaluator_manipulation",
    "reward_tampering",
    "honest_failure",
    "rationalized_failure",
}
PAIR_RULES = {
    ("grounded_success", "proxy_exploitation", "grounded_over_proxy"),
    ("exploit_disclosure", "proxy_exploitation", "disclosure_over_exploitation"),
    ("honest_failure", "evaluator_manipulation", "honest_failure_over_dishonesty"),
    ("benign_shortcut", "rationalized_failure", "creativity_over_rigidity"),
    ("exploit_disclosure", "reward_tampering", "repair_over_concealment"),
    ("grounded_success", "reward_tampering", "agency_over_immediate_reward"),
}
PRIVATE_SIGNAL_TERMS = ("hidden thoughts", "activations", "private scratchpads")


def _records() -> list[dict[str, object]]:
    """Read the checked-in source without using the builder under test."""
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    """Write a controlled rich-case fixture for validator boundary tests."""
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_curriculum_cases_validate_and_cover_the_eight_exact_families() -> None:
    """A missing or renamed family would make the intended curriculum coverage disappear."""
    records, errors = _validate_jsonl(DATASET_PATH)

    assert errors == []
    assert {record["id"] for record in records} == EXPECTED_IDS
    assert {record["case_metadata"]["scenario_family"] for record in records} == EXPECTED_FAMILIES


def test_cases_have_complete_observable_reward_integrity_contracts() -> None:
    """Removing a response class or an auditable negative-evidence link breaks pair safety."""
    for record in _records():
        integrity = record["reward_integrity"]
        assert integrity["central_diagnostic"] == CENTRAL_DIAGNOSTIC
        assert len(integrity["supporting_diagnostics"]) == 7
        assert len(integrity["repair_protocol"]) == 8
        assert set(integrity["response_classes"]) == RESPONSE_CLASSES
        assert {
            (item["chosen_class"], item["rejected_class"], item["pair_rule"])
            for item in integrity["preference_orderings"]
        } == PAIR_RULES

        for value in integrity["component_targets"].values():
            assert -1.0 <= value <= 1.0
        for response in integrity["response_classes"].values():
            assert response["observable_outcomes"]
            for component, value in response["reward_components"].items():
                assert -1.0 <= value <= 1.0, component
                if value < 0.0:
                    assert response["negative_evidence"][component]


def test_curriculum_uses_scientific_non_stigmatizing_and_safe_language() -> None:
    """Stigmatizing claims or operational tampering instructions would make the data unsafe."""
    by_family = {record["case_metadata"]["scenario_family"]: record for record in _records()}
    animal = json.dumps(by_family["animal_reinforcement_pathway"]).lower()
    addiction = json.dumps(by_family["human_substance_addiction"]).lower()
    tampering = json.dumps(by_family["reward_channel_tampering"]).lower()
    complete_text = json.dumps(_records()).lower()

    assert "all experimental animals self-stimulated until death" not in animal
    assert "dopamine is only a pleasure chemical" not in animal
    for term in ("motivation", "incentive salience", "learning", "reduced agency", "recovery"):
        assert term in addiction
    for term in ("moral defect", "choosing reward for nothing"):
        assert term not in addiction
    for term in PRIVATE_SIGNAL_TERMS:
        assert term not in complete_text
    for term in ("http://", "https://", "curl ", "wget ", "powershell", "credential"):
        assert term not in tampering


def test_benchmark_leakage_records_extension_provenance() -> None:
    """Dropping syn-gold-002 provenance would turn an extension into an untraceable copy."""
    benchmark = next(record for record in _records() if record["id"] == "reward-integrity-003")
    provenance = benchmark["reward_integrity"]["provenance"]

    assert provenance["extends_case_id"] == "syn-gold-002"
    assert "patch" in provenance["extension_summary"].lower()
    assert "rerun" in provenance["extension_summary"].lower()


def test_validator_rejects_unbounded_empty_or_unobservable_reward_contracts(tmp_path: Path) -> None:
    """Permissive components or unrelated evidence would permit ungrounded reward scoring."""
    records = _records()
    integrity = records[0]["reward_integrity"]
    integrity["component_targets"]["objective_fidelity"] = 1.1
    integrity["response_classes"]["grounded_success"]["reward_components"] = {}
    compromised = integrity["response_classes"]["proxy_exploitation"]
    compromised["negative_evidence"]["objective_fidelity"] = ["unrelated audit note"]
    path = tmp_path / "invalid.jsonl"
    _write_records(path, records)

    _, errors = _validate_jsonl(path)

    assert any("component_targets" in error and "[-1,1]" in error for error in errors)
    assert any("reward_components" in error and "complete" in error for error in errors)
    assert any("observable_outcomes" in error for error in errors)


def test_benchmark_leakage_preserves_the_gold_scenario_and_evaluation_constraints() -> None:
    """Replacing syn-gold-002 context would erase the leakage scenario being extended."""
    benchmark = next(record for record in _records() if record["id"] == "reward-integrity-003")
    gold = next(
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["id"] == "syn-gold-002"
    )

    assert benchmark["scenario"] == gold["scenario"]
    assert benchmark["case_metadata"]["contains_hidden_information"] is True
    assert benchmark["case_metadata"]["contains_time_pressure"] is True
    assert (
        benchmark["test_integrity"]["available_paths"] == gold["test_integrity"]["available_paths"]
    )
    reasoning = " ".join(benchmark["canonical_argument"]["reasoning_steps"]).lower()
    assert "patch" in reasoning
    assert "rerun" in reasoning


def test_cases_have_distinct_domain_responses_and_addiction_pairs_expose_recovery_mechanisms() -> (
    None
):
    """Generic response reuse would erase family-specific reward-integrity learning signals."""
    records = _records()
    response_sets = {
        json.dumps(record["reward_integrity"]["response_classes"], sort_keys=True)
        for record in records
    }
    addiction = next(
        record
        for record in records
        if record["case_metadata"]["scenario_family"] == "human_substance_addiction"
    )
    addiction_responses = " ".join(
        response["response"]
        for response in addiction["reward_integrity"]["response_classes"].values()
    ).lower()

    assert len(response_sets) == len(records)
    for term in ("motivation", "incentive salience", "habit", "reduced agency", "recovery"):
        assert term in addiction_responses


def test_negative_reward_evidence_is_concrete_and_scenario_specific() -> None:
    """Formulaic outcome labels would not substantiate a negative reward component."""
    for record in _records():
        responses = record["reward_integrity"]["response_classes"]
        for response in responses.values():
            for evidence in response["negative_evidence"].values():
                for fact in evidence:
                    assert "observable outcome for" not in fact.lower()
