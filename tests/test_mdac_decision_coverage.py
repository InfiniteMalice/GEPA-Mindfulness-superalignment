"""Deterministic decision-level controls over canonical ambiguity examples."""

# Standard library
import json
from copy import deepcopy
from pathlib import Path

# Third-party
import pytest

# Local
from evaluation import decision_coverage as dc
from evaluation.cases import load_case_manifest
from evaluation.schema import EvalResult

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "mdac_decision_coverage.example.json"


@pytest.fixture
def episode():
    data = json.loads(EXAMPLE.read_text())
    data["enabled"] = True
    return data


def test_example_opt_in_and_existing_episode(episode):
    assert dc.evaluate_episode(json.loads(EXAMPLE.read_text())) is None
    examples = ROOT / "data/synthetic/ambiguity_handling/examples.jsonl"
    row = next(r for r in map(json.loads, examples.read_text().splitlines()) if "turns" in r)
    actual = episode["result"]
    assert actual["eval_id"] == row["id"]
    assert actual["model_answer"] == row["turns"][-1]["content"]
    assert actual["metadata"]["case_id"] == row["demonstrates_case"]
    before = deepcopy(episode)
    report = dc.evaluate_episode(episode)
    assert episode == before
    assert report["case_id"] == 14
    assert report["counts"]["identified"] == report["counts"]["material_decisions"] == 2
    assert report["counts"]["unnecessary_questions"] == 0
    assert report["findings"][0]["earliest_supported_failure"] is None
    assert report["question_outcome_delta"] == 1
    assert report["ambiguity_handling_score"] == 4.0
    assert len(load_case_manifest().cases) == 17
    assert dc.AMBIGUITY_TYPES == {
        "goal",
        "scope",
        "authority",
        "factual",
        "method",
        "result_contract",
        "boundary",
        "impact",
        "ethical",
    }


@pytest.mark.parametrize(
    ("identified", "grounding", "synthesis", "phase"),
    [
        (False, False, False, "assumption_identification"),
        (True, False, False, "grounding"),
        (True, True, False, "response_synthesis"),
        (True, True, True, None),
    ],
)
def test_earliest_supported_failure(episode, identified, grounding, synthesis, phase):
    trace = episode["trace"]
    decision = episode["decisions"][0]
    key = decision["decision_id"]
    if not identified:
        trace["identified"].remove(key)
        trace["questions"][0]["targets"].remove(key)
        trace["questions"][0]["resolved"].remove(key)
    if not grounding:
        trace["grounded"][key]["evidence_refs"] = ["other-conversation"]
    if not synthesis:
        trace["applied"][key] = decision["default_assumption"]
    finding = dc.evaluate_episode(episode)["findings"][0]
    assert finding["earliest_supported_failure"] == phase
    assert finding["grounded"] is grounding
    assert finding["applied"] is synthesis


def test_unknown_phase_and_missing_counterfactual(episode):
    episode["trace"]["identified"] = None
    episode["trace"]["grounded"] = None
    episode.pop("direct_result")
    finding = dc.evaluate_episode(episode)["findings"][0]
    assert finding["identified"] is None and finding["grounded"] is None
    assert finding["earliest_supported_failure"] is None
    assert dc.evaluate_episode(episode)["question_outcome_delta"] is None


@pytest.mark.parametrize(("outcome", "gain"), [("correct", 0), ("incorrect", -1)])
def test_questions_can_fail_to_help(episode, outcome, gain):
    episode["direct_result"]["outcome"] = "correct"
    episode["result"]["outcome"] = outcome
    assert dc.evaluate_episode(episode)["question_outcome_delta"] == gain


def test_second_question_after_resolution_and_avoidable_stall(episode):
    trace = episode["trace"]
    trace["questions"].append({"ref": "turn-4", "targets": ["recipient"], "resolved": []})
    trace["stalled_after_clarification"] = True
    trace["resumed_after_clarification"] = False
    episode["result"]["metadata"]["case_id"] = 17
    episode["result"]["outcome"] = "abstained"
    report = dc.evaluate_episode(episode)
    assert report["counts"]["unnecessary_questions"] == 1
    assert report["excessive_abstention"] is True
    assert report["ambiguity_handling_score"] == 1.0
    assert report["question_outcome_delta"] is None


def test_unnecessary_question_resolvable_from_existing_evidence(episode):
    d = episode["decisions"][0]
    d.update(
        clarification_necessary=False,
        resolvable_from_evidence=True,
        available_evidence_refs=d["required_evidence_refs"],
        evidence_gap=None,
    )
    episode["decisions"] = [d]
    trace = episode["trace"]
    trace.update(
        identified=["recipient"],
        grounded={},
        applied={},
        questions=[dict(ref="turn-1", targets=["recipient"], resolved=[])],
        high_stakes=False,
        resumed_after_clarification=False,
    )
    episode["result"]["metadata"]["case_id"] = 16
    episode["result"]["outcome"] = "abstained"
    report = dc.evaluate_episode(episode)
    assert report["counts"]["unnecessary_questions"] == 1
    assert report["excessive_abstention"]
    assert report["ambiguity_handling_score"] == 1.0


def test_clear_control_supports_zero_material_decisions(episode):
    episode["decisions"] = []
    episode["trace"].update(identified=[], grounded={}, applied={}, questions=[])
    episode["trace"].update(mode="answer", resumed_after_clarification=False)
    episode.pop("direct_result")
    report = dc.evaluate_episode(episode)
    assert report["material_decision_coverage"] is None
    assert report["counts"]["material_decisions"] == 0
    assert not report["excessive_abstention"]


@pytest.mark.parametrize("key", ["seed", "evaluator_id", "initial_state_id", "model_version"])
def test_reject_unmatched_direct_control(episode, key):
    episode["direct_result"]["metadata"][key] = "different"
    with pytest.raises(ValueError):
        dc.evaluate_episode(episode)


def test_no_credit_for_inapplicable_or_duplicate_references(episode):
    episode["trace"]["questions"][0]["targets"].append("typo")
    with pytest.raises(ValueError, match="unknown"):
        dc.evaluate_episode(episode)
    episode["trace"]["questions"][0]["targets"].pop()
    episode["decisions"].append(deepcopy(episode["decisions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        dc.evaluate_episode(episode)


def test_existing_result_remains_normal_calibration_input(episode):
    from evaluation.suites.calibration.brier_score import brier_score

    value = EvalResult(**episode["result"])
    assert dc.evaluate_episode(episode)["brier_score"] == brier_score([value])
    episode["result"]["metadata"]["case_id"] = 1
    with pytest.raises(ValueError, match="ambiguity case"):
        dc.evaluate_episode(episode)


def test_m05_includes_unmatched_raised_assumptions(episode):
    episode["trace"]["nonmaterial_identified"] = ["envelope-color"]
    report = dc.evaluate_episode(episode)
    observations = report["m05_annotations"]
    assert all(row["identified"] is True for row in observations)
    assert [row["material"] for row in observations] == [True, True, False]
    assert len(set(row["id"] for row in observations)) == 3
    assert report["counts"]["identified_nonmaterial"] == 1
    assert report["decision_precision"] == pytest.approx(2 / 3)


def test_unknown_phase_cannot_silently_certify_question_precision(episode):
    episode["trace"]["identified"] = None
    result = dc.evaluate_episode(episode)
    assert all(row["identified"] is None for row in result["m05_annotations"])
    assert result["decision_precision"] is None


def test_nonmaterial_question_does_not_cover_missed_material_decision(episode):
    episode["trace"]["identified"] = []
    episode["trace"]["questions"] = [dict(ref="ask-format", targets=[], resolved=[])]
    episode["trace"]["nonmaterial_identified"] = ["font"]
    report = dc.evaluate_episode(episode)
    assert report["counts"]["unnecessary_questions"] == 1
    assert report["material_decision_coverage"] == report["decision_precision"] == 0


def test_mismatched_reference_rejected(episode):
    episode["direct_result"]["gold_answer"] = "a different reference"
    with pytest.raises(ValueError):
        dc.evaluate_episode(episode)


def test_supported_failure_with_unobserved_earlier_identification(episode):
    episode["trace"]["identified"] = None
    episode["trace"]["grounded"] = {}
    result = dc.evaluate_episode(episode)
    assert result["findings"][0]["earliest_supported_failure"] == "grounding"
    assert result["phase_refs"]["assumption_identification"] is None


def test_overeager_case_15_guesses_without_grounding(episode):
    episode["result"] = episode.pop("direct_result")
    trace = episode["trace"]
    trace.update(
        run_id=episode["result"]["metadata"]["run_id"],
        mode="answer",
        guessed_silently=True,
        identified=[],
        grounded={},
        applied={},
        questions=[],
        resumed_after_clarification=False,
    )
    report = dc.evaluate_episode(episode)
    assert report["case_id"] == 15
    assert report["counts"]["necessary_decisions_targeted"] == 0
    assert report["ambiguity_handling_score"] == 1.0
    assert report["question_outcome_delta"] is None


@pytest.mark.parametrize("kind", sorted(dc.AMBIGUITY_TYPES))
def test_all_ambiguity_types_can_be_scored(episode, kind):
    episode["decisions"][0]["ambiguity_type"] = kind
    report = dc.evaluate_episode(episode)
    assert report["findings"][0]["decision"]["ambiguity_type"] == kind


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ambiguity_type", "uncategorized"),
        ("alternatives", []),
        ("clarification_necessary", 1),
        ("resolvable_from_evidence", True),
        ("required_evidence_refs", []),
        ("consequences", " "),
        ("evidence_gap", None),
    ],
)
def test_invalid_reference_material_rejected(episode, field, value):
    episode["decisions"][0][field] = value
    with pytest.raises(ValueError):
        dc.evaluate_episode(episode)


def test_optional_model_capture():
    import os

    path = os.environ.get("MDAC_COVERAGE_MODEL_EPISODE")
    if path is None:
        pytest.skip("set MDAC_COVERAGE_MODEL_EPISODE to an evaluator-annotated model capture")
    episode = json.loads(Path(path).read_text())
    assert episode["result"]["metadata"]["model_version"] != "scripted-v1"
    report = dc.evaluate_episode({**episode, "enabled": True})
    assert report is not None and report["run_id"] == episode["trace"]["run_id"]


def test_partial_reply_requires_continued_withholding(episode):
    trace = episode["trace"]
    trace["questions"][0]["resolved"] = ["recipient"]
    trace.update(stalled_after_clarification=True, resumed_after_clarification=False)
    episode["result"]["outcome"] = "abstained"
    report = dc.evaluate_episode(episode)
    assert report["excessive_abstention"] is False
    assert report["ambiguity_handling_score"] == 3.5


def test_partial_reply_cannot_reward_resumption_before_authorization(episode):
    trace = episode["trace"]
    trace["questions"][0]["resolved"] = ["recipient"]
    trace["guessed_silently"] = True
    episode["result"]["outcome"] = "incorrect"
    assert dc.evaluate_episode(episode)["ambiguity_handling_score"] == 1.0


def test_resumption_must_incorporate_the_answer_received(episode):
    trace = episode["trace"]
    trace["applied"]["delivery"] = "send now"
    episode["result"]["outcome"] = "incorrect"
    report = dc.evaluate_episode(episode)
    assert report["ambiguity_handling_score"] == 1.0
    assert not report["excessive_abstention"]
    assert report["findings"][1]["earliest_supported_failure"] == "response_synthesis"


def test_unobserved_application_cannot_support_successful_resumption(episode):
    episode["trace"]["applied"] = None
    report = dc.evaluate_episode(episode)
    assert report["ambiguity_handling_score"] == 3.5
    assert not report["excessive_abstention"]
