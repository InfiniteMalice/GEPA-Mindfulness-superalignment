"""Reviewed boundary examples retain provenance and separate actor/evaluator views."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from curriculum_fixtures import boundary_inputs
from evaluation.causal_records import FactorChange, canonical_json, content_digest
from synthetic_data.cumulative_curriculum import export_cumulative_curriculum
from synthetic_data.curriculum_records import curriculum_digest


def run(inputs=None, callback=lambda p: True):
    return export_cumulative_curriculum(
        **(inputs or boundary_inputs()), authenticate=callback, enabled=True
    )


def bind(inputs, index=0):
    reviews = list(inputs["reviews"])
    reviews[index] = replace(
        reviews[index], candidate_digest=curriculum_digest(inputs["candidates"][index])
    )
    inputs["reviews"] = tuple(reviews)
    return inputs


def test_balanced_bundle_requires_independent_reviews():
    report = run()
    assert report["coverage"]["status"] == "complete"
    assert len(report["examples"]) == 4
    inputs = boundary_inputs()
    inputs["reviews"] = inputs["reviews"][:-1]
    report = run(inputs)
    assert report["coverage"] == {"status": "incomplete", "missing": ["legitimate_refusal"]}
    assert len(report["examples"]) + len(report["excluded"]) == 4
    assert not run(callback=None)["examples"]


def test_actor_prompts_exclude_labels():
    inputs = boundary_inputs()
    candidate = inputs["candidates"][0]
    pair = replace(
        candidate.pair, before=replace(candidate.pair.before, expected_actions=("SECRET_ORACLE",))
    )
    inputs["candidates"] = (replace(candidate, pair=pair),) + inputs["candidates"][1:]
    report = run(bind(inputs))
    assert "SECRET_ORACLE" not in canonical_json(report["examples"][0]["actor"])
    assert "SECRET_ORACLE" in canonical_json(report["examples"][0]["evaluator"])


def test_compound_edits_remain_compound():
    inputs = boundary_inputs()
    candidate = inputs["candidates"][0]
    pair = candidate.pair
    pair = replace(
        pair,
        intervention_kind="compound",
        before=replace(pair.before, factors=pair.before.factors + (("wording", '"a"'),)),
        after=replace(pair.after, factors=pair.after.factors + (("wording", '"b"'),)),
        changes=pair.changes + (FactorChange("wording", '"a"', '"b"'),),
    )
    inputs["candidates"] = (replace(candidate, pair=pair),) + inputs["candidates"][1:]
    assert run(bind(inputs))["examples"][0]["evaluator"]["pair"]["intervention_kind"] == "compound"


def test_decisive_case_change_keeps_each_arm_identity():
    pair = run()["examples"][0]["evaluator"]["pair"]
    assert pair["before"]["case"]["case_id"] == 1
    assert pair["after"]["case"]["case_id"] == 14


def test_hidden_parent_excluded():
    inputs = boundary_inputs()
    candidate = inputs["candidates"][0]
    source = json.loads(candidate.source_json)
    source["ancestor"] = {"training_eligibility": "HIDDEN_EVAL"}
    source.pop("result_digest")
    source["result_digest"] = content_digest(source)
    inputs["candidates"] = (replace(candidate, source_json=canonical_json(source)),)
    inputs["reviews"] = inputs["reviews"][:1]
    report = run(bind(inputs))
    assert report["examples"] == []
    assert "HIDDEN_EVAL" in canonical_json(report)


def test_generated_failure_never_becomes_target_truth():
    report = run(callback=lambda p: 1)
    assert report["examples"] == []
    assert report["optimizer_input"] is False and report["confers_authority"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_id", "unknown"),
        ("row_list", "metric_rows"),
        ("source_json", '{"result_digest":"forged"}'),
    ],
)
def test_invalid_source_selection_rejected(field, value):
    inputs = boundary_inputs()
    inputs["candidates"] = (replace(inputs["candidates"][0], **{field: value}),)
    with pytest.raises(ValueError):
        run(inputs)


def test_input_order_preserves_sorted_examples():
    inputs = boundary_inputs()
    baseline = run(inputs)
    inputs["candidates"] = inputs["candidates"][::-1]
    inputs["reviews"] = inputs["reviews"][::-1]
    assert run(inputs)["examples"] == baseline["examples"]
