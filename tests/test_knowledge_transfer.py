"""Controlled saved-answer transfer tests, never evidence of native model learning."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_records import CostEntry, ResamplingPolicy, SystemConfig
from evaluation.knowledge_transfer import evaluate_knowledge_transfer, public_transfer_prompt
from evaluation.knowledge_transfer_records import (
    TransferCondition,
    TransferObservation,
    TransferProbeConfig,
    TransferUnit,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import require_training_eligible

REFS = (EvidenceReference("fixture:source", EvidenceSourceKind.EXTERNAL_RECORD),)
CFG = TransferProbeConfig(enabled=True)


def units():
    """Author independent fictional truths and open-ended public questions."""
    return tuple(
        TransferUnit(
            key,
            entity,
            "hue",
            cohort,
            value,
            f'LOOKUP("{entity}", "hue")',
            f"What is {entity}'s hue?",
            f"Return the dispatch direction for {entity} using the learned hue map.",
            application,
            REFS,
        )
        for key, entity, cohort, value, application in (
            ("a1", "Alven", "alignment", "cobalt", "north"),
            ("a2", "Brel", "alignment", "umber", "south"),
            ("t1", "Talvek", "test", "saffron", "east"),
            ("t2", "Drez", "test", "jade", "west"),
        )
    )


def conditions():
    """Declare a common-base control and a source-acquired condition without QA alignment."""
    common = content_digest("base-checkpoint")
    return tuple(
        TransferCondition(
            name,
            SystemConfig(f"model:{name}", "harness:v1", content_digest(name)),
            common,
            common if name == "baseline" else content_digest("candidate-checkpoint"),
            None if name == "baseline" else content_digest("acquisition-manifest"),
            () if name == "baseline" else ("a1", "a2", "t1", "t2"),
            None,
            (),
            "structured_text",
            "plain_text",
            REFS,
        )
        for name in ("baseline", "candidate")
    )


def observations(repeats=("r0", "r1")):
    """Source success and target/application access deliberately disagree."""
    result = []
    for condition in conditions():
        for repeat in repeats:
            for i, unit in enumerate(units()):
                for channel in ("source", "target", "application"):
                    answer = "wrong"
                    if condition.condition_id == "candidate":
                        if channel == "source" or (channel == "target" and i % 2 == 0):
                            answer = unit.value
                        if channel == "application" and i == 3:
                            answer = "west"
                    result.append(
                        TransferObservation(
                            condition.condition_id,
                            unit.unit_id,
                            repeat,
                            channel,
                            condition.system,
                            "answered",
                            answer,
                            REFS,
                        )
                    )
    return tuple(result)


def run(captures=None, *, unit_rows=None, condition_rows=None, repeats=("r0", "r1"), config=CFG):
    """Use an explicit fixed roster; absent captures never shrink its denominator."""
    return evaluate_knowledge_transfer(
        units() if unit_rows is None else unit_rows,
        conditions() if condition_rows is None else condition_rows,
        observations(repeats) if captures is None else captures,
        repeat_ids=repeats,
        baseline_condition_id="baseline",
        evaluator_version="fixture:v1",
        config=config,
    )


def select(rows, **keys):
    """Read a single declared report stratum without reproducing its calculations."""
    found = [row for row in rows if all(row.get(k) == v for k, v in keys.items())]
    assert len(found) == 1
    return found[0]


def test_default_off_never_inspects_private_inputs():
    """A disabled experiment cannot consume labels or saved model outputs."""
    assert public_transfer_prompt(object(), object()) is None
    assert (
        evaluate_knowledge_transfer(
            None, None, None, repeat_ids=(), baseline_condition_id=None, evaluator_version=None
        )
        is None
    )


def test_public_projection_contains_only_the_authored_question():
    """A model must never receive the evaluator's truth, cohort or source references."""
    assert public_transfer_prompt(units()[2], "target", enabled=True) == "What is Talvek's hue?"
    assert public_transfer_prompt(units()[2], "source", enabled=True) == 'LOOKUP("Talvek", "hue")'
    with pytest.raises(ValueError):
        replace(units()[2], target_question="Is Talvek's hue SAFFRON?")
    with pytest.raises(ValueError):
        replace(units()[2], source_question="LOOKUP saffron")
    # This simple guard does not pretend to detect semantic hints or coded leaks.
    hinted = replace(units()[2], target_question="Name Talvek's spice-colored hue.")
    assert public_transfer_prompt(hinted, "target", enabled=True) == hinted.target_question


def test_source_recall_target_access_and_application_have_separate_metrics():
    """Source reproduction cannot masquerade as full retrieval or usable application."""
    result = run()
    source = select(result["summaries"], condition_id="candidate", cohort="test", channel="source")
    target = select(result["summaries"], condition_id="candidate", cohort="test", channel="target")
    application = select(
        result["summaries"], condition_id="candidate", cohort="test", channel="application"
    )
    assert (source["planned"], source["correct"]) == (4, 4)
    assert (target["correct"], target["wrong"]) == (2, 2)
    assert target["correct_attainment"] == 0.5
    assert application["correct_attainment"] == 0.5
    access = select(result["access_pairs"], condition_id="candidate", cohort="test")
    assert access["source_target_attainment_gap"] == 0.5
    assert access["source_correct_target_correct"] == 2
    assert access["source_correct_target_wrong"] == 2
    assert access["target_accuracy_given_source_correct"] == 0.5
    assert result["knowledge_acquisition_verified"] is False
    assert result["authority_granted"] is False and result["optimizer_eligible"] is False
    with pytest.raises(ValueError):
        require_training_eligible(result)


def test_missing_abstained_and_failed_responses_keep_the_full_roster():
    """Dropping difficult questions cannot improve the planned-cell attainment metric."""
    captures = tuple(
        o
        for o in observations()
        if not (o.condition_id == "candidate" and o.unit_id == "t1" and o.channel == "target")
    )
    captures = tuple(
        (
            replace(o, status="abstained", final_answer=None)
            if (
                o.condition_id == "candidate"
                and o.unit_id == "t2"
                and o.channel == "target"
                and o.repeat_id == "r0"
            )
            else o
        )
        for o in captures
    )
    captures = tuple(
        (
            replace(o, status="error", final_answer=None)
            if (
                o.condition_id == "candidate"
                and o.unit_id == "t2"
                and o.channel == "target"
                and o.repeat_id == "r1"
            )
            else o
        )
        for o in captures
    )
    result = run(captures)
    row = select(result["summaries"], condition_id="candidate", cohort="test", channel="target")
    assert (row["planned"], row["missing"], row["abstained"], row["errors"]) == (4, 2, 1, 1)
    assert row["correct_attainment"] == 0 and row["accuracy_when_answered"] is None
    paired = select(
        result["comparisons"], condition_id="candidate", cohort="test", channel="target"
    )
    assert paired["estimate"]["expected_count"] == 4
    assert paired["estimate"]["matched_count"] == 2
    assert "incomplete_pairs" in paired["estimate"]["reasons"]


def test_repeats_do_not_create_independent_entity_clusters():
    """Use the existing paired estimator with entity clustering and repeat averaging."""
    config = TransferProbeConfig(
        enabled=True,
        dependencies_known=True,
        resampling_policy=ResamplingPolicy(0, 100, minimum_clusters=2),
    )
    first = select(
        run(config=config)["comparisons"], condition_id="candidate", cohort="test", channel="target"
    )
    more = select(
        run(repeats=("r0", "r1", "r2", "r3"), config=config)["comparisons"],
        condition_id="candidate",
        cohort="test",
        channel="target",
    )
    assert first["estimate"]["raw_delta"] == 0.5
    assert first["estimate"]["cluster_ids"] == ["Drez", "Talvek"]
    assert first["estimate"]["cluster_means"] == more["estimate"]["cluster_means"]
    assert first["estimate"]["interval"] is not None
    default = select(
        run()["comparisons"], condition_id="candidate", cohort="test", channel="target"
    )
    assert default["estimate"]["interval"] is None
    assert "unknown_dependencies" in default["estimate"]["reasons"]


def test_negative_effects_are_retained_and_do_not_trigger_any_admission():
    """A worse candidate cannot be clipped into a nonnegative learning claim."""
    captures = tuple(
        (
            replace(o, final_answer="jade")
            if o.unit_id == "t2" and o.condition_id == "baseline" and o.channel == "target"
            else o
        )
        for o in observations()
    )
    captures = tuple(
        (
            replace(o, final_answer="wrong")
            if o.condition_id == "candidate" and o.channel == "target"
            else o
        )
        for o in captures
    )
    estimate = select(
        run(captures)["comparisons"], condition_id="candidate", cohort="test", channel="target"
    )["estimate"]
    assert estimate["raw_delta"] == -0.5


def test_source_acquisition_can_include_test_facts_but_qa_alignment_cannot():
    """Intentional source exposure differs from teaching held-out target QA answers."""
    assert run()["mode"] == "text_proxy"
    baseline, candidate = conditions()
    aligned = replace(
        candidate,
        alignment_manifest_digest=content_digest("alignment"),
        alignment_entities=("Alven",),
    )
    run(condition_rows=(baseline, aligned))
    contaminated = replace(aligned, alignment_entities=("Talvek",))
    with pytest.raises(ValueError):
        run(condition_rows=(baseline, contaminated))
    with pytest.raises(ValueError):
        run(unit_rows=units() + (replace(units()[2], unit_id="alias", cohort="alignment"),))


@pytest.mark.parametrize("kind", ("duplicate", "unknown", "model", "base"))
def test_mismatched_cells_or_checkpoint_controls_fail_closed(kind):
    """Saved outputs must belong to the exact declared condition and evaluation roster."""
    captures = observations()
    condition_rows = conditions()
    if kind == "duplicate":
        captures += (captures[0],)
    elif kind == "unknown":
        captures = (replace(captures[0], repeat_id="not_planned"),) + captures[1:]
    elif kind == "model":
        captures = (replace(captures[0], system=conditions()[1].system),) + captures[1:]
    else:
        condition_rows = (
            replace(condition_rows[0], checkpoint_digest=content_digest("changed")),
            condition_rows[1],
        )
    with pytest.raises(ValueError):
        run(captures, condition_rows=condition_rows)


def test_strict_normalization_preserves_internal_numbers_and_requires_final_answers():
    """Case/space/terminal periods normalize; verbose text or a different number does not."""
    captures = tuple(
        (
            replace(o, final_answer="  SaFfRoN 。  ")
            if o.unit_id == "t1" and o.condition_id == "candidate" and o.channel == "target"
            else o
        )
        for o in observations()
    )
    row = select(
        run(captures)["summaries"], condition_id="candidate", cohort="test", channel="target"
    )
    assert row["correct"] == 2
    captures = tuple(
        (
            replace(o, final_answer="saffron because it sounds right")
            if o.unit_id == "t1" and o.condition_id == "candidate" and o.channel == "target"
            else o
        )
        for o in captures
    )
    row = select(
        run(captures)["summaries"], condition_id="candidate", cohort="test", channel="target"
    )
    assert row["correct"] == 0


def test_entity_spelling_alias_cannot_cross_the_alignment_boundary():
    """Case/space variants must not disguise QA alignment for a held-out entity."""
    baseline, candidate = conditions()
    alias = replace(
        units()[2], unit_id="alias", entity_id="TALVEK", attribute="dispatch", cohort="alignment"
    )
    aligned = replace(
        candidate,
        alignment_manifest_digest=content_digest("alignment"),
        alignment_entities=("TALVEK",),
    )
    with pytest.raises(ValueError):
        run(unit_rows=units() + (alias,), condition_rows=(baseline, aligned))


def test_numeric_punctuation_is_not_erased_from_reference_answers():
    """Seven-point-two is not the same learned relation as seventy-two."""
    changed = tuple(replace(u, value="7.2") if u.unit_id == "t1" else u for u in units())
    captures = tuple(
        (
            replace(o, final_answer="72")
            if o.condition_id == "candidate" and o.unit_id == "t1" and o.channel == "target"
            else o
        )
        for o in observations()
    )
    summary = select(
        run(captures, unit_rows=changed)["summaries"],
        condition_id="candidate",
        cohort="test",
        channel="target",
    )
    assert summary["correct"] == 0
    captures = tuple(
        (
            replace(o, final_answer="7.2")
            if o.condition_id == "candidate" and o.unit_id == "t1" and o.channel == "target"
            else o
        )
        for o in captures
    )
    summary = select(
        run(captures, unit_rows=changed)["summaries"],
        condition_id="candidate",
        cohort="test",
        channel="target",
    )
    assert summary["correct"] == 2


def test_an_entirely_missing_capture_set_cannot_shrink_the_study():
    """An empty capture set keeps every planned opportunity and no invented matches."""
    result = run(())
    assert len(result["rows"]) == 48
    assert all(row["status"] == "missing" for row in result["rows"])
    estimate = select(
        result["comparisons"], condition_id="candidate", cohort="test", channel="target"
    )["estimate"]
    assert estimate["raw_delta"] is None and estimate["matched_count"] == 0


def test_wire_records_reject_coercion_authority_and_hidden_labels():
    """Unknown fields and generic iterable coercion cannot alter a saved study."""
    for record in (*units(), *conditions(), observations()[0], CFG):
        assert type(record).from_dict(json.loads(json.dumps(record.to_dict()))) == record
        wire = record.to_dict()
        wire["training_eligibility"] = "HIDDEN_EVAL"
        with pytest.raises(ValueError):
            type(record).from_dict(wire)
    for value in ({"t1": True}, "t1"):
        wire = conditions()[1].to_dict()
        wire["acquired_unit_ids"] = value
        with pytest.raises(ValueError):
            TransferCondition.from_dict(wire)
    wire = observations()[0].to_dict()
    wire["verified_correct"] = True
    with pytest.raises(ValueError):
        TransferObservation.from_dict(wire)


def test_cost_receipts_remain_separate_by_unit_basis_and_currency():
    """Unknown costs are not zeros, and estimated tokens cannot become billed money."""
    captures = list(observations())
    captures[0] = replace(
        captures[0],
        costs=(CostEntry(3.0, "tokens", "estimated"), CostEntry(None, "money", "billed", "USD")),
    )
    costs = run(tuple(captures))["costs"]
    tokens = select(costs, condition_id="baseline", unit="tokens", basis="estimated", currency=None)
    money = select(costs, condition_id="baseline", unit="money", basis="billed", currency="USD")
    assert tokens["total"] == 3.0 and tokens["known_values"] == 1
    assert money["total"] is None and money["unknown_values"] == 1


def test_partially_known_receipts_do_not_claim_a_complete_total():
    """A known subtotal cannot turn an unknown receipt into zero resource use."""
    captures = list(observations())
    captures[0] = replace(
        captures[0],
        costs=(CostEntry(3.0, "tokens", "estimated"), CostEntry(None, "tokens", "estimated")),
    )
    summary = select(
        run(tuple(captures))["costs"],
        condition_id="baseline",
        unit="tokens",
        basis="estimated",
        currency=None,
    )
    assert summary["total"] is None
    assert summary["known_subtotal"] == 3.0
    assert summary["unknown_values"] == 1
