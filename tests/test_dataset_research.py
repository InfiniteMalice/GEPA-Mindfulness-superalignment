"""Immutable recipe preparation must not grant dataset or verifier authority."""

# Standard library
import copy
import json
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest

# Local
from curriculum_fixtures import curation_inputs
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import DatasetCase, DatasetManifest, record_digest
from synthetic_data.curriculum_records import EvidenceCheck, GenerationReview, curriculum_digest
from synthetic_data.dataset_research import (
    DATASET_CHECKS,
    DatasetPool,
    DatasetRecipe,
    DatasetRevision,
    DatasetRole,
    DatasetRow,
    RecipeItem,
    ResearchRound,
    prepare_dataset_revision,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    base = curation_inputs()
    refs = base["episodes"][0].evidence_refs
    records = [
        json.loads(line)
        for line in (ROOT / "data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl")
        .read_text()
        .splitlines()[:2]
    ]
    rows = tuple(
        DatasetRow(
            case=DatasetCase(
                r["id"],
                content_digest(r),
                "family-" + r["id"],
                "synthetic_training",
                evaluated=False,
            ),
            record_json=canonical_json(r),
            producer=base["episodes"][0].system,
            evidence_refs=refs,
        )
        for r in records
    )
    pool = DatasetPool(
        "1",
        DatasetManifest("pool", tuple(r.case for r in rows), refs, dependencies_known=True),
        rows,
    )
    contract = ResearchRound(
        base["episodes"][0].system,
        content_digest("trainer"),
        content_digest("evaluation"),
        10000,
        10,
        1,
    )
    before = DatasetRecipe(
        "1", record_digest(contract), tuple(RecipeItem(r.case.case_id, 1, 100) for r in rows)
    )
    recipe = replace(
        before, version="2", items=(replace(before.items[0], repetitions=2), before.items[1])
    )
    roles = tuple(
        DatasetRole(role, role + "-actor", "fixture-model", content_digest(role), refs)
        for role in (
            "meta",
            "execution",
            "analysis_data",
            "analysis_failures",
            "analysis_dynamics",
            "analysis_history",
        )
    )
    proposal = DatasetRevision(
        "revision",
        "rebalancing",
        record_digest(pool),
        record_digest(before),
        record_digest(contract),
        (),
        recipe,
        roles,
        refs,
    )
    review = GenerationReview(
        candidate_digest=record_digest(proposal),
        policy_digest=curriculum_digest(base["policy"]),
        status="verified",
        human_required=False,
        checks=tuple(
            EvidenceCheck(name=k, status="pass", reason="fixture", evidence_refs=refs)
            for k in DATASET_CHECKS
        ),
        evidence_refs=refs,
    )
    return dict(
        pool=pool,
        previous=before,
        proposal=proposal,
        contract=contract,
        review=review,
        policy=base["policy"],
    )


def run(inputs=None, authenticate=lambda p: True):
    return prepare_dataset_revision(
        **(inputs or fixture()), authenticate=authenticate, enabled=True
    )


def rebind(data):
    data["review"] = replace(data["review"], candidate_digest=record_digest(data["proposal"]))
    return data


def test_reviewed_preparation_preserves_originals_and_never_admits_training():
    data = fixture()
    before = copy.deepcopy({k: v.to_dict() for k, v in data.items()})
    report = run(data)
    assert report["disposition"] == "reviewed_candidate"
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["optimizer_input"] is False and report["confers_authority"] is False
    assert report["candidate_pool"]["rows"] == data["pool"].to_dict()["rows"]
    assert report["recipe_totals"] == {"examples": 3, "tokens": 300}
    assert before == {k: v.to_dict() for k, v in data.items()}
    assert report == run(data)
    assert report["result_digest"] == content_digest(
        {k: v for k, v in report.items() if k != "result_digest"}
    )


def test_disabled_before_callback_or_input_access():
    with pytest.raises(ValueError, match="enabled"):
        prepare_dataset_revision(
            pool=None, previous=None, proposal=None, contract=None, review=None, policy=None
        )


@pytest.mark.parametrize("callback", [None, lambda p: 1, lambda p: False])
def test_missing_or_non_boolean_host_authentication_withholds(callback):
    report = run(authenticate=callback)
    assert report["disposition"] == "withheld"
    assert report["candidate_pool"] is None


def test_callback_mutation_and_exception_fail_closed():
    def mutate(p):
        p["proposal"]["operation"] = "pruning"
        return True

    def error(p):
        raise RuntimeError("fixture")

    assert run(authenticate=mutate)["candidate_pool"] is None
    assert run(authenticate=error)["candidate_pool"] is None


@pytest.mark.parametrize(
    "field", ["parent_pool_digest", "parent_recipe_digest", "round_contract_digest"]
)
def test_stale_parent_or_round_rejected_before_callback(field):
    data = fixture()
    data["proposal"] = replace(data["proposal"], **{field: content_digest("stale")})
    with pytest.raises(ValueError, match="binding"):
        run(data, lambda p: pytest.fail("must not authenticate stale inputs"))


@pytest.mark.parametrize("change", ["missing", "stale", "failed", "unknown", "human"])
def test_review_checks_cannot_be_self_certified(change):
    data = fixture()
    review = data["review"]
    if change == "missing":
        data["review"] = None
    elif change == "stale":
        data["review"] = replace(review, candidate_digest=content_digest("stale"))
    elif change == "human":
        data["review"] = replace(review, human_required=True)
    else:
        checks = (replace(review.checks[0], status="fail" if change == "failed" else "unknown"),)
        data["review"] = replace(review, checks=checks + review.checks[1:])
    assert run(data)["candidate_pool"] is None


def test_reviewer_is_not_a_proposal_role():
    data = fixture()
    roles = (replace(data["proposal"].roles[0], actor_id=data["policy"].evaluator.evaluator_id),)
    data["proposal"] = replace(data["proposal"], roles=roles + data["proposal"].roles[1:])
    assert run(rebind(data))["candidate_pool"] is None


@pytest.mark.parametrize("purpose", ["final_test", "independent_audit", "ood_combinations"])
def test_protected_source_cannot_be_relabelled_by_descendant(purpose):
    data = fixture()
    pool = data["pool"]
    parent = replace(
        pool.manifest.cases[0], case_id="heldout-parent", family_id="heldout", purpose=purpose
    )
    child = replace(pool.rows[0].case, parent_ids=(parent.case_id,))
    rows = (replace(pool.rows[0], case=child),) + pool.rows[1:]
    cases = tuple(r.case for r in rows) + (parent,)
    if purpose == "ood_combinations":
        parent = replace(parent, combination_digest=content_digest({"heldout": True}))
        cases = tuple(r.case for r in rows) + (parent,)
    manifest = replace(
        pool.manifest, cases=cases, ood_definitions=(canonical_json({"heldout": True}),)
    )
    with pytest.raises(ValueError, match="partition|protected"):
        data["pool"] = replace(pool, manifest=manifest, rows=rows)
        run(data)


def test_nested_holdout_metadata_is_not_decontamination():
    data = fixture()
    row = data["pool"].rows[0]
    record = json.loads(row.record_json)
    record["metadata"] = {"encoded": '{"training_eligibility":"HIDDEN_EVAL"}'}
    with pytest.raises(ValueError, match="protected|held.out"):
        row = replace(
            row,
            case=replace(row.case, content_digest=content_digest(record)),
            record_json=canonical_json(record),
        )
        pool = data["pool"]
        rows = (row,) + pool.rows[1:]
        data["pool"] = replace(
            pool, rows=rows, manifest=replace(pool.manifest, cases=tuple(r.case for r in rows))
        )
        run(data)


def test_rich_validation_and_content_hash_are_reused():
    row = fixture()["pool"].rows[0]
    with pytest.raises(ValueError, match="digest"):
        replace(row, record_json='{"id":"bogus"}')
    with pytest.raises(ValueError, match="rich|schema"):
        replace(
            row,
            case=replace(row.case, case_id="bogus", content_digest=content_digest({"id": "bogus"})),
            record_json='{"id":"bogus"}',
        )


@pytest.mark.parametrize("op", ["pruning", "curriculum_adjustment"])
def test_selection_and_ordering_leave_pool_bytes_immutable(op):
    data = fixture()
    items = data["previous"].items
    after = items[:1] if op == "pruning" else tuple(reversed(items))
    data["proposal"] = replace(
        data["proposal"], operation=op, recipe=replace(data["proposal"].recipe, items=after)
    )
    report = run(rebind(data))
    assert report["disposition"] == "reviewed_candidate"
    assert report["candidate_pool"]["rows"] == data["pool"].to_dict()["rows"]


@pytest.mark.parametrize(
    "op", ["repair", "replacement", "targeted_synthesis", "robustness_augmentation"]
)
def test_new_rows_extend_pool_without_overwriting_original(op):
    data = fixture()
    old = data["pool"].rows[0]
    record = json.loads(old.record_json)
    record["id"], record["version"] = "derived-001", "2"
    record["scenario"]["summary"] += " Additional public evidence."
    parents = () if op == "targeted_synthesis" else (old.case.case_id,)
    new = replace(
        old,
        case=replace(
            old.case,
            case_id=record["id"],
            content_digest=content_digest(record),
            parent_ids=parents,
        ),
        record_json=canonical_json(record),
    )
    roles = data["proposal"].roles + (
        replace(data["proposal"].roles[0], role="synthesis", actor_id="synth-actor"),
    )
    existing = data["previous"].items
    item = RecipeItem(new.case.case_id, 1, 110)
    items = (item, existing[1]) if op in ("repair", "replacement") else existing + (item,)
    data["proposal"] = replace(
        data["proposal"],
        operation=op,
        additions=(new,),
        roles=roles,
        recipe=replace(data["proposal"].recipe, items=items),
    )
    report = run(rebind(data))
    assert report["disposition"] == "reviewed_candidate"
    assert report["candidate_pool"]["rows"][:2] == data["pool"].to_dict()["rows"]
    assert report["candidate_pool"]["rows"][2] == new.to_dict()


def test_operation_labels_do_not_hide_unrelated_changes():
    data = fixture()
    data["proposal"] = replace(data["proposal"], operation="pruning")
    with pytest.raises(ValueError, match="operation"):
        run(rebind(data))


def test_fixed_budget_and_token_count_binding_precede_review():
    data = fixture()
    items = data["proposal"].recipe.items
    for modified in (replace(items[0], repetitions=100), replace(items[0], token_count=99)):
        data["proposal"] = replace(
            data["proposal"], recipe=replace(data["proposal"].recipe, items=(modified,) + items[1:])
        )
        with pytest.raises(ValueError, match="budget|token"):
            run(rebind(data), lambda p: pytest.fail("invalid budget/count must not reach host"))


@pytest.mark.parametrize("value", [True, -1, 0, 1.5, float("inf")])
def test_recipe_counts_require_positive_exact_integers(value):
    with pytest.raises(ValueError):
        RecipeItem("row", value, 10)


def test_versioned_records_reject_unknown_fields_and_round_trip():
    data = fixture()
    for record in data.values():
        if record is not None:
            assert type(record).from_dict(record.to_dict()) == record
    wire = data["proposal"].to_dict()
    assert wire["schema_version"] == "dataset-revision-v1"
    with pytest.raises(ValueError):
        DatasetRevision.from_dict({**wire, "authorization": True})
    with pytest.raises(ValueError):
        DatasetRevision.from_dict({**wire, "schema_version": "dataset-revision-v99"})


def test_complete_report_is_bounded_before_host_review():
    data = fixture()
    pool = data["pool"]
    record = json.loads(pool.rows[0].record_json)
    record["scenario"]["summary"] = "x" * 750_000
    row = replace(
        pool.rows[0],
        record_json=canonical_json(record),
        case=replace(pool.rows[0].case, content_digest=content_digest(record)),
    )
    rows = (row,) + pool.rows[1:]
    data["pool"] = replace(
        pool, rows=rows, manifest=replace(pool.manifest, cases=tuple(r.case for r in rows))
    )
    data["proposal"] = replace(data["proposal"], parent_pool_digest=record_digest(data["pool"]))
    with pytest.raises(ValueError, match="4 MiB"):
        run(rebind(data), lambda p: pytest.fail("oversized full report must not reach review"))


def test_host_sees_the_exact_complete_review_and_derivation():
    data = fixture()
    seen = []
    report = run(data, lambda p: seen.append(p) or True)
    assert seen == [report["inputs"]]
    assert seen[0]["review"] == data["review"].to_dict()
    assert seen[0]["candidate_pool"] == report["candidate_pool"]
    assert seen[0]["contract"] == data["contract"].to_dict()


@pytest.mark.parametrize("field", ["trainer_digest", "evaluation_protocol_digest"])
def test_fixed_round_configuration_cannot_drift(field):
    data = fixture()
    data["contract"] = replace(data["contract"], **{field: content_digest("different")})
    with pytest.raises(ValueError, match="binding"):
        run(data)


def test_reviewed_report_is_not_reusable_as_an_authorization_record():
    report = run()
    with pytest.raises(ValueError):
        GenerationReview.from_dict(report)


def test_historical_version_identity_cannot_be_reused_across_multiple_revisions():
    data = fixture()
    first = run(data)
    data["pool"] = DatasetPool.from_dict(first["candidate_pool"])
    data["previous"] = DatasetRecipe.from_dict(first["recipe"])
    next_items = (replace(data["previous"].items[0], repetitions=3), data["previous"].items[1])
    data["proposal"] = replace(
        data["proposal"],
        proposal_id="revision-2",
        parent_pool_digest=record_digest(data["pool"]),
        parent_recipe_digest=record_digest(data["previous"]),
        recipe=replace(data["previous"], version="1", items=next_items),
    )
    with pytest.raises(ValueError, match="version"):
        run(rebind(data), lambda p: pytest.fail("reused immutable version must precede review"))
    data["proposal"] = replace(
        data["proposal"], recipe=replace(data["proposal"].recipe, version="3")
    )
    second = run(rebind(data))
    assert second["candidate_pool"]["prior_versions"] == ["1", "2"]


def addition_inputs(op):
    data = fixture()
    pool = data["pool"]
    extra_record = json.loads(pool.rows[1].record_json)
    extra_record["id"] = "source-extra"
    extra = replace(
        pool.rows[1],
        record_json=canonical_json(extra_record),
        case=replace(
            pool.rows[1].case,
            case_id=extra_record["id"],
            content_digest=content_digest(extra_record),
        ),
    )
    rows = pool.rows + (extra,)
    data["pool"] = replace(
        pool, rows=rows, manifest=replace(pool.manifest, cases=tuple(r.case for r in rows))
    )
    data["previous"] = replace(
        data["previous"], items=data["previous"].items + (RecipeItem(extra.case.case_id, 1, 100),)
    )
    record = json.loads(pool.rows[0].record_json)
    record["id"], record["version"] = "derived-001", "2"
    parents = () if op == "targeted_synthesis" else (pool.rows[0].case.case_id,)
    new = replace(
        pool.rows[0],
        record_json=canonical_json(record),
        case=replace(
            pool.rows[0].case,
            case_id=record["id"],
            content_digest=content_digest(record),
            parent_ids=parents,
        ),
    )
    roles = data["proposal"].roles + (
        replace(data["proposal"].roles[0], role="synthesis", actor_id="synth-actor"),
    )
    item = RecipeItem(new.case.case_id, 1, 110)
    old_items = data["previous"].items
    items = (item,) + old_items[1:] if op in ("repair", "replacement") else old_items + (item,)
    data["proposal"] = replace(
        data["proposal"],
        operation=op,
        additions=(new,),
        roles=roles,
        parent_pool_digest=record_digest(data["pool"]),
        parent_recipe_digest=record_digest(data["previous"]),
        recipe=replace(data["proposal"].recipe, items=items),
    )
    return rebind(data)


@pytest.mark.parametrize(
    "op", ["targeted_synthesis", "robustness_augmentation", "repair", "replacement"]
)
def test_data_edits_cannot_silently_perform_curriculum_reordering(op):
    data = addition_inputs(op)
    assert run(data)["disposition"] == "reviewed_candidate"
    data["proposal"] = replace(
        data["proposal"],
        recipe=replace(
            data["proposal"].recipe, items=tuple(reversed(data["proposal"].recipe.items))
        ),
    )
    with pytest.raises(ValueError, match="operation"):
        run(rebind(data), lambda p: pytest.fail("mixed reordering must precede review"))


@pytest.mark.parametrize("op", ["repair", "replacement"])
def test_replacing_a_row_preserves_its_sampling_count(op):
    data = addition_inputs(op)
    items = data["proposal"].recipe.items
    data["proposal"] = replace(
        data["proposal"],
        recipe=replace(
            data["proposal"].recipe, items=(replace(items[0], repetitions=2),) + items[1:]
        ),
    )
    with pytest.raises(ValueError, match="operation"):
        run(rebind(data))
