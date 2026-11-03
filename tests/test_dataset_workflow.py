"""Active dataset roles and host admission on authored CPU records."""

# Standard library
import json
import pickle
from dataclasses import replace

# Third-party
import pytest

# isort: split
# Local
from test_dataset_research import fixture

from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import record_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import require_training_eligible
from synthetic_data.curriculum_records import curriculum_digest
from synthetic_data.dataset_research import DatasetRevision, DatasetRole
from synthetic_data.dataset_workflow import DatasetWorkflowConfig, run_dataset_workflow


def setup_workflow(tmp_path):
    data = fixture()
    rows = []
    for row in data["pool"].rows:
        record = json.loads(row.record_json)
        record["metadata"] = record.get("metadata", {}) | {"training_eligibility": "TRAIN"}
        rows.append(
            replace(
                row,
                record_json=canonical_json(record),
                case=replace(row.case, content_digest=content_digest(record)),
            )
        )
    pool = replace(
        data["pool"],
        rows=tuple(rows),
        manifest=replace(data["pool"].manifest, cases=tuple(row.case for row in rows)),
    )
    roles = data["proposal"].roles + (
        DatasetRole(
            "synthesis",
            "synthesis-actor",
            "fixture-model",
            content_digest("synthesis"),
            data["proposal"].evidence_refs,
        ),
    )
    proposal = replace(data["proposal"], parent_pool_digest=record_digest(pool), roles=roles)
    calls = []

    def analysis(name):
        def run(packet):
            calls.append(name)
            return {"summary": "Inspect admitted source coverage and labels."}

        return run

    def plan(packet):
        calls.append("meta")
        return proposal

    def synthesize(packet):
        calls.append("synthesis")
        return ()

    def execute(packet):
        calls.append("execution")
        return proposal

    def verify(packet):
        calls.append("verification")
        candidate = DatasetRevision.from_dict(packet["proposal"])
        return replace(data["review"], candidate_digest=record_digest(candidate))

    config = DatasetWorkflowConfig(
        enabled=True,
        workflow_id="dataset-workflow:1",
        roles=roles,
        policy_digest=curriculum_digest(data["policy"]),
        trainer_digest=data["contract"].trainer_digest,
        admission_authority_id="host-data:v1",
        seed=data["contract"].seed,
    )
    args = dict(
        config=config,
        analyses={
            name: analysis(name)
            for name in (
                "analysis_data",
                "analysis_failures",
                "analysis_dynamics",
                "analysis_history",
            )
        },
        plan=plan,
        synthesize=synthesize,
        execute=execute,
        verify=verify,
        render=lambda record: (
            record["scenario"]["summary"],
            record["reward_integrity"]["response_classes"]["grounded_success"]["response"],
        ),
        count_tokens=lambda prompt, response: 100,
        authorize_sources=lambda packet: True,
        authenticate=lambda packet: True,
        authorize_training=lambda packet: True,
        admission_reference=EvidenceReference(
            "host-admission:1", EvidenceSourceKind.EXTERNAL_RECORD
        ),
        output_root=tmp_path / "artifacts",
    )
    return pool, data["previous"], data["contract"], data["policy"], args, calls, proposal


def test_disabled_never_reads_inputs_or_calls_roles():
    assert run_dataset_workflow(None, None, None, None) is None


def test_roles_apply_recipe_and_materialize_only_host_admitted_immutable_data(tmp_path):
    pool, previous, contract, policy, args, calls, _ = setup_workflow(tmp_path)
    original = canonical_json(pool.to_dict())
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert calls == [
        "analysis_data",
        "analysis_failures",
        "analysis_dynamics",
        "analysis_history",
        "meta",
        "synthesis",
        "execution",
        "verification",
    ]
    assert canonical_json(pool.to_dict()) == original
    assert result.dataset is not None
    requests = result.dataset.materialize("train")
    assert len(requests) == 3
    assert requests[0].case_id == requests[1].case_id == pool.rows[0].case.case_id
    assert requests[2].case_id == pool.rows[1].case.case_id
    assert all(request.metadata["training_eligibility"] == "TRAIN" for request in requests)
    for request in requests:
        require_training_eligible(request.metadata)
        assert request.metadata["supervised_response"]
        assert request.metadata["source_sha256"]
    report = result.to_dict()
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["authority_granted"] is False
    assert report["status"] == "dataset_admitted"
    assert report["preparation"]["disposition"] == "reviewed_candidate"
    assert result.dataset.path.is_file()
    with pytest.raises(TypeError):
        pickle.dumps(result.dataset)


@pytest.mark.parametrize("role", ("authenticate", "authorize_training"))
def test_missing_host_approval_withholds_dataset_and_writes_no_artifact(tmp_path, role):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    args[role] = lambda packet: False
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert result.dataset is None
    assert not args["output_root"].exists()


def test_untrusted_sources_are_rejected_before_any_agent(tmp_path):
    pool, previous, contract, policy, args, calls, _ = setup_workflow(tmp_path)
    args["authorize_sources"] = lambda packet: False
    with pytest.raises(ValueError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not calls


def test_misleading_label_rejected_by_independent_review_cannot_be_admitted(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    verify = args["verify"]

    def reject(packet):
        reviewed = verify(packet)
        return replace(
            reviewed,
            checks=tuple(
                replace(check, status="fail") if check.name == "label_fidelity" else check
                for check in reviewed.checks
            ),
        )

    args["verify"] = reject
    admissions = []
    args["authorize_training"] = lambda packet: admissions.append(packet) or True
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert result.dataset is None
    assert not admissions


def test_mutating_analysis_packet_cannot_change_sources_or_continue(tmp_path):
    pool, previous, contract, policy, args, calls, _ = setup_workflow(tmp_path)

    def mutate(packet):
        packet["pool"]["version"] = "rewritten"
        return {"summary": "changed"}

    args["analyses"]["analysis_data"] = mutate
    with pytest.raises(ValueError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not calls
    assert pool.version == "1"


def test_executor_cannot_substitute_a_different_operation(tmp_path):
    pool, previous, contract, policy, args, _, proposal = setup_workflow(tmp_path)
    args["execute"] = lambda packet: replace(proposal, operation="pruning")
    with pytest.raises(ValueError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not args["output_root"].exists()


def test_token_counter_must_match_recipe_declarations(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    args["count_tokens"] = lambda prompt, response: 101
    with pytest.raises(ValueError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not args["output_root"].exists()


def test_materialized_dataset_tampering_invalidates_live_provider(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    result.dataset.path.write_text("{}\n")
    with pytest.raises(ValueError):
        result.dataset.materialize("train")


def test_config_is_closed_and_versioned():
    config = DatasetWorkflowConfig()
    assert DatasetWorkflowConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        DatasetWorkflowConfig.from_dict(config.to_dict() | {"unknown": True})


def test_byte_budget_withholds_before_final_training_authority(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    args["config"] = replace(args["config"], max_output_bytes=1)
    admissions = []
    args["authorize_training"] = lambda packet: admissions.append(packet) or True
    with pytest.raises(ValueError, match="byte budget"):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not admissions
    assert not args["output_root"].exists()


def test_versioned_artifacts_are_not_overwritten(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    first = run_dataset_workflow(pool, previous, contract, policy, **args)
    original = first.dataset.path.read_bytes()
    with pytest.raises(FileExistsError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert first.dataset.path.read_bytes() == original


def test_source_coverage_reports_unassigned_without_inventing_canonical_cases(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    original = args["analyses"]["analysis_data"]
    seen = []

    def inspect(packet):
        seen.append(packet["coverage"])
        return original(packet)

    args["analyses"]["analysis_data"] = inspect
    run_dataset_workflow(pool, previous, contract, policy, **args)
    assert seen[0]["unassigned_rows"] == 2
    assert len(seen[0]["canonical_cases"]) == 17
    assert all(row["rows"] == 0 for row in seen[0]["canonical_cases"])


def generated_revision(pool, previous, args, proposal, operation):
    from evaluation.cases.registry import FRAMEWORK_VERSION, load_case_manifest
    from synthetic_data.dataset_research import RecipeItem

    role = next(role for role in args["config"].roles if role.role == "synthesis")
    source = pool.rows[0]
    raw = json.loads(source.record_json)
    raw.update(id="generated-row:1", version="2.0")
    raw["metadata"] = {
        "training_eligibility": "TRAIN",
        "generator": role.actor_id,
        "source_template": source.case.case_id,
        "review_completed": True,
        "reviewed_by": "human:dataset-reviewer",
        "review_authorization": args["admission_reference"].to_dict(),
        "canonical_case_id": 1,
        "canonical_case_key": load_case_manifest().cases[0].key,
        "case_version": FRAMEWORK_VERSION,
        "stripe": "NONE",
        "stripe_subtype": None,
        "repeat_id": 0,
        "seed": 1,
        "model_version": "fixture-model",
        "harness_version": "h",
    }
    case = replace(
        source.case,
        case_id=raw["id"],
        content_digest=content_digest(raw),
        parent_ids=(source.case.case_id,),
        transformation_ids=("generated:1",),
    )
    row = replace(
        source,
        case=case,
        record_json=canonical_json(raw),
        producer=replace(
            source.producer, model_version=role.model_version, config_digest=role.config_digest
        ),
    )
    new_item = RecipeItem(case.case_id, 1, 100)
    items = (
        (new_item, previous.items[1])
        if operation in {"repair", "replacement"}
        else (*previous.items, new_item)
    )
    planned = replace(proposal, operation=operation, recipe=replace(proposal.recipe, items=items))
    args["plan"] = lambda packet: planned
    args["synthesize"] = lambda packet: (row,)
    args["execute"] = lambda packet: replace(planned, additions=(row,))
    return row


@pytest.mark.parametrize(
    "operation", ("repair", "replacement", "targeted_synthesis", "robustness_augmentation")
)
def test_generated_operations_preserve_sources_and_independently_reviewed_lineage(
    tmp_path, operation
):
    pool, previous, contract, policy, args, _, proposal = setup_workflow(tmp_path)
    generated = generated_revision(pool, previous, args, proposal, operation)
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    requests = result.dataset.materialize("train")
    assert any(request.case_id == generated.case.case_id for request in requests)
    assert len(pool.rows) == 2
    assert all(request.metadata["source_record"]["source_evidence_refs"] for request in requests)


@pytest.mark.parametrize("operation", ("pruning", "curriculum_adjustment"))
def test_non_synthesis_operations_execute_existing_recipe_contracts(tmp_path, operation):
    pool, previous, contract, policy, args, _, proposal = setup_workflow(tmp_path)
    items = previous.items[1:] if operation == "pruning" else previous.items[::-1]
    proposal = replace(proposal, operation=operation, recipe=replace(proposal.recipe, items=items))
    args["plan"] = args["execute"] = lambda packet: proposal
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert [request.case_id for request in result.dataset.materialize("train")] == [
        item.case_id for item in items
    ]


@pytest.mark.parametrize(
    "field,value", (("review_completed", False), ("reviewed_by", "synthesis-actor"))
)
def test_synthesis_cannot_self_certify_or_skip_human_review(tmp_path, field, value):
    pool, previous, contract, policy, args, _, proposal = setup_workflow(tmp_path)
    row = generated_revision(pool, previous, args, proposal, "targeted_synthesis")
    raw = json.loads(row.record_json)
    raw["metadata"][field] = value
    row = replace(
        row,
        record_json=canonical_json(raw),
        case=replace(row.case, content_digest=content_digest(raw)),
    )
    args["synthesize"] = lambda packet: (row,)
    with pytest.raises(ValueError):
        run_dataset_workflow(pool, previous, contract, policy, **args)
    assert not args["output_root"].exists()


def test_admitted_provider_supports_existing_training_resume_mode(tmp_path):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert result.dataset.materialize("resume") == result.dataset.materialize("train")


@pytest.mark.parametrize("separator", ("\u2028", "\u2029"))
def test_jsonl_materialization_preserves_unicode_separators_inside_strings(tmp_path, separator):
    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    args["render"] = lambda record: ("A" + separator + "B", "answer")
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    assert result.dataset.materialize("train")[0].prompt == "A" + separator + "B"


def test_relative_destination_remains_bound_after_working_directory_change(tmp_path, monkeypatch):
    from pathlib import Path

    pool, previous, contract, policy, args, _, _ = setup_workflow(tmp_path)
    monkeypatch.chdir(tmp_path)
    args["output_root"] = Path("artifacts")
    result = run_dataset_workflow(pool, previous, contract, policy, **args)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert len(result.dataset.materialize("resume")) == 3
    assert str(result.dataset.path) == result.to_dict()["dataset_path"]
