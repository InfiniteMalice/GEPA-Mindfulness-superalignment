"""Study overlays compile to existing provenance contracts without executing runtimes."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_records import (
    DatasetManifest,
    ImprovementProtocol,
    MetricSpec,
    ResamplingPolicy,
    SystemConfig,
)
from evaluation.research_studies import prepare_research_study
from evaluation.research_study_records import STUDIES, StudyConfiguration, StudyRuntime
from gepa_mindfulness.training.eligibility import require_training_eligible

# isort: split
from test_improvement_records import DIGEST, EVALUATOR, REF, case


def configuration(study_id="A", configured=2):
    runtimes = tuple(
        StudyRuntime(
            condition_id=name,
            availability="configured" if i < configured else "unsupported",
            reason="Host fixture declaration" if i < configured else "No external runtime",
            system=(
                SystemConfig(f"model:{i}", "harness:v1", content_digest(name))
                if i < configured
                else None
            ),
            runtime_version="software-fixture-v1" if i < configured else None,
            runtime_config_digest=DIGEST if i < configured else None,
            parent_checkpoint_digest=DIGEST if i < configured else None,
            source_refs=(REF,) if i < configured else (),
        )
        for i, name in enumerate(STUDIES[study_id])
    )
    return StudyConfiguration(
        study_id=study_id,
        runtimes=runtimes,
        manifest=DatasetManifest(
            "fixture",
            (
                case("selection"),
                case("audit-a", "independent_audit"),
                case("audit-b", "independent_audit"),
                case("final", "final_test"),
            ),
            (REF,),
            dependencies_known=True,
        ),
        metrics=(
            MetricSpec(
                "success",
                "research-study-observation-v1",
                "verified_task_success",
                content_digest(EVALUATOR.__dict__),
                "fraction",
            ),
        ),
        evaluator=EVALUATOR,
        evidence_refs=(REF,),
        seeds=(7, 19),
        repeats=2,
        inference_budget_digest=DIGEST,
        training_budget_digest=DIGEST,
        parent_checkpoint_digest=DIGEST,
        training_data_digest=DIGEST,
        protected_manifest_digest=DIGEST,
        source_revision="fixture-revision-v1",
        policy=ResamplingPolicy(41, resamples=100, minimum_clusters=2),
    )


def test_exact_six_rosters_cover_master_comparisons():
    assert tuple(STUDIES) == ("A", "B", "C", "D", "E", "F")
    assert tuple(map(len, STUDIES.values())) == (6, 7, 6, 4, 6, 4)
    assert STUDIES["A"][-1] == "locd_actionopd_sst"
    assert STUDIES["B"][-1] == "vpo_knowgpo_prm4rl"
    assert STUDIES["C"][-1] == "deloop_peo_memory_locd"
    assert STUDIES["D"][-1] == "lev_native_depth"
    assert STUDIES["E"][-1] == "autosft_rsiflow_harness_evolution"
    assert STUDIES["F"][-1] == "hybrid_memory"
    with pytest.raises(TypeError):
        STUDIES["A"] = ()


@pytest.mark.parametrize("enabled", [False, None, 1, "yes"])
def test_default_off_precedes_inputs(enabled):
    with pytest.raises(ValueError, match="enabled"):
        prepare_research_study(None, enabled=enabled)


def test_compilation_reuses_existing_slots_with_shared_seed_budget_and_versions():
    config = configuration()
    plan = prepare_research_study(config, enabled=True)
    protocol = ImprovementProtocol.from_dict(plan["protocol"])
    assert plan["runtime_qualification"] == "not_assessed"
    assert plan["execute_runtime"] is False
    assert plan["confers_authority"] is False
    assert len(protocol.candidates) == 1
    assert len(protocol.slots) == 4 * 2 * 2 * 2
    assert {s.seed for s in protocol.slots} == {7, 19}
    assert {s.repeat_id for s in protocol.slots} == {0, 1}
    assert {s.seed_policy for s in protocol.slots} == {"shared"}
    assert {s.budget_digest for s in protocol.slots} == {DIGEST}
    for slot in protocol.slots:
        assert str(slot.seed) in slot.condition_id
    assert protocol.candidates[0].baseline == config.runtimes[0].system
    assert protocol.candidates[0].candidate == config.runtimes[1].system
    assert len(plan["availability"]) == 6
    assert sum(r["status"] == "unsupported" for r in plan["availability"]) == 4
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        require_training_eligible(plan)
    assert prepare_research_study(config, enabled=True) == plan
    assert StudyConfiguration.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("study_id", ["A", "B", "C", "D", "E", "F"])
def test_unconfigured_arms_are_visible_without_fabricated_protocol(study_id):
    config = configuration(study_id, configured=0)
    plan = prepare_research_study(config, enabled=True)
    assert plan["protocol"] is None
    assert plan["status"] == "not_run"
    assert len(plan["availability"]) == len(STUDIES[study_id])
    assert all(row["status"] == "unsupported" for row in plan["availability"])
    assert "empirical_improvement" not in plan


@pytest.mark.parametrize(
    "fault",
    ["roster", "duplicate", "checkpoint", "no_baseline", "seeds", "repeat", "lineage", "capacity"],
)
def test_incomplete_incompatible_or_oversize_recipe_is_rejected(fault):
    value = configuration()
    with pytest.raises(ValueError):
        if fault == "roster":
            value = replace(value, runtimes=value.runtimes[:-1])
        if fault == "duplicate":
            value = replace(value, runtimes=value.runtimes[:-1] + (value.runtimes[0],))
        if fault == "checkpoint":
            value = replace(
                value,
                runtimes=(
                    value.runtimes[0],
                    replace(value.runtimes[1], parent_checkpoint_digest="f" * 64),
                )
                + value.runtimes[2:],
            )
        if fault == "no_baseline":
            value = replace(
                value, runtimes=(configuration(configured=0).runtimes[0],) + value.runtimes[1:]
            )
        if fault == "seeds":
            value = replace(value, seeds=(7, 7))
        if fault == "repeat":
            value = replace(value, repeats=0)
        if fault == "lineage":
            value = replace(
                value,
                manifest=replace(
                    value.manifest,
                    cases=(
                        case("one", family_id="shared"),
                        case("two", "independent_audit", family_id="shared"),
                    ),
                ),
            )
        if fault == "capacity":
            value = replace(
                value,
                seeds=tuple(range(16)),
                repeats=8,
                manifest=replace(value.manifest, cases=tuple(case(str(i)) for i in range(64))),
            )
        prepare_research_study(value, enabled=True)


def test_no_runtime_executes_from_names_or_version_strings():
    config = configuration()
    original = config.to_dict()
    assert prepare_research_study(config, enabled=True)["execute_runtime"] is False
    assert config.to_dict() == original
